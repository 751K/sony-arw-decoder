#!/usr/bin/env python3
"""Decode many ARWs with reading, decoding and writing overlapped.

    python3 base/sony_batch.py IN [IN ...] --out-dir OUT [--format jpg|tif|png] [--metal ...]

IN may be ARW files or folders (searched recursively); outputs keep each
file's path relative to its folder argument. Decode options are those of
sony_decode.py.

Three things run at once:

- a reader thread loads the next files into memory (sr2's file cache) while
  the current one decodes, so a cold disk read no longer stalls the GPU;
- the main thread decodes one file after another (Metal stages share one GPU
  queue, and the RAW decode already runs alongside the GPU inside decode());
- writer threads encode and write finished images while the next decodes.

Output is identical to decoding each file with sony_decode.py. A file that
fails is reported and skipped; the exit status is 1 if any failed.
"""
from __future__ import annotations

import os
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import sony_decode as SD        # noqa: E402
import sony_export as EXP       # noqa: E402
import sr2                      # noqa: E402

FORMATS = {"jpg": ".jpg", "jpeg": ".jpg", "tif": ".tif", "tiff": ".tif", "png": ".png",
           "hif": ".hif", "heif": ".hif", "heic": ".heic"}


def collect(inputs, out_dir, ext):
    """[(arw, output path)] for files and recursively searched folders, in order."""
    jobs, seen = [], set()
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            files = sorted(f for f in p.rglob("*") if f.suffix.lower() == ".arw" and f.is_file()
                           and not f.name.startswith("._"))
            pairs = [(f, f.relative_to(p)) for f in files]
        elif p.is_file():
            pairs = [(p, Path(p.name))]
        else:
            raise FileNotFoundError(item)
        for f, rel in pairs:
            key = f.resolve()
            if key not in seen:
                seen.add(key)
                jobs.append((f, Path(out_dir) / rel.with_suffix(ext)))
    return jobs


def run(jobs, decode_kw, write_kw, *, read_ahead=2, writers=1, skip_existing=False, log=print):
    """Decode and write every (arw, out) job; returns the list of (arw, error) failures."""
    look = decode_kw.get("look")
    decode_kw = {k: v for k, v in decode_kw.items() if k != "look"}
    if skip_existing:
        jobs = [(f, o) for f, o in jobs if not o.exists()]
    failures = []
    n = len(jobs)
    waits = dict(read=0.0, decode=0.0, write=0.0)     # main-thread time per phase
    t_start = time.perf_counter()

    def read(path):
        sr2.file_bytes(path)        # into the shared cache decode() reads from

    def write(img, out):
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(out.stem + ".partial" + out.suffix)
        EXP.write_image(img, tmp, **write_kw)
        os.replace(tmp, out)        # no half-written file under the final name

    with ThreadPoolExecutor(1, thread_name_prefix="arw-read") as reader, \
            ThreadPoolExecutor(max(1, writers), thread_name_prefix="arw-write") as writer:
        reads = {}
        pending = deque()           # (future, arw, out) in submission order

        def drain(limit):
            while len(pending) > limit:
                fut, arw, out = pending.popleft()
                t = time.perf_counter()
                try:
                    fut.result()
                except Exception as exc:                    # noqa: BLE001
                    failures.append((arw, exc))
                    log(f"FAILED writing {out}: {type(exc).__name__}: {exc}")
                waits["write"] += time.perf_counter() - t

        def schedule_reads(i):
            for j in range(i, min(n, i + 1 + read_ahead)):
                if j not in reads:
                    reads[j] = reader.submit(read, jobs[j][0])

        for i, (arw, out) in enumerate(jobs):
            schedule_reads(i)
            t = time.perf_counter()
            try:
                reads.pop(i).result()
                t1 = time.perf_counter()
                waits["read"] += t1 - t
                img = SD.decode(arw, look, **decode_kw)
            except Exception as exc:                        # noqa: BLE001
                failures.append((arw, exc))
                log(f"[{i + 1}/{n}] FAILED {arw}: {type(exc).__name__}: {exc}")
                continue
            dt = time.perf_counter() - t1
            waits["decode"] += dt
            drain(max(1, writers))  # at most writers + 1 images waiting to be written
            pending.append((writer.submit(write, img, out), arw, out))
            del img
            log(f"[{i + 1}/{n}] {arw.name} decoded in {dt * 1000:.0f} ms")
        drain(0)
    total = time.perf_counter() - t_start
    done = n - len(failures)
    if n:
        per = {k: v / n * 1000 for k, v in waits.items()}
        log(f"{done}/{n} written in {total:.1f} s ({total / n * 1000:.0f} ms per file: "
            f"decode {per['decode']:.0f}, waiting for reads {per['read']:.0f}, "
            f"for writes {per['write']:.0f})")
    return failures


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Decode many ARWs through the full Sony pipeline, "
                                             "with file reading, decoding and writing overlapped")
    ap.add_argument("inputs", nargs="+", help="ARW files and/or folders (searched recursively)")
    ap.add_argument("--out-dir", required=True, help="output folder; subfolders mirror the inputs")
    ap.add_argument("--format", default="jpg", choices=sorted(FORMATS),
                    help="output format (default jpg; tif is 16-bit, about 200 MB per A7M5 image; "
                         "hif/heic is 10-bit HEIF)")
    ap.add_argument("--skip-existing", action="store_true", help="leave files whose output exists")
    ap.add_argument("--read-ahead", type=int, default=2, help="files read ahead of the decoder")
    ap.add_argument("--writers", type=int, default=None,
                    help="image encoding/writing threads (default 2 for HEIF, whose encode is slower "
                         "than a decode, else 1); each holds one more image in memory")
    SD.add_decode_arguments(ap)
    a = ap.parse_args()
    ext = FORMATS[a.format]
    output_bits = SD.output_bits_for(a, ext)
    jobs = collect(a.inputs, a.out_dir, ext)
    failures = run(jobs, SD.decode_kwargs(a, output_bits), SD.write_kwargs(a, output_bits),
                   read_ahead=max(0, a.read_ahead),
                   writers=a.writers if a.writers else (2 if ext in EXP.HEIF_EXTS else 1),
                   skip_existing=a.skip_existing)
    for arw, exc in failures:
        print(f"failed: {arw}: {type(exc).__name__}: {exc}", file=sys.stderr)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
