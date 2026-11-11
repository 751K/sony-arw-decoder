#!/usr/bin/env python3
"""Check that sony_batch writes exactly what single-file decoding writes.

    python tests/verify_batch.py photo.ARW [more.ARW ...]

Every file is written by the overlapped batch (16-bit TIFF and JPEG, with
--metal) and compared with decode() + write_image() run one file at a time;
a copy with a damaged tile checks that a failure is reported and skipped
without stopping the batch.
"""
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'base'))
import arw_raw
import sony_batch
import sony_decode
import sony_export
import tifffile


def main(paths):
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        src = tmp / 'in'
        (src / 'sub').mkdir(parents=True)
        files = []
        for i, p in enumerate(paths):
            dst = src / ('sub' if i % 2 else '') / f'{i}_{Path(p).name}'
            dst.write_bytes(Path(p).read_bytes())
            files.append(dst)
        bad = src / 'sub' / 'damaged.ARW'
        data = bytearray(Path(paths[0]).read_bytes())
        ifd = arw_raw.raw_ifd(paths[0])
        o = int(ifd.offsets[len(ifd.offsets) // 2])
        data[o + 16:o + 144] = b'\xff\x00' * 64
        bad.write_bytes(bytes(data))
        for fmt, bits in (('.tif', 16), ('.jpg', 14), ('.hif', 16)):
            out = tmp / ('out' + fmt)
            jobs = sony_batch.collect([src], out, fmt)
            failures = sony_batch.run(jobs, dict(metal=True, output_bits=bits),
                                      dict(input_bits=bits), log=lambda *a: None)
            if [f for f, _ in failures] != [bad]:
                raise SystemExit(f'{fmt}: expected only the damaged file to fail, got {failures}')
            for f in files:
                got = out / f.relative_to(src).with_suffix(fmt)
                img = sony_decode.decode(str(f), metal=True, output_bits=bits)
                if fmt == '.tif':
                    same = np.array_equal(tifffile.imread(got), img)
                else:                       # JPEG and HEIF encoding are deterministic
                    ref = tmp / ('ref' + fmt)
                    sony_export.write_image(img, ref, input_bits=bits)
                    same = ref.read_bytes() == got.read_bytes()
                if not same:
                    raise SystemExit(f'{fmt}: {f.name} differs from single-file decoding')
            print(f'{fmt}: {len(files)} files identical to single-file decoding; damaged file skipped')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise SystemExit('Usage: python tests/verify_batch.py photo.ARW [more.ARW ...]')
    main(sys.argv[1:])
