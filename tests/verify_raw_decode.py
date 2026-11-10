#!/usr/bin/env python3
"""Compare the Rust lossless raw decoder with rawpy (LibRaw) on ARW files.

Usage: python tests/verify_raw_decode.py photo.ARW [more.ARW | folder ...]
Folders are searched for *.ARW.  Files that are not compression 7 are reported
and checked through read_raw's rawpy fallback.
"""
import sys
import time
from pathlib import Path

import numpy as np
import rawpy

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'base'))
import arw_raw


def files(args):
    for a in args:
        p = Path(a)
        if p.is_dir():
            yield from sorted(p.rglob('*.ARW'))
        else:
            yield p


def main(args):
    n = skipped = 0
    t_rust = t_lib = 0.0
    sizes = {}
    for p in files(args):
        ifd = arw_raw.raw_ifd(p)
        t = time.perf_counter()
        with rawpy.imread(str(p)) as r:
            ref = np.array(r.raw_image, np.uint16, copy=True)
            vis = np.array(r.raw_image_visible, np.uint16, copy=True)
        t_lib += time.perf_counter() - t
        if ifd is None or ifd.compression != 7:
            got = arw_raw.read_raw(p)
            assert np.array_equal(got, vis), f'{p}: rawpy fallback differs'
            skipped += 1
            print(f'{p.name}: compression {None if ifd is None else ifd.compression}, rawpy fallback')
            continue
        t = time.perf_counter()
        full = arw_raw.decode_lossless(p)
        t_rust += time.perf_counter() - t
        if not np.array_equal(full, ref):
            d = full.astype(np.int32) - ref
            raise AssertionError(f'{p}: {np.count_nonzero(d)} differ, first at {tuple(np.argwhere(d)[0])}')
        if not np.array_equal(arw_raw.read_raw(p), vis):
            raise AssertionError(f'{p}: visible crop differs')
        sizes[full.shape] = sizes.get(full.shape, 0) + 1
        n += 1
    print(f'{n} compression-7 files identical to rawpy (raw and visible crop), sizes {sizes}; '
          f'{skipped} other files via rawpy')
    if n:
        print(f'per file: Rust {t_rust / n * 1000:.0f} ms, rawpy read + unpack {t_lib / (n + skipped) * 1000:.0f} ms '
              f'(rawpy also copies two arrays here)')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    main(sys.argv[1:])
