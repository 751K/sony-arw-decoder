#!/usr/bin/env python3
"""Compare the Metal RawNR with sony_rawnr.rawnr.

Usage: python tests/verify_metal_rawnr.py photo.ARW   (any ARW; NR need not be on in the file)
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'base'))
import arw_raw
import decoder_params
import metal_rawnr
import sony_rawnr
import sr2


def same(name, expected, actual):
    if not np.array_equal(expected, actual):
        d = np.abs(expected.astype(np.int32) - actual.astype(np.int32))
        raise AssertionError(f'{name}: {np.count_nonzero(d)} differences, max {d.max()}, first at {tuple(np.argwhere(d)[0])}')


def main(path):
    look = sony_rawnr.look_from_sr2(sr2.sr2_subifd(path))
    iso = int(decoder_params.decoder_params(path)['iso'])
    raw = arw_raw.read_raw(path)
    rng = np.random.default_rng(11)
    n = 0
    for iso_k in (100, 400, 800, 1600, 6400):
        for settings in ((0, 0, 0), (40, 30, 0), (-60, -50, 0), (100, -100, 0), (-100, 100, 0)):
            for h, w in ((96, 128), (75, 101)):
                y0, x0 = 2 * rng.integers(0, (raw.shape[0] - h) // 2), 2 * rng.integers(0, (raw.shape[1] - w) // 2)
                crop = np.ascontiguousarray(raw[y0:y0 + h, x0:x0 + w])
                noisy = np.clip(crop.astype(np.int64) + rng.normal(0, 200, crop.shape), 0, 16383).astype(np.uint16)
                for img in (crop, noisy):
                    same(f'ISO {iso_k} settings {settings} {h}x{w}',
                         sony_rawnr.rawnr(img, look, iso=iso_k, settings=settings),
                         metal_rawnr.apply(img, look, iso=iso_k, settings=settings))
                    n += 1
    print(f'{n} crops (5 ISO x 5 NR settings x 2 sizes x real/noisy): identical to NumPy')
    extreme = rng.integers(0, 65536, (64, 80), dtype=np.uint16)
    same('full 16-bit range', sony_rawnr.rawnr(extreme, look, iso=iso), metal_rawnr.apply(extreme, look, iso=iso))
    print('random full 16-bit input: identical')

    metal_rawnr.apply(raw[:64, :64], look, iso=iso)
    t = time.perf_counter()
    actual = metal_rawnr.apply(raw, look, iso=iso)
    tm = time.perf_counter() - t
    t = time.perf_counter()
    expected = sony_rawnr.rawnr(raw, look, iso=iso)
    tn = time.perf_counter() - t
    same('full raw', expected, actual)
    print(f'full raw {raw.shape[1]} x {raw.shape[0]} (ISO {iso}): identical; NumPy {tn:.2f} s, Metal {tm:.3f} s')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    main(sys.argv[1])
