#!/usr/bin/env python3
"""Compare Metal GeoCorr with Rust and NumPy on sensor-coordinate regions."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'base'))
import fused_rust
import metal_geo
import sony_lens


def check(planes, origin, region, lens):
    cpu = fused_rust.geocorr_u16(planes, origin, region, fused_rust.lens_config(lens))
    gpu = metal_geo.apply(planes, origin, region, lens)
    reference = sony_lens.geocorr_u16(planes, origin, region, lens)
    for name, a, b, c in zip('RGB', cpu, gpu, reference):
        if not np.array_equal(a, c):
            raise AssertionError(f'Rust differs from NumPy for {name}')
        if not np.array_equal(a, b):
            delta = np.abs(a.astype(np.int32) - b.astype(np.int32))
            raise AssertionError(f'Metal differs from Rust for {name}: max {delta.max()}')


def main():
    rng = np.random.default_rng(19)
    origin, region = (-32, -20), (-32, -20, 201, 151)
    for seed in range(4):
        shape = (171, 233)
        planes = [rng.integers(0, 65536, shape, dtype=np.uint16) for _ in range(3)]
        knots = np.linspace(0, 1, 19, dtype=np.float32)
        lens = dict(cx=np.float32(100.3), cy=np.float32(70.6),
                    s=np.float32(.96 + seed*.02), s2=np.float32(.93 + seed*.03),
                    step=np.float32(12.8), n=19,
                    tables=[np.asarray(1 + knots*t, dtype=np.float32) for t in (.1, -.08, .18)],
                    off=[(-1.3, .6), (0, 0), (.42, -.67)])
        check(planes, origin, region, lens)
    print('synthetic distortion, CA, offsets and borders: identical')
    if len(sys.argv) > 1:
        lens = sony_lens.lens_from_arw(sys.argv[1])
        shape = (512, 700)
        planes = [rng.integers(0, 16384, shape, dtype=np.uint16) for _ in range(3)]
        check(planes, (1000, 1800), (1020, 1820, 1680, 2260), lens)
        print('camera lens parameters: identical')


if __name__ == '__main__':
    main()
