#!/usr/bin/env python3
"""Compare Metal SSCS's integer path with Rust and NumPy."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'base'))
import fused_rust
import metal_sscs
import sony_sscs


def check(planes, params):
    gpu = metal_sscs.apply(planes, params)
    cpu = fused_rust.sscs_u16(np.stack(planes, axis=-1), fused_rust.sscs_config(params))
    with np.errstate(divide='ignore', invalid='ignore'):
        reference = np.stack(sony_sscs.sscs_int(*planes, params), axis=-1)
    if not np.array_equal(cpu, reference):
        raise AssertionError('Rust and NumPy SSCS differ')
    if not np.array_equal(cpu, gpu):
        diff = np.abs(cpu.astype(np.int32) - gpu.astype(np.int32))
        raise AssertionError(f'Metal and Rust SSCS differ at {np.count_nonzero(diff)} values; max {diff.max()}')


def main():
    rng = np.random.default_rng(634)
    cases = [dict(T=14000, tR=11000, tG=10000, tB=10500, w=(32, 64, 32)),
             dict(T=14000, tR=-300, tG=3000, tB=1000, w=(255, 510, 255)),
             dict(T=5000, tR=5000, tG=5000, tB=5000, w=(0, 0, 0)),
             dict(T=7934, tR=2380, tG=2380, tB=2380, w=(32, 64, 32))]
    if len(sys.argv) > 1:
        cases.append(sony_sscs.params_from_arw(sys.argv[1]))
    for params in cases:
        planes = [rng.integers(0, 65536, (343, 519), dtype=np.uint16) for _ in range(3)]
        thresholds = np.array([0, 1, params['T'], params['tR'], params['tG'], params['tB'], 65535], dtype=np.int64)
        thresholds = np.clip(np.concatenate((thresholds-1, thresholds, thresholds+1)), 0, 65535).astype(np.uint16)
        for plane in planes:
            plane[0, :len(thresholds)] = thresholds
        check(planes, params)
    print(f'{len(cases)} parameter sets: Metal, Rust and NumPy identical')


if __name__ == '__main__':
    main()
