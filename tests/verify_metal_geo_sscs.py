#!/usr/bin/env python3
"""Compare fused Metal GeoCorr+SSCS with the separate Metal and NumPy paths."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "base"))
import fused_rust
import metal_geo
import metal_itp
import metal_sscs
import sony_lens
import sony_sscs


def check(planes, origin, region, lens, params):
    geo = metal_geo.apply(planes, origin, region, lens)
    separate = metal_sscs.apply(geo, params)
    fused = metal_geo.apply_sscs(planes, origin, region, lens, params)
    rust_geo = fused_rust.geocorr_u16(
        planes, origin, region, fused_rust.lens_config(lens))
    rust = fused_rust.sscs_u16(
        np.stack(rust_geo, axis=-1), fused_rust.sscs_config(params))
    reference = np.stack(
        sony_sscs.sscs_int(*sony_lens.geocorr_u16(
            planes, origin, region, lens), params), axis=-1)

    if not np.array_equal(rust, reference):
        raise AssertionError("Rust and NumPy GeoCorr+SSCS differ")
    if not np.array_equal(separate, fused):
        diff = np.abs(separate.astype(np.int32) - fused.astype(np.int32))
        raise AssertionError(
            f"fused Metal differs at {np.count_nonzero(diff)} values; "
            f"max {diff.max()}")
    if not np.array_equal(rust, fused):
        diff = np.abs(rust.astype(np.int32) - fused.astype(np.int32))
        raise AssertionError(
            f"fused Metal differs from Rust at {np.count_nonzero(diff)} values; "
            f"max {diff.max()}")

    metal_itp.begin()
    try:
        batched_geo = metal_geo.apply(planes, origin, region, lens)
        batched = metal_sscs.apply(batched_geo, params)
        metal_itp.end()
    except BaseException:
        metal_itp.abort()
        raise
    if not np.array_equal(separate, batched):
        raise AssertionError("batched Metal stage submission differs from separate calls")


def main():
    rng = np.random.default_rng(819)
    origin, region = (-32, -20), (-32, -20, 201, 151)
    knots = np.linspace(0, 1, 19, dtype=np.float32)
    params = [
        dict(T=14000, tR=11000, tG=10000, tB=10500, w=(32, 64, 32)),
        dict(T=5000, tR=5000, tG=5000, tB=5000, w=(0, 0, 0)),
        dict(T=7934, tR=2380, tG=2380, tB=2380, w=(32, 64, 32)),
    ]
    for seed in range(4):
        shape = (171, 233)
        planes = [rng.integers(0, 65536, shape, dtype=np.uint16) for _ in range(3)]
        lens = dict(
            cx=np.float32(100.3), cy=np.float32(70.6),
            s=np.float32(.96 + seed*.02), s2=np.float32(.93 + seed*.03),
            step=np.float32(12.8), n=19,
            tables=[
                np.asarray(1 + knots*t, dtype=np.float32)
                for t in (.1, -.08, .18)
            ],
            off=[(-1.3, .6), (0, 0), (.42, -.67)],
        )
        check(planes, origin, region, lens, params[seed % len(params)])
    print("synthetic GeoCorr+SSCS: fused, separate Metal, Rust and NumPy identical")

    if len(sys.argv) > 1:
        lens = sony_lens.lens_from_arw(sys.argv[1])
        params = sony_sscs.params_from_arw(sys.argv[1])
        shape = (512, 700)
        planes = [rng.integers(0, 16384, shape, dtype=np.uint16) for _ in range(3)]
        check(planes, (1000, 1800), (1020, 1820, 1680, 2260), lens, params)
        print("camera lens and SSCS parameters: identical")


if __name__ == "__main__":
    main()
