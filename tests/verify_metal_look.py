#!/usr/bin/env python3
"""Compare the GPU look front (LinearMatrix16 through ChromaSuppress) with the Rust and NumPy references."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'base'))
import fused_rust
import metal_look
import sony_look
import sony_render
import sony_sscs
import sony_ycc
import sony_ycc_stages


def numpy_ycc(encoded, look, area):
    y, cr, cb = sony_ycc.rgb2ycc(*(encoded[..., i] for i in range(3)), look.ycc_params, look.luma, clamp=True)
    if area is not None:
        cr, cb = sony_ycc_stages.area_comp_apply(cr, cb, area)
    return sony_ycc_stages.chroma_suppress_apply(y, cr, cb, look.cs)


def area_sets(rng, look):
    """Real AreaComp parameters when the file has them, plus synthetic sets.  The tiny
    multipliers make m * p[12] a tiny fraction, where float64 truncation is easy to get wrong."""
    sets = [] if look.area_comp is None else [list(look.area_comp[0])]
    for scale in (1.0, 1e-3, 1e-9, 1e-30):
        p = rng.uniform(-3000, 3000, 16).astype(np.float32)
        p[4:12] = rng.uniform(-1, 1, 8)
        p[12:14] = rng.uniform(-1, 1, 2) * scale
        sets.append([float(v) for v in p])
    return sets


def same(name, expected, actual):
    for label, e, a in zip(('Y', 'Cr', 'Cb'), expected, actual):
        if not np.array_equal(e, a):
            d = np.abs(e.astype(np.int32) - a.astype(np.int32))
            raise AssertionError(f'{name} {label}: {np.count_nonzero(d)} differences, max {d.max()}')


def check_ycc(rng, looks, atan):
    for name, mode in (('Standard', 0), ('FL2', 0), ('Vivid', 1)):
        look = sony_render.Look(name, looks, output_mode=mode, use_rust=True)
        cfg = fused_rust.config_from_look(look, atan)
        encoded = rng.integers(-32768, 32768, (301, 257, 3), dtype=np.int16)
        encoded[0, :6] = [[-32768] * 3, [32767] * 3, [0] * 3, [16383] * 3, [-1, 0, 1], [32767, -32768, 0]]
        for area in [None] + area_sets(rng, look):
            args = metal_look.front_args(look)
            cfg.params.area_flag = args.area_flag = 0 if area is None else 1
            for i in range(16):
                cfg.params.area[i] = args.area[i] = 0.0 if area is None else area[i]
            actual = metal_look.encoded_front(encoded, look, args)
            same(f'{name} Rust', fused_rust.encoded_front_u16(encoded, cfg), actual)
            same(f'{name} NumPy', numpy_ycc(encoded, look, None if area is None else (area, 1)), actual)
            rgb = rng.integers(0, 16384, (97, 133, 3), dtype=np.uint16)
            same(f'{name} fused', fused_rust.front_u16(rgb, cfg), metal_look.front(rgb, look, atan, args))
        print(f'{name}, output mode {mode}: YCC, AreaComp and ChromaSuppress identical (Rust and NumPy)')


def main(path):
    rng = np.random.default_rng(651)
    looks = sony_render.load_looks(path)
    atan, _ = sony_look.load_static_tables()
    for name, mode in (('Standard', 0), ('FL2', 0), ('Vivid', 1)):
        look = sony_render.Look(name, looks, output_mode=mode, use_rust=True)
        cfg = fused_rust.config_from_look(look, atan)
        shape = (1024, 1024, 3) if name == 'Standard' else (512, 517, 3)
        rgb = rng.integers(0, 16384, shape, dtype=np.uint16)
        expected = np.stack(sony_look.render_look([rgb[..., i] for i in range(3)], look.table, look.gamma, atan), axis=-1)
        encoded = metal_look.apply(rgb, look.table, look.gamma, atan)
        if not np.array_equal(expected, encoded):
            d = np.abs(expected.astype(np.int32)-encoded.astype(np.int32))
            raise AssertionError(f'{name}: encoded RGB differs at {np.count_nonzero(d)} values, max {d.max()}')
        for flag in (0, 1):
            cfg.params.area_flag = flag
            fused = fused_rust.front_u16(rgb, cfg)
            split = fused_rust.encoded_front_u16(encoded, cfg)
            if any(not np.array_equal(a, b) for a, b in zip(fused, split)):
                raise AssertionError(f'{name}: split YCC/chroma differs from fused Rust with area flag {flag}')
        print(f'{name}, output mode {mode}: matrix, gamma and split front identical')
    check_ycc(rng, looks, atan)
    # Match band overlap, enabled SSCS, and the NumPy downstream fallback.
    params = sony_sscs.params_from_arw(path)
    for rust in (False, True):
        look = sony_render.Look('Standard', looks, use_rust=rust)
        rgb = rng.integers(0, 16384, (73, 101, 3), dtype=np.uint16)
        expected = look.render(rgb, sscs=params, strip=32)
        actual = look.render(rgb, sscs=params, strip=32, metal_look=True)
        if not np.array_equal(expected, actual):
            diff = np.abs(expected.astype(np.int32) - actual.astype(np.int32))
            raise AssertionError(f'render rust={rust}: max difference {diff.max()}')
        print(f'band render with SSCS, rust={rust}: identical')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit('Usage: python tests/verify_metal_look.py photo.ARW')
    main(sys.argv[1])
