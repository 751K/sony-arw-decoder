#!/usr/bin/env python3
"""Compare the Metal spatial stages (Sharpness, YGamma, Spica, YCC2RGB) with Rust and NumPy.

Run after building both native libraries: python tests/verify_metal_spatial.py photo.ARW
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'base'))
import fused_rust
import metal_look
import sony_detail
import sony_look
import sony_render
import sony_sscs
import sony_ycc
import sony_ycc_stages


def test_planes(rng, h, w):
    """Smooth gradients (Spica's flat-pattern branch), edges and noise of mixed amplitude."""
    yy, xx = np.mgrid[0:h, 0:w]
    base = 2000 + 9000 * (xx / w) + 3000 * np.sin(yy / 9.0)
    base += np.where((xx // 23 + yy // 17) % 2 == 0, 2500, 0)
    amp = np.where(xx < w // 3, 0, np.where(xx < 2 * w // 3, 40, 3000))
    y = np.clip(base + rng.normal(0, 1, (h, w)) * amp, 0, 32767).astype(np.uint16)
    y[:4, :8] = [0, 1, 32767, 16383, 8191, 0, 32767, 1]
    cr = rng.integers(0x8000 - 6000, 0x8000 + 6000, (h, w), dtype=np.uint16)
    cb = rng.integers(0x8000 - 6000, 0x8000 + 6000, (h, w), dtype=np.uint16)
    cr[0, :3] = [0, 0xffff, 0x8000]
    return y, cr, cb


def detail_sets(srs):
    sets = []
    for iso, sliders in ((100, dict()), (800, dict(dev208=3, dev20c=2, dev210=-2, dev214=1, dev2c4=30)),
                         (6400, dict(dev208=-4, dev20c=-3, dev210=3, dev214=4, dev2c4=80))):
        P = sony_detail.sharpness_params(**sliders)
        S = sony_detail.spica_params(iso)
        for full, gain_on in ((S['full'], S['gain_on']), (1, True), (1, False), (0, False)):
            S2 = dict(S, full=full, gain_on=gain_on)
            sets.append((f'ISO {iso} {sliders or "defaults"} full={full} gain={int(bool(gain_on))}',
                         dict(sharp=P, spica=(srs, S2), dev208=sliders.get('dev208', 0),
                              dev2c4=sliders.get('dev2c4', 50))))
    return sets


def numpy_spatial(y, cr, cb, look, detail):
    h, w = y.shape
    y = sony_detail.sharpness(y, (0, 0, w, h), detail['sharp'])
    y = sony_ycc_stages.ygamma_apply(y, look.ylut, look.yoff, look.ygain, look.black_level, look.white_level)
    y = sony_detail.spica(np.asarray(y, np.uint16), (0, 0, w, h), *detail['spica'],
                          dev208=detail['dev208'], dev2c4=detail['dev2c4'])
    return sony_ycc.ycc2rgb(y, cr, cb)


def same(name, expected, actual):
    for label, e, a in zip('RGB', expected, actual):
        e = np.asarray(e, np.int32)
        a = np.asarray(a, np.int32)
        if not np.array_equal(e, a):
            d = np.abs(e - a)
            yx = np.argwhere(d)[0]
            raise AssertionError(f'{name} {label}: {np.count_nonzero(d)} differences, max {d.max()}, first at {tuple(yx)}')


def main(path):
    rng = np.random.default_rng(1729)
    looks = sony_render.load_looks(path)
    atan, _ = sony_look.load_static_tables()
    srs = sony_detail.load_srsettings()
    sets = detail_sets(srs)
    for look_name in ('Standard', 'FL2'):
        look = sony_render.Look(look_name, looks, use_rust=True)
        y, cr, cb = test_planes(rng, 157, 211)
        for name, detail in sets:
            sp = metal_look.Spatial(look, detail)
            actual = metal_look.spatial(y, cr, cb, sp)
            cfg = fused_rust.spatial_config_from_look(look, detail)
            h = y.shape[0]
            same(f'{look_name} {name} Rust', fused_rust.spatial_u16(y, cr, cb, 0, 0, h, 0, h, cfg), actual)
            same(f'{look_name} {name} NumPy', numpy_spatial(y, cr, cb, look, detail), actual)
        print(f'{look_name}: {len(sets)} parameter sets, Metal, Rust and NumPy identical')

    # Full render path: band overlap, SSCS, and both CPU backends.
    params = sony_sscs.params_from_arw(path)
    for rust in (False, True):
        look = sony_render.Look('Standard', looks, use_rust=rust)
        rgb = rng.integers(0, 16384, (97, 131, 3), dtype=np.uint16)
        for name, detail in sets[::4]:
            expected = look.render(rgb, detail=detail, sscs=params, strip=32)
            actual = look.render(rgb, detail=detail, sscs=params, strip=32, metal_look=True, metal_spatial=True)
            same(f'render rust={rust} {name}', np.moveaxis(expected, -1, 0), np.moveaxis(actual, -1, 0))
        print(f'band render with SSCS and detail, rust={rust}: identical')

    # Timing on a full-resolution-sized plane set.
    look = sony_render.Look('Standard', looks, use_rust=True)
    detail = sets[0][1]
    y, cr, cb = test_planes(rng, 4672, 7008)
    sp = metal_look.Spatial(look, detail)
    cfg = fused_rust.spatial_config_from_look(look, detail)
    metal_look.spatial(y[:64, :64], cr[:64, :64], cb[:64, :64], sp)
    t = time.perf_counter()
    actual = metal_look.spatial(y, cr, cb, sp)
    tm = time.perf_counter() - t
    t = time.perf_counter()
    expected = fused_rust.spatial_u16(y, cr, cb, 0, 0, y.shape[0], 0, y.shape[0], cfg)
    tr = time.perf_counter() - t
    same('7008 x 4672', expected, actual)
    print(f'7008 x 4672: identical; Rust single band {tr:.2f} s, Metal {tm:.2f} s')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit('Usage: python tests/verify_metal_spatial.py photo.ARW')
    main(sys.argv[1])
