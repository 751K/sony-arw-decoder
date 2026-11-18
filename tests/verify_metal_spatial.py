#!/usr/bin/env python3
"""Compare the Metal spatial stages (Sharpness, YGamma, Spica, YCC2RGB) with Rust and NumPy.

Run after building both native libraries: python tests/verify_metal_spatial.py photo.ARW
"""
import os
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


def check_marble_input(path, looks, sets, rng):
    """Marble's input conversion written by the spatial pass (spatial_marble) against
    Marble converting the spatial output itself: luma and chroma paths, every
    width and height mod 4, sizes crossing the 32 x 32 tiles, all output modes."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from verify_metal_marble import looks as marble_looks
    blk4, _, blk8 = marble_looks(path)
    look = sony_render.Look('Standard', looks, use_rust=True)
    dec = dict(iso=800, f0=1, d14c=0, f22780=1.0, f22784=1.0, d1b0=0)
    y0, cr0, cb0 = test_planes(rng, 131, 197)
    n = 0
    for h, w in ((10, 10), (11, 13), (32, 64), (33, 62), (63, 97), (64, 95), (131, 197), (130, 196)):
        y, cr, cb = (np.ascontiguousarray(v[:h, :w]) for v in (y0, cr0, cb0))
        for mlook, d218, lum, mode in ((blk4, 1, 0, 0), (blk8, 1, 30, 1), (blk4, 0, 40, 3), (blk8, 1, 60, 4)):
            dev = dict(d218=d218, d224=20, d2c0=lum)
            sp = metal_look.Spatial(look, sets[n % len(sets)][1])
            mcfg = dict(look=mlook, dev=dev, dec=dec, mode=mode)
            for inset in (False, True):
                plain = metal_look.spatial(y, cr, cb, sp)
                expected = metal_look.marble(*plain, mlook, dev, dec, mode=mode, inset=inset)
                pre = metal_look.spatial(y, cr, cb, sp, marble=mcfg)
                same(f'spatial+marble {h} x {w} pre', plain, pre)
                same(f'spatial+marble {h} x {w} d218={d218} lum={lum} mode={mode} inset={inset}',
                     expected, metal_look.marble(*pre, mlook, dev, dec, mode=mode, inset=inset))
                n += 1
    # A staged conversion that does not match the Marble call is not used.
    y, cr, cb = (np.ascontiguousarray(v[:64, :95]) for v in (y0, cr0, cb0))
    dev = dict(d218=1, d224=20, d2c0=30)
    sp = metal_look.Spatial(look, sets[0][1])
    plain = metal_look.spatial(y, cr, cb, sp)
    pre = metal_look.spatial(y, cr, cb, sp, marble=dict(look=blk4, dev=dev, dec=dec, mode=0))
    same('mismatched mode', metal_look.marble(*plain, blk4, dev, dec, mode=1),
         metal_look.marble(*pre, blk4, dev, dec, mode=1))
    print(f'Marble input from the spatial pass: {n} cases identical to separate conversion')
    # Through render(): front_spatial, where Marble's Cb/Cr overwrite the front's
    # Cr/Cb temporaries in place, with and without the command batch.
    m = 0
    for batch in ('1', '0'):
        os.environ['SONYLOOK_METAL_BATCH'] = batch
        for h, w in ((10, 10), (33, 62), (64, 95), (97, 131)):
            rgb = rng.integers(0, 16384, (h, w, 3), dtype=np.uint16)
            for mlook, d218, lum in ((blk4, 1, 0), (blk8, 1, 30), (blk4, 0, 40)):
                mcfg = dict(look=mlook, dev=dict(d218=d218, d224=20, d2c0=lum), dec=dec)
                kw = dict(detail=sets[m % len(sets)][1], marble=mcfg, marble_inset=16, strip=32,
                          metal_look=True, metal_spatial=True, metal_marble=True)
                os.environ['SONYLOOK_SPATIAL_MARBLE'] = '0'
                expected = look.render(rgb, **kw)
                os.environ.pop('SONYLOOK_SPATIAL_MARBLE')
                same(f'render {h} x {w} batch={batch} d218={d218} lum={lum}',
                     np.moveaxis(expected, -1, 0), np.moveaxis(look.render(rgb, **kw), -1, 0))
                m += 1
    os.environ.pop('SONYLOOK_METAL_BATCH')
    print(f'render with Marble input from front_spatial: {m} cases identical')


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

    # Image edges crossing the fused kernel's 32 x 32 tiles, down to 7 x 7:
    # the fused kernel against the three separate ones.
    look = sony_render.Look('Standard', looks, use_rust=True)
    y0, cr0, cb0 = test_planes(rng, 100, 257)
    sizes = ((7, 7), (7, 40), (33, 9), (64, 65), (95, 31), (100, 257))
    for h, w in sizes:
        y, cr, cb = (np.ascontiguousarray(v[:h, :w]) for v in (y0, cr0, cb0))
        for name, detail in sets[::3]:
            sp = metal_look.Spatial(look, detail)
            os.environ['SONYLOOK_SPATIAL_FUSED'] = '0'
            separate = metal_look.spatial(y, cr, cb, sp)
            os.environ.pop('SONYLOOK_SPATIAL_FUSED')
            same(f'fused {h} x {w} {name}', separate, metal_look.spatial(y, cr, cb, sp))
    print(f'fused kernel: {len(sizes)} image sizes from 7 x 7, identical to the separate kernels')

    check_marble_input(path, looks, sets, rng)

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
