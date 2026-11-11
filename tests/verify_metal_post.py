#!/usr/bin/env python3
"""Compare the Metal Effect / ToneCurve / ShadeCorrection pass with the NumPy reference.

Run after building the Metal bridge: python tests/verify_metal_post.py photo.ARW
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'base'))
import metal_look
import sony_render

CURVES = (None,
          [((0, 128, 384, 511), (0, 170, 350, 511))] * 4,
          [((0, 511), (0, 511)), ((0, 200, 511), (40, 230, 480)),
           ((0, 100, 300, 511), (0, 60, 330, 511)), ((0, 64, 256, 448, 511), (20, 90, 256, 420, 500))])
SHADES = (None,
          dict(amount=80, radius=50, strength=0),
          dict(amount=30, radius=100, strength=60),
          dict(amount=100, radius=20, strength=-40),
          dict(amount=60, radius=70, strength=30, flag=1, angle=0),
          dict(amount=60, radius=70, strength=30, flag=1, angle=45000))


def image(h, w, seed):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    out = []
    for c in range(3):
        base = 8000 + 7000 * np.sin(xx / (19 + 5 * c)) * np.cos(yy / (27 + 3 * c))
        out.append(np.clip(base + rng.normal(0, 1500, (h, w)), 0, 16383).astype(np.uint16))
    out[0][0, :4] = [0, 16383, 0, 16383]
    out[1][0, :4] = [0, 16383, 16383, 0]
    out[2][0, :4] = [0, 16383, 0, 0]
    return out


def same(name, expected, actual):
    for label, e, a in zip('RGB', expected, actual):
        e = np.asarray(e, np.int32)
        a = np.asarray(a, np.int32)
        if not np.array_equal(e, a):
            d = np.abs(e - a)
            raise AssertionError(f'{name} {label}: {np.count_nonzero(d)} differences, max {d.max()}, '
                                 f'first at {tuple(np.argwhere(d)[0])}')


def main(path):
    looks = sony_render.load_looks(path)
    n = fixed = 0
    for name in ('Standard', 'BW', 'Sepia'):
        for mode in (0, 1, 3, 4):     # sRGB, AdobeRGB, Display P3, ProPhoto
            for ci, curves in enumerate(CURVES):
                for si, shade in enumerate(SHADES):
                    look = sony_render.Look(name, looks, output_mode=mode, tone_curves=curves, shade=shade)
                    h, w = (203, 301) if (ci + si) % 2 else (240, 320)
                    pp = metal_look.Post(look, w, h)
                    if not pp.active:
                        continue
                    r, g, b = image(h, w, n)
                    stats = {}
                    actual = metal_look.post(r, g, b, pp, stats=stats)
                    same(f'{name} mode={mode} curve={ci} shade={si}', pp.reference(r, g, b), actual)
                    fixed += stats['fixed']
                    n += 1
    print(f'{n} configurations: identical to NumPy ({fixed} pixels settled on the CPU)')

    # Capacity overflow falls back to the CPU chain for the whole image.
    look = sony_render.Look('Sepia', looks, tone_curves=CURVES[2], shade=SHADES[2])
    r, g, b = image(240, 320, 99)
    pp = metal_look.Post(look, 320, 240)
    stats = {}
    same('capacity 0', pp.reference(r, g, b), metal_look.post(r, g, b, pp, capacity=0, stats=stats))
    print(f'capacity overflow ({stats["fixed"]} flagged): CPU fallback identical')

    # Render path.
    for name in ('Standard', 'Sepia'):
        look = sony_render.Look(name, looks, tone_curves=CURVES[1], shade=SHADES[4])
        rgb = np.random.default_rng(3).integers(0, 16384, (97, 131, 3), dtype=np.uint16)
        same(f'render {name}', np.moveaxis(look.render(rgb, strip=32), -1, 0),
             np.moveaxis(look.render(rgb, strip=32, metal_post=True), -1, 0))
    print('band render with Effect, ToneCurve and Shade: identical')

    # Full resolution.
    h, w = 4672, 7008
    look = sony_render.Look('Sepia', looks, tone_curves=CURVES[2], shade=SHADES[4])
    r, g, b = image(h, w, 5)
    pp = metal_look.Post(look, w, h)
    metal_look.post(r[:64, :64].copy(), g[:64, :64].copy(), b[:64, :64].copy(), metal_look.Post(look, 64, 64))
    t = time.perf_counter()
    stats = {}
    actual = metal_look.post(r, g, b, pp, stats=stats)
    tm = time.perf_counter() - t
    t = time.perf_counter()
    expected = pp.reference(r, g, b)
    tc = time.perf_counter() - t
    same('7008 x 4672', expected, actual)
    print(f'7008 x 4672 Sepia + curves + shade: identical ({stats["fixed"]} pixels settled on the CPU); '
          f'NumPy {tc:.2f} s, Metal {tm:.2f} s')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit('Usage: python tests/verify_metal_post.py photo.ARW')
    main(sys.argv[1])
