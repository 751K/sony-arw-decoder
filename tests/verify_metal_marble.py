#!/usr/bin/env python3
"""Compare the Metal Marble with the Rust core (luma path) and the NumPy reference (both paths).

Run after building both native libraries: python tests/verify_metal_marble.py photo.ARW [nr.ARW]
The optional second file is decoded through the chroma path at full resolution (an ARW
shot with noise reduction on); the first file's look is used for the synthetic cases.
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'base'))
import fused_rust
import metal_look
import sony_marble
import sony_render
import sr2


def image(h, w, seed):
    """Smooth gradients plus texture, 14-bit encoded, so Marble has detail to boost."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    out = []
    for c in range(3):
        base = 6000 + 4000 * np.sin(xx / (23 + 7 * c)) * np.cos(yy / (31 + 5 * c))
        out.append(np.clip(base + rng.normal(0, 600, (h, w)), 0, 16383).astype(np.uint16))
    out[0][:2, :4] = [[0, 16383, 0, 16383], [16383, 0, 16383, 0]]
    return out


def same(name, expected, actual):
    for label, e, a in zip('RGB', expected, actual):
        e = np.asarray(e, np.int32)
        a = np.asarray(a, np.int32)
        if not np.array_equal(e, a):
            d = np.abs(e - a)
            raise AssertionError(f'{name} {label}: {np.count_nonzero(d)} differences, max {d.max()}, '
                                 f'first at {tuple(np.argwhere(d)[0])}')


def looks(path):
    tags = {k: (v[1] if isinstance(v[1], list) else [v[1]]) for k, v in sr2.sr2_subifd(path).items()}
    with_tab = sony_marble.look_from_tags(tags)
    no_tab = sony_marble.look_from_tags({k: v for k, v in tags.items() if k not in range(0x7810, 0x781b)})
    b8 = bytearray(with_tab.b)
    b8[0x109e - 0x1000:0x109e - 0x1000 + 2] = (1).to_bytes(2, 'little')
    return with_tab, no_tab, sony_marble.Look(bytes(b8))


def cases(with_tab, no_tab, blk8):
    base = dict(iso=100, f0=0, d14c=0, f22780=1.0, f22784=1.0, d1b0=0)
    dev = lambda lum, d224=0: dict(d218=0, d224=d224, d2c0=lum)
    out = []
    for h in (200, 201, 202, 203):                                  # H % 4 = 0..3
        for name, look in (('table', with_tab), ('no-table', no_tab), ('block8', blk8)):
            out.append((name, h, 160, look, dev(10), base, 0, True))
    for w in (161, 162, 163):                                       # odd / unaligned widths: Rust only
        out.append(('width', 120, w, with_tab, dev(30), base, 0, False))
    for iso in (100, 3200, 12800, 51200, 102400, 409600, 819200):  # every ISO-factor branch
        for lum in (5, 37, 100, 109):
            out.append(('no-table', 120, 96, no_tab, dev(lum), dict(base, iso=iso), 0, True))
    for mode in (0, 1):
        for look in (with_tab, no_tab):
            out.append(('mode', 121, 96, look, dev(25), base, mode, True))
            out.append(('dec-None', 122, 96, look, dev(25), None, mode, True))
    for d224 in (-80, 0, 20, 45, 100):                              # strength() branches
        out.append(('strength', 120, 96, with_tab, dev(60, d224), dict(base, f0=1, iso=400), 0, True))
        out.append(('strength-d14c', 120, 96, with_tab, dev(60, d224), dict(base, d14c=3), 0, True))
    return out


def main(path):
    with_tab, no_tab, blk8 = looks(path)
    n = 0
    for i, (name, h, w, look, dev, dec, mode, check_numpy) in enumerate(cases(with_tab, no_tab, blk8)):
        r, g, b = image(h, w, i)
        actual = metal_look.marble(r, g, b, look, dev, dec, mode=mode, inset=False)
        rust = fused_rust.marble_slab(r, g, b, fused_rust.marble_config(look, dev, dec, mode))
        same(f'{name} h={h} w={w} Rust', rust[0], actual)
        if check_numpy:
            same(f'{name} h={h} w={w} NumPy', sony_marble.marble(r, g, b, look, dev, dec, mode=mode)[0], actual)
        n += 1
    print(f'{n} cases: Metal identical to Rust (whole planes) and NumPy')

    # FUN_100129470 (decoder +0x22780 / +0x22784 != 1): NumPy only, the Rust core falls back.
    for k, (s1, s2) in enumerate(((0.7, 1.0), (1.0, 1.3), (0.5, 0.8))):
        for lum, d1b0 in ((40, 0), (40, 90), (95, 50)):
            r, g, b = image(120, 96, 100 + k)
            dev = dict(d218=0, d224=0, d2c0=lum)
            dec = dict(iso=100, f0=0, d14c=0, f22780=s1, f22784=s2, d1b0=d1b0)
            same(f'strength_470 {s1} {s2} lum={lum}', sony_marble.marble(r, g, b, with_tab, dev, dec)[0],
                 metal_look.marble(r, g, b, with_tab, dev, dec, inset=False))
    print('decoder strength tables (f22780 / f22784 != 1): identical to NumPy')

    # Render: slab path with halo and the 16-px inset, for both CPU backends.
    rl = sony_render.load_looks(path)
    dec = dict(iso=400, f0=1, d14c=0, f22780=1.0, f22784=1.0, d1b0=0)
    for rust in (False, True):
        look = sony_render.Look('Standard', rl, use_rust=rust)
        rgb = np.random.default_rng(7).integers(0, 16384, (300, 120, 3), dtype=np.uint16)
        for mlook in (with_tab, blk8):
            mcfg = dict(look=mlook, dev=dict(d218=0, d224=20, d2c0=40), dec=dec)
            expected = look.render(rgb, marble=mcfg, marble_inset=16, strip=32)
            actual = look.render(rgb, marble=mcfg, marble_inset=16, strip=32, metal_marble=True)
            same(f'render rust={rust}', np.moveaxis(expected, -1, 0), np.moveaxis(actual, -1, 0))
    print('band render with Marble slabs and inset, rust=False/True: identical')

    check_chroma(with_tab, blk8, path)

    # Full-resolution timing: Rust as one slab vs Metal.
    r, g, b = image(4672, 7008, 9)
    dev = dict(d218=0, d224=0, d2c0=30)
    dec = dict(iso=100, f0=0, d14c=0, f22780=1.0, f22784=1.0, d1b0=0)
    metal_look.marble(r[:64, :64], g[:64, :64], b[:64, :64], with_tab, dev, dec)
    t = time.perf_counter()
    actual = metal_look.marble(r, g, b, with_tab, dev, dec, inset=False)
    tm = time.perf_counter() - t
    t = time.perf_counter()
    expected = fused_rust.marble_slab(r, g, b, fused_rust.marble_config(with_tab, dev, dec, 0))[0]
    tr = time.perf_counter() - t
    same('7008 x 4672', expected, actual)
    print(f'7008 x 4672: identical; Rust one slab {tr:.2f} s, Metal {tm:.2f} s')

    if len(sys.argv) > 2:
        full_chroma(sys.argv[2])


def check_chroma(blk4, blk8, path):
    """Chroma path (chroma NR on): NumPy reference only; the Rust core does not cover it."""
    assert blk4.s16(0x109c) != 0, 'this ARW look has no chroma-path NR (look+0x109c == 0)'
    base = dict(iso=100, f0=0, d14c=0, f22780=1.0, f22784=1.0, d1b0=0)
    n = 0
    for name, look in (('B=4', blk4), ('B=8', blk8)):
        for lum in (0, 10, 60):
            for d224 in (-60, 0, 40, 100):
                for dec, mode, h in ((base, 0, 200), (dict(base, f0=1, iso=800), 0, 201),
                                     (dict(base, f0=1, iso=3200), 1, 202), (None, 0, 203)):
                    r, g, b = image(h, 160, 1000 + n)
                    dev = dict(d218=1, d224=d224, d2c0=lum)
                    same(f'chroma {name} lum={lum} d224={d224} mode={mode} h={h}',
                         sony_marble.marble(r, g, b, look, dev, dec, mode=mode)[0],
                         metal_look.marble(r, g, b, look, dev, dec, mode=mode, inset=False))
                    n += 1
    print(f'chroma path: {n} cases (B 4/8, Clarity 0/10/60, d224, dec, mode, H % 4) identical to NumPy')
    rl = sony_render.load_looks(path)
    for rust in (False, True):
        look = sony_render.Look('Standard', rl, use_rust=rust)
        rgb = np.random.default_rng(8).integers(0, 16384, (300, 120, 3), dtype=np.uint16)
        for mlook in (blk4, blk8):
            mcfg = dict(look=mlook, dev=dict(d218=1, d224=0, d2c0=10), dec=base)
            same(f'chroma render rust={rust}', np.moveaxis(look.render(rgb, marble=mcfg, marble_inset=16, strip=32), -1, 0),
                 np.moveaxis(look.render(rgb, marble=mcfg, marble_inset=16, strip=32, metal_marble=True), -1, 0))
    print('chroma path band render with slabs and inset: identical')


def full_chroma(path):
    tags = {k: (v[1] if isinstance(v[1], list) else [v[1]]) for k, v in sr2.sr2_subifd(path).items()}
    look = sony_marble.look_from_tags(tags)
    dev = dict(d218=1, d224=0, d2c0=10)
    dec = dict(iso=100, f0=0, d14c=0, f22780=1.0, f22784=1.0, d1b0=0)
    r, g, b = image(4672, 7008, 21)
    metal_look.marble(r[:64, :64], g[:64, :64], b[:64, :64], look, dev, dec)
    t = time.perf_counter()
    actual = metal_look.marble(r, g, b, look, dev, dec, inset=False)
    tm = time.perf_counter() - t
    t = time.perf_counter()
    expected = sony_marble.marble(r, g, b, look, dev, dec)[0]
    tn = time.perf_counter() - t
    same('chroma 7008 x 4672', expected, actual)
    print(f'chroma + Clarity 7008 x 4672: identical; NumPy {tn:.2f} s, Metal {tm:.2f} s')


if __name__ == '__main__':
    if len(sys.argv) not in (2, 3):
        raise SystemExit(__doc__)
    main(sys.argv[1])
