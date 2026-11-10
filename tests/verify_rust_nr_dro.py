#!/usr/bin/env python3
"""Compare the Rust RawNR, DRO and Marble chroma path with the NumPy reference.

Usage: python tests/verify_rust_nr_dro.py photo.ARW
photo.ARW needs a DRO curve (SR2 0x781b/0x781c) and chroma-path NR in its look
(look+0x109c != 0); NR and DRO need not be on in the file.
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'base'))
import arw_raw
import decoder_params
import fused_rust
import sony_dro
import sony_marble
import sony_rawnr
import sr2


def same(name, expected, actual):
    for k, (e, a) in enumerate(zip(expected, actual)):
        e, a = np.asarray(e, np.int32), np.asarray(a, np.int32)
        if not np.array_equal(e, a):
            d = np.abs(e - a)
            raise AssertionError(f'{name} [{k}]: {np.count_nonzero(d)} differences, max {d.max()}, '
                                 f'first at {tuple(np.argwhere(d)[0])}')


def check_rawnr(path, raw):
    look = sony_rawnr.look_from_sr2(sr2.sr2_subifd(path))
    iso = int(decoder_params.decoder_params(path)['iso'])
    rng = np.random.default_rng(11)
    n = 0
    for iso_k in (100, 800, 6400):
        for settings in ((0, 0, 0), (40, 30, 0), (-60, -50, 0), (100, -100, 0)):
            for h, w in ((96, 128), (75, 101)):
                y0, x0 = 2 * rng.integers(0, (raw.shape[0] - h) // 2), 2 * rng.integers(0, (raw.shape[1] - w) // 2)
                crop = np.ascontiguousarray(raw[y0:y0 + h, x0:x0 + w])
                noisy = np.clip(crop.astype(np.int64) + rng.normal(0, 200, crop.shape), 0, 16383).astype(np.uint16)
                for img in (crop, noisy):
                    same(f'RawNR ISO {iso_k} {settings} {h}x{w}', [sony_rawnr.rawnr(img, look, iso=iso_k, settings=settings)],
                         [fused_rust.rawnr(img, look, iso=iso_k, settings=settings)])
                    n += 1
    extreme = rng.integers(0, 65536, (64, 80), dtype=np.uint16)
    same('RawNR full 16-bit range', [sony_rawnr.rawnr(extreme, look, iso=iso)], [fused_rust.rawnr(extreme, look, iso=iso)])
    t = time.perf_counter(); actual = fused_rust.rawnr(raw, look, iso=iso); tr = time.perf_counter() - t
    t = time.perf_counter(); expected = sony_rawnr.rawnr(raw, look, iso=iso); tn = time.perf_counter() - t
    same('RawNR full raw', [expected], [actual])
    print(f'RawNR: {n} crops and the full raw identical; NumPy {tn:.2f} s, Rust {tr:.3f} s')


def check_dro(path, raw):
    x0, y0, x1, y1 = decoder_params.crop_rect(path)
    G = sony_dro.build_grid(raw[y0:y1, x0:x1], *sony_dro.curve_points_from_arw(path))
    rng = np.random.default_rng(5)
    n = 0
    for strength, k_up, k_down in ((1.0, 1.0, 1.0), (0.6, 1.3, 0.7), (1.4, 0.5, 1.0)):
        for h, w, ox, oy in ((64, 96, -32, -32), (50, 77, 3000, 2000), (40, 40, 6990, 4650)):
            planes = [rng.integers(0, 65536, (h, w), dtype=np.uint16) for _ in range(3)]
            planes[0][:2, :3] = [[0, 65535, 32768], [1, 2, 3]]
            X, Y = np.meshgrid(np.arange(ox, ox + w), np.arange(oy, oy + h))
            f32 = np.float32
            same(f'DRO s={strength} k={k_up}/{k_down} at {ox},{oy}',
                 sony_dro.vatr(planes, X, Y, G, f32(strength), f32(k_up), f32(k_down)),
                 fused_rust.dro_vatr(planes, (ox, oy), G, strength, k_up, k_down))
            n += 1
    H, W = y1 - y0, x1 - x0
    planes = [rng.integers(0, 16384, (H, W), dtype=np.uint16) * 2 for _ in range(3)]
    t = time.perf_counter(); actual = fused_rust.dro_vatr(planes, (0, 0), G); tr = time.perf_counter() - t
    t = time.perf_counter()
    expected = [np.empty((H, W), np.uint16) for _ in range(3)]
    for ya in range(0, H, 256):
        yb = min(ya + 256, H)
        X, Y = np.meshgrid(np.arange(W), np.arange(ya, yb))
        for e, c in zip(expected, sony_dro.vatr([p[ya:yb] for p in planes], X, Y, G)):
            e[ya:yb] = c
    tn = time.perf_counter() - t
    same('DRO full image', expected, actual)
    print(f'DRO: {n} crops and a {W} x {H} image identical; NumPy {tn:.2f} s, Rust {tr:.3f} s')


def image(h, w, seed):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    out = []
    for c in range(3):
        base = 6000 + 4000 * np.sin(xx / (23 + 7 * c)) * np.cos(yy / (31 + 5 * c))
        out.append(np.clip(base + rng.normal(0, 600, (h, w)), 0, 16383).astype(np.uint16))
    return out


def check_marble(path):
    tags = {k: (v[1] if isinstance(v[1], list) else [v[1]]) for k, v in sr2.sr2_subifd(path).items()}
    blk4 = sony_marble.look_from_tags(tags)
    assert blk4.s16(0x109c) != 0, 'this ARW look has no chroma-path NR (look+0x109c == 0)'
    b8 = bytearray(blk4.b)
    b8[0x109e - 0x1000:0x109e - 0x1000 + 2] = (1).to_bytes(2, 'little')
    blk8 = sony_marble.Look(bytes(b8))
    base = dict(iso=100, f0=0, d14c=0, f22780=1.0, f22784=1.0, d1b0=0)
    n = 0
    for name, look in (('B=4', blk4), ('B=8', blk8)):
        for lum in (0, 10, 60):
            for d224 in (-60, 0, 40, 100):
                for dec, mode, h, w in ((base, 0, 200, 160), (dict(base, f0=1, iso=800), 0, 201, 160),
                                        (dict(base, f0=1, iso=3200), 1, 202, 160), (None, 0, 203, 160)):
                    r, g, b = image(h, w, 1000 + n)
                    dev = dict(d218=1, d224=d224, d2c0=lum)
                    cfg = fused_rust.marble_config(look, dev, dec, mode)
                    actual = fused_rust.marble_slab(r, g, b, cfg)
                    assert actual is not None, 'Rust Marble declined the chroma path'
                    expected = sony_marble.marble(r, g, b, look, dev, dec, mode=mode)
                    vx0, vy0, vx1, vy1 = expected[1]
                    sl = (slice(vy0, vy1), slice(vx0, vx1))
                    same(f'Marble chroma {name} lum={lum} d224={d224} mode={mode} {h}x{w}',
                         [p[sl] for p in expected[0]], [p[sl] for p in actual[0]])
                    n += 1
    r, g, b = image(1200, 1600, 7)
    dev = dict(d218=1, d224=0, d2c0=10)
    cfg = fused_rust.marble_config(blk4, dev, base, 0)
    t = time.perf_counter(); actual = fused_rust.marble_slab(r, g, b, cfg); tr = time.perf_counter() - t
    t = time.perf_counter(); expected = sony_marble.marble(r, g, b, blk4, dev, base); tn = time.perf_counter() - t
    sl = (slice(16, 1200 - 16), slice(16, 1600 - 16))
    same('Marble chroma 1600 x 1200', [p[sl] for p in expected[0]], [p[sl] for p in actual[0]])
    print(f'Marble chroma path: {n} cases and a 1600 x 1200 slab identical (valid region); '
          f'NumPy {tn:.2f} s, Rust {tr:.3f} s (one thread)')


def main(path):
    raw = arw_raw.read_raw(path)
    check_rawnr(path, raw)
    check_dro(path, raw)
    check_marble(path)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    main(sys.argv[1])
