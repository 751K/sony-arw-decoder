#!/usr/bin/env python3
"""Output color spaces beyond Edit's sRGB and AdobeRGB: Display P3 and ProPhoto RGB.

Edit renders AdobeRGB by folding an sRGB -> AdobeRGB linear matrix into the
LinearMatrix16 table (output_mode 1, which works on white-balanced camera RGB),
reading MainGamma's output as gamma-2.2 encoded, and giving Marble and the
BW/Sepia effect a pair of 13-bit matrices between the output space and Edit's
wide working space W (sRGB -> W is the LinearMatrix16 output_mode 2 matrix).
The two spaces here are built the same way:

    output_mode 3  Display P3   P3 primaries, D65, encoded with the sRGB curve
                                (MainGamma output read as for sRGB; Marble and the
                                effect use the sRGB tables)
    output_mode 4  ProPhoto RGB ROMM primaries, D50 (Bradford from D65); the render
                                runs gamma-2.2 encoded as for AdobeRGB, and the
                                finished image is re-encoded to ProPhoto's gamma 1.8
                                (461/256, as in the usual ProPhoto ICC profile).

Edit has no such modes, so nothing here is checked against an Edit export.
Marble and the effect clip to W, which holds Display P3 (to a sliver of red)
but not all of ProPhoto. The ICC profiles are generated here (ICC v2,
matrix/TRC) rather than taken from Edit.
"""
from __future__ import annotations

import struct
from functools import lru_cache

import numpy as np

SRGB, ADOBE, WORKING, DISPLAY_P3, PROPHOTO = 0, 1, 2, 3, 4
NAMES = {SRGB: "srgb", ADOBE: "adobe", DISPLAY_P3: "p3", PROPHOTO: "prophoto"}
MODES = {v: k for k, v in NAMES.items()}

D65 = (0.3127, 0.3290)
D50 = (0.3457, 0.3585)
PRIMARIES = {
    SRGB: ((0.64, 0.33), (0.30, 0.60), (0.15, 0.06), D65),
    DISPLAY_P3: ((0.680, 0.320), (0.265, 0.690), (0.150, 0.060), D65),
    PROPHOTO: ((0.7347, 0.2653), (0.1596, 0.8404), (0.0366, 0.0001), D50),
}
SONY_GAMMA = 563 / 256          # 2.19921875, Edit's AdobeRGB encoding (gamma_dec / gamma_enc)
PROPHOTO_GAMMA = 461 / 256      # 1.80078125, u8Fixed8 1.8 as in ProPhoto ICC profiles

BRADFORD = np.array([[0.8951, 0.2664, -0.1614], [-0.7502, 1.7135, 0.0367], [0.0389, -0.0685, 1.0296]])


def _xyz(xy):
    x, y = xy
    return np.array([x / y, 1.0, (1 - x - y) / y])


def rgb_to_xyz(mode):
    """Linear RGB -> XYZ in the space's own white."""
    r, g, b, w = PRIMARIES[mode]
    P = np.stack([_xyz(r), _xyz(g), _xyz(b)], axis=1)
    return P * np.linalg.solve(P, _xyz(w))


def adapt(src, dst):
    """Bradford chromatic adaptation between two white points (xy)."""
    s, d = BRADFORD @ _xyz(src), BRADFORD @ _xyz(dst)
    return np.linalg.inv(BRADFORD) @ np.diag(d / s) @ BRADFORD


def srgb_to(mode):
    """Linear sRGB -> linear RGB of `mode` (float64 3x3)."""
    w = PRIMARIES[mode][3]
    return np.linalg.inv(rgb_to_xyz(mode)) @ adapt(D65, w) @ rgb_to_xyz(SRGB)


# LinearMatrix16 output matrices (row-major float32), folded in like Edit's
# mode-1 sRGB -> AdobeRGB matrix.
LM16_MATRIX = {m: tuple(float(np.float32(v)) for v in srgb_to(m).ravel()) for m in (DISPLAY_P3, PROPHOTO)}


def fixed13(m):
    """A 3x3 matrix in Edit's 13-bit fixed point, rounded so every row keeps its
    float sum (each row of these sums to 1, so white stays white)."""
    m = np.asarray(m, np.float64) * 8192
    out = np.round(m).astype(int)
    for i in range(3):
        err = int(round(m[i].sum())) - int(out[i].sum())
        if err:      # move the entry whose rounding went furthest the other way
            j = int(np.argmax((m[i] - out[i]) * np.sign(err)))
            out[i, j] += err
    return tuple(tuple(int(v) for v in row) for row in out)


def derive_marble_matrices(srgb_to_working):
    """{mode: (out -> W, W -> out)} 13-bit matrices for Marble / the effect, from
    Edit's sRGB -> W matrix (build_color_table constant mode2_param8_zero)."""
    s2w = np.asarray(srgb_to_working, np.float64).reshape(3, 3)
    out = {}
    for m in (DISPLAY_P3, PROPHOTO):
        o2w = s2w @ np.linalg.inv(srgb_to(m))
        out[m] = (fixed13(o2w), fixed13(np.linalg.inv(o2w)))
    return out


# derive_marble_matrices(build_color_table.constants()['mode2_param8_zero']['values']),
# stored so decoding does not need Edit's binary (tests/verify_color_spaces.py
# re-derives them).
MARBLE_MATRICES = {
    DISPLAY_P3: (((6041, 2243, -92), (616, 6918, 658), (-31, 6, 8217)),
                 ((11491, -3726, 427), (-1028, 10035, -815), (45, -22, 8169))),
    PROPHOTO: (((9517, 328, -1653), (-50, 7796, 446), (33, -497, 8656)),
               ((7046, -210, 1356), (46, 8579, -433), (-24, 493, 7723))),
}


def srgb_trc(mode):
    """True when the encoded output uses the sRGB curve (else Edit's gamma 2.2)."""
    return mode in (SRGB, DISPLAY_P3)


@lru_cache(maxsize=None)
def prophoto_lut(bits=14):
    """Re-encode the gamma-2.2 render (0..16383, or the 16-bit `<< 2` form) to gamma 1.8.

    v -> round(16383 * (v / 16383) ** (2.2 / 1.8)) on the 14-bit codes, returned
    for 14-bit input (16384 entries) or 16-bit input (65536 entries, v >> 2 in,
    << 2 out, as decode(output_bits=16) produces)."""
    v = np.arange(16384, dtype=np.float64) / 16383
    lut = np.round(16383 * v ** (SONY_GAMMA / PROPHOTO_GAMMA)).astype(np.uint16)
    lut.flags.writeable = False
    if bits == 14:
        return lut
    wide = (lut[np.arange(65536) >> 2].astype(np.uint16) << 2)
    wide.flags.writeable = False
    return wide


def reencode(img, mode, output_bits=14):
    """The finished image in its final encoding: ProPhoto's gamma 1.8, else unchanged.

    A contiguous GPU-resident image (the Metal path's output) is rewritten in
    place on the GPU; anything else gets a new array from a NumPy lookup."""
    if mode != PROPHOTO:
        return img
    lut = prophoto_lut(16 if output_bits == 16 else 14)
    if img.dtype == np.uint16 and img.flags.c_contiguous and img.flags.writeable:
        try:
            import metal_itp
            if metal_itp.lut_u16(img, lut):
                return img
        except (ImportError, OSError):
            pass
    out = np.empty(img.shape, np.uint16)
    np.take(lut, img, out=out)
    return out


# ------------------------------------------------------------------ ICC (v2, matrix/TRC)

def _s15(v):
    return struct.pack(">i", int(round(v * 65536)))


def _xyz_tag(xyz):
    return b"XYZ " + b"\0" * 4 + b"".join(_s15(v) for v in xyz)


def _desc(text):
    a = text.encode("ascii") + b"\0"
    return (b"desc" + b"\0" * 4 + struct.pack(">I", len(a)) + a
            + struct.pack(">II", 0, 0) + struct.pack(">HB", 0, 0) + b"\0" * 67)


def _curve(mode):
    if mode == PROPHOTO:
        return b"curv" + b"\0" * 4 + struct.pack(">IH", 1, 461)
    x = np.arange(1024) / 1023
    y = np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)
    vals = np.round(y * 65535).astype(">u2")
    return b"curv" + b"\0" * 4 + struct.pack(">I", 1024) + vals.tobytes()


@lru_cache(maxsize=None)
def icc_profile(mode):
    """ICC v2 display profile bytes for DISPLAY_P3 or PROPHOTO."""
    name = {DISPLAY_P3: "Display P3 (sonylook)", PROPHOTO: "ProPhoto RGB (sonylook)"}[mode]
    w = PRIMARIES[mode][3]
    m = adapt(w, D50) @ rgb_to_xyz(mode)        # columns: primaries in the D50 PCS
    curve = _curve(mode)
    tags = [(b"desc", _desc(name)),
            (b"cprt", b"text" + b"\0" * 4 + b"No copyright, use freely\0"),
            (b"wtpt", _xyz_tag(_xyz(D50) * 1.0)),
            (b"rXYZ", _xyz_tag(m[:, 0])), (b"gXYZ", _xyz_tag(m[:, 1])), (b"bXYZ", _xyz_tag(m[:, 2])),
            (b"rTRC", curve), (b"gTRC", curve), (b"bTRC", curve)]
    if w != D50:
        a = adapt(w, D50)
        tags.append((b"chad", b"sf32" + b"\0" * 4 + b"".join(_s15(v) for v in a.ravel())))
    table, data, offsets = b"", b"", {}
    base = 128 + 4 + 12 * len(tags)
    for sig, body in tags:
        if body in offsets:                       # the three TRCs share one curve
            off = offsets[body]
        else:
            while (base + len(data)) % 4:
                data += b"\0"
            off = base + len(data)
            offsets[body] = off
            data += body
        table += sig + struct.pack(">II", off, len(body))
    size = base + len(data)
    header = (struct.pack(">I", size) + b"\0" * 4 + struct.pack(">I", 0x02100000) + b"mntrRGB XYZ "
              + struct.pack(">6H", 2025, 1, 1, 0, 0, 0) + b"acsp" + b"APPL" + b"\0" * 4
              + b"\0" * 4 + b"\0" * 4 + b"\0" * 8 + struct.pack(">I", 0)
              + _s15(0.9642) + _s15(1.0) + _s15(0.8249) + b"\0" * 4 + b"\0" * 16 + b"\0" * 28)
    assert len(header) == 128
    return header + struct.pack(">I", len(tags)) + table + data
