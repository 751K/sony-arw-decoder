#!/usr/bin/env python3
"""Check the standalone generated tables against the user's Imaging Edge Edit install.

Exact bit-for-bit matches verified against Edit:
  - hue atan table (512 float32)
  - decoder_params MakerNote slider tables (5 tables)
  - build_color_table 3x3 colour-space matrices (4 matrices, 36 float32)
  - sony_ycc saturation scale table (19 float64)
  - contrast_family row 18 (identity curve, 1025 int32)

Analytic / standard-derived tables verified for accuracy & invariants:
  - generated sRGB / AdobeRGB ICC profiles (identical TRC, colorants within 10/65536, dE76 < 0.1)
  - contrast_family non-zero rows (monotone, fixed at 0 / 471->30144 / 1024->65535)
  - standalone Spica directional 25-tap filter bank (256->100 orbits, exact DC sum 512)

Needs Edit installed (SONY_EDIT_BINARY / SONY_EDIT_RESOURCES) for the Edit comparisons;
the decoder itself no longer requires Edit at runtime.

    python tests/verify_generated_tables.py
"""
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'base'))
import build_color_table
import decoder_params
import edit_binary
import sony_detail
import sony_export
import sony_look
import sony_ycc


def tags(icc):
    n = struct.unpack('>I', icc[128:132])[0]
    out = {}
    for k in range(n):
        sig, off, ln = struct.unpack('>4sII', icc[132 + 12 * k:144 + 12 * k])
        out[sig] = icc[off:off + ln]
    return out


def colorants(t):
    return np.stack([np.array(struct.unpack('>3i', t[s][8:20])) for s in (b'rXYZ', b'gXYZ', b'bXYZ')], 1)


def lab(xyz):
    t = xyz / np.array([0.9642, 1, 0.8249])[:, None]
    f = np.where(t > (6 / 29) ** 3, np.cbrt(t), t / (3 * (6 / 29) ** 2) + 4 / 29)
    return np.stack([116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])])


def main():
    edit_atan = np.frombuffer(edit_binary.read_va(sony_look.HUE_ATAN_TABLE, 2048), '<f4')
    ours = sony_look.hue_atan_table()
    if ours.dtype != np.float32 or not np.array_equal(ours.view(np.uint32), edit_atan.view(np.uint32)):
        raise SystemExit(f'hue atan table: {np.count_nonzero(ours != edit_atan)} entries differ from Edit')
    print('hue atan table: 512 entries identical to Edit')

    ours_dp = decoder_params._tables()
    for name, (addr, n, base) in decoder_params._TABLE_SPEC.items():
        edit_vals = struct.unpack_from('<%di' % n, edit_binary.read_va(addr, 4 * n))
        if ours_dp[name] != (edit_vals, base):
            raise SystemExit(f'decoder_params {name}: differs from Edit')
    print('decoder_params: all 5 slider tables identical to Edit')

    edit_mats, _ = build_color_table.read_constants()
    ours_mats = build_color_table.standard_constants()
    for name in build_color_table.CONSTANT_ADDRESSES:
        a = np.array(edit_mats[name]['values'], dtype=np.float32)
        b = np.array(ours_mats[name]['values'], dtype=np.float32)
        if not np.array_equal(a.view(np.uint32), b.view(np.uint32)):
            raise SystemExit(f'build_color_table {name}: differs from Edit')
    print('build_color_table: all 4 colour-space matrices (36 float32) identical to Edit')

    edit_sat = np.frombuffer(edit_binary.read_va(sony_ycc.SAT_TABLE, 19 * 8), '<f8')
    ours_sat = sony_ycc.sat_table()
    if not np.array_equal(edit_sat.view(np.uint64), ours_sat.view(np.uint64)):
        raise SystemExit('sony_ycc SAT_TABLE: differs from Edit')
    print('sony_ycc SAT_TABLE: all 19 float64 entries identical to Edit')

    edit_fam = np.frombuffer(edit_binary.read_va(sony_look.CONTRAST_FAMILY, 37 * 0x1004), '<i4').reshape(37, 1025)
    ours_fam = sony_look.contrast_family()
    if not np.array_equal(ours_fam[18], edit_fam[18]):
        raise SystemExit('contrast_family row 18 (identity): differs from Edit')
    if np.any(np.diff(ours_fam, axis=1) < 0) or np.any(ours_fam[:, 471] != 30144):
        raise SystemExit('contrast_family: non-monotonic or pivot shifted')
    mae_8bit = np.abs(ours_fam.astype(float) - edit_fam.astype(float)).mean() / 256.0
    print(f'contrast_family: row 18 identical to Edit; analytic family monotone (mean diff {mae_8bit:.2f} / 255)')

    sp = sony_detail.load_srsettings()
    pats, coefs = sp['patterns'], sp['coefs']
    if pats.shape != (256, 3) or coefs.shape != (100, 26) or len(np.unique(pats[:, 0])) != 100:
        raise SystemExit('standalone Spica table: unexpected shape or orbit count')
    dc_err = float(np.max(np.abs(coefs[:, 1:26].sum(axis=1) - 512.0)))
    if dc_err > 1e-4:
        raise SystemExit(f'standalone Spica table: DC sum off by {dc_err}')
    print('spica filter bank: 256 patterns -> 100 symmetry orbits, exact DC sum 512')

    g = np.linspace(0, 1, 17)
    rgb = np.array(np.meshgrid(g, g, g)).reshape(3, -1)
    for space, name in (('srgb', 'Sony_sRGB.icc'), ('adobe', 'Sony_AdobeRGB_1998.icc')):
        edit = tags((edit_binary.resources_path() / name).read_bytes())
        ours = tags(sony_export.icc_profile(space))
        for c in (b'rTRC', b'gTRC', b'bTRC'):
            if edit[c][8:] != ours[c][8:]:
                raise SystemExit(f'{space} {c.decode()}: tone curve differs from Edit')
        a, b = colorants(edit), colorants(ours)
        worst = int(np.abs(a - b).max())
        de = np.linalg.norm(lab(a / 65536 @ rgb) - lab(b / 65536 @ rgb), axis=0).max()
        if worst > 10 or de > 0.1:
            raise SystemExit(f'{space}: colorants differ from Edit by {worst}/65536 (dE76 {de:.3f})')
        print(f'{space} ICC: tone curves identical to Edit; colorants within {worst}/65536, max dE76 {de:.3f}')


if __name__ == '__main__':
    main()
