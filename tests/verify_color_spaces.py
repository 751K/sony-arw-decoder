#!/usr/bin/env python3
"""Check the Display P3 and ProPhoto outputs (base/color_spaces.py).

    python tests/verify_color_spaces.py photo.ARW

- the stored 13-bit Marble/effect matrices re-derive from Edit's sRGB -> W
  constant, and the same method reproduces Edit's own sRGB/AdobeRGB matrices to
  within 2/8192;
- the generated ICC profiles load in LittleCMS and convert like the matrices;
- the ProPhoto gamma 2.2 -> 1.8 re-encode is monotonic and keeps 0 and 16383;
- Metal and Rust decodes agree in every output space (14- and 16-bit), the
  16-bit ProPhoto output is the 14-bit one << 2, and written JPEGs carry the profile.
Edit has no P3/ProPhoto modes, so nothing here compares with an Edit export.
"""
import io
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'base'))
import build_color_table
import color_spaces as CS
import sony_decode
import sony_effect
import sony_export
from PIL import Image, ImageCms


def main(path):
    s2w = build_color_table.constants()['mode2_param8_zero']['values']
    if CS.derive_marble_matrices(s2w) != CS.MARBLE_MATRICES:
        raise SystemExit('stored Marble matrices do not re-derive from the Edit constant')
    w = np.array(s2w, np.float64).reshape(3, 3)
    a = np.array(build_color_table.constants()['mode1_param8_zero']['values'], np.float64).reshape(3, 3)
    for mode, fwd in ((0, w), (1, w @ np.linalg.inv(a))):
        for ours, edit in ((CS.fixed13(fwd), sony_effect.M_FWD[mode]),
                           (CS.fixed13(np.linalg.inv(fwd)), sony_effect.M_INV[mode])):
            if np.abs(np.array(ours) - np.array(edit)).max() > 2:
                raise SystemExit(f'matrix method is off for Edit mode {mode}')
    print('matrices: stored values re-derive; the method gives Edit\'s own within 2/8192')

    srgb = ImageCms.ImageCmsProfile(io.BytesIO(sony_export.icc_profile('srgb')))
    px = np.array([[255, 255, 255], [128, 128, 128], [200, 120, 60], [30, 200, 90]], float)
    for name, mode in (('p3', CS.DISPLAY_P3), ('prophoto', CS.PROPHOTO)):
        prof = ImageCms.ImageCmsProfile(io.BytesIO(sony_export.icc_profile(name)))
        t = ImageCms.buildTransform(prof, srgb, 'RGB', 'RGB', renderingIntent=1)
        got = np.array(ImageCms.applyTransform(Image.fromarray(px[None].astype(np.uint8)), t))[0]
        x = px / 255
        lin = (np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4) if mode == CS.DISPLAY_P3
               else x ** CS.PROPHOTO_GAMMA) @ np.linalg.inv(CS.srgb_to(mode)).T
        lin = np.clip(lin, 0, 1)
        want = np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * lin ** (1 / 2.4) - 0.055) * 255
        if np.abs(got - want).max() > 1.0:
            raise SystemExit(f'{name} ICC profile converts differently from its matrix')
    print('ICC profiles: LittleCMS conversions match the matrices to 8-bit rounding')

    lut = CS.prophoto_lut(14)
    if lut[0] != 0 or lut[-1] != 16383 or np.any(np.diff(lut.astype(int)) < 0):
        raise SystemExit('ProPhoto re-encode LUT is not monotonic 0..16383')
    if not np.array_equal(CS.prophoto_lut(16)[::4], lut.astype(np.uint16) << 2):
        raise SystemExit('16-bit ProPhoto LUT is not the 14-bit one << 2')

    for space in ('srgb', 'adobe', 'p3', 'prophoto'):
        for bits in (14, 16):
            m = sony_decode.decode(path, metal=True, output=space, output_bits=bits)
            r = sony_decode.decode(path, rust=True, output=space, output_bits=bits)
            if not np.array_equal(m, r):
                raise SystemExit(f'{space} {bits}-bit: Metal and Rust differ')
            if bits == 14:
                m14 = m
            elif not np.array_equal(m, m14.astype(np.uint16) << 2):
                raise SystemExit(f'{space}: 16-bit output is not the 14-bit one << 2')
        with tempfile.TemporaryDirectory() as tmp:
            out = sony_export.write_image(m14, Path(tmp) / 'x.jpg', color_space=space)
            if Image.open(out).info.get('icc_profile') != sony_export.icc_profile(space):
                raise SystemExit(f'{space}: JPEG does not carry its profile')
    print('decode: Metal and Rust identical in srgb/adobe/p3/prophoto, 14- and 16-bit; JPEGs carry the profile')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit('Usage: python tests/verify_color_spaces.py photo.ARW')
    main(sys.argv[1])
