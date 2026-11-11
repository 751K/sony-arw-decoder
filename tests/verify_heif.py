#!/usr/bin/env python3
"""Check HEIF (.hif) output: 10- and 8-bit files in every output space.

    python tests/verify_heif.py photo.ARW

Each file is read back through ImageIO: it must have the requested bit depth,
the output space's profile, and stay close to the decoded image (10-bit
PSNR >= 48 dB, 8-bit >= 42 dB, on the 16-bit scale); writing the same image
twice gives the same bytes.
"""
import ctypes
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'base'))
import sony_decode
import sony_export

PROFILE = {'srgb': 'sRGB', 'adobe': 'Adobe RGB', 'p3': 'Display P3', 'prophoto': 'ProPhoto RGB'}


def read(lib, path):
    w, h, d = ctypes.c_uint32(), ctypes.c_uint32(), ctypes.c_uint32()
    name = ctypes.create_string_buffer(256)
    if lib.sonylook_heif_info(str(path).encode(), ctypes.byref(w), ctypes.byref(h), ctypes.byref(d), name, 256):
        raise SystemExit(lib.sonylook_heif_error().decode())
    out = np.empty((h.value, w.value, 3), np.uint16)
    if lib.sonylook_heif_read(str(path).encode(), out.ctypes.data, w.value, h.value):
        raise SystemExit(lib.sonylook_heif_error().decode())
    return out, d.value, name.value.decode()


def main(path):
    lib = sony_export._heif_lib()
    with tempfile.TemporaryDirectory() as tmp:
        for space in ('srgb', 'adobe', 'p3', 'prophoto'):
            img = sony_decode.decode(path, metal=True, output=space, output_bits=16)
            ref = img.astype(np.float64)
            for bits, floor in ((10, 48.0), (8, 42.0)):
                out = Path(tmp) / f'{space}{bits}.hif'
                sony_export.write_image(img, out, color_space=space, bits=bits, input_bits=16)
                got, depth, profile = read(lib, out)
                psnr = 10 * np.log10(65535.0 ** 2 / np.mean((got - ref) ** 2))
                if depth != bits or not profile.startswith(PROFILE[space]) or got.shape != img.shape or psnr < floor:
                    raise SystemExit(f'{space} {bits}-bit: depth {depth}, profile {profile!r}, PSNR {psnr:.1f} dB')
                print(f'{space:8s} {bits:2d}-bit: depth {depth}, profile {profile!r}, PSNR {psnr:.1f} dB, '
                      f'{out.stat().st_size / 1e6:.1f} MB')
            again = Path(tmp) / 'again.hif'
            sony_export.write_image(img, again, color_space=space, bits=8, input_bits=16)
            if again.read_bytes() != (Path(tmp) / f'{space}8.hif').read_bytes():
                raise SystemExit(f'{space}: writing the same image twice gave different files')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit('Usage: python tests/verify_heif.py photo.ARW')
    main(sys.argv[1])
