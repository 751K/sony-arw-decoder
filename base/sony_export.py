#!/usr/bin/env python3
"""Write a decoded image the way Edit exports it: 8/16-bit TIFF or JPEG, with an ICC profile.

The decoder image is 14-bit encoded (0..16383).  Edit's 16-bit TIFF stores exactly
`value << 2` (every sample of the reference export is a multiple of 4; the 16-bit
maximum is 65532), so the 8-bit value is `value >> 6`.  No rounding is involved.

The ICC profiles are the ones bundled with Edit, read from the user's own
installation (SONY_EDIT_RESOURCES, default /Applications/Imaging Edge/Edit.app/
Contents/Resources):

    sRGB      Sony_sRGB.icc            sRGB primaries, 1024-point sRGB TRC
    AdobeRGB  Sony_AdobeRGB_1998.icc   Adobe RGB primaries, gamma 2.19921875

TIFF gets tag 34675, JPEG an APP2 marker, PNG an iCCP chunk.  Edit itself embeds
the profile only for AdobeRGB (byte-identical to Sony_AdobeRGB_1998.icc, EXIF
ColorSpace 0xFFFF); its sRGB TIFFs carry none (ColorSpace 1).  Both are embedded
here, so the file is self-describing.  Checked on DSC03591 (AdobeRGB 16-bit and
8-bit exports): 8-bit = 16-bit >> 8 for all but a few pixels.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

SPACES = {"srgb": "Sony_sRGB.icc", "adobe": "Sony_AdobeRGB_1998.icc"}
ALIASES = {"srgb": "srgb", "adobergb": "adobe", "adobergb1998": "adobe", "adobe": "adobe",
           "p3": "p3", "displayp3": "p3", "prophoto": "prophoto", "prophotorgb": "prophoto",
           "romm": "prophoto", "rommrgb": "prophoto"}


def normalize_space(space="srgb"):
    """'sRGB', 'AdobeRGB', 'Display P3', 'ProPhoto RGB', ... -> 'srgb' / 'adobe' / 'p3' / 'prophoto'."""
    key = str(space).strip().lower().replace(" ", "").replace("-", "").replace("_", "")
    if key in ALIASES:
        return ALIASES[key]
    raise ValueError(f"unknown color space {space!r}; use 'srgb', 'adobe', 'p3' or 'prophoto'")


@lru_cache(maxsize=4)
def icc_profile(space="srgb"):
    """Bytes of the profile for this output space: Edit's bundled one for sRGB and
    AdobeRGB, a generated ICC v2 profile for Display P3 and ProPhoto (color_spaces.py)."""
    space = normalize_space(space)
    if space not in SPACES:
        import color_spaces as CS
        return CS.icc_profile(CS.MODES[space])
    import edit_binary
    name = SPACES[space]
    p = edit_binary.resources_path() / name
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found; install Imaging Edge Edit or set SONY_EDIT_RESOURCES")
    return p.read_bytes()


def to_16bit(img):
    """14-bit encoded 0..16383 -> the uint16 values Edit writes (exactly value << 2)."""
    return (np.asarray(img, np.uint16) << 2)


def to_8bit(img, input_bits=14):
    """Encoded 14/16-bit image -> uint8 (Edit's 16-bit value >> 8)."""
    if input_bits not in (14, 16):
        raise ValueError("input_bits must be 14 or 16")
    return (np.asarray(img, np.uint16) >> (6 if input_bits == 14 else 8)).astype(np.uint8)


HEIF_EXTS = (".hif", ".heif", ".heic")


@lru_cache(maxsize=1)
def _heif_lib():
    import ctypes
    import os
    path = os.environ.get("SONYLOOK_METAL_LIB") or str(Path(__file__).resolve().parents[1] / "metal" / "libsonylook_metal.dylib")
    try:
        lib = ctypes.CDLL(path)
        fn = lib.sonylook_heif_write
    except (OSError, AttributeError) as exc:
        raise RuntimeError("HEIF output needs the macOS bridge: run ./metal/build.sh") from exc
    fn.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_char_p, ctypes.c_size_t,
                   ctypes.c_float, ctypes.c_int, ctypes.c_char_p]
    lib.sonylook_heif_error.restype = ctypes.c_char_p
    lib.sonylook_heif_info.argtypes = [ctypes.c_char_p] + [ctypes.c_void_p] * 3 + [ctypes.c_char_p, ctypes.c_size_t]
    lib.sonylook_heif_read.argtypes = [ctypes.c_char_p, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32]
    return lib


def write_heif(arr16, path, icc, bits=10, quality=95):
    """HEIF through Core Image (hardware HEVC): arr16 is (H, W, 3) uint16 on the
    16-bit scale; bits 10 or 8; the profile is embedded."""
    arr16 = np.ascontiguousarray(arr16, np.uint16)
    h, w = arr16.shape[:2]
    lib = _heif_lib()
    if lib.sonylook_heif_write(arr16.ctypes.data, w, h, icc, len(icc), float(quality) / 100.0,
                               int(bits), str(path).encode()):
        raise RuntimeError(lib.sonylook_heif_error().decode())


def write_image(img, path, *, color_space="srgb", bits=None, quality=95, input_bits=14):
    """Write (H, W, 3) encoded image with the matching ICC profile.

    path: .tif/.tiff (default 16-bit), .jpg/.jpeg (8-bit), .png (default 8-bit),
          .hif/.heif/.heic (HEIF, default 10-bit, as Sony's cameras write it; macOS).
    bits: 8 or 16 (8 or 10 for HEIF); None picks the natural depth for the format.
          16-bit is not a JPEG option.
    input_bits: 14 (decoder values) or the 16-bit TIFF encoding produced by
          decode(..., output_bits=16). No second shift is applied in the latter case.
    quality: JPEG / HEIF quality 1..100.
    """
    img = np.asarray(img)
    if img.ndim != 3 or img.shape[2] < 3:
        raise ValueError(f"expected an (H, W, 3) image, got {img.shape}")
    if img.dtype.kind not in "ui":
        raise TypeError(f"expected an integer 14-bit image, got {img.dtype}")
    if input_bits not in (14, 16):
        raise ValueError("input_bits must be 14 or 16")
    img = img[..., :3]
    path = Path(path)
    space = normalize_space(color_space)
    icc = icc_profile(space)
    ext = path.suffix.lower()
    is_tiff = ext in (".tif", ".tiff")
    is_jpeg = ext in (".jpg", ".jpeg")
    if ext in HEIF_EXTS:
        bits = 10 if bits is None else int(bits)
        if bits not in (8, 10):
            raise ValueError("HEIF output is 10-bit or 8-bit")
        arr16 = img.astype(np.uint16, copy=False) if input_bits == 16 else to_16bit(img)
        write_heif(arr16, path, icc, bits=bits, quality=quality)
        return path
    if not (is_tiff or is_jpeg or ext == ".png"):
        raise ValueError(f"unsupported extension {path.suffix!r}; use .tif, .jpg, .png or .hif")
    bits = (16 if is_tiff else 8) if bits is None else int(bits)
    if bits not in (8, 16):
        raise ValueError("bits must be 8 or 16")
    if is_jpeg and bits == 16:
        raise ValueError("JPEG is 8-bit only")
    if ext == ".png" and bits == 16:
        raise ValueError("PNG output is 8-bit; use a .tif for 16-bit")

    arr = (img.astype(np.uint16, copy=False) if input_bits == 16 else to_16bit(img)) \
        if bits == 16 else to_8bit(img, input_bits=input_bits)
    if is_tiff:
        import tifffile
        # TIFF type 7 (UNDEFINED), as Capture One and the TIFF ICC convention use
        tifffile.imwrite(path, arr, photometric="rgb",
                         extratags=[(34675, 7, len(icc), icc, False)])
    else:
        from PIL import Image
        im = Image.fromarray(arr, "RGB")
        if is_jpeg:
            im.save(path, quality=int(quality), icc_profile=icc)
        else:
            im.save(path, icc_profile=icc)
    return path
