"""Experimental Metal ITP demosaic using the standard Sony ITP tables.

Direction-map accumulation uses float32 on Metal, while the Rust reference
uses float64 for some sums. Individual output values can therefore differ.

Build with: ./metal/build.sh
SONYLOOK_METAL_LIB selects a different library path.
"""
from __future__ import annotations

import ctypes
import os
import weakref
from pathlib import Path
import numpy as np

_DIR = Path(__file__).resolve().parents[1] / "metal"
_lib = None


def _load():
    global _lib
    if _lib is None:
        lib = ctypes.CDLL(os.environ.get("SONYLOOK_METAL_LIB") or str(_DIR / "libsonylook_metal.dylib"))
        lib.sonylook_metal_error.restype = ctypes.c_char_p
        lib.sonylook_metal_init.argtypes = [ctypes.c_char_p]
        lib.sonylook_metal_itp_debug.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32,
                                                  ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32,
                                                  ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
        lib.sonylook_metal_itp_tile.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32,
                                                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32,
                                                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        lib.sonylook_metal_itp_image.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32,
                                                 ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32,
                                                 ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32,
                                                 ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                                 ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        lib.sonylook_metal_alloc.argtypes = [ctypes.c_size_t]
        lib.sonylook_metal_alloc.restype = ctypes.c_void_p
        lib.sonylook_metal_free.argtypes = [ctypes.c_void_p]
        lib.sonylook_metal_trim.argtypes = []
        lib.sonylook_metal_pool_bytes.restype = ctypes.c_size_t
        lib.sonylook_metal_begin.argtypes = []
        lib.sonylook_metal_begin.restype = ctypes.c_int
        lib.sonylook_metal_end.argtypes = []
        lib.sonylook_metal_end.restype = ctypes.c_int
        lib.sonylook_metal_abort.argtypes = []
        lib.sonylook_metal_abort.restype = ctypes.c_int
        lib.sonylook_metal_is_active.argtypes = []
        lib.sonylook_metal_is_active.restype = ctypes.c_int
        lib.sonylook_metal_flush.argtypes = []
        lib.sonylook_metal_flush.restype = ctypes.c_int
        lib.sonylook_metal_raw_stream_begin.argtypes = []
        lib.sonylook_metal_raw_stream_begin.restype = ctypes.c_uint64
        lib.sonylook_metal_raw_release.argtypes = [ctypes.c_uint64]
        lib.sonylook_metal_raw_release.restype = None
        if lib.sonylook_metal_init(str(_DIR / "shaders").encode()):
            raise RuntimeError(lib.sonylook_metal_error().decode())
        _lib = lib
    return _lib


def empty(shape, dtype=np.uint16):
    """An uninitialised array in GPU-visible (shared Metal) memory.

    Metal stages read such arrays in place and write their results straight into
    them, so chaining stages through these arrays avoids copies and fresh
    allocations.  The memory returns to the Metal buffer pool when the last view
    of the array is gone."""
    dtype = np.dtype(dtype)
    shape = tuple(int(v) for v in (shape if np.iterable(shape) else (shape,)))
    n = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
    lib = _load()
    p = lib.sonylook_metal_alloc(max(n, 1))
    if not p:
        return np.empty(shape, dtype)
    holder = (ctypes.c_uint8 * max(n, 1)).from_address(p)
    weakref.finalize(holder, lib.sonylook_metal_free, ctypes.c_void_p(p))
    return np.frombuffer(holder, np.uint8, count=n).view(dtype).reshape(shape)


def trim():
    """Release the idle buffers held by the Metal buffer pool."""
    _load().sonylook_metal_trim()


def begin():
    """Start a command buffer shared by successive Metal stage calls."""
    lib = _load()
    if lib.sonylook_metal_begin():
        raise RuntimeError(lib.sonylook_metal_error().decode())


def end():
    """Submit the shared command buffer once and wait for all stages."""
    lib = _load()
    if lib.sonylook_metal_end():
        raise RuntimeError(lib.sonylook_metal_error().decode())


def abort():
    """Discard a shared command buffer without submitting it."""
    _load().sonylook_metal_abort()


def flush():
    """Commit the shared command buffer without waiting and keep batching."""
    lib = _load()
    if lib.sonylook_metal_flush():
        raise RuntimeError(lib.sonylook_metal_error().decode())


class RawStream(ctypes.Structure):
    """ITP input read in place from a sensor raw that is still decoding.

    Image row y is row y + src_y of `src` (a src_rows x src_stride GPU-resident
    array) and may be read once the decoder has reported that many rows."""
    _fields_ = [("src", ctypes.c_void_p), ("src_stride", ctypes.c_uint32),
                ("src_rows", ctypes.c_uint32), ("src_x", ctypes.c_uint32),
                ("src_y", ctypes.c_uint32), ("base", ctypes.c_uint64)]


def active():
    """True while a shared command buffer is open."""
    return bool(_load().sonylook_metal_is_active())


def debug_stage(raw, black, gains, scale, params, stage):
    """Return ITP float intermediate (0–6, 18–19, 22–27) for an RGGB tile including its halo."""
    raw = np.ascontiguousarray(raw, dtype=np.uint16)
    if raw.ndim != 2 or min(raw.shape) < 12:
        raise ValueError("raw must be a 2D tile at least 12 by 12")
    black = np.ascontiguousarray(black, dtype=np.int32)
    gains = np.ascontiguousarray(gains, dtype=np.float32)
    if black.shape != (4,) or gains.shape != (4,):
        raise ValueError("black and gains must each contain four values")
    blend = np.array([params.blend_k, params.blend_lo, params.blend_hi, params.blend_half], dtype=np.float32)
    out = np.empty(raw.shape, dtype=np.float32)
    lib = _load()
    rc = lib.sonylook_metal_itp_debug(raw.ctypes.data, raw.shape[1], raw.shape[0], black.ctypes.data,
                                     gains.ctypes.data, int(scale), blend.ctypes.data, int(stage), out.ctypes.data)
    if rc:
        raise RuntimeError(lib.sonylook_metal_error().decode())
    return out


def process_tile(raw, black, gains, scale, params):
    """Experimental full ITP tile; raw includes the original tile halo."""
    raw = np.ascontiguousarray(raw, dtype=np.uint16)
    if raw.ndim != 2 or min(raw.shape) < 12:
        raise ValueError("raw must be a 2D tile at least 12 by 12")
    black = np.ascontiguousarray(black, dtype=np.int32)
    gains = np.ascontiguousarray(gains, dtype=np.float32)
    if black.shape != (4,) or gains.shape != (4,):
        raise ValueError("black and gains must each contain four values")
    p = np.array([params.blend_k, params.blend_lo, params.blend_hi, params.blend_half,
                  params.k1, params.k2, params.k3, params.k4, params.k5, params.k6], dtype=np.float32)
    out = [np.empty(raw.shape, dtype=np.uint16) for _ in range(3)]
    lib = _load()
    rc = lib.sonylook_metal_itp_tile(raw.ctypes.data, raw.shape[1], raw.shape[0], black.ctypes.data,
                                    gains.ctypes.data, int(scale), p.ctypes.data, *(x.ctypes.data for x in out))
    if rc:
        raise RuntimeError(lib.sonylook_metal_error().decode())
    return tuple(out)


def full_image(raw, black, gains, scale, params, tile=384, halo=12,
               nr_args=None, nr_thresholds=None, stream=None):
    """Run experimental Metal ITP over an RGGB image with Sony's tile edges.

    stream: optional RawStream. `raw` is then only the (h, w) view of the
    stream's source that the GPU reads in place as rows arrive (never read
    here); tiles start as soon as their rows are decoded."""
    if stream is None:
        raw = np.ascontiguousarray(raw, dtype=np.uint16)
    elif raw.dtype != np.uint16:
        raise ValueError("a streamed raw must be uint16")
    if raw.ndim != 2 or min(raw.shape) < 12 or tile <= 0 or halo < 10:
        raise ValueError("expected a 2D Bayer image, positive tile and halo >= 10")
    if tile % 2 or halo % 2:
        raise ValueError("tile and halo must preserve the RGGB phase")
    black = np.ascontiguousarray(black, dtype=np.int32)
    gains = np.ascontiguousarray(gains, dtype=np.float32)
    if black.shape != (4,) or gains.shape != (4,):
        raise ValueError("black and gains must each contain four values")
    p = np.array([params.blend_k, params.blend_lo, params.blend_hi, params.blend_half,
                  params.k1, params.k2, params.k3, params.k4, params.k5, params.k6], dtype=np.float32)
    if (nr_args is None) != (nr_thresholds is None):
        raise ValueError("nr_args and nr_thresholds must be provided together")
    if nr_thresholds is not None:
        nr_thresholds = np.ascontiguousarray(nr_thresholds, dtype=np.int32)
        if nr_thresholds.shape != (3, 0x8000):
            raise ValueError("nr_thresholds must have shape (3, 0x8000)")
    out = [empty(raw.shape) for _ in range(3)]
    lib = _load()
    rc = lib.sonylook_metal_itp_image(stream.src if stream is not None else raw.ctypes.data, raw.shape[1], raw.shape[0], black.ctypes.data,
                                     gains.ctypes.data, int(scale), p.ctypes.data, tile, halo,
                                     ctypes.byref(nr_args) if nr_args is not None else None,
                                     nr_thresholds.ctypes.data if nr_thresholds is not None else None,
                                     ctypes.byref(stream) if stream is not None else None,
                                     *(x.ctypes.data for x in out))
    if rc:
        raise RuntimeError(lib.sonylook_metal_error().decode())
    return tuple(out)
