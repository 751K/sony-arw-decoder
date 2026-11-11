"""Metal DRO grid analysis and local tone mapping."""
from __future__ import annotations

import ctypes
import numpy as np

import metal_itp
import sony_dro


def _common(raw, black, curve, origin):
    raw = np.ascontiguousarray(raw, dtype=np.uint16)
    black = np.ascontiguousarray(black, dtype=np.int32)
    curve = np.ascontiguousarray(curve, dtype=np.float32)
    if raw.ndim != 2 or raw.shape[0] < 336 or raw.shape[1] < 448:
        raise ValueError("DRO raw grid requires at least 448 x 336 pixels")
    if black.shape != (3,) or curve.shape != (104,):
        raise ValueError("expected three black levels and a 104-element curve")
    if len(origin) != 2:
        raise ValueError("origin must contain x and y")
    return raw, black, curve


def prepare(raw, black, curve, *, origin=(0, 0), strength=1.0,
            k_up=1.0, k_down=1.0, scale=1.0):
    """Start DRO grid analysis on the auxiliary Metal queue and return W/S arrays."""
    raw, black, curve = _common(raw, black, curve, origin)
    weights = metal_itp.empty((6, 8, 14), np.float32)
    sums = metal_itp.empty((6, 8, 14), np.float32)
    lib = metal_itp._load()
    lib.sonylook_metal_dro_grid.argtypes = [
        ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p,
        ctypes.c_int32, ctypes.c_int32, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float,
        ctypes.c_void_p, ctypes.c_void_p]
    rc = lib.sonylook_metal_dro_grid(
        raw.ctypes.data, raw.shape[1], raw.shape[0], black.ctypes.data,
        int(origin[0]), int(origin[1]), sony_dro.LOG_T.ctypes.data, curve.ctypes.data,
        sony_dro.LOG2_MAX, float(strength), float(k_up), float(k_down), float(scale),
        weights.ctypes.data, sums.ctypes.data)
    if rc:
        raise RuntimeError(lib.sonylook_metal_error().decode())
    return weights, sums


def prepare_view(source, offset, size, black, curve, *, origin=(0, 0), strength=1.0,
                 k_up=1.0, k_down=1.0, scale=1.0, wait=0):
    """prepare() on the size = (w, h) crop at offset = (x, y) of a GPU-resident
    2D array, read in place. wait: a RAW stream value (stream base + sensor rows);
    the grid then starts on the GPU only once that many rows are decoded."""
    if source.ndim != 2 or source.dtype != np.uint16 or not source.flags.c_contiguous:
        raise ValueError("source must be a contiguous 2D uint16 array")
    black = np.ascontiguousarray(black, dtype=np.int32)
    if black.shape != (3,) or len(offset) != 2 or len(size) != 2:
        raise ValueError("expected three black levels, an offset and a size")
    w, h = map(int, size)
    if w < 448 or h < 336:
        raise ValueError("DRO raw grid requires at least 448 x 336 pixels")
    weights = metal_itp.empty((6, 8, 14), np.float32)
    sums = metal_itp.empty((6, 8, 14), np.float32)
    lib = metal_itp._load()
    lib.sonylook_metal_dro_grid_view.argtypes = [
        ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
        ctypes.c_uint64, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p,
        ctypes.c_int32, ctypes.c_int32, ctypes.c_void_p, ctypes.c_float, ctypes.c_float,
        ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_void_p, ctypes.c_void_p]
    rc = lib.sonylook_metal_dro_grid_view(
        source.ctypes.data, source.shape[1], source.shape[0], int(offset[0]), int(offset[1]),
        int(wait), w, h, black.ctypes.data, int(origin[0]), int(origin[1]),
        sony_dro.LOG_T.ctypes.data, sony_dro.LOG2_MAX, float(strength), float(k_up),
        float(k_down), float(scale), weights.ctypes.data, sums.ctypes.data)
    if rc:
        raise RuntimeError(lib.sonylook_metal_error().decode())
    return weights, sums


def apply_prepared(planes, grid, black, curve, *, origin=(0, 0), strength=1.0,
                   k_up=1.0, k_down=1.0, scale=1.0, raw_size=None):
    """Apply a DRO grid started by prepare(), waiting on the shared event."""
    planes = tuple(np.ascontiguousarray(p, dtype=np.uint16) for p in planes)
    weights, sums = (np.ascontiguousarray(v, dtype=np.float32) for v in grid)
    curve = np.ascontiguousarray(curve, dtype=np.float32)
    black = np.ascontiguousarray(black, dtype=np.int32)
    if len(planes) != 3 or any(p.shape != planes[0].shape for p in planes):
        raise ValueError("expected three identically shaped 2D planes")
    if weights.shape != (6, 8, 14) or sums.shape != (6, 8, 14):
        raise ValueError("invalid DRO grid")
    if raw_size is None:
        raw_size = (planes[0].shape[1] + 2 * max(origin[0], 0),
                    planes[0].shape[0] + 2 * max(origin[1], 0))
    h, w = planes[0].shape
    out = [metal_itp.empty(p.shape) for p in planes]
    lib = metal_itp._load()
    lib.sonylook_metal_dro_grid_apply.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32,
        ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_float, ctypes.c_float,
        ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    rc = lib.sonylook_metal_dro_grid_apply(
        *(p.ctypes.data for p in planes), w, h, int(raw_size[0]), int(raw_size[1]),
        black.ctypes.data, int(origin[0]), int(origin[1]), sony_dro.LOG_T.ctypes.data,
        curve.ctypes.data, sony_dro.LOG2_MAX, float(strength), float(k_up), float(k_down),
        float(scale), weights.ctypes.data, sums.ctypes.data, *(p.ctypes.data for p in out))
    if rc:
        raise RuntimeError(lib.sonylook_metal_error().decode())
    return tuple(out)


def apply(raw, planes, black, curve, *, origin=(0, 0), strength=1.0,
          k_up=1.0, k_down=1.0, scale=1.0, debug=False):
    """Build the DRO grid from raw and apply it to three uint16 ITP planes."""
    raw, black, curve = _common(raw, black, curve, origin)
    if len(planes) != 3 or planes[0].ndim != 2 or any(p.shape != planes[0].shape for p in planes):
        raise ValueError("expected three identically shaped 2D planes")
    out = [metal_itp.empty(p.shape) for p in planes]
    sub = np.empty((84, 112), np.float32) if debug else None
    W = np.empty((6, 8, 14), np.float32) if debug else None
    S = np.empty((6, 8, 14), np.float32) if debug else None
    lib = metal_itp._load()
    lib.sonylook_metal_dro.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32,
        ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    rc = lib.sonylook_metal_dro(raw.ctypes.data, raw.shape[1], raw.shape[0],
        *(p.ctypes.data for p in planes), planes[0].shape[1], planes[0].shape[0],
        black.ctypes.data, int(origin[0]), int(origin[1]), sony_dro.LOG_T.ctypes.data,
        curve.ctypes.data, sony_dro.LOG2_MAX, float(strength), float(k_up), float(k_down), float(scale),
        *(p.ctypes.data for p in out), *(p.ctypes.data if p is not None else None for p in (sub, W, S)))
    if rc:
        raise RuntimeError(lib.sonylook_metal_error().decode())
    return (tuple(out), (sub, W, S)) if debug else tuple(out)
