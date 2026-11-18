"""Metal lens distortion and lateral chromatic aberration correction."""
from __future__ import annotations

import ctypes
import os
import numpy as np

import metal_itp


def _prepare(planes, origin, region, lens):
    planes = tuple(np.ascontiguousarray(p, dtype=np.uint16) for p in planes)
    if len(planes) != 3 or planes[0].ndim != 2 or min(planes[0].shape) == 0 or any(p.shape != planes[0].shape for p in planes):
        raise ValueError("expected three identically shaped nonempty 2D planes")
    if len(origin) != 2 or len(region) != 4:
        raise ValueError("origin and region must contain 2 and 4 coordinates")
    x0, y0, x1, y1 = map(int, region)
    if x1 <= x0 or y1 <= y0:
        raise ValueError("output region must be nonempty")
    n = int(lens['n'])
    tables = tuple(np.ascontiguousarray(t, dtype=np.float32) for t in lens['tables'])
    if len(tables) != 3 or n < 2 or any(t.shape != (n,) for t in tables):
        raise ValueError("expected three lens tables with n >= 2 entries")
    params = np.array([lens['cx'], lens['cy'], lens['s'], lens['s2'], lens['step'],
                       *(v for pair in lens['off'] for v in pair)], dtype=np.float32)
    if params.shape != (11,) or params[4] <= 0:
        raise ValueError("lens offsets or knot spacing are invalid")
    return planes, params, tables, n, (x0, y0, x1, y1)


def apply(planes, origin, region, lens):
    """Correct three uint16 planes for a region in decoder-image coordinates."""
    planes, params, tables, n, region = _prepare(planes, origin, region, lens)
    x0, y0, x1, y1 = region
    out = [metal_itp.empty((y1-y0, x1-x0)) for _ in range(3)]
    lib = metal_itp._load()
    lib.sonylook_metal_geo.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_uint32, ctypes.c_uint32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
        ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_uint32, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    rc = lib.sonylook_metal_geo(*(p.ctypes.data for p in planes), planes[0].shape[1], planes[0].shape[0],
        int(origin[0]), int(origin[1]), x0, y0, x1-x0, y1-y0, params.ctypes.data,
        *(t.ctypes.data for t in tables), n, *(p.ctypes.data for p in out))
    if rc:
        raise RuntimeError(lib.sonylook_metal_error().decode())
    return tuple(out)


def _prepare_sscs(planes, origin, region, lens, params, over_input=False):
    planes, lens_params, tables, n, region = _prepare(planes, origin, region, lens)
    x0, y0, x1, y1 = region
    sscs = np.asarray([params['T'], params['tR'], params['tG'], params['tB'],
                       *params['w']], dtype=np.int32)
    if sscs.shape != (7,):
        raise ValueError("SSCS parameters must include three channel weights")
    # over_input: the output array is the input planes' own block (planes_block),
    # for the final image to be written there later. GeoCorr itself never writes
    # it: it runs inside the look front (take), or into a new array (resolve).
    out = metal_itp.block_view(planes) if over_input else None
    if out is None or out.shape != (y1 - y0, x1 - x0, 3):
        out, over_input = metal_itp.empty((y1-y0, x1-x0, 3)), False
    return dict(planes=planes, origin=(int(origin[0]), int(origin[1])), region=region,
                lens_params=lens_params, tables=tables, n=n, sscs=sscs, out=out, over_input=over_input)


def apply_sscs(planes, origin, region, lens, params):
    """Fused GeoCorr and SSCS, returning interleaved RGB in one GPU pass."""
    return _run_sscs(_prepare_sscs(planes, origin, region, lens, params))


# GeoCorr + SSCS left for the look front to run (metal_look.front_spatial with
# geo=): the front then computes them per threadgroup and the interleaved image
# between the two stages is never written.  One record at most; it keeps the
# input planes alive until it is taken or resolved.
_pending = None


def defer_sscs(planes, origin, region, lens, params):
    """apply_sscs, deferred: returns the output array, not yet written.  Whoever uses it next
    either take()s the record and runs the fused front, or resolve()s it (the plain kernel)
    and uses the array resolve() returns.  The input planes must not be used otherwise:
    the returned array may share their memory."""
    global _pending
    _pending = _prepare_sscs(planes, origin, region, lens, params,
                             over_input=os.environ.get("SONYLOOK_PLANE_REUSE", "1") != "0")
    return _pending["out"]


def pending(out):
    return _pending is not None and _pending["out"] is out


def take(out):
    """The deferred record for `out` (now the caller's to run), or None."""
    global _pending
    rec = _pending if pending(out) else None
    if rec is not None:
        _pending = None
    return rec


def resolve(out):
    """Run a deferred GeoCorr + SSCS now and return its result: `out`, or a new array when
    `out` shares the input planes' memory.  Returns `out` when none is pending for it."""
    rec = take(out)
    if rec is not None:
        if rec["over_input"]:
            rec["out"] = metal_itp.empty(out.shape)
        return _run_sscs(rec)
    return out


def _run_sscs(rec):
    planes, origin, (x0, y0, x1, y1) = rec["planes"], rec["origin"], rec["region"]
    lens_params, tables, n, sscs, out = rec["lens_params"], rec["tables"], rec["n"], rec["sscs"], rec["out"]
    lib = metal_itp._load()
    lib.sonylook_metal_geo_sscs.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_uint32, ctypes.c_uint32,
        ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
        ctypes.c_uint32, ctypes.c_uint32,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_uint32, ctypes.c_void_p, ctypes.c_void_p,
    ]
    rc = lib.sonylook_metal_geo_sscs(
        *(p.ctypes.data for p in planes), planes[0].shape[1], planes[0].shape[0],
        int(origin[0]), int(origin[1]), x0, y0, x1-x0, y1-y0,
        lens_params.ctypes.data, *(t.ctypes.data for t in tables), n,
        sscs.ctypes.data, out.ctypes.data)
    if rc:
        raise RuntimeError(lib.sonylook_metal_error().decode())
    return out
