#!/usr/bin/env python3
"""Compare Metal DRO's analysis grid and tone mapping with NumPy (bit-exact)."""
import ctypes
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'base'))
import metal_dro
import metal_itp
import sony_dro


def check_math():
    """DRO's float32 division and float32(exp2(float64(e))) on Metal against NumPy."""
    lib = metal_itp._load()
    lib.sonylook_metal_dro_math.argtypes = [ctypes.c_void_p] * 2 + [ctypes.c_uint32] + [ctypes.c_void_p] * 2

    def run(a, b):
        a, b = np.ascontiguousarray(a, np.float32), np.ascontiguousarray(b, np.float32)
        e2, q = np.empty_like(a), np.empty_like(a)
        if lib.sonylook_metal_dro_math(a.ctypes.data, b.ctypes.data, a.size, e2.ctypes.data, q.ctypes.data):
            raise RuntimeError(lib.sonylook_metal_error().decode())
        return e2, q

    rng = np.random.default_rng(5)
    total = 0
    u = np.arange(0x00800000, 0x41a00000, 7, dtype=np.uint32)          # |e| in [2^-126, 20]
    for sign in (0, 0x80000000):
        for chunk in np.array_split(u, 4):
            a = (chunk | sign).view(np.float32)
            e2, _ = run(a, np.ones_like(a))
            ref = np.exp2(a.astype(np.float64)).astype(np.float32)
            bad = np.flatnonzero(e2.view(np.uint32) != ref.view(np.uint32))
            if bad.size:
                raise AssertionError(f'exp2 differs at {a[bad[:4]]}: {e2[bad[:4]]} vs {ref[bad[:4]]}')
            total += a.size
    tiny = np.array([0.0, -0.0, 1e-30, -1e-30, 1e-40, -1e-40, -2.0 ** -25, 1 - 2.0 ** -24, -126.5, 127.5], np.float32)
    e2, _ = run(tiny, np.ones_like(tiny))
    if not np.array_equal(e2.view(np.uint32), np.exp2(tiny.astype(np.float64)).astype(np.float32).view(np.uint32)):
        raise AssertionError('exp2 edge values differ')
    a = np.concatenate([rng.uniform(0, 50, 1 << 22), rng.uniform(-1, 1, 1 << 20), np.ones(4096)]).astype(np.float32)
    b = np.concatenate([rng.uniform(0.01, 50, 1 << 22), rng.uniform(0.5, 300, 1 << 20), np.arange(1, 4097)]).astype(np.float32)
    _, q = run(a, b)
    if not np.array_equal(q.view(np.uint32), (a / b).view(np.uint32)):
        raise AssertionError('division differs')
    print(f'exp2: {total + tiny.size} inputs, division: {a.size} inputs, identical to NumPy')


def main():
    check_math()
    rng = np.random.default_rng(78)
    knots = np.linspace(0, float(sony_dro.LOG2_MAX), 10, dtype=np.float32)
    black = (512, 500, 530)
    for index, origin in enumerate(((-32, -32), (250, 190), (550, 440))):
        raw = rng.integers(0, 16384, (421, 533), dtype=np.uint16)
        ctrl = (np.linspace(0, float(sony_dro.LOG2_MAX), 28) ** (0.9 + index * 0.1)).astype(np.float32)
        grid = sony_dro.build_grid(raw, knots, ctrl, black=black)
        planes = [rng.integers(0, 16384, (256, 333), dtype=np.uint16) for _ in range(3)]
        x, y = np.meshgrid(np.arange(origin[0], origin[0] + 333), np.arange(origin[1], origin[1] + 256))
        for st, ku, kd in ((1.0, 1.0, 1.0), (.8, .7, .9), (.5, 1.3, .6)):
            expected = sony_dro.vatr(planes, x, y, grid, strength=np.float32(st),
                                     k_up=np.float32(ku), k_down=np.float32(kd))
            actual, debug = metal_dro.apply(raw, planes, black, grid['curve'], origin=origin,
                                            strength=st, k_up=ku, k_down=kd, debug=True)
            for name, cpu, gpu in zip(('subcells', 'W', 'S'), (grid['sub'], grid['W'], grid['S']), debug):
                if not np.array_equal(cpu, gpu):
                    raise AssertionError(f'{origin} {name}: max difference {np.max(np.abs(cpu-gpu))}')
            for name, cpu, gpu in zip('RGB', expected, actual):
                difference = np.abs(cpu.astype(np.int32) - gpu.astype(np.int32))
                if difference.any():
                    raise AssertionError(f'{origin} {st} {ku} {kd} {name}: {np.count_nonzero(difference)} differ, max {difference.max()}')
        print(f'{origin} strength/k variants: grid and RGB identical')


if __name__ == '__main__':
    main()
