#!/usr/bin/env python3
"""Verify that Metal command-buffer batching does not change a decode.

Besides the full --metal path, this covers settings and partial stage
selections where a CPU stage (or a self-submitting Metal call) reads the
output of a batched GPU stage, so the batch must be submitted first.
"""
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from base.sony_decode import decode

CASES = {
    "metal": dict(metal=True),
    "metal, RAW read before the GPU starts": dict(metal=True, _env={"SONYLOOK_RAW_STREAM": "0"}),
    "metal, no detail (Metal front + CPU spatial)": dict(metal=True, detail=False),
    "metal_nr only": dict(metal_nr=True, nr=True),
    "metal_itp only": dict(metal_itp=True),
    "metal_itp + metal_dro (CPU geo)": dict(metal_itp=True, metal_dro=True),
    "RAW chain on Metal, CPU render": dict(metal_nr=True, metal_itp=True, metal_dro=True,
                                           metal_geo=True, metal_sscs=True),
    "metal_marble without metal_post": dict(metal_itp=True, metal_geo=True, metal_look=True,
                                            metal_spatial=True, metal_marble=True),
}


def run(path, kwargs, batch):
    kwargs = dict(kwargs)
    env = {"SONYLOOK_METAL_BATCH": "1" if batch else "0", "SONYLOOK_RAW_STREAM": "1",
           **kwargs.pop("_env", {})}
    os.environ.update(env)
    return decode(path, rust=True, **kwargs)


def main():
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    failed = 0
    for name, kwargs in CASES.items():
        settings = {k: v for k, v in kwargs.items() if not k.startswith(("metal", "_"))}
        reference = decode(sys.argv[1], rust=True, **settings)
        batched = run(sys.argv[1], kwargs, True)
        separate = run(sys.argv[1], kwargs, False)
        for label, other in (("separate", separate), ("CPU reference", reference)):
            if not np.array_equal(batched, other):
                diff = np.abs(batched.astype(np.int32) - other.astype(np.int32))
                print(f"FAIL {name}: batched vs {label} differs at "
                      f"{np.count_nonzero(diff)} values, max {diff.max()}")
                failed += 1
                break
        else:
            print(f"ok   {name}")
    if failed:
        raise SystemExit(1)
    print("batched, separately submitted and CPU decodes are identical")


if __name__ == "__main__":
    main()
