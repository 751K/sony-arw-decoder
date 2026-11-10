#!/usr/bin/env python3
"""Verify Metal's direct 16-bit encoding and TIFF write against the 14-bit path."""
import hashlib
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from base.sony_decode import decode
from base.sony_export import write_image


def main():
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    path = sys.argv[1]
    encoded14 = decode(path, metal=True)
    encoded16 = decode(path, metal=True, output_bits=16)
    if not np.array_equal(encoded16, encoded14 << 2):
        diff = np.abs(encoded16.astype(np.int32) - (encoded14.astype(np.int32) << 2))
        raise AssertionError(
            f"direct 16-bit output differs at {np.count_nonzero(diff)} values; "
            f"max {diff.max()}")

    with tempfile.TemporaryDirectory() as directory:
        direct = write_image(
            encoded16, Path(directory) / "direct.tif", bits=16, input_bits=16)
        reference = write_image(
            encoded14, Path(directory) / "reference.tif", bits=16, input_bits=14)
        direct_hash = hashlib.sha256(direct.read_bytes()).hexdigest()
        reference_hash = hashlib.sha256(reference.read_bytes()).hexdigest()
        if direct_hash != reference_hash:
            raise AssertionError("direct and converted 16-bit TIFF files differ")
    print("Metal direct 16-bit decode and TIFF write are identical to value << 2")


if __name__ == "__main__":
    main()
