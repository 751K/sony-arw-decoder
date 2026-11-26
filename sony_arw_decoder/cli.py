"""Console script entry points for sony-decode and sony-batch."""
from __future__ import annotations

from . import _ensure_base_on_path


def decode_main() -> None:
    _ensure_base_on_path()
    import sony_decode
    sony_decode.main()


def batch_main() -> None:
    _ensure_base_on_path()
    import sony_batch
    sony_batch.main()
