"""Standalone Sony ARW RAW decoder with Rust CPU and Apple Metal GPU backends."""
from __future__ import annotations

import sys
from pathlib import Path

__version__ = "0.1.0"


def _ensure_base_on_path() -> Path:
    pkg_dir = Path(__file__).resolve().parent
    bundled = pkg_dir / "_base"
    base_dir = bundled if bundled.is_dir() else pkg_dir.parent / "base"
    base_str = str(base_dir)
    if base_str not in sys.path:
        sys.path.insert(0, base_str)
    return base_dir


_ensure_base_on_path()

from sony_decode import decode  # noqa: E402
from sony_export import icc_profile, normalize_space, write_image  # noqa: E402
from sony_render import Look, load_looks  # noqa: E402

__all__ = [
    "__version__",
    "Look",
    "decode",
    "icc_profile",
    "load_looks",
    "normalize_space",
    "write_image",
]
