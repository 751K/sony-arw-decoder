"""Cross-platform native library and resource discovery for sony-arw-decoder.

Searches packaged wheel/bundle locations first, then falls back to the source
checkout layout so both installed packages and `python3 base/sony_decode.py`
work identically.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import os
import sys
from functools import lru_cache
from pathlib import Path

_BASE_DIR = Path(__file__).resolve().parent


def _candidate_native_dirs() -> list[Path]:
    dirs: list[Path] = []
    # PyInstaller / frozen bundle
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        dirs.append(Path(meipass) / "sony_arw_decoder" / "_native")
        dirs.append(Path(meipass) / "_native")
    # Installed wheel layout: sony_arw_decoder/_base/ + sony_arw_decoder/_native/
    dirs.append(_BASE_DIR.parent / "_native")
    # Source checkout layout when sony_arw_decoder/_native exists next to base/
    dirs.append(_BASE_DIR.parent / "sony_arw_decoder" / "_native")
    return dirs


def _rust_lib_filename() -> str:
    if sys.platform == "darwin":
        return "libsonylook_core.dylib"
    if sys.platform == "win32":
        return "sonylook_core.dll"
    return "libsonylook_core.so"


def rust_lib_path() -> Path:
    env = os.environ.get("SONYLOOK_RUST_LIB")
    if env:
        return Path(env)
    name = _rust_lib_filename()
    for d in _candidate_native_dirs():
        p = d / name
        if p.exists():
            return p
    return _BASE_DIR.parent / "rust" / "sonylook_core" / "target" / "release" / name


def metal_lib_path() -> Path:
    env = os.environ.get("SONYLOOK_METAL_LIB")
    if env:
        return Path(env)
    for d in _candidate_native_dirs():
        p = d / "libsonylook_metal.dylib"
        if p.exists():
            return p
    return _BASE_DIR.parent / "metal" / "libsonylook_metal.dylib"


def metal_shaders_dir() -> Path:
    env = os.environ.get("SONYLOOK_METAL_SHADERS")
    if env:
        return Path(env)
    env_lib = os.environ.get("SONYLOOK_METAL_LIB")
    if env_lib:
        p = Path(env_lib).resolve().parent / "shaders"
        if p.is_dir():
            return p
    for d in _candidate_native_dirs():
        p = d / "shaders"
        if p.is_dir():
            return p
    return _BASE_DIR.parent / "metal" / "shaders"


@lru_cache(maxsize=1)
def load_libm() -> ctypes.CDLL:
    """Load the platform C math library providing C99 fma / fmaf / powf."""
    if sys.platform == "darwin":
        return ctypes.CDLL("/usr/lib/libSystem.B.dylib")
    if sys.platform == "win32":
        for name in ("ucrtbase", "msvcrt"):
            try:
                return ctypes.CDLL(name)
            except OSError:
                continue
    for candidate in ("libm.so.6", ctypes.util.find_library("m"), "libm.so"):
        if not candidate:
            continue
        try:
            return ctypes.CDLL(candidate)
        except OSError:
            continue
    raise OSError("Could not load platform C math library (libm)")
