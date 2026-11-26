"""Build helper that bundles base/ modules plus native Rust and Metal libraries into wheels.

Environment controls:
  SONYLOOK_PURE_WHEEL=1        Build a pure-Python wheel (skip native Rust/Metal libs).
  SONYLOOK_SKIP_NATIVE_BUILD=1 Reuse already-built native artifacts instead of recompiling.
  MACOSX_DEPLOYMENT_TARGET     Minimum macOS version for compiled binaries (default 13.0).
"""
from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from setuptools import Distribution, setup
from setuptools.command.build_py import build_py as _build_py

try:
    from setuptools.command.bdist_wheel import bdist_wheel as _bdist_wheel
except ImportError:
    try:
        from wheel.bdist_wheel import bdist_wheel as _bdist_wheel
    except ImportError:
        _bdist_wheel = None

ROOT = Path(__file__).resolve().parent
MIN_MACOS = os.environ.get("MACOSX_DEPLOYMENT_TARGET", "13.0")


def _pure_wheel() -> bool:
    return os.environ.get("SONYLOOK_PURE_WHEEL", "0") == "1"


def _rust_lib_name() -> str:
    if sys.platform == "darwin":
        return "libsonylook_core.dylib"
    if sys.platform == "win32":
        return "sonylook_core.dll"
    return "libsonylook_core.so"


def _build_env() -> dict[str, str]:
    env = dict(os.environ)
    if sys.platform == "darwin":
        env.setdefault("MACOSX_DEPLOYMENT_TARGET", MIN_MACOS)
    return env


def _ensure_rust_lib() -> Path | None:
    manifest = ROOT / "rust" / "sonylook_core" / "Cargo.toml"
    out = ROOT / "rust" / "sonylook_core" / "target" / "release" / _rust_lib_name()
    if os.environ.get("SONYLOOK_SKIP_NATIVE_BUILD") == "1" and out.exists():
        return out
    cargo = shutil.which("cargo")
    if cargo and manifest.exists():
        subprocess.run(
            [cargo, "build", "--release", "--manifest-path", str(manifest)],
            check=True,
            env=_build_env(),
        )
    return out if out.exists() else None


def _ensure_metal_artifacts() -> tuple[Path | None, Path | None]:
    if sys.platform != "darwin":
        return None, None
    dylib = ROOT / "metal" / "libsonylook_metal.dylib"
    shaders = ROOT / "metal" / "shaders"
    build_sh = ROOT / "metal" / "build.sh"
    if os.environ.get("SONYLOOK_SKIP_NATIVE_BUILD") != "1" and build_sh.exists() and shutil.which("clang"):
        subprocess.run(["sh", str(build_sh)], check=True, env=_build_env())
    return (dylib if dylib.exists() else None), (shaders if shaders.is_dir() else None)


class build_py(_build_py):
    def run(self) -> None:
        super().run()
        pkg_out = Path(self.build_lib) / "sony_arw_decoder"

        # Copy base/*.py (excluding Edit-only helper) and JSON data files into _base/
        base_src = ROOT / "base"
        base_dst = pkg_out / "_base"
        base_dst.mkdir(parents=True, exist_ok=True)
        for p in sorted(base_src.iterdir()):
            if p.is_file() and (p.suffix in (".py", ".json")) and p.name != "edit_binary.py":
                shutil.copy2(p, base_dst / p.name)

        if _pure_wheel():
            return

        native_dst = pkg_out / "_native"
        rust_lib = _ensure_rust_lib()
        if rust_lib is not None:
            native_dst.mkdir(parents=True, exist_ok=True)
            shutil.copy2(rust_lib, native_dst / rust_lib.name)

        metal_lib, shaders_dir = _ensure_metal_artifacts()
        if metal_lib is not None:
            native_dst.mkdir(parents=True, exist_ok=True)
            shutil.copy2(metal_lib, native_dst / metal_lib.name)
            if shaders_dir is not None:
                sh_dst = native_dst / "shaders"
                sh_dst.mkdir(parents=True, exist_ok=True)
                for p in sorted(shaders_dir.iterdir()):
                    if p.is_file() and p.suffix in (".metal", ".metallib"):
                        shutil.copy2(p, sh_dst / p.name)


cmdclass: dict[str, type] = {"build_py": build_py}

if _bdist_wheel is not None:
    class bdist_wheel(_bdist_wheel):
        def finalize_options(self) -> None:
            super().finalize_options()
            if not _pure_wheel():
                self.root_is_pure = False

        def get_tag(self) -> tuple[str, str, str]:
            _, _, plat = super().get_tag()
            if _pure_wheel():
                return "py3", "none", "any"
            # ctypes C-ABI libraries do not link against libpython, so one wheel per
            # OS/architecture works across all Python 3.11+ versions.
            if sys.platform == "darwin":
                arch = platform.machine().lower()
                if arch == "aarch64":
                    arch = "arm64"
                ver = MIN_MACOS.replace(".", "_")
                plat = f"macosx_{ver}_{arch}"
            return "py3", "none", plat

    cmdclass["bdist_wheel"] = bdist_wheel


class BinaryDistribution(Distribution):
    def has_ext_modules(self) -> bool:
        return not _pure_wheel()


setup(cmdclass=cmdclass, distclass=BinaryDistribution)
