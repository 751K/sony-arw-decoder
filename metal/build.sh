#!/bin/sh
set -eu

ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
SHADERS="$ROOT/metal/shaders"
TMP="$(mktemp -t sonylook-shaders).metal"
AIR="$(mktemp -t sonylook-shaders).air"
trap 'rm -f "$TMP" "$AIR"' EXIT

if xcrun --find metal >/dev/null 2>&1 && xcrun --find metallib >/dev/null 2>&1; then
    for name in itp dro geo sscs look spatial marble post nr; do
        cat "$SHADERS/$name.metal" >> "$TMP"
    done
    # Match the runtime's MTLMathModeSafe path: no relaxed IEEE arithmetic.
    xcrun -sdk macosx metal -fno-fast-math -c "$TMP" -o "$AIR"
    xcrun -sdk macosx metallib "$AIR" -o "$SHADERS/shaders.metallib"
else
    # Never let a stale metallib mask edited shader sources on a machine
    # without the full Metal toolchain.
    rm -f "$SHADERS/shaders.metallib"
    printf '%s\n' 'Metal compiler unavailable; the runtime will compile the shader sources.'
fi

clang -O2 -fobjc-arc -dynamiclib -framework Foundation -framework Metal \
    "$ROOT/metal/bridge.m" -o "$ROOT/metal/libsonylook_metal.dylib"

if [ -f "$SHADERS/shaders.metallib" ]; then
    printf 'Built %s\n' "$SHADERS/shaders.metallib"
else
    printf 'Built %s\n' "$ROOT/metal/libsonylook_metal.dylib"
fi
