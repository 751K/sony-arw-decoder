# Optional Rust render core

`sonylook_core` fuses the per-pixel half of the full-resolution band loop:

```
blur_rows -> hue index -> LinearMatrix16 -> MainGamma -> RGB2YCC
          -> AreaComp -> ChromaSuppress
Sharpness -> YGamma -> Spica -> YCC2RGB
RawNR (whole raw, rows in parallel)
ITP demosaic (whole image: edge extension, 1024x1024 tile grid and parallel tiles all in Rust)
DRO per-pixel operator (the grid is built in NumPy, ~0.1 s)
GeoCorr (distortion + lateral CA, u16 bilinear path)
Marble (one slab per call; the luma path and the NR chroma path)
SSCS (integer path; the float preview path stays on NumPy)
MainGamma and Effect table construction
Lossless compressed raw (TIFF compression 7) decoding, tiles in parallel
```

The Python/NumPy implementation stays the reference and the default backend.
The Rust path is bit-identical to it (verified stage by stage and by comparing
the SHA-256 of a whole 7008x4672 decode), and is opt-in.

## Build

```bash
cargo build --release --manifest-path rust/sonylook_core/Cargo.toml
```

The shared library is written to `rust/sonylook_core/target/release/`
(`libsonylook_core.dylib` on macOS, `libsonylook_core.so` elsewhere).
`base/fused_rust.py` finds it automatically; set `SONYLOOK_RUST_LIB` to
use a different path.

## Use

```bash
python3 base/sony_decode.py /path/to/A7M5.ARW --rust --out out.tif
```

or from Python:

```python
from sony_decode import decode
img = decode("A7M5.ARW", rust=True)
```

`SONYLOOK_RUST=1` enables it for every `Look` as well.

## Notes

- All float32 operations keep the original evaluation order. Fused
  multiply-adds use `f32::mul_add`, which lowers to the hardware FMA
  (`fmadd`) on aarch64, so the rounding matches the original ARM64 code.
- The kernel consumes the uint16 band produced by SSCS (or the front end) and
  returns the planes that `Look.finish` feeds to Marble / Effect / ToneCurve.
- ITP's 1024x1024 tiles, Spica's row loops, Marble's row slabs and the render
  bands run on multiple threads (the work items are independent, and the output
  is unchanged).  `SONYLOOK_THREADS` is the total budget (default 6); the band
  pool uses `SONYLOOK_BAND_THREADS` (default min(6, bands)) and gives each band
  `budget // workers` Spica threads, so the total in flight stays bounded.
  `1` forces the sequential path.  Marble has its own
  `SONYLOOK_MARBLE_THREADS` override because each concurrent slab holds
  ~200 MB of int64 planes: reducing it trades speed for peak memory.
- In band-parallel mode the `render.look` / `render.spatial` entries in the
  timing dict are sums across workers (CPU time), so they can exceed the wall
  time of `render`; the parent entry is the wall time.
- MainGamma and the four Effect tables are built in Rust (`tables.rs`).
  LinearMatrix16 and YGamma stay in Python but are cached in-process and on
  disk (`~/.cache/sonylook`), keyed by the look blob, slider values, the Edit
  binary SHA and a cache version.  `SONYLOOK_CACHE=0` disables the disk cache;
  `SONYLOOK_CACHE_DIR` moves it.  Measured `Look` construction (M6): 0.15 s with
  the Python builders, 0.03 s with the Rust ones, 0.01 s from a warm cache.
- Pointwise/LUT-grid renders (`Look.render(pointwise=True)`), float input and
  non-zero Hue/Saturation fall back to NumPy.
- The NumPy path remains the reference used by the differential verifiers; do
  not change the Rust kernel without re-running the whole-image SHA-256 check.

## Measured on an M6 (Mac mini, 16 GB; A7M5, 7008x4672)

Wall time per stage, decoding the same file again in one process:

| Stage | NumPy | Rust |
| --- | ---: | ---: |
| raw read | 0.26 s (rawpy) | 0.02 s |
| RawNR (DSC05740) | 4.2 s | 0.09 s |
| ITP | 4.4 s | 0.41 s |
| DRO (DSC05740) | 5.2 s | 0.21 s |
| GeoCorr | 5.0 s | 0.37 s |
| band render: SSCS, look chain, Sharpness to YCC2RGB, Marble (DSC03591) | 18.5 s | 0.83 s |
| full decode, DSC03591 (NR / DRO off) | 28.1 s | 1.7 s |
| full decode, DSC05740 (BW, NR / DRO on) | 37.0 s | 2.8 s |

Effect, ToneCurve and ShadeCorrection stay on NumPy: Effect (BW) takes about
0.7 s, and all three together about 5.5 s on a full image.  They have Metal
implementations (see below).  The one-time tables take about 0.03 s with the
Rust builders and about 0.01 s from the disk cache.

The Rust render, RawNR, DRO, GeoCorr, ITP, Marble, SSCS and table-builder paths are
verified against the NumPy version (interior and border tiles / slabs, the
full-image SHA-256, and direct table comparisons), all bit-identical.  `itp_debug_u16` and
the `debug_stage` argument of `marble_slab_u16` expose intermediate planes for
that comparison; they are not used by the decode path.  Marble's rare
`_strength_470` branch (decoder +0x22780 / +0x22784 != 1, never set by the ARW
path) is not ported: the FFI returns 0 and the caller uses NumPy for that slab.

RawNR, DRO and the Marble chroma path are checked by
`tests/verify_rust_nr_dro.py photo.ARW` (crops, synthetic extremes and whole
images, all identical to NumPy).

Verification scripts and reports live in the separate local research workspace,
including `analysis/reverse/pixel_pipeline/verify_marble_rust.py` and the
`verify_rust_*.py` checks. Locations and environment setup are listed in the
[research index](../docs/research.md).

## Raw decoding

With `--rust`, `sony_decode.py` reads lossless compressed ARWs (TIFF compression
7, 512x512 lossless JPEG tiles) through `ljpeg_tiles_u16`, which decodes the
tiles on all cores. The result is identical to rawpy's `raw_image` and
`raw_image_visible`; rawpy is still used for the sensor margins and for every
other raw format. On an M6, one 7168x5120 raw takes about 0.02 s with a warm
file cache, against about 0.27 s for LibRaw's single-threaded unpack.

```sh
python tests/verify_raw_decode.py photo.ARW [folder ...]
```

## Metal acceleration

On macOS, every stage after the raw read also has an optional Metal GPU
implementation, enabled with `--metal` (or the individual `--metal-*`
options). The Metal path does not need this library, but `--rust` still
speeds up the raw read (about 0.02 s against 0.3 s with rawpy) and the table
construction. Build steps, coverage, exactness, memory use and measurements
are in the [Metal notes](../docs/metal.md) (Chinese).

```sh
python3 base/sony_decode.py photo.ARW --rust --metal --out photo.tif
```
