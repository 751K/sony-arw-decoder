# Sony ARW Decoder

English | [中文](README.zh-CN.md)

A standalone complete Sony RAW decoder based on the Imaging Edge Edit development pipeline. NumPy is the reference implementation, with separate Rust and Metal acceleration for the later processing stages.

## Pipeline

```text
RAW -> [RawNR] -> ITP -> [DRO] -> [GeoCorr] -> SSCS
    -> LinearMatrix16 -> MainGamma -> RGB2YCC -> AreaComp -> ChromaSuppress
    -> Sharpness -> YGamma -> [HueSaturation] -> Spica -> YCC2RGB
    -> [Marble] -> [Effect] -> [ToneCurve] -> [ShadeCorrection]
```

Bracketed stages run according to the development settings stored in the ARW. The output is the image corresponding to the decoder rectangle as 16-bit sRGB or Adobe RGB data (plus Display P3 and ProPhoto RGB, which this project adds; Edit has no such options) and can be written to TIFF, JPEG or PNG.

The current implementation uses A7 V samples and development settings as its initial calibration target and has been validated against Sony Imaging Edge Edit 4.0 / 4.1 for RGGB sensor output.

## Backends

| Backend | Platform | Requirements |
| --- | --- | --- |
| NumPy | any | Python 3.11+, `requirements.txt` |
| Rust | any | Rust toolchain |
| Metal | macOS | Xcode Command Line Tools, a Metal-capable Mac |

Rust and Metal are optional and may be combined. NumPy always serves as the reference for stage-by-stage and whole-image validation.

## Speed

Benchmark environment: Mac mini with Apple M6, 16 GB memory, macOS 27.2, arm64. Input: `chart/SonyChart.ARW`; output: a full-resolution 7008 × 4672 image.

Each backend was run once to initialise caches and Metal shaders, then the second full-image decode was timed in the same process.

| Backend | Time | Speed-up vs NumPy |
| --- | ---: | ---: |
| NumPy | 24.92 s | 1.0× |
| Rust | 1.79 s | 13.9× |
| Metal | 0.088 s | 283.2× |
| Rust + Metal | 0.088 s | 283.2× |

`--metal` automatically uses the Rust RAW reader and table builders when the Rust library is available. GPU stages are submitted in two command buffers: the RAW chain and the render chain. GeoCorr with SSCS, and Marble with Effect/ToneCurve/ShadeCorrection, run in fused kernels. The compressed RAW is decoded on a background thread while the GPU already processes the rows that are ready (`SONYLOOK_RAW_STREAM=0` decodes it first). Set `SONYLOOK_METAL_BATCH=0` to restore independent stage submission. All four paths produced identical full-image hashes. The first decode in a new process is slower because it includes initialisation, table construction and Metal shader compilation. These figures describe one synthetic ARW and should not be treated as a general benchmark across all camera models or development settings.

## Install

```bash
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -r requirements.txt
```

Build the Rust core:

```bash
cargo build --release --manifest-path rust/sonylook_core/Cargo.toml
```

Build the Metal bridge on macOS:

```bash
./metal/build.sh
```

When the full Metal toolchain is available, the script precompiles `metal/shaders/shaders.metallib`; otherwise it builds the bridge and the runtime compiles the shader sources.

## Decode

```bash
python3 base/sony_decode.py photo.ARW --out out.tif
python3 base/sony_decode.py photo.ARW --rust --out out-rust.tif
python3 base/sony_decode.py photo.ARW --metal --out out-metal.tif
python3 base/sony_decode.py photo.ARW --rust --metal --out out-fast.tif
```

Python usage:

```python
from base.sony_decode import decode

image = decode("photo.ARW")
fast_image = decode("photo.ARW", rust=True, metal=True)
export_image = decode("photo.ARW", metal=True, output_bits=16)
```

`decode()` returns a `uint16` array shaped `(height, width, 3)` with values 0–16383 by default. `output_bits=16` returns Edit's 16-bit TIFF encoding, exactly the 14-bit result shifted left by two; the Metal path performs that shift in the final interleave kernel. Command-line 16-bit TIFF export selects this path automatically.

`--color-space` (`output=` in Python) is `srgb` (default), `adobe`, `p3` or `prophoto`. Edit offers only the first two. Display P3 and ProPhoto RGB are built the way Edit builds Adobe RGB: an extra matrix in the LinearMatrix16 table, which works on camera RGB so colors outside sRGB survive, plus matching matrices for Marble and the BW/Sepia effect. ProPhoto is re-encoded to gamma 1.8 at the end. The ICC profiles are generated. There is no Edit export to compare these two against; all backends agree bit for bit. Marble and the effect clip to Edit's working gamut, which holds Display P3 but not all of ProPhoto. Details: docs/pipeline.md §9.2. ProPhoto is best written as 16-bit TIFF.

### Many files

```bash
python3 base/sony_batch.py ~/Pictures/2025 more.ARW --out-dir out --metal
python3 base/sony_batch.py ~/Pictures/2025 --out-dir out --format tif --metal --skip-existing
python3 base/sony_batch.py ~/Pictures/2025 --out-dir out-p3 --metal --color-space p3
```

Folders are searched recursively and the output keeps their subfolders; the default format is 8-bit JPEG (`--format tif` writes 16-bit TIFFs of about 200 MB each). The decode options are those of `sony_decode.py`. Reading the next files, decoding, and encoding/writing the previous images run at the same time, and the output is byte-identical to decoding each file on its own. A file that fails (for example a damaged one) is reported and skipped. On the Mac mini above, with JPEG output and `--metal`, a batch takes about 87 ms per A7M5 file when the files are in the OS cache (151 ms one after another) and 126–162 ms per file from an external USB SSD (267–354 ms one after another), where the disk's ~300 MB/s read is then the limit. The summary line splits the time into decoding and waiting for reads and writes.

## Verification

```bash
python3 tests/verify_raw_decode.py photo.ARW
python3 tests/verify_rust_nr_dro.py photo.ARW
python3 tests/verify_metal_itp.py photo.ARW
python3 tests/verify_metal_geo.py photo.ARW
python3 tests/verify_metal_geo_sscs.py photo.ARW
python3 tests/verify_metal_batch.py photo.ARW
python3 tests/verify_batch.py photo.ARW [more.ARW ...]
python3 tests/verify_color_spaces.py photo.ARW
python3 tests/verify_metal_export.py photo.ARW
python3 tests/verify_metal_sscs.py photo.ARW
python3 tests/verify_metal_look.py photo.ARW
python3 tests/verify_metal_spatial.py photo.ARW
python3 tests/verify_metal_marble.py photo.ARW [nr_photo.ARW]
python3 tests/verify_metal_post.py photo.ARW
python3 tests/verify_metal_rawnr.py photo.ARW
```

Rust and Metal outputs are bit-identical to NumPy. The exception is Metal ITP on synthetic full-range noise, where a few direction-map evaluations may differ by one code value. This difference has not been observed in whole-image decoding of the validated real ARW files.

## Layout

| Path | Contents |
| --- | --- |
| `base/` | ARW readers, decoder stages, look engine and backend bindings |
| `rust/sonylook_core/` | Rust RAW reader and accelerated processing core |
| `metal/` | Objective-C bridge and Metal shaders |
| `tests/` | Stage-by-stage differential backend verification |
| `docs/pipeline.md` | Complete stage order, settings and implementation notes |
| `docs/metal.md` | Metal build, memory and verification details |
| `chart/SonyChart.ARW` | Synthetic ARW for local testing |

## License and compatibility

The source is released under the MIT license. This project is not affiliated with Sony and contains no Sony binaries or proprietary application data.
