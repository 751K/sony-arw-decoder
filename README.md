# Sony ARW Decoder

English | [中文](README.zh-CN.md)

A standalone complete Sony RAW decoder that reproduces the Imaging Edge Edit development pipeline stage by stage and takes the camera's own output as the reference, with Rust and Metal acceleration backends for the CPU and the GPU. It decodes a 33-megapixel RAW in 75 ms, 11–22× faster than macOS's own RAW decoder (Core Image) and 40–80× faster than LibRaw; peak memory for one image is about 0.83 GB, below Core Image's 1.58 GB for the same 16-bit output (LibRaw: 0.61 GB); and it writes sRGB, Adobe RGB, Display P3 and ProPhoto RGB as TIFF, JPEG, PNG or HEIF.

## Pipeline

```text
RAW -> [RawNR] -> ITP -> [DRO] -> [GeoCorr] -> [SSCS]
    -> LinearMatrix16 -> MainGamma -> RGB2YCC -> AreaComp -> [ChromaSuppress]
    -> Sharpness -> YGamma -> [HueSaturation] -> Spica -> YCC2RGB
    -> [Marble] -> [Effect] -> [ToneCurve] -> [ShadeCorrection]
```

Bracketed stages run according to the development settings stored in the ARW. SSCS and ChromaSuppress are the exception: the camera's own output skips them, so they run only with `--highlights edit`. The output is the image corresponding to the decoder rectangle as 16-bit sRGB or Adobe RGB data (plus Display P3 and ProPhoto RGB, which this project adds; Edit has no such options) and can be written to TIFF, JPEG, PNG or HEIF (`.hif`, 10-bit, as Sony's cameras save it).

The current implementation uses A7 V samples and development settings as its initial calibration target. Each stage has been checked bit for bit against Sony Imaging Edge Edit 4.0 / 4.1 (RGGB sensors); the output as a whole takes the camera's own JPEG and HEIF as the reference and is validated with charts and real photos (docs/pipeline.md §10.3).

## Backends

| Backend | Platform | Requirements |
| --- | --- | --- |
| NumPy | any | Python 3.11+, `requirements.txt` |
| Rust | any | Rust toolchain |
| Metal | macOS | Xcode Command Line Tools, a Metal-capable Mac |

Rust and Metal are optional and may be combined. NumPy always serves as the reference for stage-by-stage and whole-image validation.

## Speed

Benchmark environment: Mac mini with Apple M6, 16 GB memory, macOS 27.2, arm64. Input: `chart/SonyChart.ARW`; output: a full-resolution 7008 × 4672 image.

Each backend was run once to initialise caches and Metal shaders, then the second full-image decode was timed in the same process (Metal: the median of 9 decodes after 3 warm-up runs).

| Backend | Time | Speed-up vs NumPy |
| --- | ---: | ---: |
| NumPy | 24.92 s | 1.0× |
| Rust | 1.79 s | 13.9× |
| Metal (`--metal`) | 0.075 s | 332× |

`--metal` automatically uses the Rust RAW reader and table builders when the Rust library is available, so `--rust --metal` is the same path. GPU stages are submitted in two command buffers: the RAW chain and the render chain. GeoCorr with SSCS, and Marble with Effect/ToneCurve/ShadeCorrection, run in fused kernels. The compressed RAW is decoded on a background thread while the GPU already processes the rows that are ready (`SONYLOOK_RAW_STREAM=0` decodes it first). Set `SONYLOOK_METAL_BATCH=0` to restore independent stage submission. All three backends produced identical full-image hashes. The first decode in a new process is slower because it includes initialisation, table construction and Metal shader compilation. These figures describe one synthetic ARW and should not be treated as a general benchmark across all camera models or development settings.

Compared on the same machine with macOS's own RAW decoder (Core Image's `CIRAWFilter`, used by Photos and Preview) and with LibRaw 0.22.2 (through rawpy 0.28, whose build is single-threaded), from file to a full-resolution 16-bit bitmap in memory, median of the warm runs. Core Image gets a new `CIRAWFilter` for each decode (reusing one filter on the same file re-renders a cached decode and is not a decode); LibRaw runs `imread` + `postprocess(output_bps=16)` with its default AHD demosaic, and with its fastest (linear):

| File | Core Image, defaults | Core Image, noise reduction / sharpening / local tone mapping off | LibRaw, AHD | LibRaw, linear | This decoder (`--metal`) |
| --- | ---: | ---: | ---: | ---: | ---: |
| SonyChart (A7M5) | 857 ms | 1184 ms | 4995 ms | 3047 ms | 74 ms |
| DSC02741 (A7M5) | 1522 ms | 1370 ms | 5988 ms | 3680 ms | 76 ms |
| DSC06657 (A7M5) | 1469 ms | 1376 ms | 6034 ms | 3667 ms | 75 ms |
| DSC05212 (A7M4) | 1546 ms | 1410 ms | 5426 ms | 3482 ms | 69 ms |

Peak memory for one decode of DSC02741 in its own process (`/usr/bin/time -l`, peak memory footprint, GPU memory included):

| Decoder and output | Peak memory |
| --- | ---: |
| This decoder, `--metal`, 16-bit TIFF or JPEG | 0.83 GB |
| Core Image, 16-bit RGBA | 1.58 GB |
| Core Image, 8-bit RGBA | 1.06 GB |
| LibRaw, 16-bit RGB | 0.61 GB |
| LibRaw, 8-bit RGB | 0.50 GB |

Core Image and LibRaw apply their own demosaic and tone (Core Image also its own noise reduction), and LibRaw keeps the whole sensor area (7168 × 5120 rather than the 7008 × 4672 crop); this decoder reproduces Sony's pipeline, so only times and memory are comparable, not the images.

## Install

### Prebuilt Wheel (recommended, no Rust or Xcode needed)

Download the platform `.whl` from GitHub Releases (macOS arm64 wheels bundle both the Rust core and the Metal bridge/shaders) or install a locally built wheel:

```bash
pip install sony_arw_decoder-*.whl
# or with uv:
uv tool install sony_arw_decoder-*.whl
```

This installs the `sony-decode` and `sony-batch` commands and the `sony_arw_decoder` Python package:

```bash
sony-decode photo.ARW --metal --out out.tif
sony-batch ~/Pictures/2025 --out-dir out --format hif --metal
```

```python
from sony_arw_decoder import decode, write_image

image = decode("photo.ARW", metal=True, output_bits=16)
write_image(image, "out.tif", input_bits=16)
```

### From source and building distributable wheels

```bash
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -r requirements.txt
```

Build a distributable platform wheel (automatically compiles and bundles the Rust core and, on macOS, the Metal bridge and shaders):

```bash
uv build --wheel
# or: python3 -m build --wheel --no-isolation --outdir dist
```

Or build the native libraries individually for running directly from a checkout:

```bash
cargo build --release --manifest-path rust/sonylook_core/Cargo.toml
./metal/build.sh
```

When the full Metal toolchain is available, `./metal/build.sh` precompiles `metal/shaders/shaders.metallib` (default `MACOSX_DEPLOYMENT_TARGET=13.0`); if a host cannot load the precompiled library, the runtime automatically falls back to compiling the bundled `.metal` sources.

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

`--color-space` (`output=` in Python) is `srgb` (default), `adobe`, `p3` or `prophoto`. Edit offers only the first two. Display P3 and ProPhoto RGB are built the way Edit builds Adobe RGB: an extra matrix in the LinearMatrix16 table, which works on camera RGB so colors outside sRGB survive, plus matching matrices for Marble and the BW/Sepia effect. ProPhoto is re-encoded to gamma 1.8 at the end. All four ICC profiles are generated from the standard primaries and curves; for sRGB and Adobe RGB they match the curves of the profiles Edit bundles exactly and their colorants to within ΔE 0.05. There is no Edit export to compare these two against; all backends agree bit for bit. Marble and the effect clip to Edit's working gamut, which holds Display P3 but not all of ProPhoto. Details: docs/pipeline.md §9.2. ProPhoto is best written as 16-bit TIFF.

`--highlights` (`highlights=` in Python) is `camera` (default) or `edit`. The camera's own JPEG and HEIF skip Edit's near-saturation colour suppression (SSCS) and ChromaSuppress. With them, bright colours whose green channel has clipped turn white: in a chart exposure bracket, +2 EV is 1.93 ΔE2000 from the camera HEIF with them and 0.43 without. `edit` keeps both and reproduces Edit's export bit for bit. Details: docs/pipeline.md §10.3.

`--dro-model` (`dro_model=`) is `camera` (default) or `edit` and only matters when DRO is on. The two share the curve stored in the ARW and the per-pixel operator; they differ in how the local brightness is estimated. `camera` uses the grid fitted to the camera's own HEIFs (luma-weighted samples, half-stop bins that also sum their samples' values, a wider range kernel): on a colour chart at Lv1–Lv5 the error against the camera HEIF falls from 0.55–1.86 to 0.27–0.72 ΔE2000, and smooth areas follow the camera closely. `edit` reproduces Edit's grid bit for bit. Both read the whole sensor buffer, as Edit does. Details: docs/pipeline.md §3.3.

### Many files

```bash
python3 base/sony_batch.py ~/Pictures/2025 more.ARW --out-dir out --metal
python3 base/sony_batch.py ~/Pictures/2025 --out-dir out --format tif --metal --skip-existing
python3 base/sony_batch.py ~/Pictures/2025 --out-dir out-p3 --metal --color-space p3
python3 base/sony_batch.py ~/Pictures/2025 --out-dir out-hif --format hif --metal
```

`--format hif` (or `--out photo.hif`) writes 10-bit HEIF through Core Image, which encodes HEVC on Apple's hardware encoder (macOS only; `--bits 8` gives 8-bit). For an A7M5 image at the default quality 95 the file is about 6–9 MB, and it stays within PSNR 51 dB of the decoded image (8-bit JPEG at the same quality: 4 MB, 46 dB). Writing one takes about 150 ms (a JPEG about 100 ms, more than a decode), so batches use two writer threads by default; HEIF then takes about 79 ms per file when the files are in the OS cache. The output profile is embedded as for the other formats.

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
python3 tests/verify_generated_tables.py
python3 tests/verify_heif.py photo.ARW
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

The source is released under the MIT license. This decoder runs completely standalone without Sony Imaging Edge Edit installed; it is not affiliated with Sony and only draws on the sharpening and processing design ideas.
