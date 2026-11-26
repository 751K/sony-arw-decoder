# Sony ARW Decoder

[English](README.md) | 中文

本仓库为 Sony RAW 完整解码器：以 Imaging Edge Edit 的显影管线为蓝本逐阶段复现，输出以相机直出为参照，针对 CPU 与 GPU 分别提供基于 Rust 和 Metal 的加速后端。在速度上，解码一张 3300 万像素的 RAW 仅需 75 ms，比 macOS 自带的 RAW 解码（Core Image）快 11–22 倍，比 LibRaw 快 40–80 倍；在内存占用上，单张峰值约 0.83 GB，低于 Core Image 同样输出 16 位时的 1.58 GB（LibRaw 为 0.61 GB）；并支持 sRGB、Adobe RGB、Display P3、ProPhoto RGB 多种色彩空间，以及 TIFF、JPEG、PNG、HEIF 多种格式输出。

## 处理管线

```text
RAW → [RawNR] → ITP → [DRO] → [GeoCorr] → [SSCS]
    → LinearMatrix16 → MainGamma → RGB2YCC → AreaComp → [ChromaSuppress]
    → Sharpness → YGamma → [HueSaturation] → Spica → YCC2RGB
    → [Marble] → [Effect] → [ToneCurve] → [ShadeCorrection]
```

方括号中的阶段根据 ARW 中保存的显影设置运行；SSCS 与 ChromaSuppress 例外，相机直出不做这两步，因此默认不运行，`--highlights edit` 时运行。输出为解码器矩形对应的
16 位 sRGB 或 Adobe RGB 图像（另可输出本项目新增的 Display P3 和 ProPhoto RGB，Edit 没有这两个选项），可写入 TIFF、JPEG、PNG 或 HEIF（`.hif`，10 位，和 Sony 相机保存的格式相同）。

当前实现以 A7 V 样张和显影设置为初始标定目标。各阶段已与 Sony Imaging Edge Edit 4.0 / 4.1 逐位核对（RGGB 传感器）；整体输出以相机直出为参照，用色卡和实拍照片验证，详见 docs/pipeline.md §10.3。

## 后端

| 后端  | 平台  | 要求                                        |
| ----- | ----- | ------------------------------------------- |
| NumPy | 任意  | Python 3.11+、`requirements.txt`            |
| Rust  | 任意  | Rust 工具链                                 |
| Metal | macOS | Xcode Command Line Tools、支持 Metal 的 Mac |

Rust 与 Metal 可选，也可以组合使用。NumPy 始终作为逐阶段和整图验证的参考。

## 速度

测试环境：Mac mini，Apple M6，16 GB 内存，macOS 27.2，arm64。输入为 `chart/SonyChart.ARW`，输出为完整的 7008 × 4672 图像。

每个后端先运行一次以初始化缓存和 Metal 着色器，再在同一进程中测量第二次完整解码（Metal 取预热 3 次后 9 次解码的中位数）。

| 后端 | 耗时 | 相对 NumPy 加速 |
| --- | ---: | ---: |
| NumPy | 24.92 秒 | 1.0× |
| Rust | 1.79 秒 | 13.9× |
| Metal（`--metal`） | 0.075 秒 | 332× |

使用 `--metal` 时，如果 Rust 库可用，会自动使用 Rust 读取 RAW 并构建外观表，所以 `--rust --metal` 与它是同一条路径。GPU 阶段分为 RAW 链和渲染链两条 command buffer 提交；GeoCorr 与 SSCS、Marble 与 Effect/ToneCurve/ShadeCorrection 分别在同一个 kernel 内完成。压缩 RAW 在后台线程解码，GPU 同时处理已经解出的行（`SONYLOOK_RAW_STREAM=0` 改为先解码完再开始）。设置 `SONYLOOK_METAL_BATCH=0` 可恢复逐阶段提交。三种后端的整图输出哈希完全一致。新进程中的首次解码还会包含初始化、表构建和 Metal 着色器编译，因此耗时更高。以上数据基于一张合成 ARW，不代表所有相机型号或显影设置下的通用性能。

在同一台机器上，与 macOS 自带的 RAW 解码（Core Image 的 `CIRAWFilter`，照片和预览都用它）以及 LibRaw 0.22.2（通过 rawpy 0.28，该版本为单线程）对比：从文件到内存中的全分辨率 16 位位图，取热运行中位数。Core Image 每次解码都新建 `CIRAWFilter`（同一个 filter 反复渲染同一个文件时，用的是缓存的解码结果，不算真正解码）；LibRaw 执行 `imread` + `postprocess(output_bps=16)`，分别用默认的 AHD 去马赛克和最快的线性插值：

| 文件 | Core Image 默认 | Core Image 关闭降噪、锐化、局部色调 | LibRaw AHD | LibRaw 线性插值 | 本解码器（`--metal`） |
| --- | ---: | ---: | ---: | ---: | ---: |
| SonyChart（A7M5） | 857 ms | 1184 ms | 4995 ms | 3047 ms | 74 ms |
| DSC02741（A7M5） | 1522 ms | 1370 ms | 5988 ms | 3680 ms | 76 ms |
| DSC06657（A7M5） | 1469 ms | 1376 ms | 6034 ms | 3667 ms | 75 ms |
| DSC05212（A7M4） | 1546 ms | 1410 ms | 5426 ms | 3482 ms | 69 ms |

DSC02741 单张解码的峰值内存（每项单独一个进程，`/usr/bin/time -l` 的 peak memory footprint，含 GPU 内存）：

| 解码器与输出 | 峰值内存 |
| --- | ---: |
| 本解码器，`--metal`，16 位 TIFF 或 JPEG | 0.83 GB |
| Core Image，16 位 RGBA | 1.58 GB |
| Core Image，8 位 RGBA | 1.06 GB |
| LibRaw，16 位 RGB | 0.61 GB |
| LibRaw，8 位 RGB | 0.50 GB |

Core Image 和 LibRaw 用的是各自的去马赛克和色调（Core Image 还有自己的降噪），LibRaw 还保留整个传感器区域（7168 × 5120，而不是 7008 × 4672 的裁切）；本解码器复现的是 Sony 的流程，所以这里只比较耗时和内存，画面不能直接比较。

## 安装

```bash
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -r requirements.txt
```

构建 Rust 内核：

```bash
cargo build --release --manifest-path rust/sonylook_core/Cargo.toml
```

在 macOS 上构建 Metal 桥接层：

```bash
./metal/build.sh
```

完整 Metal 工具链可用时，脚本会预编译 `metal/shaders/shaders.metallib`；否则生成动态库，并让运行时从 shader 源码编译。

## 解码

```bash
python3 base/sony_decode.py photo.ARW --out out.tif
python3 base/sony_decode.py photo.ARW --rust --out out-rust.tif
python3 base/sony_decode.py photo.ARW --metal --out out-metal.tif
python3 base/sony_decode.py photo.ARW --rust --metal --out out-fast.tif
```

Python 调用：

```python
from base.sony_decode import decode

image = decode("photo.ARW")
fast_image = decode("photo.ARW", rust=True, metal=True)
export_image = decode("photo.ARW", metal=True, output_bits=16)
```

`decode()` 默认返回形状为 `(height, width, 3)`、范围 0–16383 的 `uint16` 数组。`output_bits=16` 返回 Edit 的 16 位 TIFF 编码，即每个值严格等于 14 位结果左移两位；Metal 路径在最终交错 kernel 中直接完成这一步。命令行导出 16 位 TIFF 时自动采用该路径。

`--color-space`（Python 中为 `output=`）可选 `srgb`（默认）、`adobe`、`p3`、`prophoto`。Edit 只提供前两种。Display P3 和 ProPhoto RGB 按 Edit 构造 Adobe RGB 的方式实现：在 LinearMatrix16 表中多乘一个矩阵（这张表作用在相机 RGB 上，所以 sRGB 之外的颜色能保留下来），并给 Marble 和 BW/Sepia 效果配上对应的矩阵；ProPhoto 最后重新编码为 gamma 1.8。四种 ICC 配置文件都按标准原色和曲线由代码生成；sRGB 和 Adobe RGB 的曲线与 Edit 自带的配置文件完全相同，原色相差在 ΔE 0.05 以内。这两种没有 Edit 导出图可以对照，各后端之间逐位一致。Marble 和效果会截到 Edit 的工作色域，它包含 Display P3，但不能完整包含 ProPhoto。细节见 docs/pipeline.md §9.2。ProPhoto 建议输出 16 位 TIFF。

`--highlights`（Python 中为 `highlights=`）可选 `camera`（默认）或 `edit`。相机直出的 JPEG 和 HEIF 不做 Edit 的近饱和色彩抑制（SSCS）和 ChromaSuppress。若做这两步，G 通道已饱和的亮色会偏白：色卡曝光包围中，+2 EV 与相机 HEIF 的 ΔE2000 为 1.93，不做时为 0.43。`edit` 保留这两步，与 Edit 导出逐位一致。细节见 docs/pipeline.md §10.3。

`--dro-model`（Python 中为 `dro_model=`）可选 `camera`（默认）或 `edit`，只在 DRO 开启时起作用。两者使用同一条 ARW 曲线和同一个逐像素算子，区别在于局部亮度的估计：`camera` 采用按相机 HEIF 拟合的网格（按亮度加权的样本、半档宽且同时累加样本值的 bin、更宽的亮度维核），色卡 Lv1–Lv5 与相机 HEIF 的 ΔE2000 由 0.55–1.86 降至 0.27–0.72，平滑区域与相机的一致性也明显提高；`edit` 与 Edit 的网格逐位一致。两者都读取整个传感器缓冲区，与 Edit 相同。细节见 docs/pipeline.md §3.3。

### 批量处理

```bash
python3 base/sony_batch.py ~/Pictures/2025 more.ARW --out-dir out --metal
python3 base/sony_batch.py ~/Pictures/2025 --out-dir out --format tif --metal --skip-existing
python3 base/sony_batch.py ~/Pictures/2025 --out-dir out-p3 --metal --color-space p3
python3 base/sony_batch.py ~/Pictures/2025 --out-dir out-hif --format hif --metal
```

`--format hif`（或 `--out photo.hif`）通过 Core Image 写出 10 位 HEIF，由 Apple 的硬件编码器做 HEVC 编码（仅限 macOS；`--bits 8` 输出 8 位）。A7M5 图像在默认质量 95 下约 6–9 MB，与解码结果相比 PSNR 51 dB（同质量的 8 位 JPEG 为 4 MB、46 dB）。写一张约 150 ms（JPEG 约 100 ms，都比解码一张慢），所以批量处理默认用两个写出线程；输出 HEIF、文件已在系统缓存中时每张约 79 ms。ICC 配置文件和其他格式一样嵌入。

文件夹会递归查找，输出保留原来的子目录结构；默认输出 8 位 JPEG（`--format tif` 输出 16 位 TIFF，每张约 200 MB）。解码选项与 `sony_decode.py` 相同。读取后面的文件、解码、编码写出前面的图片三者同时进行，输出与逐张单独解码逐字节相同。出错的文件（例如损坏的文件）会报告并跳过。在上面的 Mac mini 上，输出 JPEG 并使用 `--metal` 时：文件已在系统缓存中，每张约 87 ms（逐张串行为 151 ms）；从外接 USB 固态硬盘读取，每张 126–162 ms（串行为 267–354 ms），此时瓶颈是硬盘约 300 MB/s 的读取速度。结束时的汇总行会把时间拆分为解码、等待读取和等待写出。

## 验证

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

Rust 与 Metal 输出与 NumPy 逐位一致。例外是 Metal ITP 在随机的满量程噪声输入上，
少数方向图求值可能相差 1；已验证的真实 ARW 整图中未出现该差异。

## 目录

| 路径                  | 内容                                   |
| --------------------- | -------------------------------------- |
| `base/`               | ARW 读取、解码阶段、外观引擎和后端绑定 |
| `rust/sonylook_core/` | Rust RAW 读取与加速核心                |
| `metal/`              | Objective-C 桥接层与 Metal 着色器      |
| `tests/`              | 后端逐阶段差分验证                     |
| `docs/pipeline.md`    | 完整阶段顺序、设置与实现说明           |
| `docs/metal.md`       | Metal 构建、内存与验证                 |
| `chart/SonyChart.ARW` | 可用于本地测试的合成 ARW               |

## 许可与兼容性

源码使用 MIT 许可证。本项目为完全独立运行的解码器，无需安装 Sony Imaging Edge Edit；本项目与 Sony 无关联，锐化等阶段仅借鉴了相关处理思路。
