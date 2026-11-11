# Sony ARW Decoder

[English](README.md) | 中文

本仓库为 Sony RAW 完整解码器，基于 Imaging Edge Edit 的显影管线构建。NumPy 是参考实现，并另提供基于 Rust 与 Metal 的加速后段。

## 处理管线

```text
RAW → [RawNR] → ITP → [DRO] → [GeoCorr] → SSCS
    → LinearMatrix16 → MainGamma → RGB2YCC → AreaComp → ChromaSuppress
    → Sharpness → YGamma → [HueSaturation] → Spica → YCC2RGB
    → [Marble] → [Effect] → [ToneCurve] → [ShadeCorrection]
```

方括号中的阶段根据 ARW 中保存的显影设置运行。输出为解码器矩形对应的
16 位 sRGB 或 Adobe RGB 图像，可写入 TIFF、JPEG 或 PNG。

当前实现使用 A7 V 样张和显影设置作为初始标定目标，并已对 Sony Imaging
Edge Edit 4.0 / 4.1 的 RGGB 传感器结果进行验证。

## 后端

| 后端  | 平台  | 要求                                        |
| ----- | ----- | ------------------------------------------- |
| NumPy | 任意  | Python 3.11+、`requirements.txt`            |
| Rust  | 任意  | Rust 工具链                                 |
| Metal | macOS | Xcode Command Line Tools、支持 Metal 的 Mac |

Rust 与 Metal 可选，也可以组合使用。NumPy 始终作为逐阶段和整图验证的参考。

## 速度

测试环境：Mac mini，Apple M6，16 GB 内存，macOS 27.2，arm64。输入为 `chart/SonyChart.ARW`，输出为完整的 7008 × 4672 图像。

每个后端先运行一次以初始化缓存和 Metal 着色器，再在同一进程中测量第二次完整解码。

| 后端 | 耗时 | 相对 NumPy 加速 |
| --- | ---: | ---: |
| NumPy | 24.92 秒 | 1.0× |
| Rust | 1.79 秒 | 13.9× |
| Metal | 0.090 秒 | 276.9× |
| Rust + Metal | 0.090 秒 | 276.9× |

使用 `--metal` 时，如果 Rust 库可用，会自动使用 Rust 读取 RAW 并构建外观表。GPU 阶段分为 RAW 链和渲染链两条 command buffer 提交；GeoCorr 与 SSCS、Marble 与 Effect/ToneCurve/ShadeCorrection 分别在同一个 kernel 内完成。压缩 RAW 在后台线程解码，GPU 同时处理已经解出的行（`SONYLOOK_RAW_STREAM=0` 改为先解码完再开始）。设置 `SONYLOOK_METAL_BATCH=0` 可恢复逐阶段提交。四种路径的整图输出哈希完全一致。新进程中的首次解码还会包含初始化、表构建和 Metal 着色器编译，因此耗时更高。以上数据基于一张合成 ARW，不代表所有相机型号或显影设置下的通用性能。

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

## 验证

```bash
python3 tests/verify_raw_decode.py photo.ARW
python3 tests/verify_rust_nr_dro.py photo.ARW
python3 tests/verify_metal_itp.py photo.ARW
python3 tests/verify_metal_geo.py photo.ARW
python3 tests/verify_metal_geo_sscs.py photo.ARW
python3 tests/verify_metal_batch.py photo.ARW
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

源码使用 MIT 许可证。本项目与 Sony 无关联，不包含 Sony 二进制或专有应用数据。
