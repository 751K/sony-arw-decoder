# Sony ARW Decoder

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
clang -O2 -fobjc-arc -dynamiclib -framework Foundation -framework Metal \
  metal/bridge.m -o metal/libsonylook_metal.dylib
```

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
```

`decode()` 返回形状为 `(height, width, 3)` 的 `uint16` 数组。

## 验证

```bash
python3 tests/verify_raw_decode.py photo.ARW
python3 tests/verify_rust_nr_dro.py photo.ARW
python3 tests/verify_metal_itp.py photo.ARW
python3 tests/verify_metal_geo.py photo.ARW
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
