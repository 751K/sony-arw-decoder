# Metal 加速

`metal/` 为整图解码器 `sony_decode.py` 提供可选的 Metal GPU 实现，适用于 macOS。NumPy 实现仍是参考实现和默认后端。除 ITP 外，各 Metal 阶段的输出与 NumPy 逐位一致（见[输出一致性](#输出一致性)）。

## 覆盖范围

管线顺序见[处理管线 §9.1](pipeline.md#91-流程sony_decodedecode)。在 RAW 读取之后，所有阶段都有 Metal 实现：

| 选项 | 阶段 | 说明 |
| --- | --- | --- |
| `--metal-nr` | RawNR | 设置中开启降噪时运行 |
| `--metal-itp` | ITP 去马赛克 | 整图一次处理，按 Sony 的 1024×1024 分块边界计算；使用标准 ITP 表 |
| `--metal-dro` | DRO | 设置中开启 DRO 时运行 |
| `--metal-geo` | GeoCorr（畸变、横向色差） | 镜头校正开启时运行 |
| `--metal-sscs` | SSCS | 整数路径 |
| `--metal-look` | LinearMatrix16 → MainGamma → RGB2YCC → AreaComp → ChromaSuppress | 一次 GPU 调用完成 |
| `--metal-spatial` | Sharpness → YGamma → Spica → YCC2RGB | 需同时指定 `--metal-look`；条件与 Rust 空间路径相同：Sharpness 和 Spica 均开启、色相 / 饱和度为 0，否则由 CPU 处理 |
| `--metal-marble` | Marble | 亮度路径（清晰度）和降噪开启时的色度路径；清晰度超出 0–109 时由 CPU 处理 |
| `--metal-post` | Effect（BW、Sepia）→ ToneCurve → ShadeCorrection | 一次 GPU 调用完成 |
| `--metal` | 以上全部 | |

白平衡增益的计算量可以忽略，仍在 CPU 上进行。Metal 路径不依赖 Rust 内核；同时指定 `--rust` 时，RAW 由 Rust 多线程解码，外观表也由 Rust 构建，因此两者一起使用最快。

## 构建

需要 Xcode Command Line Tools（`clang`）和支持 Metal 的 Mac：

```bash
clang -O2 -fobjc-arc -dynamiclib -framework Foundation -framework Metal \
  metal/bridge.m -o metal/libsonylook_metal.dylib
```

`SONYLOOK_METAL_LIB` 可指定其他路径的动态库。着色器源码（`metal/shaders/*.metal`）在首次调用时由 Metal 运行时编译，使用安全数学模式和精确数学函数（`MTLMathModeSafe`、`MTLMathFloatingPointFunctionsPrecise`）。修改 `.metal` 文件后无需重新构建动态库；修改 `bridge.m` 后需要重新构建。

## 使用

```bash
python3 base/sony_decode.py /path/to/A7M5.ARW --rust --metal --out out.tif
```

也可以只启用部分阶段，例如 `--rust --metal-itp --metal-geo`。在 Python 中使用时，`metal=True` 启用全部 Metal 阶段，也可以单独传入 `metal_itp=True` 等关键字参数：

```python
from base.sony_decode import decode
for f in files:
    img = decode(f, rust=True, metal=True)
```

批量处理时，宜在同一进程中连续调用 `decode()`：Metal 初始化、着色器编译和缓冲池只在进程内第一次调用时建立，后续调用直接复用。

指定 Metal 选项而 Metal 不可用时报错，不会退回 CPU。

## 内存

- Metal 阶段的输出位于 GPU 可见的共享内存中（`metal_itp.empty`），以普通 NumPy 数组返回。后续阶段直接在原处读取这些数组，并把结果写入新的共享内存数组，相邻阶段之间不发生复制。最终的交错 RGB 图像也在 GPU 上组装。
- 临时缓冲区放入缓冲池，供后续调用复用，上限 3 GB；一次全分辨率解码后约占用 1.1 GB。`metal_itp.trim()` 释放空闲缓冲区。
- 每个 ARW 的元数据（SR2 子 IFD、机型、Spica 设置等）按（路径、修改时间、文件大小）缓存；文件内容只读取一次，由各元数据读取模块共用。

## 输出一致性

NumPy 中的部分运算在 float64 下进行，或依赖 NumPy 的正确舍入；Metal 没有 double 类型，内置的除法和 `exp2` 也不保证正确舍入。各阶段的处理方式如下：

- **RawNR、SSCS、GeoCorr、Marble、外观链和空间阶段**：整数运算，或与参考实现顺序相同的 float32 运算（显式 FMA），结果逐位一致。AreaComp 中 float64 的截断用整数运算等价实现。
- **DRO**：除法使用自行实现的正确舍入 float32 除法（在近似商附近的候选值中，选 FMA 余数最小者）；`exp2` 使用 64 位整数定点运算，依次舍入到 53 位和 24 位，与 NumPy 的 `float32(exp2(float64(e)))` 一致。
- **Effect 和 ShadeCorrection**：参考实现中的亮度和增益是 float64 运算。Metal 用双 float（hi + lo）运算求值；距离舍入边界过近、无法确定结果的像素会被记录下来，由 CPU 用参考实现重新计算。这类像素通常很少；超过容量时整张图改由 CPU 处理。
- **ITP**：方向图中有几处求和，Metal 用 float32，Rust 参考实现用 float64，因此个别输出值可能相差 1。各中间阶段的逐块比较完全一致；对随机的满量程 16 位噪声输入做整图比较，2048×2048 中有不到 10 个值相差 1。下文列出的真实 ARW 整图解码中没有出现差异。

## 性能

测量环境为 Mac mini（Apple M6，16 GB），A7M5 7008×4672，在同一进程中再次解码同一文件：

| 文件 | 设置 | NumPy | `--rust` | `--metal` | `--rust --metal` |
| --- | --- | ---: | ---: | ---: | ---: |
| DSC03591 | NR、DRO 关闭 | 28.1 s | 1.7 s | 0.40 s | 0.15 s |
| DSC05740 | BW，NR、DRO 开启 | 37.0 s | 2.8 s | 0.48 s | 0.19 s |

单独使用 `--metal` 时 RAW 由 rawpy 读取（约 0.3 s），外观表由 Python 构建；加上 `--rust` 后这两项分别约为 0.02 s 和 0.03 s。

DSC05740 的分阶段耗时（`--rust --metal`）：

| 阶段 | 耗时 |
| --- | ---: |
| 读取 RAW | 21 ms |
| RawNR | 8 ms |
| ITP | 72 ms |
| DRO | 19 ms |
| GeoCorr | 5 ms |
| SSCS | 4 ms |
| 外观链 + 空间阶段 | 20 ms |
| Marble | 17 ms |
| Effect / ToneCurve / ShadeCorrection | 5 ms |

- 同一进程中首次解码某个文件时，需要读取文件并解析元数据，每张另需约 0.1–0.3 s。
- 进程中第一次使用 Metal 时需要编译着色器。命令行单张解码（含 Python 启动、Metal 初始化和写 TIFF）约 0.35–0.45 s；开机后首次运行，文件和着色器缓存均未命中时，约 1.8 s。

## 验证

每个 Metal 阶段都有对应的差分验证脚本，比较 Metal 与 NumPy（部分阶段同时比较 Rust）的输出。除 ITP 的整图比较允许相差 1 外，均要求完全相同：

```bash
python3 tests/verify_metal_itp.py photo.ARW
python3 tests/verify_metal_dro.py
python3 tests/verify_metal_geo.py photo.ARW
python3 tests/verify_metal_sscs.py photo.ARW
python3 tests/verify_metal_look.py photo.ARW
python3 tests/verify_metal_spatial.py photo.ARW
python3 tests/verify_metal_marble.py photo.ARW [nr_photo.ARW]
python3 tests/verify_metal_post.py photo.ARW
python3 tests/verify_metal_rawnr.py photo.ARW
python3 tests/verify_raw_decode.py photo.ARW [folder ...]
```

- `verify_metal_dro.py` 另行检查 3.1 亿个 `exp2` 输入和 520 万个除法输入。
- `verify_metal_marble.py` 的第二个参数为开启降噪的 ARW，用于检查色度路径的全分辨率输出。
- 整图检查：DSC03591 和 `Test/DSC05740`–`DSC05749`（含 BW、NR、DRO 开启的文件）在 `--rust --metal` 下的输出，与 `--rust` 逐位一致。

## 源码

| 文件 | 内容 |
| --- | --- |
| `metal/bridge.m` | Objective-C 桥接层：设备、着色器加载、缓冲池、共享内存数组，以及各阶段的 C 入口 |
| `metal/shaders/itp.metal` | ITP |
| `metal/shaders/nr.metal` | RawNR |
| `metal/shaders/dro.metal` | DRO，以及正确舍入的除法和定点 `exp2` |
| `metal/shaders/geo.metal` | GeoCorr |
| `metal/shaders/sscs.metal` | SSCS |
| `metal/shaders/look.metal` | LinearMatrix16 至 ChromaSuppress |
| `metal/shaders/spatial.metal` | Sharpness、YGamma、Spica、YCC2RGB |
| `metal/shaders/marble.metal` | Marble 亮度路径和色度路径 |
| `metal/shaders/post.metal` | Effect、ToneCurve、ShadeCorrection，以及 RGB 交错 |
| `base/metal_itp.py` | 动态库加载、共享内存数组（`empty`、`trim`）和 ITP 封装 |
| `base/metal_rawnr.py`、`metal_dro.py`、`metal_geo.py`、`metal_sscs.py`、`metal_look.py` | 其余阶段的 Python 封装 |
