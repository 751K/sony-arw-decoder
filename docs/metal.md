# Metal 加速

`metal/` 为整图解码器 `sony_decode.py` 提供可选的 Metal GPU 实现，适用于 macOS。NumPy 实现仍是参考实现和默认后端。除 ITP 外，各 Metal 阶段的输出与 NumPy 逐位一致（见[输出一致性](#输出一致性)）。

## 覆盖范围

管线顺序见[处理管线 §9.1](pipeline.md#91-流程sony_decodedecode)。在 RAW 读取之后，所有阶段都有 Metal 实现：

| 选项 | 阶段 | 说明 |
| --- | --- | --- |
| `--metal-nr` | RawNR | 设置中开启降噪时运行 |
| `--metal-itp` | ITP 去马赛克 | 整图一次处理，按 384×384 分块、12 px halo 计算（Rust/NumPy 用 Sony 的 1024×1024 分块，halo 覆盖算子范围，整图结果一致）；每块 3 个 kernel，最后一个直接写入整图；使用标准 ITP 表 |
| `--metal-dro` | DRO | 设置中开启 DRO 时运行 |
| `--metal-geo` | GeoCorr（畸变、横向色差） | 镜头校正开启时运行；与 SSCS 同时启用时融合为一个 kernel |
| `--metal-sscs` | SSCS | 整数路径；与 GeoCorr 融合时直接输出交错 RGB |
| `--metal-look` | LinearMatrix16 → MainGamma → RGB2YCC → AreaComp → ChromaSuppress | 一次 GPU 调用完成 |
| `--metal-spatial` | Sharpness → YGamma → Spica → YCC2RGB | 需同时指定 `--metal-look`；条件与 Rust 空间路径相同：Sharpness 和 Spica 均开启、色相 / 饱和度为 0，否则由 CPU 处理 |
| `--metal-marble` | Marble | 亮度路径（清晰度）和降噪开启时的色度路径；清晰度超出 0–109 时由 CPU 处理 |
| `--metal-post` | Effect（BW、Sepia）→ ToneCurve → ShadeCorrection | 一次 GPU 调用完成；与 Marble 同时启用时融合到 Marble 的最终 kernel |
| `--metal` | 以上全部 | |

白平衡增益的计算量可以忽略，仍在 CPU 上进行。Metal 路径不依赖 Rust 内核；如果 Rust 库可用，使用 `--metal` 时会自动采用 Rust RAW 读取和表构造。显式指定 `rust=False` 可保留仅用 rawpy 和 Python 表构造的行为。

## 构建

需要 macOS 开发工具（`clang`）和支持 Metal 的 Mac：

```bash
./metal/build.sh
```

完整 Metal 工具链可用时，脚本把 `metal/shaders/*.metal` 预编译为 `shaders.metallib`；不可用时仍生成动态库，并由运行时编译 shader 源码，使用安全数学模式和精确数学函数（`MTLMathModeSafe`、`MTLMathFloatingPointFunctionsPrecise`）。脚本不会保留可能落后于源码的旧 metallib。`SONYLOOK_METAL_LIB` 可指定其他路径的动态库。修改 `.metal` 文件后需再次运行构建脚本；修改 `bridge.m` 后同样需要重新构建。

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

- Metal 阶段的输出位于 GPU 可见的共享内存中（`metal_itp.empty`），以普通 NumPy 数组返回。后续阶段直接在原处读取这些数组，并把结果写入新的共享内存数组，相邻阶段之间不发生复制。最终的交错 RGB 图像也在 GPU 上组装；16 位 TIFF 路径在交错时直接执行 `value << 2`，不再分配一整幅转换后的临时数组。
- 临时缓冲区放入缓冲池，供后续调用复用，上限 3 GB；一次全分辨率解码后约占用 1.1 GB。`metal_itp.trim()` 释放空闲缓冲区。
- 每个 ARW 的元数据（SR2 子 IFD、机型、Spica 设置等）按（路径、修改时间、文件大小）缓存；文件内容只读取一次，由各元数据读取模块共用。

## 输出一致性

NumPy 中的部分运算在 float64 下进行，或依赖 NumPy 的正确舍入；Metal 没有 double 类型，内置的除法和 `exp2` 也不保证正确舍入。各阶段的处理方式如下：

- **RawNR、SSCS、GeoCorr、Marble、外观链和空间阶段**：整数运算，或与参考实现顺序相同的 float32 运算（显式 FMA），结果逐位一致。AreaComp 中 float64 的截断用整数运算等价实现。
- **GeoCorr+SSCS 融合**：重采样结果先写入与独立 kernel 相同的 ushort 值，再执行同一整数 SSCS 逻辑，因此输出与两段式 Metal、Rust 和 NumPy 完全一致。
- **Marble+Post 融合**：Marble 的最终像素计算直接进入 Effect、ToneCurve 和 ShadeCorrection。不确定像素同时记录其 Post 前 RGB，CPU 仍以阶段输入重算；结果与分离 kernel 逐位一致。
- **DRO**：除法使用自行实现的正确舍入 float32 除法（在近似商附近的候选值中，选 FMA 余数最小者）；`exp2` 的精确实现使用 64 位整数定点运算，依次舍入到 53 位和 24 位，与 NumPy 的 `float32(exp2(float64(e)))` 一致。逐像素路径先用双 float 快速计算（查表加短级数，误差小于 2⁻³⁷），只有结果距 float32 舍入边界不足 2⁻³⁴ 时，才把像素记入列表，由另一个 kernel 用精确实现重算；列表溢出时由间接 dispatch 对整图重算，不需要 CPU 往返。真实照片中这类像素通常少于 0.1%。验证脚本逐个检查了 |e| < 24 的全部 float。
- **Effect 和 ShadeCorrection**：参考实现中的亮度和增益是 float64 运算。Metal 用双 float（hi + lo）运算求值；距离舍入边界过近、无法确定结果的像素会被记录下来，由 CPU 用参考实现重新计算。这类像素通常很少；超过容量时整张图改由 CPU 处理。
- **ITP**：方向图中有几处求和，Metal 用 float32，Rust 参考实现用 float64，因此个别输出值可能相差 1。各中间阶段的逐块比较完全一致；对随机的满量程 16 位噪声输入做整图比较，2048×2048 中有不到 10 个值相差 1。下文列出的真实 ARW 整图解码中没有出现差异。

## 性能

测量环境为 Mac mini（Apple M6，16 GB），A7M5 7008×4672，在同一进程中再次解码同一文件：

| 后端 | `SonyChart.ARW`，7008×4672 |
| --- | ---: |
| NumPy | 24.92 s |
| `--rust` | 1.79 s |
| `--metal`（自动使用 Rust 读取和表构造） | 0.088 s |
| `--rust --metal` | 0.088 s |

`--metal` 在 Rust 库可用时自动使用 Rust RAW 读取和表构造。热运行 9 次的中位数为 0.088 s；四种后端生成的整图哈希完全一致。

Metal 阶段按两条 command buffer 提交：RawNR → ITP → DRO → GeoCorr/SSCS 为 RAW 链；front/spatial → Marble/Post → interleave 为渲染链。Post 需要 CPU 回填时会在融合调用前完成已排队的 command buffer。设置 `SONYLOOK_METAL_BATCH=0` 可恢复逐阶段提交。

ITP 每个分块只用 3 个 kernel：`load_scale_tile` 生成 s；`itp_edges` 一次算出 ah/av 和 eh/ev/e1/e2 六个平面（马赛克平面就是奇数相位上的 s，各抽头与中心同相位，因此不再单独存储）；`itp_final` 在每个像素上就地计算权重、方向、绿色插值和色度插值（色度输入在它读取的 ±1 邻点上由 s 重算），直接写入整图。原先只在本像素读取的十余个中间平面不再落地，ITP 由约 45 ms 降到约 25 ms，结果逐位不变。`SONYLOOK_ITP_FUSED=1` 使用原来的 9 个 kernel，`=0` 使用逐个 kernel，用于对照。

Marble 的颜色转换与半宽色度合为一个 kernel（`marble_in_half`）：每个线程转换一对像素，写出 Y/Cb/Cr 和这对像素的半宽色度，不再整平面重读 Cb/Cr；GPU 时间由 7.6 ms 降到 5.5 ms。`SONYLOOK_MARBLE_HALFC=1` 恢复两个独立 kernel。

Sharpness、YGamma、Spica 和 YCC2RGB 合为一个 kernel（`spatial_fused`）。每个线程组负责 32×32 的输出块：先把带 6 像素边的 Y（44×44）读进线程组内存，再依次算水平模糊（44×38）、竖直模糊 + Sharpness + YGamma（38×38），最后做 Spica 和 YCC2RGB。这样不再生成整幅的 float `hb` 平面和 `yg` 平面（共约 200 MB 临时缓冲），运算顺序与原来三个 kernel 相同。这几步主要受计算限制，合并本身只快约 0.2 ms；省时间的是 Spica 改用 32 位整数：GPU 上的 64 位整数是软件模拟的，而截断结果只要落在 int 范围内就与 64 位相同，超出时最后的 clamp 也得到同样的值（增益曲线在参数端点超过 2^30 时退回 64 位）。合计由 14.5 ms 降到 12.4 ms。`SONYLOOK_SPATIAL_FUSED=0` 恢复三个独立 kernel。其余只在本像素读取的中间平面已检查过：RawNR 的 hp 平面改为在滤波时重算没有收益（两个 kernel 受计算限制）；空间阶段的 Cr/Cb 和 Marble 输出读取的 Y/Cb/Cr 若就地重算，需要重复整个 front 或颜色转换，计算量大于省下的内存流量，因此保留。

`SONYLOOK_GPU_PROFILE=1` 让每个 encoder 在开始和结束时记录 GPU 时间戳；`sonylook_metal_profile_collect()` 在 GPU 工作完成后累计，`sonylook_metal_profile_report(runs)` 按 kernel 打印每次运行的 GPU 毫秒数（不设置时不采样）。

RAW 读取与 GPU 并行：压缩 RAW 由 Rust 在后台线程解码，直接写入 GPU 可见内存。每解完一行分块（512 行），Rust 回调 bridge，把一个 `MTLSharedEvent` 设为已解码的行数。RAW 链在 RAW 解完之前就提交，GPU 按 ITP 的分块行推进：每行分块之前等待所需的 RAW 行，再依次做 RawNR（`nr_split` 读 ±2 行，`nr_filter` 读 ±8 行）和这一行的 ITP。RawNR 和 ITP 按行跨度与偏移直接读取完整传感器 RAW 中的裁剪区域，不再先拷贝出裁剪图（省去约 3 ms 的 `raw_crop_rows`，GPU 时间由约 86 ms 降到约 82 ms）。DRO 网格同样就地读取，在辅助队列上等到裁剪区域的最后一行解码完成后开始。只有 ITP、RawNR、DRO 都用 Metal 且白平衡不是取样模式 3 时才启用；`SONYLOOK_RAW_STREAM=0` 可关闭。引入时 SonyChart 由 0.117 s 降到 0.110 s，开启 NR 和 DRO 的 DSC05740 由 0.127 s 降到 0.109 s。解码失败时会解除 GPU 的等待并报告失败的分块。

DSC05740 的分阶段耗时（`--metal`，`SONYLOOK_METAL_BATCH=0 SONYLOOK_RAW_STREAM=0` 逐阶段测量；默认设置下 RAW 读取与 RawNR/ITP 重叠）：

| 阶段 | 耗时 |
| --- | ---: |
| 读取 RAW | 21 ms |
| RawNR + ITP | 30 ms |
| DRO | 6 ms |
| GeoCorr + SSCS | 5 ms |
| 外观链 + 空间阶段 | 20 ms |
| Marble + Effect / ToneCurve / ShadeCorrection | 17 ms |

- 同一进程中首次解码某个文件时，需要读取文件并解析元数据，每张另需约 0.1–0.3 s。
- 有 `shaders.metallib` 时直接加载预编译库；否则进程中第一次使用 Metal 时需要编译 shader 源码。命令行单张解码（含 Python 启动、Metal 初始化和写 TIFF）约 0.22 s，其中进程内第一次解码约 0.11 s（热运行 0.088 s）；开机后首次运行，文件和着色器缓存均未命中时约 1.8 s。

## 验证

每个 Metal 阶段都有对应的差分验证脚本，比较 Metal 与 NumPy（部分阶段同时比较 Rust）的输出。除 ITP 的整图比较允许相差 1 外，均要求完全相同：

```bash
python3 tests/verify_metal_itp.py photo.ARW
python3 tests/verify_metal_dro.py
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
| `metal/shaders/sscs.metal` | SSCS，以及 GeoCorr+SSCS 融合 kernel |
| `metal/shaders/look.metal` | LinearMatrix16 至 ChromaSuppress |
| `metal/shaders/spatial.metal` | Sharpness、YGamma、Spica、YCC2RGB |
| `metal/shaders/marble.metal` | Marble 亮度路径和色度路径 |
| `metal/shaders/post.metal` | Marble+Post 融合、Effect、ToneCurve、ShadeCorrection，以及带可选移位的 RGB 交错 |
| `base/metal_itp.py` | 动态库加载、共享内存数组（`empty`、`trim`）和 ITP 封装 |
| `base/metal_rawnr.py`、`metal_dro.py`、`metal_geo.py`、`metal_sscs.py`、`metal_look.py` | 其余阶段的 Python 封装 |
