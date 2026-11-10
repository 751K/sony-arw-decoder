# Viewer 的 AI 超分（Extended RAW Processing / SuperDemosaic）

本文是技术笔记：Viewer 的 AI 超分在管线里怎么做、已经确认的部分和还没解决的问题。本仓库**不提供**实现、模型或获取模型的方法。目标程序是 Imaging Edge Viewer 4.1.00.03062（arm64 切片 SHA-256 `219a46fb…8095`）。

原始实验、张量和 AXR 工具在本地 `sony/analysis/reverse/sdm/`，见[研究资料索引](research.md)。

## 1. 功能

Viewer 的 *File → Extended RAW Processing* 有两种模式：

- **Extended NR**：原分辨率，深度学习去马赛克 + 降噪。
- **Extended Hi-Res**：在上面基础上长宽各 ×2（A7M5 为 14016×9344）。

模型按相机区分，第一次使用时由 Viewer 下载。

## 2. 模型的输入输出

```
input  x : float32[1, 8, 256, 256]
output y : float32[1, 3, 1024, 1024]
```

每块覆盖 512 个原始像素，输出 1024，即 2× 线性去马赛克。

## 3. 张量格式（已确认）

拦截 `OrtGetApiBase`，把 `RunWithBinding`/`BindInput` 包一层，就能把 Viewer 真正喂给模型的张量 dump 出来。和 ARW 逐像素对照后，输入格式是：

```
ch0..3 = (raw - black) / 16384                逐相位，不乘白平衡、不乘 0x7311
ch4..7 = sqrt(13/64 · (raw - black) + 4) / 16384   Viewer 自己的噪声图
```

- **绿色两个相位逐位相同**（float32 误差 0），所以 `/16384`、黑电平、通道顺序都是确定的。
- **R/B 在进网络前被按镜头的横向色差重采样**：实测 R、B 相对 G 有一个随位置平滑变化的亚像素位移，和 `sony_lens.src_coords` 的 R−G、B−G 分量吻合到 0.02–0.05 px；G 完全不插值，也没有畸变校正。加上这一步后 R/B 的平均误差从 ~7 个计数降到 ~1 个计数。
- 噪声图对绿色相位也是**逐位精确**的；R/B 的噪声图跟随上面重采样后的值。
- 输出 `y` 与输入同尺度（未白平衡）。**白平衡在模型之后**：`linear = y × (s7311/32) × gain`，s7311 = 256 时就是 `y × 8 × gain`，量纲回到 `itp_scale`。
- 分块：Viewer 用 512×512、**步长 416**（重叠 96），17×12 = 204 块；输入 8×256×256 覆盖 512 个原始像素，输出 3×1024×1024，即 2× 线性去马赛克。

复现结果（DSC03591，ISO 100，对照 Viewer 导出的 **16 位** TIFF `DSC03591_ExHiRes (12).TIF`，全程 float64）：

- 几何：FFT 相关的最佳平移 (0, 0)，相关系数 0.99–0.9995。
- 8× 缩小 MAE **0.72/255**（DRO 增益混合 0.45）。按亮度分桶：暗部 +17、中间调 −47、亮部 +7（14 位计数，除以 64 就是 8 位色阶，即 +0.26 / −0.73 / +0.10）——中间调略暗、阴影略亮，是 DRO 形状的残差。
- 用 8 位导出对比时 MAE 是 0.84：因为 Viewer 的 8 位转换是**截断**的（16 位导出平均比 8 位高 ~0.47 级），而且 8 位版把噪声量化了。16 位参考才是干净口径。
- **Viewer 的输出里有额外的加性噪声**：早期测到亮部平坦区高通标准差 ~31–49 个 14 位计数，实验输出只有约 8；跨次导出基本确定。最初把它归因于 `sdm_noise_add`、随后排除 `ZcTaskDither` 的判断均已被后续验证修正。**2026-11-02 按 tile 对齐并用 MAD 稳健估计后，噪声来源已定位到 `ZcTaskDither`，见 §7.4。**
- 模型输入在**全亮度范围**都验证过：dump 到的 tile 覆盖 raw 80–8529，噪声图公式 `sqrt(13/64·p+4)` 在这些 tile 上全部逐位吻合。
- 上一版（输入格式还没对上、用估计噪声图 + 后置颗粒）报过 0.53/2.60 等数字，那些是在 float32 上求均值算出来的（393M 元素的 float32 均值会算错），而且输入本身不对，不能和这组直接比。

## 4. 复现思路

- RAW 按 512×512 分块、96 px 重叠（步长 416，和 Viewer 一致）；边界夹取补齐。
- 按 §3 的公式打包 8 个通道；R/B 的色差重采样用 `sony_lens.src_coords`，每块现算 256×256 的位移场。
- 每块的结果按升余弦窗加权融合，避免接缝（Viewer 自己怎么融合还没确认）。
- 输出乘 `(s7311/32) × gain` 回到 `itp_scale` 量纲，然后才做白平衡之后的各阶段。
- **DRO 按半强度应用**（相机 DRO 打开时）：把恒等和完整 Vatr 增益按 0.45 混合。全强度会明显过冲（8× MAE 3.0）；0.4 / 0.45 / 0.5 / 0.6 的 8× MAE 分别是 0.61（天空条带口径）/ 0.84（全图）/ 0.87（全图）/ 1.01（天空条带）。也试过 Vatr 真正的 `strength` 参数，形状不对（strength=0.8 时天空条带 MAE 2.5），所以 Viewer 用的更像是增益混合系数。**注意**：天空条带内部最优仍是 0.5，全图最优是 0.45，说明这个 DRO 模型和 Viewer 的真实阶段仍有形状差异；具体是哪个参数/哪一步还没从汇编确认。
- 关掉进网络前的色差重采样、改回“模型之后做完整 GeoCorr”，R/B 误差会明显变大（见 §3）。
- 颗粒不在网络之后另加：Viewer 的 `sdm_noise_add` 在 macOS 上没有被 dispatch，导出图里的颗粒来自后面的 `ZcTaskDither`（§7.4）。
- 之后运行 `sony_render.Look`（SSCS + 外观）。

## 5. 还没做的部分

- **噪声图公式只验证了 ISO 100**：`sqrt(13/64·(raw-black)+4)/16384` 在 DSC03591 上逐位精确，但 13/64 和 4 是否随 ISO 变化还不知道。要确认需要在 ISO 12800 的照片上再 dump 一次输入。
- **R/B 色差重采样的亚像素误差**：用 `sony_lens.src_coords` 的相对位移重采样后，R/B 与 dump 的平均差约 1e-4（0.1%），绿色是 0。剩余差异可能是插值核（bilinear 对双三次）或坐标取整方式不同，量级远小于 8 位输出的一级。
- **sdm_noise_add**：已经逆向出着色器（拦截 `vkCreateShaderModule`，dump 40 个 SPIR-V，见分析工作区 `analysis/reverse/sdm/spv_dump/`）。公式是 `Y=0.3R+0.6G+0.1B`、`sigma=sqrt(aY²+bY+c)`、Box-Muller 正态、`out += n*sigma*amount`。但**它在 macOS 上没有被 dispatch**：跟踪了 NR Standard 和 High 两次运行，实际只出现 shader 0/7/11/12/13，没有 036。所以 mac 上的颗粒差异不是这一步造成的。
- **高 ISO 颗粒对不上**：在 DSC05749（黑帧）上，无论用 sigma 通道、细节残差通道、负值通道，还是调输入尺度，都无法同时匹配 Viewer 的 mean 9.04 和 high-pass 9.24；后置高斯颗粒能匹配其中一个，但不能兼顾。说明 Viewer 的颗粒是空间相关的（固定图案/色度噪声），不是独立高斯，而且黑帧本身就是极端测试图。
- **CAS 锐化**：检查了 macOS 实际 dispatch 的 shader（0/7/11/12/13），它们分别是黑电平、缩放/夹紧、重采样，**没有 CAS**。在已经对齐、色调匹配的裁块上试了自适应 CAS（按局部对比度调制的 unsharp），任何强度都会让 MAE 变差（1.75 → 1.76–1.84）。说明 mac 上的锐化很可能已经包含在模型里，没有单独的 CAS 阶段可以补。
- **DRO 的确切参数**：见 §4。增益混合系数 π 0.45–0.5 只是拟合值；Viewer 内部是哪个参数、哪一步（`sdm_post`？），还没从汇编确认。
- **Sharpness / Spica / Marble**：没有接；Marble 在测试裁块上也没有改善。
- **分块融合**：网格已经确定（512、步长 416、204 块），但 Viewer 重叠区怎么融合（硬切、平均还是加权）还没确认；我们用自己的升余弦窗。
- **Extended NR**：用的是另一个模型（输入输出尺寸相同，8×256×256 → 3×1024×1024）。它的输入和 Hi-Res 完全不同：ch0–3 是零均值的高频残差（约一半像素为负），ch4–7 是 0.03 左右的平滑图；输出也是零均值残差。所以它很可能是“base + 细节修正”的 NR 路径，但需要再跑一次 NR 模式并 dump 才能确认。

## 6. 和仓库其他部分的关系

- 模型属于 Sony。本仓库不包含模型，也不提供获取、解出或运行它的代码。
- 外观链、SSCS、`itp_scale` 直接复用现有模块；这次没有改动它们。
- 如果以后把 CAS、镜头校正接上，优先复用 `sony_lens.py` 和 `sony_detail.py`，但要按 2× 坐标重新标定。

## 7. AXR 中间文件与导出链（2026-11-02 补充）

分析工作区：`analysis/reverse/sdm/`，详细记录在 `axr_dev_findings.md`，可复用代码在 `axr.py`。

### 7.1 AXR 是什么

SDM 阶段把结果写成临时文件 `_tmp<名字>.AXR`，再单独起一个开发任务把它变成最终 TIFF。AXR 是个 TIFF 容器：

```
IFD0      JPEG 预览，tag 330 → 主 Sub-IFD
Sub-IFD   14064×9376，Compression=8（Deflate），Photometric=34892（LinearRaw），
          SamplesPerPixel=3，SampleFormat=float，单条带 14064×9376×12 字节
```

负载解压后是 **4 个字节平面**（每平面 3N 字节，N = w·h）：P0=float 最高字节 … P3=最低，
再做步长 3 的后向差分；解码就是按 mod 3 的残差类做前向累加，然后
`P0<<24|P1<<16|P2<<8|P3` 当 float32，得到交错 RGB。`axr.py` 里 `decode_axr()` 实现，
在 (5000,4000) 上和运行中的内存逐位一致。

### 7.2 最终 TIFF 与 AXR 的几何关系：纯裁剪

`DSC03591_ExHiRes.TIF` 是 `9344×14016 uint16`、未压缩、RowsPerStrip=1，值全是 4 的倍数
（内部 14 位，写出时 `<<2`）。全画幅 9 个位置搜索最佳平移，结果恒为 **(16, 24)**，
没有尺度漂移 —— 之前"scale 0.9966 / 有重采样核"的结论作废。AXR 的线性 float32 图像经过 §7.3 的显影链，再裁剪输出 TIFF。

### 7.3 开发链

格式码 `0x10000201` → `FUN_1000e1f18(..., bitdepth=16)` 写文件，交给它的图像对象
（14016×9344、stride 84096）**已经带着最终噪声**（与输出 TIFF 相关系数 1.000000）。

开发商是 `TileDevForAXRFull`（vtable `0x1006a4a40`），任务链：

```
ZcTaskTileDiv → ZcTaskDemosaicCompRaw → [ZcTaskVatrFloat] → ZcTaskSSCS
→ ZcTaskLinearMatrix16 → ZcTaskMainGamma → ZcTaskRGB2YCC
→ [ZcTaskAreaComp/SIMD] → ZcTaskChromaSuppres
→ ZcTaskSharpness/SIMDSharpness → ZcTaskYGamma → … → ZcTaskDither
```

### 7.4 噪声来源：`ZcTaskDither`（2026-11-02 更正）

之前「dither 不是这层噪声」的**结论是错的**。按 tile 对齐、并用 MAD 稳健估计重测后：

- 整条任务链到 `ZcTaskMarble` 为止都是**干净的**（稳健 σ 1.5–11，16 位计数）；
- 只有 `ZcTaskDither` 前后，σ 从 5 跳到 42（p0）、3→30（p1）、6→48（p2），
  增量是白噪声（自相关 ≈0），幅度与最终 TIFF 的噪声（43/33/53，14 位计数）一致。

`FUN_100193434`（vtable+0x40）反编译还原：

```c
img = *(param_2 + 8);  p0 = img[0]; p2 = img[2]; p1 = img[1];   // *(img+8+8*i)
rng_init(state, seed = u16@p0[32行 + 32像素], shifts = (2, 7, 25));
for (y = y0; y < y1; y++)
  for (x = x0; x < x1; x++) {
    rand(state, &a, &_);   // 1 步 xorshift
    rand(state, &c, &d);   // 1 步 xorshift
    p0[x] = clamp(p0[x] + a);      // 原地
    p2[x] = clamp(p2[x] + c);
    p1[x] = clamp(p1[x] + d);
  }
```

每次 `rand` 取 `u = (s >> shift) & 0x7fff`，输出
`(u&0xFF) + ((u>>8)&0xFF) + ((u>>16)&1) - 256*[(u&0xFF)>=128] - 256*[((u>>8)&0xFF)>=128]`；
15 位切片使后两项大部分恒为 0 → 范围 [-128,254]、均值 63、σ=82.6。

实测 ROI `(28,28)-(1060,1060)`，tile 1088×1088、stride 2176 B（u16），seed 每 tile 不同
（如 9250 / 12176 / 12565 / 12811）。实现见 `analysis/reverse/sdm/dither.py`。

### 7.5 dither 前后的 YCC 变换（噪声为什么变弱、为什么跟信号有关）

dither 的 process 并不是直接在 RGB 上加噪：

```
FUN_100192f4c   RGB → YCC（原地换成新图）
                Y  = (r*0xb44 + g*0xdc3 + b*0x271) >> 11, clamp 0xefff, 存 Y+0x1000
                C2 = (r*-0x6ab + g*-0x964 + b*0x1011) >> 11, 存 C2^0x8000
                C1 = (r*0x1acc + g*-0x1ac0 + b*-9)      >> 11, 存 C1^0x8000
   dither      三个平面各加 σ=82.6 的白噪声（Y / C2 / C1）
FUN_1001814a4   YCC → RGB
                s = LUT_in[plane];  矩阵 (1.195, -0.226, 0.032 / -0.190, 1.296, -0.106 /
                -0.019, -0.033, 1.052);  clamp 0x3fff;  out = LUT_out[clamp]
```

抖动在 **YCC 域**加入，再经过一层非线性 `LUT_out` 才回到 14 位——中调斜率约 0.4，
所以最终噪声 σ ≈ 82.6 × 1.22 × 0.4 ≈ 40，与实测（43 / 33 / 53）吻合；暗部斜率趋小，
噪声几乎消失（实测全黑块 hp σ = 0.00）。这解释了此前「幅度差 2 倍、且随信号变化」的现象。

装配几何（`FUN_1000556ec`，反编译 + 实测）：目标 tile **1024×1024**，源区域 (32,32)-(1056,1056)，
1:1 三通道拷贝，无缩放无混合。

**剩下要做的**：把 `DAT_1007048d2`（LUT_in）和 `DAT_1007448d2`（LUT_out）两张表 dump 出来，
加上 `>>11` / `>>13` 的舍入细节，就能复现整条链。
