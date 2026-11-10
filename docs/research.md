# 本地研究资料与验证索引

公开代码和文档位于 `base/`。原始 RAW、Edit 导出、反推脚本和验证报告保存在仓库之外的本机研究目录中。

| 内容 | 本地位置 |
| --- | --- |
| 0x780f 的 276 字节解包与色表构建 | `reverse/decoder_verification/`、`reverse/color_table_verification/` |
| 像素链、YCC 和各空间阶段 | `reverse/pixel_pipeline/verify_*.py` |
| ToneCurve、ShadeCorrection | `reverse/pixel_pipeline/verify_tonecurve_shade.py` |
| Rust 的 Marble、参数复用、滑块与边界 | `reverse/pixel_pipeline/verify_marble_rust.py`、`verify_rust_config.py`、`verify_rust_sliders.py`、`verify_rust_bounds.py` |
| 完整解码 ST / FL2 / FL3 重测 | `readme_update/decode_looks.py`、`measure_results.py`、`results.json` |
| Edit 参考图 | `reverse/reference/` 及 `analysis/` 根目录 |
| Edit 4.0 / 4.1 函数及常量地址对照 | `reverse/edit41_diff/` |
| AXR、Viewer 张量和 Dither | `reverse/sdm/` |

研究环境通过 `.pth` 文件指向本仓库的 `base/`。部分早期脚本仍引用历史会话路径，重跑时需要调整导入和数据位置。
