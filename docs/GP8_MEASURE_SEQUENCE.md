# Guitar Pro 8 小节事件序列路线

## 目标

输入是 Guitar Pro 8 导出的规则 PDF 或整页图片，输出是可验证、可写回 GP5 的多声部事件，而不是另一张外观相似的图片。当前模型以单个小节图为识别单元；页面切分负责保持页、谱表、系统和小节顺序。

```text
page image
  → staff/system/measure boxes
  → measure image
  → GLM-OCR 0.9B LoRA + previous-measure C2 context
  → M2 measure sequence
  → deterministic parser, music constraints and bounded self-correction
  → GP5
```

## 为什么使用事件序列

单符号 CNN 能回答“这个小块像哪个符号”，但不能单独解决以下问题：

- 一个时间点可能包含和弦、多个声部以及重叠的符杆、横梁、附点和技法；
- TAB 数字和五线谱音符必须属于同一个事件；
- 延音、击勾弦、滑音和反复需要连接相邻事件；
- 时值由音符头、符杆、横梁、附点、连音组和上下文共同决定；
- 纯五线谱无法唯一决定吉他弦/品，纯 TAB 在特殊定弦未知时也无法唯一决定绝对音高。

因此监督目标按“图中可见语义”设计：

| 模式 | 音符目标 |
| --- | --- |
| `tab` | string + fret；不要求猜绝对音高 |
| `notation` | pitch；不要求猜吉他指法 |
| `both` | string + fret + pitch |

## M2 语法

一个小节是一行：

```text
M2 time=4/4 tempo=100 key=CMajor | V0{@0:q:s1f3p67 @960:q:r} || V1{@0:h:p60}
```

- `M2`：协议版本；
- 小节元数据：拍号、速度、调号、反复、段落和导航符号；
- `V0` / `V1`：声部；
- `@start:duration:payload`：事件起点、时值和内容；起点单位与 GP quarter time 一致；
- `s1f3p67`：1 弦 3 品、MIDI 音高 67；具体字段随显示模式变化；
- `r` / `e`：休止与 GP 空拍；
- 括号保存音符技法，尖括号保存拍级技法。

解析与格式化在 `guitarocr/data/gp_measure_sequence.py`；写回 GP5 在 `guitarocr/export/measure_sequence_to_gp5.py`。2026-07-15 的 v2 标签门禁对 447,378 个真实目标执行格式化→解析→格式化和语义约束，失败为 0。测试集的 `tab` 与 `both` 写回均为 15,304/15,304 小节一致；纯五线谱为 15,296/15,304，8 个差异均来自原始 GP 中没有同弦前音的孤立 tie。

## 数据构建

构建器从 GP3/GP4/GP5/GTP 语料进行固定种子、分格式候选抽样，再按稀有技法、多声部和格式覆盖选择来源曲目。可解析但不能合法写成 GP5 的脏文件会记录并自动补位。

```powershell
.\scripts\build_gp8_measure_sequence_dataset.ps1 `
  -Corpus D:\guitarOCR\music-scores-collection\files\guitar_pro `
  -Output D:\guitarOCR\database\gp8_measure_sequence_v2 `
  -SourceCount 2000
```

2026-07-15 v1：

- 完整候选语料 285,362 份可考虑的 GP3/GP4/GP5/GTP；另有 GPX 不在此构建器范围；
- 最终 600 首：GP3 182、GP4 181、GP5 217、GTP 20；
- 官方 GP8 PDF 和 native layout：三模式各 600，1,800/1,800 成功；
- 138,618 个小节图：三模式各 46,206；
- train / validation / test：110,016 / 12,936 / 15,666；
- 473 / 58 / 69 首，按原始文件 SHA 隔离；
- 小节框与标签数量不一致：0；
- 42 首包含可见多声部。

最终 v2 从同一 285,362 份候选中按格式、稀有技法和多声部重新选择 2,000 首，不是取目录前 N 个：

- GP3 / GP4 / GP5 / GTP：509 / 661 / 777 / 53 首；
- 149,126 个唯一小节，三模式共 447,378 个图文目标；一首会稳定终止官方 GP8 worker 的脏源已由一首从未进入 v2、且三版式渲染通过的复杂 GP4 补位；
- 包含推弦 569 首、普通/移位滑音 485/810 首、泛音 522 首、颤音 44 首、快速重复（tremolo picking）73 首和多声部 139 首；
- 全量标签 canonical、版式字段、事件顺序、弦/品/音高一致性门禁已在最终标签上逐项执行，失败为 0；移除不可见 playback velocity 后最大目标 1,887 tokens，叠加真实图像与提示后的完整样本最大为 2,657 tokens，因此 v2 cutoff 使用 3,072；
- 训练划分另生成有上限的长尾难例文件；稀有类全部保留，常见技法按模式限额，避免普通音符淹没推弦、滑音等类别。

若只修改 M2 schema，不必重新扫描整个语料或重新选择曲目，可以对已有稳定来源执行 `-Phase relabel`；当官方 PDF 正在读取 prepared GP5 时可用 `-Phase relabel-labels` 只重建标签。GP8 worker 的偶发连续 I/O 失败会自动以更短的重启间隔重试缺失项。正式训练前建议运行：

```powershell
D:\guitarOCR\guitar-hero-main\.venv\Scripts\python.exe `
  -m guitarocr.evaluation.validate_m2_dataset `
  --dataset database\gp8_measure_sequence_v2 `
  --report reports\m2_dataset_v2_validation.json
```

数据目录被 `.gitignore` 排除。原始曲谱、PDF、layout、crop 和训练缓存都不能提交。

## 整页小节切分

官方 GP8 PDF 优先读取矢量谱表行。纯 TAB 的弦线会被数字截断，因此先合并同一高度的短矢量段来定位完整谱表，再在 180 DPI 栅格页上检测真正跨越弦线、且没有符杆长尾的小节线。矢量层和像素层按谱表纵坐标互证，任一层发现的完整谱表都会保留。

```powershell
python -m guitarocr.evaluation.validate_gp8_measure_geometry `
  --dataset database\gp8_measure_sequence_v2 `
  --report reports\gp8_measure_sequence_v2_geometry.json
```

200 首 source-disjoint v2 测试曲的结果：

| 模式 | 整曲 | 页面 | 小节 |
| --- | ---: | ---: | ---: |
| `notation` | 200/200 | 572/572 | 15,304/15,304 |
| `both` | 200/200 | 904/904 | 15,304/15,304 |
| `tab` | 200/200 | 612/612 | 15,304/15,304 |

这是几何切分指标，不是 M2/音符识别指标。

## M2 写回 GP5

`tab` 和 `both` 直接使用可见弦/品；`notation` 不监督不可见指法，写回时由确定性束搜索分配弦号。约束包括音高可演奏、同一和弦不重复用弦，以及延音音符必须沿用该弦上一个音高。和弦内的 notation 音符按音高 canonical 排序，不依赖源 GP 的隐藏弦号顺序。

```powershell
D:\guitarOCR\guitar-hero-main\.venv\Scripts\python.exe `
  -m guitarocr.evaluation.validate_m2_gp5_roundtrip
```

200 首测试曲的三种显示模式均成功写入 GP5 并重新解析。`tab / both` 各 15,304/15,304；`notation` 为 15,296/15,304。纯五线谱的 8 个差异集中在 5 份源文件的孤立 tie：源 GP 没有可继承的同弦前音，且图中没有弦号，因此不存在唯一的原弦位逆解。

## 训练

官方 GLM-OCR 是 0.9B 视觉语言模型。RTX 2080 Ti 使用 FP16 LoRA。v1 冻结视觉塔，只训练 3,735,552 个语言侧 LoRA 参数作为基线；v2 在减小 micro-batch 后同时训练语言层和视觉塔的 LoRA，使细小品位数字、音符头和技法标记也能做领域适配：

```powershell
.\scripts\train_glm_ocr_measure_sequence.ps1
```

默认训练成功后会自动运行 3000 条 source-disjoint v2 正式评测和发布门禁；调试时可用 `-SkipEvaluation` 跳过。已经在后台运行的训练可由 `wait_for_glm_ocr_training.ps1` 监控，训练进程退出且完整 adapter 落盘后再启动评测。

基线配置位于 `configs/glm_ocr_measure_sequence_lora_fp16.yaml`，v2 配置位于 `configs/glm_ocr_measure_sequence_v2_lora_fp16.yaml`。v2 最大目标为 1,887 tokens；对 447,378 个目标结合各自 layout 尺寸、图像 token 和 `C2` 提示做上界扫描后，完整样本最大为 2,657 tokens，cutoff 因此设置为 3,072。

v2 直接从 GLM-OCR 基座在修正后的标签上训练，不继承 v1 中已确认不可见的 playback velocity 监督。每个非首小节会携带上一小节的紧凑 `C2`：只保留每个声部的最后事件和已打印的拍号/速度/调号，使 Transformer 能学习延音与声部连续性，又不会把整段答案塞进提示。推理时 `C2` 来自上一小节预测，不读取源 GP。

模型输出还要通过 `measure_sequence_constraints.py`：三版式字段、事件顺序、时值、弦/品/音高、技法词表和 `both` 模式的 `pitch = tuning[string] + fret` 都是硬约束。失败时模型基于同一图像和约束错误最多自纠三次；该约束已在构建期真值上做全量零误杀门禁。

## 评估

测试必须按整首来源隔离。训练 loss 不能替代结构指标：

```powershell
.\scripts\evaluate_glm_ocr_measure_sequence.ps1 -MaxSamples 3000
```

评估项包括：M2 语法/约束有效率、核心字段全对率、节奏整小节全对率、音符字段整小节全对率、完整 exact、元数据、事件起点 F1、对齐事件时值、休止状态、音符 exact、弦/品/音高和逐技法 F1，并分别报告三种显示模式。3000 条正式抽样按版式、来源和技法覆盖分层；运行签名绑定测试清单、模型、adapter 与生成参数，防止旧预测续写到新模型报告。训练 loss 不能替代这些来源隔离指标。

`configs/measure_sequence_release_gate.json` 是发布门槛而非训练目标装饰：总体核心字段整小节全对率至少 98%，节奏和音符字段至少 99%，每个版式也有独立下限；测试集中正例不少于 20 的技法要求 F1 至少 90%。检查命令：

```powershell
python -m guitarocr.evaluation.check_measure_sequence_release `
  reports\glm_ocr_measure_sequence_v2_test_metrics.json
```

## 仍需单独解决

- 跨页/跨系统连线以及复杂导航后的实际演奏顺序；
- 标题、作者、调弦、变调夹等页头信息；
- 纯五线谱的指法选择只能确定性优化，不能宣称恢复原作者指法；
- PDF 中不可见的混音、MIDI 控制和隐藏速度变化无法从像素恢复；
- 多轨合奏、手写谱、拍照透视和其他制谱软件字体仍需独立数据域。
