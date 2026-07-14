# GuitarOCR 已有模型与实验记录

> 本文保留各阶段的实现说明、指标和历史命令。项目已经改为分层包结构；当前目录和可直接执行的命令以根目录 [README](../README.md) 为准。下文出现的根目录旧脚本路径仅用于还原当时的实验记录。

源代码目前位于 `guitarocr`、`scripts` 和 `java`；`database` 只保存生成的数据、标签、模型和日志。

## 历史模块索引

| 文件 | 用途 |
| --- | --- |
| `build_database.ps1` | 从 GP 文件构建 PDF、整页 PNG 和语义标签数据集 |
| `validate_database.ps1` | 校验源文件、PDF、图片、标签和数据划分 |
| `TuxGuitarDatasetBuilder.java` | 调用 TuxGuitar 导入 GP 并渲染三种常用版式 |
| `TuxGuitarAtomicSymbolBuilder.java` | 使用 TuxGuitar 原生绘制器生成符号模板 |
| `build_symbol_dataset.py` | 把模板扩增成 64×64 的符号分类数据集 |
| `symbol_model.py` | 轻量 CNN 模型定义 |
| `train_symbol_cnn.py` | CUDA 训练、测试及模型导出 |
| `infer_symbol.py` | 对一张已经裁切好的单符号图片推理 |
| `run_symbol_cnn.ps1` | 从模板生成到训练完成的一键脚本 |
| `TuxGuitarTabAnnotationBuilder.java` | 从 TuxGuitar 内部导出小节、弦线和品位符号坐标 |
| `build_tuxguitar_page_annotations.ps1` | 生成页面坐标标签及彩色叠框检查图 |
| `validate_tuxguitar_page_annotations.py` | 对照原 GP 语义严格校验坐标标注 |
| `build_tab_detector_dataset.py` | 把页面标签变成 512×128 小节检测训练集 |
| `tab_detector_model.py` | 轻量 CenterNet 风格 TAB 符号定位模型 |
| `train_tab_detector.py` | 训练与评估数字/X 定位模型 |
| `infer_tab_detector.py` | 对单个小节图执行数字/X 检测 |
| `run_tab_detector.ps1` | 一键构建并训练 TAB 检测器 |
| `infer_tuxguitar_tab_page.py` | 只读取整页 PNG，恢复 TAB 谱表、小节、品位、弦号和起音事件 |
| `validate_tuxguitar_tab_geometry.py` | 验证纯像素谱表/小节定位，不向算法提供坐标标签 |
| `evaluate_tab_detector.py` | 统计小节块上的逐类检测指标 |
| `evaluate_tuxguitar_tab_pages.py` | 统计纯图片整页端到端指标 |
| `TuxGuitarScoreRhythmAnnotationBuilder.java` | 从 `score_tab` 版式导出五线谱事件、声部、音符和节奏坐标真值 |
| `build_score_rhythm_dataset.py` | 生成节奏像素标签、事件上下文图和可视化叠框 |
| `validate_score_rhythm_dataset.py` | 对照原 GP 语义严格验证节奏数据与来源隔离划分 |
| `rhythm_context_model.py` | 轻量双声部节奏上下文 CNN |
| `train_rhythm_context.py` | 训练并分别评估主声部、第二声部和各节奏子任务 |
| `infer_rhythm_event.py` | 对一张以事件为中心的 `score_tab` 图块识别节奏语义 |
| `run_rhythm_context.ps1` | 一键构建节奏数据并训练上下文 CNN |
| `build_score_event_locator_dataset.py` | 将五线谱小节生成保持尺度的事件定位图块 |
| `score_event_locator_model.py` | 轻量一维事件横坐标热力图网络 |
| `train_score_event_locator.py` | 训练并评估五线谱事件定位器 |
| `score_tab_geometry.py` | 仅从像素检测并配对五线谱、TAB 谱和小节 |
| `infer_tuxguitar_score_tab_page.py` | 整页 `score_tab` 事件定位并串联节奏 CNN |
| `evaluate_score_event_pages.py` | 评估纯图片页面几何和事件位置 |
| `evaluate_detected_rhythm_pages.py` | 评估自动定位事件后的联合节奏准确率 |
| `score_tab_fingering.py` | 在 `score_tab` TAB 区域检测指法并构建统一 Score/Event IR |
| `evaluate_score_tab_fingering.py` | 单独评估 `score_tab` 可见弦号/品位识别 |
| `evaluate_merged_event_ir.py` | 评估事件定位、节奏和可见指法的联合结果 |
| `time_signature_recognizer.py` | 从五线谱像素识别印刷拍号并按文档顺序传播 |
| `measure_rhythm_constraints.py` | 用精确有理数审计小节容量并生成非破坏性修正建议 |
| `evaluate_time_signatures.py` | 评估拍号出现位置、数值和跨页传播 |
| `evaluate_measure_rhythm_constraints.py` | 比较原始 CNN 与高可信小节约束候选的真实指标 |
| `infer_tuxguitar_score_tab_document.py` | 对 PDF 或多页图片一次加载模型并输出整首 Score IR |
| `pdf_page_renderer.py` | 用固定 180 DPI 灰度 Poppler 渲染直接输入的 PDF，并缓存页图清单 |
| `build_tie_event_dataset.py` | 从真实 PDF 事件图构造延音存在、数量和纵坐标关系标签 |
| `tie_context_model.py` | 延音关系轻量 CNN 定义 |
| `train_tie_context.py` | 从节奏 CNN 迁移骨干并训练延音多任务模型 |
| `tie_inference.py` | 运行延音 CNN 并结合 Score/TAB 缺失音符约束 |
| `evaluate_tie_event_pages.py` | 评估整页延音候选与保守自动连接 |
| `run_tie_context.ps1` | 构建、训练并评估延音关系阶段 |
| `run_score_event_locator.ps1` | 一键构建、训练并测试整页事件定位器 |
| `EVENT_IR_SCHEMA.md` | 当前 Event IR 字段、关联规则和未解决语义 |
| `RECOGNITION_ARCHITECTURE.md` | PDF → 小节 → 事件 → 符号 → 中间态 → GP 的设计 |

## 数据与模型

```text
D:\guitarOCR\database\
  source\                 已接收的 GP 真值文件
  output\pdf\             TuxGuitar 生成的 PDF
  output\images\          PDF 渲染后的整页 PNG
  labels\                 曲目和事件级语义真值
    layout\tab_only\      TuxGuitar 的 550×800 原始布局坐标
    pages\tab_only\       换算后的逐页 PNG 像素坐标
    layout\score_tab_rhythm\  五线谱+TAB 的事件和节奏逻辑坐标
    pages\score_tab_rhythm\   换算后的逐页 PNG 节奏像素标签
  manifests\              清单、统计和数据划分
  output\annotation_overlays\  带小节、TAB 区域和符号框的检查图
  tab_detector\           小节检测数据、模型、指标和日志
  rhythm_events\          256×192 事件上下文数据、模型、指标和日志
  score_event_locator\    512×192 小节图块、事件定位模型、整页推理与指标
  symbol_cnn\
    templates\            TuxGuitar 原生矢量符号模板
    dataset\              64×64 train/validation/test 图片
    models\               CNN、TorchScript 和测试报告
    reports\              训练日志
```

`database` 下没有项目源代码；Python、PowerShell 和 Java 源码分别位于 `guitarocr/`、`scripts/` 和 `java/`。

## 当前符号 CNN 基线

- 28 个语义类，15,680 张合成图片；训练/验证/测试分别为 11,200 / 2,240 / 2,240 张。
- 模型参数量 218,884，模型文件约 0.9 MB。
- 使用 PyTorch 2.7.1 CUDA，在 RTX 2080 Ti 上训练。
- 最佳验证准确率 99.96%，独立合成测试集准确率 99.78%。
- 已覆盖：数字 0–9、闷音 X、三类音符头、六类休止符形状、升降还原号、四类谱号和点。

这个结果只表示“已经裁切好的单个合成符号”的分类能力，不代表真实 PDF 的端到端识别率。整页符号定位、谱线/小节线、符杆/横梁、连音线以及符号间关系仍是下一阶段。

有些语义不能只看孤立图形：全休止和二分休止共用 `rest_block`，需要通过相对谱线位置区分；中音和次中音 C 谱号共用 `clef_c`，需要通过谱号垂直位置区分；`dot` 的附点、断奏点或反复记号含义也必须结合上下文判断。

## 重新训练

```powershell
& .\scripts\run_symbol_cnn.ps1
```

## 单张符号推理

```powershell
python -m guitarocr.pipeline.infer_symbol `
  D:\path\to\cropped_symbol.png `
  --model D:\guitarOCR\database\symbol_cnn\models\atomic_symbol_cnn.pt
```

详细指标在 `D:\guitarOCR\database\symbol_cnn\models\test_metrics.json`。

## TuxGuitar 页面坐标标注

```powershell
& .\scripts\build_tuxguitar_page_annotations.ps1
python -m guitarocr.evaluation.validate_tuxguitar_page_annotations
```

当前 `tab_only` 标注包含 31 首、63 页、958 个小节和 10,534 个品位数字/X 框。红框表示数字，紫框表示 X，绿色表示 TAB 区域，蓝色表示小节。

## 训练 TAB 符号定位模型

```powershell
& .\scripts\run_tab_detector.ps1
```

模型是 592,623 参数的轻量 CenterNet 风格检测器。训练时根据坐标真值裁出 TAB 小节；为避免把很宽的小节压扁，保持谱表高度并用重叠窗口横向切块。模型输出数字 0–9 和 X 的类别与位置。

在按曲目隔离的测试集中，小节块检测 F1 为 99.76%；其中 X 为 104/104 全部检出。坐标真值只用于构造训练集和评测，不会作为实际推理输入。

## 只用整页图片推理

```powershell
python -m guitarocr.pipeline.infer_tuxguitar_tab_page `
  D:\path\to\tuxguitar_tab_only_page.png
```

整页流程是：像素检测 TAB 横线 → 定位小节线 → 保持高度切块 → 检测数字/X → 映射到弦 → 合并多位品位 → 按横向位置组合起音事件。输出 JSON 和带框 PNG，默认写入 `D:\guitarOCR\database\tab_detector\page_inference`。

当前 TuxGuitar `tab_only` 数据上的验证结果：

- 63 页、334 个 TAB 谱表、958 个小节全部由像素正确定位；
- 独立测试曲目的整页符号检测 F1 为 99.61%；
- 已匹配符号的弦号准确率为 100%；
- 品位/弦组合后的精确起音事件 F1 为 99.01%。

这完成的是 Level A TAB 转录，不等于完整 GP 重建。当前还没有从图片恢复时值、声部、符杆/横梁、休止符、附点、延音线、拍号、调弦和速度；这些属于下一阶段的节奏与乐谱中间态模块。当前实现也只针对 TuxGuitar 的 `tab_only` 渲染风格，其他软件和 `score_tab` 版式需要后续适配与单独评测。

详细结果：

- `D:\guitarOCR\database\tab_detector\models\test_detailed_metrics.json`
- `D:\guitarOCR\database\tab_detector\models\page_end_to_end_test_metrics.json`

## TuxGuitar `score_tab` 节奏上下文基线

这一阶段把原 GP 的节奏语义与真实 PDF 渲染坐标对齐，生成了 31 首曲目、110 页、958 个小节、6,558 个事件图块和 6,746 个可见声部实例。事件图块保留相邻音符、符杆、横梁、休止符和附点的上下文，不能当作互不重叠的单符号切图。

```powershell
& .\scripts\run_rhythm_context.ps1

python -m guitarocr.pipeline.infer_rhythm_event `
  D:\path\to\event_crop.png
```

当前 CNN 有 879,358 个参数。按曲目来源隔离的测试集上，主声部完整节奏语义准确率为 89.59%，主声部时值准确率为 95.05%；双声部整事件完全正确率为 76.04%。第二声部完整语义准确率只有 2.97%，因为现有数据中仅两首曲目含可见第二声部，且训练和测试的时值分布差异很大。因此当前可作为主声部节奏基线，第二声部尚未解决。

节奏 CNN 的上述独立指标使用 TuxGuitar 真值事件中心裁出的 256×192 图块。现在已经增加纯图片事件定位器，可以从整页自动产生这些中心并串联节奏 CNN；仍不能直接输出完整 GP。严格校验结果在 `D:\guitarOCR\database\rhythm_events\models\metrics.json`，坐标可视化在 `D:\guitarOCR\database\output\annotation_overlays\score_tab_rhythm`。

## TuxGuitar `score_tab` 整页事件定位

```powershell
& .\scripts\run_score_event_locator.ps1

python -m guitarocr.pipeline.infer_tuxguitar_score_tab_page `
  D:\path\to\score_tab_page.png
```

整页流程只读取 PNG 像素：先检测并配对五线谱/TAB 谱表，从 TAB 竖线取得可靠的小节边界，再用 161,538 参数的一维 CenterNet 风格 CNN 定位每个事件的横坐标。检测出的中心直接用于裁切节奏上下文，不再读取 TuxGuitar 坐标标签。

来源隔离测试集包含 14 页、39 个谱表、95 个小节和 793 个事件。谱表与小节全部正确定位；事件定位召回率 100%、精确率 99.87%、F1 99.94%，平均横向误差 0.33 像素。自动中心送入节奏 CNN 后，主声部可见实例完整语义准确率仍为 89.59%，双声部整事件准确率仍为 76.04%，说明定位误差没有降低已有节奏分类结果。第二声部数据不足的问题仍然存在。

当前实现只针对本项目的 TuxGuitar `score_tab` 固定渲染尺度和样式。拍号与小节时值约束已经接入；延音线、调弦/速度和 GP 写出尚未完成。

## 节奏与 TAB 指法合并

整页推理现在还会在同一组 TAB 谱线上运行已有数字/X 检测器，再按系统、小节和横坐标与五线谱事件关联，输出独立的 `score_ir.json`。无法确认的声部、调弦和速度保持为 `null`，不会用默认值伪造完整结果。印刷拍号从五线谱像素识别，后续小节使用文档顺序传播的值。

在来源隔离测试集上，`score_tab` 可见 TAB 事件定位 F1 为 99.74%，匹配事件的完整弦/品位准确率为 98.59%，逐音符 F1 为 98.92%。把定位、主声部节奏和可见 TAB 指法同时计入后，当前受支持的主声部核心事件完全正确率为 87.39%；要求两个声部同时正确时为 74.78%。这里尚未检查延音续接、调弦和演奏技巧，因此不能解释为完整 GP 重建率。

## 拍号传播与小节时值约束

拍号识别复用 218,884 参数的原子符号 CNN 读取上下排列的数字，并用当前 TuxGuitar 数据中实际出现的拍号语法消除数字歧义。31 首、958 小节的全量检查中，44 处实际印刷拍号的出现位置和数值均全部正确，传播后的 958 个小节拍号也全部正确。这个结果仅适用于当前 TuxGuitar 固定渲染语料，不代表其他软件排版的泛化结果。

每个声部的时值使用 `Fraction` 精确求和，附点和连音比例不会产生浮点误差；按声部顺序同时写出精确分数形式的 `onset`、`duration_fraction` 和 `end`。若总时值不等于拍号容量，系统在节奏 CNN 的候选概率中搜索最小代价的填满方案；结果写入 `rhythm_audit.correction_proposal`，不覆盖原预测。阈值只在验证集选择为相对概率 0.20：验证集高可信建议 6/6 正确；独立测试集也是 6/6 正确。测试集主声部节奏 exact F1 从 88.97% 提升到 89.66%，完全正确小节从 64/95 提升到 69/95，主声部起点准确率从 81.85% 提升到 85.15%。

直接输入 PDF：

```powershell
python -m guitarocr.pipeline.infer_tuxguitar_score_tab_document `
  D:\path\to\score.pdf `
  --output D:\guitarOCR\database\document_inference\my_score
```

也可以把最后一个位置参数换成页图片目录。PDF 使用 Poppler 按当前模型固定的 180 DPI 灰度渲染，页图和 `render_manifest.json` 缓存在输出目录；PDF 大小、修改时间或参数未变化时不会重复渲染。输出包括整首 `document_score_ir.json`、逐页 `score_ir.json`、节奏裁图和叠加检查图。

现有 TuxGuitar PDF 回归中，6 个直接渲染页面与训练数据库对应 PNG 的 16,500,000 个灰度像素逐一相同，最大像素差为 0。PDF 入口与原 PNG 入口产生相同的拍号、44 个小节、310 个事件及全部离散 Score IR 语义；GPU 置信度与坐标浮点值的重复运行差异不超过 `4e-6`。

## 延音关系基线

真实语料共有 510 个延音音符、347 个延音事件；173 个事件是“部分和弦延音并同时加入新音”，25 个延音跨谱表或跨页，因此不能把检测到弧线的整个和弦都直接标成延音。当前数据集同时监督延音存在、延音音符数量、五线谱音符总数和目标音符纵坐标。

延音 CNN 有 888,096 个参数，视觉骨干从节奏 CNN 迁移。划分严格沿用节奏模型的曲目隔离：训练/验证/测试分别为 4,562 / 1,203 / 793 个事件，其中延音正例为 294 / 37 / 16。测试曲既未参与节奏骨干训练，也未参与延音训练。

单看弧线会混淆 tie 与 slur/hammer-pull。生产流程因此要求：视觉延音成立，并且 CNN 预测的五线谱音符数大于当前 TAB 实际新发音数。经过这一 Score/TAB 约束后：

- 验证集延音事件候选精确率 100%、召回率 72.97%、F1 84.38%；
- 独立测试集精确率 100%、召回率 81.25%、F1 89.66%；
- 测试集候选延音数量 10/13 个事件完全正确；目标纵坐标 F1 为 57.14%；
- 仅对“相邻、无新发音、整事件全部续接”的情况自动写入边：测试集恢复 9 个延音音符，9/9 正确，覆盖 23 个真值延音音符的 39.13%。

其余部分和弦、非相邻和跨谱表候选保留在 `tie_relation` 中，不会被猜测成具体弦/品。重新训练与评估：

```powershell
& .\scripts\run_tie_context.ps1
```

Event IR 说明见 `docs/EVENT_IR_SCHEMA.md`；测试指标见 `database/score_event_locator/models/merged_event_ir_test_metrics.json`。

## 最小 PDF 到 GP5 闭环

当前已经能够把 TuxGuitar `score_tab` 风格 PDF 的主声部识别结果写成可由
TuxGuitar 打开和播放的 GP5，并把该 GP5 重新渲染成预览 PDF：

```powershell
python .\pdf_to_gp.py `
  D:\path\to\score.pdf `
  -o D:\guitarOCR\output\gp\score_ocr.gp5
```

如果已经有 `document_score_ir.json`，可以跳过视觉推理，只测试写谱阶段：

```powershell
python -m guitarocr.export.export_score_ir_to_gp `
  D:\path\to\document_score_ir.json `
  -o D:\guitarOCR\output\gp\score_ocr.gp5
```

第一版支持小节、拍号、主声部起点/时值、休止、弦号、品位和已安全解析的
延音。无法从 PDF 确认时默认使用标准六弦调弦、120 BPM 和 capo 0，并在
`*.report.json` 中记录；结构性空缺会用休止补齐。第二声部、反复、演奏技巧、
非标准调弦识别以及未解析的部分/跨系统延音尚未写出。

## 2026-07-14：331 首扩容、技法模型与目标曲闭环

本轮把原先 31 个源文件扩大到 331 个 GP/GTP 源文件。数据库包含 26,682
小节、180,417 个节奏事件裁块和 318,376 个音符。`tab_only` 与
`score_tab` 坐标真值合并后，TAB 检测集有 65,953 个小节图块、752,401
个数字/X 标注；131,906 个 PNG/JSON 文件的完整性检查未发现损坏。训练、
验证和测试继续按源曲目隔离。

### 独立测试集

| 组件 | 参数量 | 独立测试结果 |
| --- | ---: | --- |
| 节奏上下文 CNN | 882,956 | 主声部完整语义 99.596%；全事件 exact 99.456% |
| TAB 数字/X 检测器 | 592,623 | P 99.614%、R 99.970%、F1 99.791%；85,423 个真值中 26 FN |
| 延音上下文 CNN | 888,096 | presence F1 99.119%；数量准确率 88.950%；目标 y F1 94.422% |
| 技法上下文 CNN | 874,989 | dead 99.650%、vibrato 95.833%、bend 98.876%、palm mute 96.970%、slide 48.415% F1 |
| PickStroke 上下文 CNN | 875,503 | 测试只有 1 个上拨、0 个下拨真值；不可据此宣称独立泛化 |

技法模型是 13 类多标签分类器。独立测试 macro F1（只计有 support 的类）
为 57.432%。slide、hammer、let-ring 的域内变化仍明显，ghost/accent 等稀有
类在本次测试上没有形成可靠召回，不能因为常见类很准就宣称“全部技法已解决”。
GP7/8 `.gp` ZIP 可以用 `guitarocr.data.augment_gpif_technique_labels` 从内部
GPIF 覆盖 TuxGuitar 兼容层丢失的精确技法标签。

### 目标曲最多 10 轮回归

目标是 `finaltest/若能绽放光芒 Final 教学版.gp`。每轮执行
GP → GT.pdf → 像素 OCR → IR，并用内部 GPIF 逐事件对齐。第 10 轮结果：

| 指标 | 结果 |
| --- | ---: |
| 小节 exact | 168 / 168 |
| 事件 exact | 701 / 701 |
| 节奏 exact | 701 / 701 |
| 音符（弦/品） | 1,425 / 1,425，P/R/F1 100% |
| muted X | 61 / 61 |
| tie-in 音符 | 177 |
| palm mute | 173 个事件，P/R/F1 100% |
| slide | 13 个事件，P/R/F1 100% |
| vibrato | 2 个事件，P/R/F1 100% |
| pick up / pick down | 153 / 389 个事件，两类 P/R/F1 均 100% |

这里的 100% 是**指定 hard case 的回归验收**。该曲参与了困难样本微调、
阈值选择和窄范围规则修正，不能作为未知曲目的独立泛化指标；泛化能力应引用
上一节 source-disjoint 测试结果。

最终公共 CLI 输出 168 小节、701 事件、1,425 音符、77 休止、177 个
tie-in 音符和 542 个 PickStroke 事件。TuxGuitar 重新读入 GP5 时 701/701 事件通过节奏、音符和可表示
技法校验。13 个音符同时具有 dead/X 与 slide，而 TuxGuitar GP5 模型把这两种
状态视为互斥；导出选择保留页面可见的 X，并在报告中逐项记录
`gp5_dead_slide_conflict`。IR 仍保留两种语义。

视觉 QA 比较了首尾和包含 X/和弦的中间页面。PRE.pdf 可正常渲染且无裁切；
由于当前不恢复标题、调号和分页排版元数据，PRE 为 10 页而 GT 为 11 页，调号
差异由逐音临时升降号表达，不影响上述音高/弦品事件指标。

末页视觉 QA 最初发现 `V/Π` 上下拨弦符号没有写回。后续把 Beat 级
`PickStroke` 加入 TuxGuitar 坐标真值、GPIF 读取、事件 CNN、IR、GP5 写入和
回读验证。为避免联合微调导致原 13 个技法类别遗忘，发布运行时保留原技法
CNN，并让第二个同规模 CNN 只覆盖 `pick_up/pick_down` 两项。休止符泄漏、
双方向冲突和严格交替拨弦中的单点漏检再经过可审计的窄序列约束。目标回归
达到上拨 153/153、下拨 389/389；但独立测试 support 严重不足，仍需从其他
曲目补充 PickStroke 样本。

## 2026-07-14：纯 TAB 从 Level A 升级到完整主流程

旧版 `tab_only` 只找六条谱线并识别数字/X，因此看不到无品位数字的休止事件、
延音续接、声部、时值、附点和连音组。本轮复用 `score_tab` 的分层语义，但为
纯 TAB 的视觉域分别构建事件定位、节奏、延音和技法数据并训练独立 checkpoint：

| 数据/组件 | 规模或结果 |
| --- | ---: |
| `tab_event_locator` | 26,682 小节，27,720 图块，180,417 唯一事件 |
| `tab_rhythm_events` | 180,417 个 256×192 事件上下文裁块 |
| `tab_tie_events` | 测试 18,397 事件，724 个延音正例、1,267 个延音音符 |
| 纯 TAB 事件定位器 | 161,538 参数 |
| 纯 TAB 节奏 CNN | 882,956 参数 |
| 纯 TAB 技法 CNN | 875,503 参数 |
| 纯 TAB 延音 CNN | 888,096 参数 |

划分继续按 `source_id` 隔离。149 个独立测试页面只向推理端提供页面像素；GP
坐标和语义仅用于训练标签与最终评分：

| 整页指标 | 结果 |
| --- | ---: |
| 页面/小节几何 | 149/149 页，2,874/2,874 小节 exact |
| 全事件定位 | P 99.978%、R 100%、F1 99.989%（18,397 TP、4 FP、0 FN） |
| 自动事件中心后的主声部节奏 exact | 99.750% |
| 自动事件中心后的双声部节奏 exact | 99.706% |
| 可见 TAB 事件整组弦/品/X exact recall | 99.056% |
| 节奏 + 可见指法主核心事件 exact | 99.005% |
| 节奏 + 可见指法双声部核心事件 exact | 98.962% |

单模块纯 TAB 延音 presence F1 为 99.653%，真值事件裁块上的目标纵坐标 F1
为 99.249%。把页面几何、学习式事件定位、节奏、TAB 指法和延音串联后，18,397
个事件全部匹配，视觉 presence 与纯 TAB 语义候选 F1 均为 99.653%，目标弦纵坐标
F1 为 99.365%。按目标弦和同弦事件序列自动续接 716 个事件，事件 precision
98.464%、音符 precision 99.281%，覆盖 98.106% 的真值延音音符；无明确目标弦的
候选继续保留为未解析，而不是强行写错。

纯 TAB 印刷拍号单独识别时，出现位置 precision 98.507%、recall 92.958%，检测到
真拍号后的数值准确率 93.939%。普通品位数字会放大传播误差，因此最终流程还用
节奏 CNN 的精确小节容量交叉确认：33 首、149 页、2,874 小节的来源隔离测试为
2,874/2,874 正确。弱视觉形状只用于拍号容量等价时的唯一消歧，不会独立改写拍号。

技法模型 macro F1（只计有 support 的类）为 83.181%；dead 100%、vibrato
99.213%、bend 99.187%、hammer 99.531%、slide 95.172%、palm mute
98.160%、let ring 98.869%。staccato 的 13 个测试正例未召回，tapping 和
pick-down 没有测试正例，不能宣称这些稀有技法已解决。

公共 `guitarocr` CLI 现在默认自动区分 `score_tab` / `tab_only` 并路由到对应
模型。GP5 写入计划格式升级到 v3，可把两个识别声部分别写入同一 TGBeat；回读
校验也由 beat 级升级为 voice-event 级。测试曲 `Bass string on 2nd voice` 的
纯 TAB PDF 被恢复为 4 小节、32 个时间位置、40 个声部事件和 35 个音符，包含
3 个 tie-in；TuxGuitar GP5 回读 40/40 个声部事件完全匹配。输出 PRE PDF 经
180-DPI Poppler 重渲染检查，无截断或损坏。

另一个含 bend、slide、hammer、let ring 和大量延音的 39 小节来源隔离样例，
最终恢复 440/440 个事件和全部 667 个实际弹奏音符；109 个真值续音中自动写回
104 个，GP5 回读 440/440 个事件完全匹配。生成的 6 页预览 PDF 重新以 180 DPI
渲染并检查首尾页，未见截断、空白页或损坏。

一键复现入口为 `scripts/run_tab_only_models.ps1`。它构建四类纯 TAB 数据、训练
四个专用模型，并依次执行整页事件、节奏、指法、合并 IR、延音和拍号测试。

## 2026-07-14：跨制谱软件零微调基准

为了避免把 TuxGuitar 域内指标误当成 Guitar Pro/MuseScore 泛化能力，新建
`guitarocr.data.build_cross_renderer_benchmark`。它从 source-disjoint `test`
划分选择同一组 GP3/GP4/GP5 单轨曲目，分别生成 `score_only`、`tab_only` 和
`score_tab` PDF。所有记录明确写入 `training_eligible=false`，基准样本不得回流
训练。TuxGuitar 和 MuseScore 4 当前可自动导出；Guitar Pro 在完成安装与激活前
保留为 manual/GUI export queue。

首轮使用 3 首曲目，共规划 27 个 PDF：TuxGuitar 9 个、MuseScore 9 个已生成，
Guitar Pro 9 个待导出。OCR 当前只评 `tab_only` 与 `score_tab`，所以每个已安装
渲染器各评 6 个样本。结果完全在任何跨软件微调之前获得：

| 渲染器/版式 | 样本成功 | 小节数 | 事件 P/R | 核心事件 exact | 节奏 exact | 指法事件 exact |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TuxGuitar `score_tab` | 3/3 | 122/122 | 99.416% / 100% | 91.856% | 99.832% | 91.940% |
| TuxGuitar `tab_only` | 3/3 | 122/122 | 99.582% / 100% | 92.024% | 99.664% | 92.107% |
| MuseScore `tab_only` | 3/3 | 122/133 | 77.516% / 49.790% | 0.169% | 42.159% | 1.012% |
| MuseScore `score_tab` | 0/3 | 未形成 IR | — | — | — | — |

TuxGuitar 基准中约 8% 的核心事件差距集中在一首双声部曲目：同一时间横坐标同时
出现两个声部时，部分 TAB 数字没有可靠绑定到具体声部。MuseScore 则首先暴露
几何域偏移：纯 TAB 谱线/小节被误分组，`score_tab` 的上下谱表配对失败。因此后续
顺序应是先适配页面几何和谱表配对，再共享数字/X CNN，最后训练声部归属与节奏
上下文；不能直接把低分归因于数字分类器。

复现命令：

```powershell
.\scripts\run_cross_renderer_benchmark.ps1 -SourceCount 12
```

Guitar Pro 导出的 PDF 放入 `manual_export_queue.jsonl` 指定路径后再次执行脚本，
即可在不改变参考语义和评测口径的前提下补齐第三个视觉域。

## 2026-07-14：Guitar Pro 8.1.2.37 域适配

后续接入本地固定版本的 Guitar Pro 8.1.2.37 worker，自动生成 `tab_only`、
`score_only` 和 `score_tab` 三种矢量 PDF 与 native layout JSON。v1 数据包含
127 首曲目、381 份 PDF、1,215 页和 26,103 个小节区域；训练与验证按源曲目
隔离，独立基准记录固定为 `training_eligible=false`。

域适配加入事件条件下的 blank/X/0–36 品位 CNN，并分别微调纯 TAB 与
score+TAB 的节奏、声部和技法上下文模型。扩大的独立基准找到了全部 9 首满足
当前单轨条件的测试曲，共 18 份 PDF、431 个小节：

| Guitar Pro 8.1.2.37 版式 | 样本成功 | 小节数 | 事件 P/R | 核心事件 exact | 节奏 exact | 弦/品 exact |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `tab_only` | 9/9 | 431/431 | 92.949% / 99.750% | 86.991% | 93.981% | 91.379% |
| `score_tab` | 9/9 | 431/431 | 95.382% / 99.468% | 79.063% | 90.695% | 87.111% |

几何回归还覆盖 Guitar Pro 的交替 15/17 像素弦距、和弦图伪谱表、首页反复线
前缀，以及同一横行结束/开始两个系统时的跨谱表长小节线。GP8 来源隔离验证集
达到纯 TAB 34/34 页、score+TAB 55/55 页和两种版式各 980/980 小节精确；
TuxGuitar 旧域保持 63 页、334 个谱表、958 个小节零失败。

最终 9 份纯 TAB 与 9 份 score+TAB IR 导出的 GP5 回读分别保留
2,744/2,744 和 3,364/3,364 个核心声部事件。TuxGuitar GP5 分别对 140 和
744 个事件的附加技法位发生降级，因此导出验收拆成强制核心层与可审计技法层；
技法预测的完整版本仍保存在 IR。该结果仍不覆盖纯五线谱反推指法、多轨复杂
结构、其他 Guitar Pro 版本和高精度跨事件 hammer/slide 连接关系。

## 2026-07-15：GP8 矢量真值、伪事件清理和横梁约束

本轮没有继续把官方 Guitar Pro 当作只提供像素的黑盒。PyMuPDF 从官方矢量 PDF 读取 TAB 字符边界框和绘图路径，源 GP 标签提供事件语义。二者对齐后得到实际官方排版坐标。GP8 字体签名用于隔离域专用逻辑，TuxGuitar 与栅格路径不受影响。

实现包括：矢量 TAB 攻击/弦品覆盖、速度文字、连音组括号、技法文字、延音音高墨迹候选、延音解析后的无音高伪事件清理、近邻重复谱头清理、事件重编号后的 `tie_edges` 重映射，以及两层矢量横梁确认十六分音符。两层横梁信号在独立 GP8 基准的 703 个覆盖事件上 703/703 为十六分音符，因此只把该高精度条件作为覆盖；一层横梁和多声部不覆盖 CNN。

高可信 GP8 节奏数据为 18,097/2,596/793 train/validation/test，含 14,928 个 GP8 高可信事件、29,262 个矢量 token、253 页和 2,749 个矢量锚定小节；4,227 个弱事件被排除。新节奏模型 GP8 validation 全语义 exact 99.307%，TuxGuitar test 全事件 exact 94.578%、可见主声部 97.462%。新延音模型 validation presence F1 100%，test F1 93.3%。

九首 source-disjoint GP8 `score_tab`、431 小节的 round22 结果：

| 指标 | round19 | round22 |
| --- | ---: | ---: |
| 事件 precision | 96.815% | 99.781% |
| 事件 recall | 99.812% | 99.750% |
| 核心事件 exact | 89.630% | 90.658% |
| 节奏 exact | 96.617% | 97.618% |
| 弦/品事件 exact | 92.701% | 92.759% |

同源三曲配对口径如下。两侧均为 122 小节、1,196 个参考事件：

| 渲染器 | 事件 P/R | 核心事件 exact | 节奏 exact | 弦/品 exact | 小节 exact |
| --- | ---: | ---: | ---: | ---: | ---: |
| Guitar Pro 8 | 100% / 99.916% | 97.741% | 97.992% | 99.582% | 90.164% |
| TuxGuitar | 100% / 100% | 98.495% | 98.913% | 99.415% | 91.803% |

因此核心、节奏和弦/品达到了配对差距不超过 1 个百分点的阶段门槛；完整九曲难例仍显著受未打印延音和复杂和弦关系影响，不能用三曲配对结果替代。

`finaltest/test.gp5` 的官方 `GT.pdf` 回归为 34/34 小节、278/278 事件、节奏 exact 100%、核心/弦品事件 exact 98.201%、音符 P/R 100%/98.361%、30/34 小节 exact。缺失项为 5 个第三弦延音音符。GP5 写出后 TuxGuitar 回读 278/278 核心事件，无结构或语义写出损失；官方 GP8 预览位于 `lab/20260715_gp8_target_round22/PRE_GP8.pdf`。
