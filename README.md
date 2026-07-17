# GuitarOCR

> Guitar Pro 8 的闭源数据构建、六条迭代工作流、验收门槛和最新配对结果见 [docs/GUITAR_PRO_PARITY_PLAN.md](docs/GUITAR_PRO_PARITY_PLAN.md)。

GuitarOCR 把规则排版的吉他“纯六线 TAB”或“五线谱 + 六线 TAB”PDF 恢复为可编辑的 GP5。当前主流程适配 TuxGuitar 2.0.1 与官方 Guitar Pro 8.1.2.37 的打印样式，不是手写谱或任意制谱软件的通用 OMR。

```text
PDF / 整页图片
  → 版式与谱表/小节几何
  → 时间事件定位
  → 节奏、TAB 弦/品、X、延音和演奏技法 CNN
  → Score Event IR + 小节时值/上下文约束
  → GP5
  → TuxGuitar 回读校验 + 预览 PDF
```

本分支同时提供已经完成 step 27403 训练的 Guitar Pro 8 结构化序列管线：

```text
GP8 PDF / 图片
  → 页面、谱表与小节切分
  → GLM-OCR 0.9B + LoRA
  → 可逆 M2 多声部事件序列
  → M2 解析与音乐约束
  → GP5
  → 官方 Guitar Pro 8 回渲染验收
```

这不是把 PDF 翻译成另一张 PDF。M2 显式保存图中可见的拍号、速度、事件起点、时值、休止、声部、弦/品、音高、反复和常见技法；`tab_only` 只监督图中可见的弦/品，`score_only` 只监督可见音高，`score_tab` 同时监督二者，避免要求模型猜测不可见信息。GP 文件中的 MIDI velocity 是播放参数，官方 GP8 三种打印版式均不显示，因此不作为 OCR 真值，写回时使用稳定默认值。M2 已实现双向解析；对普通节奏、多声部及含推弦/击勾弦/滑音/三连音的 GP5 做语义往返时逐小节完全一致，并通过官方 GP8 回渲染冒烟测试。

仓库包含源码、数据构建/训练/评估脚本、十二个传统 CNN 和一个合并的
GLM-OCR v2 LoRA adapter；不提交 GLM-OCR 基座、原始曲谱、批量渲染图、
训练数据库或 Guitar Pro 运行时。

### Guitar Pro v2 一键运行

完整安装、权重结构、指标和限制见
[docs/GUITAR_PRO_END_TO_END.md](docs/GUITAR_PRO_END_TO_END.md)。最短运行命令：

```powershell
git lfs pull
python -m pip install -e ".[glm-ocr]"
.\scripts\download_glm_ocr_base.ps1

.\scripts\run_guitarpro_pdf_to_gp5.ps1 `
  -InputPath "D:\scores\song.pdf" `
  -OutputDirectory "D:\scores\song_result" `
  -Mode tab
```

页面/小节切分没有模型 checkpoint。现役 v2 只加载一个约 30 MiB 的
`adapter_model.safetensors`；这个文件内部同时包含 4,128,768 个视觉塔
LoRA 参数和 3,735,552 个语言 Transformer LoRA 参数。`weights/*.pt` 的
十二个 CNN 不参与上述 GLM 端到端入口。

## 当前能做到什么

| PDF 版式 | 状态 | 当前能力 |
| --- | --- | --- |
| `score_tab`：五线谱 + 六线 TAB | **主流程支持** | PDF/整页图 → 事件、节奏、弦/品、X、延音与部分技法 → IR → GP5/PDF |
| `tab_only`：纯六线 TAB | **主流程支持** | PDF/整页图 → 音符与休止事件、双声部节奏、弦/品、X、延音与部分技法 → IR → GP5/PDF |
| `score_only`：纯五线谱 | **仅版式识别** | 能分类版式；仅凭音高无法唯一反推出吉他的弦/品指法，当前不导出 GP5 |

主流程的有效输入范围：

- TuxGuitar 2.0.1 或 Guitar Pro 8.1.2.37 导出的规则 `tab_only` / `score_tab` 矢量 PDF；CLI 默认自动判别两种版式；
- 完整页面 `PNG/JPG/JPEG/BMP/TIF/TIFF`，缩放和留白接近主流程的 180-DPI 页面；
- 常规拍号、音符/和弦事件、休止、附点、连音组、TAB 多位数品位和 X；
- 延音候选与部分和弦延音；dead/muted、vibrato、bend、hammer、slide、ghost、accent、harmonic、grace、palm mute、staccato、let ring、tapping 等技法标签；
- 单吉他轨、最多两个声部、六弦标准或 IR 中已知的调弦。

尚不承诺：手写谱、拍照/透视、严重扫描噪声、复杂多声部、多轨、纯五线谱反推指法、全部反复结构与跨系统连线，以及其他 Guitar Pro 版本、MuseScore、Sibelius 等未验收字体和排版。其他软件或版本导出的规则谱需要加入对应真实 PDF 做域适配。

GP5 也有格式限制：TuxGuitar 的 GP5 模型不能让同一音同时保留 dead/X 与 slide，也不能同时保留 dead/X 与 tie-in。导出时优先保留页面上可见的 X，冲突会写入 `*.report.json`；完整语义仍保存在 IR，未来可由 GPIF 写入器使用。TuxGuitar 的 GP5 写入/回读还可能丢弃部分附加技法位，报告中的 `readback_lossy_semantic_events` 会明确计数。标题、调号和分页样式也不保证与源 PDF 一致，但音高可用逐音临时升降号表达。

## 已验证环境

| 项目 | 已验证配置 |
| --- | --- |
| 操作系统 | Windows 10/11 x86-64 |
| Python | 3.10–3.12；本次开发/回归为 Python 3.11 |
| PyTorch | CPU 可推理；NVIDIA CUDA 推荐训练和批量推理 |
| 制谱运行时 | TuxGuitar 2.0.1；Guitar Pro 8.1.2.37 仅用于对应域的数据生成与验收 |
| Java | JDK 17+，命令行可运行 `javac` |
| PDF 渲染 | Poppler，180 DPI 灰度 |
| 输出 | GP5、Score IR JSON、计划 TSV、回读报告 JSON、预览 PDF |

## Windows 快速开始

先安装 Git、Python 3.11 和 JDK 17+：

```powershell
git clone https://github.com/LZJU-1/guitarOCR.git
cd guitarOCR
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\setup_windows.ps1
.\.venv\Scripts\Activate.ps1
guitarocr-check
```

安装脚本创建 `.venv`，并把 TuxGuitar 2.0.1 与 Poppler 下载到仓库本地。如果已有运行时，可以设置：

```powershell
$env:GUITAROCR_TUXGUITAR_ROOT = 'D:\software\tuxguitar'
$env:GUITAROCR_POPPLER_BIN = 'D:\software\poppler\Library\bin'
$env:GUITAROCR_JAVAC = 'C:\Program Files\Java\jdk-21\bin\javac.exe'
```

### PDF 直接转 GP5

```powershell
guitarocr D:\scores\input.pdf `
  -o D:\scores\result.gp5 `
  --preview-pdf D:\scores\result_preview.pdf
```

中间文件默认在 `database/end_to_end/<PDF文件名>/`：

- `rendered_pages/`：180-DPI 页图；
- `pages/`：逐页 IR、事件裁块和 QA 可视化；
- `document_score_ir.json`：跨页合并后的 Score Event IR；
- 输出旁的 `.plan.tsv` / `.report.json`：写入计划、假设、格式降级和 TuxGuitar 回读结果。

程序写完 GP5 后会重新读入，并分层校验：小节数、声部事件位置、节奏、弦/品、延音和 X 属于强制核心层，任一不一致都会让命令失败；附加演奏技法属于可审计语义层，GP5/TuxGuitar 未保留时记录 `readback_fully_matched_events` 与 `readback_lossy_semantic_events`，但不把一份结构正确、可打开的 GP5 误判为导出失败。

### 整页图片输入

```powershell
python -m guitarocr.pipeline.infer_tuxguitar_score_tab_document `
  D:\pages\page_001.png D:\pages\page_002.png `
  --output D:\work\song

# 纯 TAB 整页图改用这一入口
python -m guitarocr.pipeline.infer_tuxguitar_tab_document `
  D:\tab_pages\page_001.png D:\tab_pages\page_002.png `
  --output D:\work\tab_song

python -m guitarocr.export.export_score_ir_to_gp `
  D:\work\song\document_score_ir.json `
  -o D:\work\song.gp5 `
  --preview-pdf D:\work\song_preview.pdf
```

必须按页序传入完整页面，不要先把页面切成单个符号。单独查看某页属于哪种版式：

```powershell
guitarocr-layout D:\pages\page_001.png
```

### 有 GP 真值时做端到端验收

```powershell
.\scripts\run_gp_acceptance.ps1 `
  -Gp 'D:\scores\song.gp' `
  -OutputDir 'D:\scores\song_acceptance'
```

该脚本依次生成 `GT.pdf`、执行 `GT.pdf → PRE.gp5 → PRE.pdf`，再把 OCR IR 与 GP7/8 内部 GPIF 语义逐事件比较。

### 跨制谱软件基准（实验）

同一个 GP 文件经不同软件导出后，字体、谱线间距、系统配对和技法字形都会变化；因此不能用 TuxGuitar 测试集代替 Guitar Pro 的真实验收。仓库提供来源隔离的三渲染器基准：

```powershell
.\scripts\run_cross_renderer_benchmark.ps1 `
  -SourceCount 12 `
  -Renderers tuxguitar,musescore,guitarpro
```

TuxGuitar、MuseScore 4 与 Guitar Pro 8 现在都能自动生成基准 PDF。Guitar Pro 分支使用本地 `guitar-hero-main` 的固定 8.1.2.37 worker，先把曲目改写为目标显示模式，再后台导出 PDF 和 native layout JSON。它只用于 Windows，会在启动前结束现有 `GuitarPro.exe` 进程；运行时与注入 DLL 不进入本仓库。基准只从 source-disjoint `test` 划分选曲，每条记录均为 `training_eligible=false`，不能反向加入训练集。

```powershell
$env:GUITAROCR_GP8_DATAGEN_ROOT = 'D:\guitarOCR\guitar-hero-main'
$env:GUITAROCR_GP8_PYTHON = 'D:\guitarOCR\guitar-hero-main\.venv\Scripts\python.exe'

# 单文件：同时输出官方 GP8 PDF 与 native layout JSON
python -m guitarocr.export.render_gp_to_guitarpro_pdf `
  D:\scores\song.gp5 D:\scores\song_gp8.pdf
```

适配器默认严格校验 `GuitarPro.exe` 和注入 DLL 的 SHA-256，只接受已验证的 8.1.2.37 配对。安全边界、依赖准备和数据目录见 [docs/GUITAR_PRO_DATASET.md](docs/GUITAR_PRO_DATASET.md)。

2026-07-14 的 GP8 source-disjoint 端到端基准只找到 9 首满足“单轨、4–8 弦、GP3/4/5、从未进入训练”的曲目，共 18 份 PDF、431 个小节。域适配后两种版式均恢复 431/431 小节：`tab_only` 事件 P/R 92.949%/99.750%、节奏 exact 93.981%、弦/品 exact 91.379%、核心事件 exact 86.991%；`score_tab` 事件 P/R 95.382%/99.468%、节奏 exact 90.695%、弦/品 exact 87.111%、核心事件 exact 79.063%。最终 9 份 `tab_only` 和 9 份 `score_tab` IR 导出的 GP5 经回读分别保留 2,744/2,744 与 3,364/3,364 个核心声部事件；附加技法位分别有 140 和 744 个事件未被 TuxGuitar GP5 回读完整保留，已单独报告而不计入视觉核心指标。这些是独立曲目结果，不等于“所有 Guitar Pro 排版都已解决”；尤其纯五线谱、复杂结构和若干连接型技法仍在范围外。

## 2026-07-15：Guitar Pro 8 矢量结构与配对验收

官方 GP8 矢量 PDF 现在优先使用字符边界框恢复 TAB 攻击和弦/品，并读取速度、连音组文字及横梁绘图路径；栅格图片仍走检测器/CNN。源 GP 语义与官方 PDF 坐标对齐构成训练真值，不需要读取闭源 Guitar Pro 的内部 IR。

独立九曲 `score_tab` 难例（431 小节）当前事件 P/R 99.781%/99.750%、核心事件 exact 90.658%、节奏 exact 97.618%、弦/品事件 exact 92.759%。在与 TuxGuitar 完全相同的三首配对曲上，GP8 核心/节奏/弦品分别为 97.741%/97.992%/99.582%，TuxGuitar 为 98.495%/98.913%/99.415%；三个核心指标差距均不超过 1 个百分点。九曲绝对难例仍未达到最终 98% 目标，不能解释为“所有 Guitar Pro PDF 已解决”。

指定 `finaltest/test.gp5` 回归达到 34/34 小节、278/278 事件、节奏 100%、核心事件 98.201%；仍缺 5 个未打印延音音符。最新官方 GP8 回渲染产物在 `lab/20260715_gp8_target_round22/`。

## 模型和本次指标

默认推理读取 `weights/` 下共 28.81 MiB 的十二个 checkpoint。数字/X 和拍号模型共享，事件/节奏/延音/技法按版式使用独立模型：

| 模型 | 参数量 | 作用 |
| --- | ---: | --- |
| `score_event_locator.pt` | 161,538 | 在五线谱小节中定位时间事件横坐标 |
| `rhythm_context_cnn.pt` | 882,956 | 音符/休止、时值、附点、连音组等上下文 |
| `tab_event_locator.pt` | 161,538 | 在纯 TAB 小节中定位全部音符/休止事件横坐标 |
| `tab_rhythm_context_cnn.pt` | 882,956 | 从纯 TAB 符杆、横梁、休止与附点恢复双声部节奏 |
| `tab_symbol_detector.pt` | 592,623 | 定位并分类 TAB 数字与 X |
| `fret_token_cnn.pt` | 133,031 | 在已定位事件的每根弦上分类空白、X 与 0–36 品，负责 Guitar Pro 字体域适配 |
| `atomic_symbol_cnn.pt` | 218,884 | 拍号数字等原子印刷符号 |
| `tie_context_cnn.pt` | 888,096 | 延音存在、数量和目标音高关系 |
| `technique_context_cnn.pt` | 874,989 | 多标签演奏技法上下文 |
| `pick_stroke_context_cnn.pt` | 875,503 | 上拨/下拨事件上下文；只覆盖主技法模型的 PickStroke 两项 |
| `tab_tie_context_cnn.pt` | 888,096 | 纯 TAB 延音存在、数量和目标弦关系 |
| `tab_technique_context_cnn.pt` | 875,503 | 纯 TAB 多标签演奏技法上下文 |

扩大后的 TuxGuitar 语料含 331 个 GP/GTP 源文件、26,682 小节、180,417 个节奏事件裁块、318,376 个音符；TAB 检测集合含 65,953 个混合版式小节图块和 752,401 个数字/X 标注。划分按源曲目隔离。

主要独立测试结果：

- `score_tab` 节奏 CNN：主声部完整语义 99.596%，全事件 99.456%；
- TAB 检测器：precision 99.614%、recall 99.970%、F1 99.791%，85,423 个真值中漏检 26 个；
- 延音 CNN：presence F1 99.119%，延音数量准确率 88.950%，目标纵坐标 F1 94.422%；
- 技法 CNN：dead F1 99.650%、vibrato 95.833%、bend 98.876%、palm mute 96.970%；slide 只有 48.415%，ghost/accent 等稀有类仍需更多真实样本。PickStroke 的独立测试只有 1 个上拨、0 个下拨，不足以证明跨曲泛化。

纯 TAB 的 source-disjoint 测试共 149 页、2,874 个小节、18,397 个事件：

- 页面几何 149/149 页、2,874/2,874 小节完全正确；事件定位 P 99.978%、R 100%、F1 99.989%；
- 自动定位后主声部节奏 exact 99.750%，双声部事件 exact 99.706%；
- 可见 TAB 事件（整组弦/品/X）exact recall 99.056%，节奏+可见指法主核心事件 exact 99.005%；
- 视觉拍号经小节节奏容量交叉确认后为 2,874/2,874 小节正确；
- 纯 TAB 整页延音视觉/语义候选 F1 均为 99.653%，目标弦纵坐标 F1 99.365%；自动续接事件 precision 98.464%、音符 precision 99.281%、真值音符覆盖率 98.106%；
- 纯 TAB 技法 macro F1 83.181%，其中 slide F1 95.172%；staccato 仍为 0，tapping 与 pick-down 在测试集没有正例。

目标回归曲《若能绽放光芒 Final 教学版》在最多 10 轮 hard-case 修正后达到 168/168 小节、701/701 事件、1,425/1,425 音符、61/61 个 X，以及已纳入现有技法口径的 palm mute/普通 slide/vibrato、153 个上拨和 389 个下拨全部命中。另有 7 个 `X + pick-scrape/slide-like` 复合标记尚未作为独立语义建模；TuxGuitar GP5 模型也不能同时表达 dead/X 与 slide。**该曲被加入过训练和规则修正，因此这是回归验收，不是独立泛化成绩。**

模型文件大小和 SHA-256 见 [weights/README.md](weights/README.md)，完整实验口径见 [docs/EXPERIMENT_RESULTS.md](docs/EXPERIMENT_RESULTS.md)。

## 用自己的 GP 扩大数据集

### 1. 准备语料

把合法的 `.gp/.gp3/.gp4/.gp5/.gtp/.gpx` 放到一个目录。只使用有权处理的数据，不要把第三方曲谱提交到 GitHub。

```powershell
.\scripts\build_database.ps1 `
  -CorpusRoot D:\my_gp_corpus `
  -DatabaseRoot D:\guitarocr_database `
  -PerFormat 100

$env:GUITAROCR_DATABASE_ROOT = 'D:\guitarocr_database'
```

基础构建会让 TuxGuitar 输出 `tab_only`、`score_tab`、`score_only` 三种渲染图和曲谱语义。坐标真值只在生成训练标签时使用；正式 OCR 只有 PDF/图片像素。

### 2. 生成坐标、节奏和技法标签

```powershell
.\scripts\build_tuxguitar_page_annotations.ps1 -Layout tab_only
.\scripts\build_tuxguitar_page_annotations.ps1 -Layout score_tab
.\scripts\build_score_rhythm_dataset.ps1

# 可选：GP7/8 .gp ZIP 的技法以内部 GPIF 精确覆盖渲染器标签
python -m guitarocr.data.augment_gpif_technique_labels <source_id> `
  --database D:\guitarocr_database
```

数据构建器会按源曲目创建 train/validation/test，避免同一首曲子的不同页面泄漏到测试集。`source_id` 可在 `database/manifests/sources.jsonl` 查看。

### 3. 生成 Guitar Pro 8 三版式域数据

把 `guitar-hero-main` 放在仓库根目录，并为它准备仅含 `PyGuitarPro / PyMuPDF / Pillow / fonttools` 的 `.venv` 后运行：

```powershell
.\scripts\build_guitarpro_multimode_dataset.ps1 `
  -RealDatasetDir D:\guitarocr_database\v2\source\gp `
  -WorkDir D:\guitarocr_database\guitarpro8_multimode_v1
```

默认配置为 50 首技法覆盖合成曲和最多 80 首真实 GP5，分别生成 `tab / notation / both`，并输出 PDF、PNG、native layout、小节 TNL crop 标签及版面 COCO。训练必须使用 `layout_coco_source_disjoint`；它按曲目把三种版式和全部页面绑定到同一划分，不能使用外部工具原始的 `layout_coco` 随机划分。

native layout 的自动真值是页面区域和小节语义，不是逐数字/逐音符像素框。仓库会把 native 小节、源 GP 事件和像素检测候选对齐，生成事件条件下的逐弦品位/X、节奏和技法训练裁块；source-disjoint `test` 曲目会被强制排除。

新的多声部 M2 序列数据直接从完整 GP3/GP4/GP5/GTP 语料分层抽样，并使用官方 GP8 同时生成 `tab / notation / both`：

```powershell
.\scripts\build_gp8_measure_sequence_dataset.ps1 `
  -Corpus D:\my_gp_corpus `
  -Output D:\guitarOCR\database\gp8_measure_sequence_v2 `
  -SourceCount 2000
```

2026-07-15 的本地 v1 数据由 600 首来源曲目产生 138,618 个小节样本：训练 110,016、验证 12,936、测试 15,666。三种显示模式各 46,206，所有划分按原始文件 SHA 隔离，数据构建的坐标/小节数失败为 0。生成数据位于 `database/`，不会提交到 Git。

v2 已从完整 285,362 份 GP3/GP4/GP5/GTP 候选中按格式、稀有技法和多声部筛选 2,000 首，得到 149,126 个唯一小节、三版式 447,378 个目标。一首会稳定终止官方 GP8 worker 的脏源已由一首从未进入 v2、且三版式渲染通过的复杂 GP4 补位。标签会在训练前全部经过 parse→canonical 和音乐约束门禁；训练集还会额外生成按模式限额的长尾难例文件。修改 schema 后可用 `-Phase relabel` 复用同一来源选择，不必重新扫描整个语料。

官方 GP8 矢量 PDF 优先读取矢量谱表行。纯 TAB 常常不画内部竖直小节线，和弦符杆又会跨越全部弦线，因此 PDF 路径使用连续打印的小节号恢复边界；扁平 PDF/图片才回退到像素竖线检测。200 首 source-disjoint v2 测试曲上，`notation / both / tab` 分别为 572/572、904/904、612/612 页完全匹配，三种模式均恢复 15,304/15,304 个小节。这个数字只评估页面到小节框，不代表音符 OCR 准确率。可复现命令：

```powershell
python -m guitarocr.evaluation.validate_gp8_measure_geometry `
  --dataset database\gp8_measure_sequence_v2 `
  --report reports\gp8_measure_sequence_v2_geometry.json
```

M2→GP5 使用确定性指法约束；纯五线谱中不可见的弦号由约束搜索分配，并强制和弦内弦号不重复、音高可演奏、有效延音沿用同一根弦。200 首 v2 测试曲中，`tab` 与 `both` 均为 15,304/15,304 小节语义往返一致；`notation` 为 15,296/15,304。后者的 8 个差异来自 5 份原始 GP 的孤立 tie：源文件没有可继承的同弦前音，而纯五线谱又不包含弦号，不能声称可唯一恢复原作者弦位。

```powershell
D:\guitarOCR\guitar-hero-main\.venv\Scripts\python.exe `
  -m guitarocr.evaluation.validate_m2_gp5_roundtrip
```

### 4. 微调 GLM-OCR 事件序列模型

本项目使用官方 GLM-OCR 推荐的 LLaMA-Factory LoRA 方案。RTX 20 系显卡使用 FP16，不能照搬 BF16 配置：

```powershell
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git tools\LLaMA-Factory
D:\anaconda3\envs\raftstereo\python.exe -m pip install -e tools\LLaMA-Factory

D:\anaconda3\envs\raftstereo\Scripts\hf.exe download `
  zai-org/GLM-OCR --local-dir tools\models\GLM-OCR

.\scripts\train_glm_ocr_measure_sequence.ps1
```

脚本默认使用 v2。v1 基线配置见 `configs/glm_ocr_measure_sequence_lora_fp16.yaml`：LoRA rank 8、3,735,552 个可训练参数、输入上限 2,304 tokens，并遍历全部训练样本；它冻结视觉塔，只用于验证输出语法和语言侧基线。v2 配置 `configs/glm_ocr_measure_sequence_v2_lora_fp16.yaml` 直接在修正后的大数据上训练 7,864,320 个语言层与视觉塔 LoRA 参数，完整输入上限 3,072 tokens，混合全量小节与有上限的长尾难例；减小 micro-batch 并增加梯度累积以适配 22GB 显存。非首小节加入上一小节的紧凑 `C2` 上下文来学习延音和声部连续性；推理时上下文来自上一小节预测，不读取源 GP。独立测试集的结构指标使用：

```powershell
.\scripts\evaluate_glm_ocr_measure_sequence.ps1 -MaxSamples 3000
```

报告分别给出 M2 语法/约束有效率、核心字段整小节全对率、节奏全对率、音符字段全对率、完整 exact、事件起点 F1、时值/休止、弦、品、音高和逐技法 F1；3000 条正式抽样会按版式、来源和技法覆盖分层，600 条只用于快速冒烟。评测脚本固定读取 source-disjoint v2 测试集，并把测试清单、模型、adapter 与生成参数写入运行签名；只有显式 `--resume` 且签名一致时才续跑，防止旧 checkpoint 的预测混入新报告。不能用训练 loss 代替端到端准确率。推理输出若违反版式字段、事件顺序、时值/弦品范围或 `both` 的音高-弦品关系，会基于原图和约束错误进行有上限的自纠，不静默写入坏 GP5。

一键构建并微调 Guitar Pro 域模型：

```powershell
.\scripts\run_guitarpro_domain_adaptation.ps1 `
  -GuitarProDataset D:\guitarocr_database\guitarpro8_multimode_v1 `
  -Epochs 12 -Promote
```

技法发布默认要求验证 precision ≥ 25% 且至少 10 个正例；不达标的类别在 checkpoint 中禁用。`score_tab` hammer 目前也由发布安全门主动关闭，等待相邻事件对模型替代单事件分类器。

### 5. 训练传统 CNN 与评估

```powershell
.\scripts\run_symbol_cnn.ps1
.\scripts\run_tab_detector.ps1
.\scripts\run_rhythm_context.ps1 -SkipDatasetBuild
.\scripts\run_score_event_locator.ps1
.\scripts\run_tie_context.ps1
.\scripts\run_tab_only_models.ps1

python -m guitarocr.training.train_technique_context `
  --database D:\guitarocr_database --epochs 20 --batch-size 64

# 联合模型也可作为 PickStroke 专用覆盖模型；分别验收后再发布
Copy-Item D:\guitarocr_database\technique_events\models\technique_context_cnn.pt `
  D:\guitarocr_database\technique_events\models\pick_stroke_context_cnn.pt
```

`run_tab_only_models.ps1` 会依次构建纯 TAB 事件、节奏、技法、延音数据，训练四个版式专用模型，并执行整页事件、节奏、弦/品、合并 IR、延音和拍号的独立测试。已有数据或模型时可传 `-SkipDatasetBuild` / `-SkipTraining`。

训练建议使用 NVIDIA GPU。只有 validation/test、整页 IR 和重渲染 PDF 都通过后再发布权重：

```powershell
.\scripts\promote_models.ps1 -DatabaseRoot D:\guitarocr_database -GuitarProDomain
guitarocr-check
```

## 少量新样本微调

节奏、TAB、延音和技法训练器都能从现有 checkpoint 初始化：

```powershell
python -m guitarocr.training.train_rhythm_context `
  --database D:\guitarocr_database --epochs 8 --learning-rate 0.0002 `
  --init-checkpoint .\weights\rhythm_context_cnn.pt

python -m guitarocr.training.train_tab_detector `
  --database D:\guitarocr_database --epochs 8 --learning-rate 0.0002 `
  --init-checkpoint .\weights\tab_symbol_detector.pt

python -m guitarocr.training.train_tie_context `
  --database D:\guitarocr_database --epochs 8 --learning-rate 0.0001 `
  --init-checkpoint .\weights\tie_context_cnn.pt

python -m guitarocr.training.train_technique_context `
  --database D:\guitarocr_database --epochs 8 --learning-rate 0.0001 `
  --init-checkpoint .\weights\technique_context_cnn.pt

# 纯 TAB 使用相同网络结构、独立数据域和独立 checkpoint
python -m guitarocr.training.train_rhythm_context `
  --database D:\guitarocr_database --task-root tab_rhythm_events `
  --epochs 8 --learning-rate 0.0002 `
  --init-checkpoint .\weights\tab_rhythm_context_cnn.pt

python -m guitarocr.training.train_tie_context `
  --database D:\guitarocr_database --task-root tab_tie_events `
  --epochs 8 --learning-rate 0.0001 `
  --init-checkpoint .\weights\tab_tie_context_cnn.pt
```

实践要求：

- 新域测试集必须按源曲目隔离；目标 hard case 的成绩不能当独立测试成绩；
- 保留多拍号、时值、和弦、休止、附点、连音、多位数品位和稀有技法；
- 技法是长尾多标签任务，slide、ghost、accent、tapping 等不能只靠过采样，仍需要足够不同曲目的正例；
- 适配其他制谱软件时，加入该软件真实导出的 PDF，并留一套完全未训练的整页测试集；
- 不只看裁块准确率，最终看整页事件、小节时值、弦/品和 GP5 回读/重渲染。

## CNN 与 0.9B 视觉语言模型如何分工

十二个专用 CNN 仍是小体积、低延迟的稳定主线，也适合提供事件候选。GLM-OCR 0.9B 分支不直接生成二进制 GP，而是生成受严格语法约束、可解析和可评测的 M2 事件序列；它负责联合理解节奏、声部和上下文。确定性代码继续负责页面顺序、调弦/指法约束、M2→GP5 写入和回渲染验收。

通用预训练模型本身不会音乐转写。未微调的 GLM-OCR 在 M2 冒烟样本上语法有效率为 0，因此只有 source-disjoint 微调与结构指标通过后才能替换现有主流程。模型选择依据与官方项目链接见 [docs/OCR_CONTEXT_MODEL_ASSESSMENT.md](docs/OCR_CONTEXT_MODEL_ASSESSMENT.md)。

## 代码结构

```text
guitarocr/
├─ models/       # 轻量 CNN
├─ pipeline/     # 页面解析、版式、事件、上下文约束和 IR 合并
├─ data/         # GP/GPIF 读取与数据集构建
├─ training/     # 训练、微调、阈值选择
├─ evaluation/   # 数据、模型、M2、整页和 GPIF 评估
├─ export/       # Score IR/M2 → GP5、GP → PDF
└─ cli/          # 命令行入口、版式分类和环境检查
scripts/         # Windows 安装、数据构建、训练、验收和发布
java/            # TuxGuitar 标注、写入、回读和渲染桥接
weights/         # 十一个随仓库发布的 checkpoint
docs/            # 架构、IR、实验和上下文模型评估
database/        # 本地生成，不提交
```

进一步阅读：

- [识别与重建架构](docs/RECOGNITION_ARCHITECTURE.md)
- [Score Event IR 协议](docs/EVENT_IR_SCHEMA.md)
- [Guitar Pro 8 小节事件序列与 M2](docs/GP8_MEASURE_SEQUENCE.md)
- [实验与结果](docs/EXPERIMENT_RESULTS.md)
