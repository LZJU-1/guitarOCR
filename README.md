# GuitarOCR

GuitarOCR 把规则排版的吉他“纯六线 TAB”或“五线谱 + 六线 TAB”PDF 恢复为可编辑的 GP5。项目当前聚焦 TuxGuitar 2.0.1 的印刷样式，不是手写谱或任意制谱软件的通用 OMR。

```text
PDF / 整页图片
  → 版式与谱表/小节几何
  → 时间事件定位
  → 节奏、TAB 弦/品、X、延音和演奏技法 CNN
  → Score Event IR + 小节时值/上下文约束
  → GP5
  → TuxGuitar 回读校验 + 预览 PDF
```

仓库包含源码、数据构建/训练/评估脚本和十一个轻量推理模型；不提交原始曲谱、批量渲染图、训练数据库或第三方运行时。

## 当前能做到什么

| PDF 版式 | 状态 | 当前能力 |
| --- | --- | --- |
| `score_tab`：五线谱 + 六线 TAB | **主流程支持** | PDF/整页图 → 事件、节奏、弦/品、X、延音与部分技法 → IR → GP5/PDF |
| `tab_only`：纯六线 TAB | **主流程支持** | PDF/整页图 → 音符与休止事件、双声部节奏、弦/品、X、延音与部分技法 → IR → GP5/PDF |
| `score_only`：纯五线谱 | **仅版式识别** | 能分类版式；仅凭音高无法唯一反推出吉他的弦/品指法，当前不导出 GP5 |

主流程的有效输入范围：

- TuxGuitar 2.0.1 导出的规则 `tab_only` 或 `score_tab` 矢量 PDF；CLI 默认自动判别两种版式；
- 完整页面 `PNG/JPG/JPEG/BMP/TIF/TIFF`，缩放和留白接近 180-DPI TuxGuitar 页面；
- 常规拍号、音符/和弦事件、休止、附点、连音组、TAB 多位数品位和 X；
- 延音候选与部分和弦延音；dead/muted、vibrato、bend、hammer、slide、ghost、accent、harmonic、grace、palm mute、staccato、let ring、tapping 等技法标签；
- 单吉他轨、最多两个声部、六弦标准或 IR 中已知的调弦。

尚不承诺：手写谱、拍照/透视、严重扫描噪声、复杂多声部、多轨、歌词/和弦图、反复结构、全部跨系统连线，以及 Guitar Pro、MuseScore、Sibelius 等其他软件的字体和排版。其他软件导出的规则谱需要加入对应真实 PDF 做域适配。

GP5 也有格式限制：TuxGuitar 的 GP5 模型不能让同一音同时保留 dead/X 和 slide。导出时优先保留页面上可见的 X，冲突会写入 `*.report.json`；完整语义仍保存在 IR，未来可由 GPIF 写入器使用。标题、调号和分页样式也不保证与源 PDF 一致，但音高可用逐音临时升降号表达。

## 已验证环境

| 项目 | 已验证配置 |
| --- | --- |
| 操作系统 | Windows 10/11 x86-64 |
| Python | 3.10–3.12；本次开发/回归为 Python 3.11 |
| PyTorch | CPU 可推理；NVIDIA CUDA 推荐训练和批量推理 |
| 制谱运行时 | TuxGuitar 2.0.1 |
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

程序写完 GP5 后会重新读入，并逐声部检查小节数、事件位置、节奏、弦/品、延音及可表示的技法；回读不一致时命令失败。

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

TuxGuitar 与 MuseScore 4 可自动生成 PDF。Guitar Pro 8 需要先由用户完成安装和试用/许可证激活；当前脚本会把待导出的源 GP、版式和目标 PDF 路径写入 `database/cross_renderer_benchmark/manual_export_queue.jsonl`。把 Guitar Pro PDF 放到队列指定位置后重新运行命令，即会自动纳入同一套像素 OCR 评估。基准只从 source-disjoint `test` 划分选曲，每条记录均为 `training_eligible=false`，不能反向加入训练集。

2026-07-14 的首个 3 首曲目、零微调基线表明域差异很大：TuxGuitar 两种版式均恢复 122/122 小节，事件召回 100%，核心事件 exact 约 91.9%–92.0%（主要缺口为双声部同横坐标的音符归属）；MuseScore `tab_only` 的核心事件 exact 只有 0.169%，`score_tab` 3/3 均未通过谱表配对。该结果用于确定适配优先级，不代表训练后的目标性能。

## 模型和本次指标

默认推理读取 `weights/` 下共 28.29 MiB 的十一个 checkpoint。数字/X 和拍号模型共享，事件/节奏/延音/技法按版式使用独立模型：

| 模型 | 参数量 | 作用 |
| --- | ---: | --- |
| `score_event_locator.pt` | 161,538 | 在五线谱小节中定位时间事件横坐标 |
| `rhythm_context_cnn.pt` | 882,956 | 音符/休止、时值、附点、连音组等上下文 |
| `tab_event_locator.pt` | 161,538 | 在纯 TAB 小节中定位全部音符/休止事件横坐标 |
| `tab_rhythm_context_cnn.pt` | 882,956 | 从纯 TAB 符杆、横梁、休止与附点恢复双声部节奏 |
| `tab_symbol_detector.pt` | 592,623 | 定位并分类 TAB 数字与 X |
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

### 3. 训练与评估

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
.\scripts\promote_models.ps1 -DatabaseRoot D:\guitarocr_database
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

## 为什么没有默认接入 1B OCR/LLM

GLM-OCR 0.9B、HunyuanOCR 等是通用文档视觉语言模型，不直接输出本项目所需的弦号、品位、精确 onset、时值和跨事件约束。当前约 28.3 MiB 的专用 CNN + 确定性音乐约束更小、更快，也能用 GP 真值精确监督。它们可作为弱标注教师、标题/速度文字 OCR 或失败页路由器；真正适合下一步的是在 CNN 事件特征上训练约 10–30M 参数的小型事件序列 Transformer，而不是让通用 OCR 模型直接生成 GP。

评估依据与官方项目链接见 [docs/OCR_CONTEXT_MODEL_ASSESSMENT.md](docs/OCR_CONTEXT_MODEL_ASSESSMENT.md)。

## 代码结构

```text
guitarocr/
├─ models/       # 轻量 CNN
├─ pipeline/     # 页面解析、版式、事件、上下文约束和 IR 合并
├─ data/         # GP/GPIF 读取与数据集构建
├─ training/     # 训练、微调、阈值选择
├─ evaluation/   # 数据、模型、整页和 GPIF 评估
├─ export/       # Score IR → GP5、GP → PDF
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
- [实验与结果](docs/EXPERIMENT_RESULTS.md)
