# GuitarOCR

GuitarOCR 将规则排版的吉他“五线谱 + 六线 TAB”PDF 恢复为可编辑的 GP5。当前目标不是通用乐谱 OCR，而是先可靠覆盖网上常见的、由制谱软件导出的标准吉他谱。

```text
PDF / 整页图片
  → 谱表与小节几何
  → 事件位置、节奏、休止、附点、连音与 TAB 指法识别
  → Score Event IR
  → 小节时值和延音约束
  → GP5
  → TuxGuitar 回读校验与预览 PDF
```

仓库包含完整源码、数据构建/训练/评估脚本和五个推理模型；不包含原始曲谱语料、批量渲染图片以及第三方 TuxGuitar/Poppler 运行时。

## 已验证运行环境

| 项目 | 当前范围 |
| --- | --- |
| 操作系统 | Windows 10/11 x86-64（当前完整流程的已验证平台） |
| Python | 3.10–3.12；开发与回归环境为 Python 3.11 |
| PyTorch | CPU 可推理；NVIDIA CUDA 推荐用于训练和批量推理 |
| 制谱运行时 | TuxGuitar 2.0.1 |
| Java | JDK 17+，必须能在命令行运行 `javac` |
| PDF 渲染 | Poppler，固定以 180 DPI 灰度渲染 |
| 输出 | GP5、Score IR JSON、事件计划 TSV、回读报告 JSON、可选预览 PDF |

TuxGuitar 和 Poppler 都由安装脚本从各自的 GitHub Release 下载到仓库本地目录，不会提交到 Git。

## 支持什么谱面

当前预训练模型的有效范围：

- TuxGuitar 2.0.1 导出的标准 `score_tab` 页面：上方五线谱、下方六弦 TAB；
- 规则的矢量 PDF，横向或纵向分页均可，输入 PDF 会统一渲染为 180 DPI；
- 整页 `PNG/JPG/JPEG/BMP/TIF/TIFF`，但尺寸、缩放和留白应接近 TuxGuitar 180-DPI 页面；
- 常规拍号、音符/和弦事件、休止、附点、连音、横梁、TAB 品位数字与 X；
- 单吉他轨和主声部是当前 GP5 重建重点。

以下情况目前不承诺正确：

- 手写谱、相机拍照、明显倾斜/透视、低清扫描、重噪声或严重裁边；
- Guitar Pro、MuseScore、Sibelius 等其他软件的字体与页面布局——后续需要用对应软件的样本做域适配；
- 纯数字简谱、鼓谱、钢琴谱、歌词 OCR；
- 多轨、复杂多声部、和弦图、歌词、完整演奏技法和所有跨系统连线的无损复原；
- 只改了文件扩展名、内部并非有效图片/PDF 的文件。

换句话说：当前最适合“从 TuxGuitar 导出的规则 PDF 再转回 GP5”，其他软件导出的规则谱可以尝试，但应先检查 `report.json` 和重渲染 PDF。

## Windows 快速开始

先安装 Git、Python 3.11 和 JDK 17+，然后：

```powershell
git clone git@github.com:LZJU-1/guitarOCR.git
cd guitarOCR
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\setup_windows.ps1
.\.venv\Scripts\Activate.ps1
guitarocr-check
```

`setup_windows.ps1` 会创建 `.venv`，安装 Python 依赖，并下载 TuxGuitar 2.0.1 与 Poppler。若已经自行安装，可以设置：

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

也可以不激活虚拟环境：

```powershell
& .\.venv\Scripts\python.exe .\pdf_to_gp.py D:\scores\input.pdf `
  -o D:\scores\result.gp5 `
  --preview-pdf D:\scores\result_preview.pdf
```

推理中间文件默认位于 `database/end_to_end/<PDF文件名>/`：

- `rendered_pages/`：180-DPI 页面；
- `pages/`：逐页 IR、事件裁块和可视化；
- `document_score_ir.json`：跨页合并后的中间语义；
- 输出旁的 `.plan.tsv` 和 `.report.json`：GP5 写入计划与 TuxGuitar 回读统计。

### 整页图片输入

根命令针对 PDF。已有整页图片时先生成 IR，再导出：

```powershell
python -m guitarocr.pipeline.infer_tuxguitar_score_tab_document `
  D:\pages\page_001.png D:\pages\page_002.png `
  --output D:\work\song

python -m guitarocr.export.export_score_ir_to_gp `
  D:\work\song\document_score_ir.json `
  -o D:\work\song.gp5 `
  --preview-pdf D:\work\song_preview.pdf
```

图片必须是完整页面并按页序传入；不要先切成单个音符。

## 模型

默认推理直接读取 `weights/` 下约 10.6 MiB 的五个 PyTorch checkpoint：

| 模型 | 作用 |
| --- | --- |
| `score_event_locator.pt` | 在五线谱小节中定位时间事件的横坐标 |
| `rhythm_context_cnn.pt` | 识别音符/休止、时值、附点、连音组等上下文 |
| `tab_symbol_detector.pt` | 定位并分类 TAB 数字和 X |
| `atomic_symbol_cnn.pt` | 识别拍号数字等原子印刷符号 |
| `tie_context_cnn.pt` | 判断延音候选及其音高关系 |

文件大小和 SHA-256 见 [weights/README.md](weights/README.md)。模型不是大语言模型，CPU 可以运行；CUDA 主要改善速度。

## 用新数据重新训练

### 1. 准备源文件

把合法的 `.gp3/.gp4/.gp5/.gtp/.gpx` 放在任意目录中。脚本会递归读取三层、用 TuxGuitar 渲染 `tab_only/score_tab/score_only` 三种布局，并按源曲目隔离为 train/validation/test，避免同一首曲子跨集合泄漏。

不要把来源不明或无权分发的曲谱提交到 GitHub。

### 2. 构建基础数据库和坐标真值

```powershell
.\.venv\Scripts\Activate.ps1

.\scripts\build_database.ps1 `
  -CorpusRoot D:\my_gp_corpus `
  -DatabaseRoot D:\guitarocr_database `
  -PerFormat 100

$env:GUITAROCR_DATABASE_ROOT = 'D:\guitarocr_database'
.\scripts\build_tuxguitar_page_annotations.ps1
.\scripts\build_score_rhythm_dataset.ps1
```

坐标真值来自 TuxGuitar 自身的排版/绘制过程，只用于生成训练标签；正式推理只读取 PDF 或图片，不会得到这些坐标。

### 3. 训练与完整评估

```powershell
.\scripts\run_symbol_cnn.ps1
.\scripts\run_tab_detector.ps1
.\scripts\run_rhythm_context.ps1 -SkipDatasetBuild
.\scripts\run_score_event_locator.ps1
.\scripts\run_tie_context.ps1
```

训练产物写入数据库各任务的 `models/`，评估程序会分别检查符号、TAB、事件定位、节奏、拍号、小节约束和延音。训练建议使用 NVIDIA GPU；CPU 训练会很慢。

只有在新模型通过 validation/test 和整页重渲染对照后，才替换仓库默认权重：

```powershell
.\scripts\promote_models.ps1 -DatabaseRoot D:\guitarocr_database
guitarocr-check
```

## 在少量新样本上微调

定位、节奏、TAB、原子符号和延音训练器都支持从现有 checkpoint 初始化。先按上一节生成新域的数据和 source-disjoint 划分，再使用较小学习率，例如：

```powershell
python -m guitarocr.training.train_score_event_locator `
  --database D:\guitarocr_database --epochs 8 --batch-size 16 --learning-rate 0.0002 `
  --init-checkpoint .\weights\score_event_locator.pt

python -m guitarocr.training.train_rhythm_context `
  --database D:\guitarocr_database --epochs 8 --learning-rate 0.0002 `
  --init-checkpoint .\weights\rhythm_context_cnn.pt

python -m guitarocr.training.train_tab_detector `
  --database D:\guitarocr_database --epochs 8 --learning-rate 0.0002 `
  --init-checkpoint .\weights\tab_symbol_detector.pt

python -m guitarocr.training.train_tie_context `
  --database D:\guitarocr_database --epochs 8 --learning-rate 0.0001 `
  --init-checkpoint .\weights\tie_context_cnn.pt

python -m guitarocr.training.train_symbol_cnn `
  --data D:\guitarocr_database\symbol_cnn\dataset `
  --output D:\guitarocr_database\symbol_cnn\models `
  --epochs 8 --learning-rate 0.0002 `
  --init-checkpoint .\weights\atomic_symbol_cnn.pt
```

注意：

- TAB/原子符号 checkpoint 的类别集合必须与新数据一致；新增类别需要重新定义模型输出并重训；
- 微调集不能只包含一种简单谱面，应保留不同拍号、时值、和弦、休止、附点、连音和多位数品位；
- 测试集必须按“源曲目”隔离，不能把同一 PDF 的不同页拆到训练和测试；
- 适配 Guitar Pro/MuseScore 等新软件时，优先加入该软件真实导出 PDF，并保留一套完全未参与训练的整页测试集；
- 不要只看裁块分类准确率，最终以整页事件匹配、小节时值正确率和 GP5 重渲染结果为准。

## 代码结构

```text
guitarocr/
├─ models/       # 轻量 CNN 定义
├─ pipeline/     # 页面解析、事件识别、音乐约束和 IR 合并
├─ data/         # 数据集构建
├─ training/     # 训练与微调
├─ evaluation/   # 数据/模型/整页评估
├─ export/       # Score IR→GP5、GP→PDF
└─ cli/          # 命令行入口和环境检查
scripts/         # Windows 安装、数据构建、训练和模型发布脚本
java/            # TuxGuitar 标注、写入、回读和渲染桥接
weights/         # 随仓库发布的五个推理 checkpoint
docs/            # 架构、IR 协议和历史实验记录
database/        # 本地生成，不提交
```

进一步阅读：

- [识别与重建架构](docs/RECOGNITION_ARCHITECTURE.md)
- [Score Event IR 协议](docs/EVENT_IR_SCHEMA.md)
- [已有模型与实验记录](docs/EXPERIMENT_RESULTS.md)
