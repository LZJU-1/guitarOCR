# Guitar Pro PDF → M2 → GP5 端到端管线

本页只描述 `agent/guitar-pro-end-to-end` 分支中的 GLM-OCR v2 管线。它面向
Guitar Pro 8 风格的规则矢量 PDF 或清晰整页图，不包含手写谱、拍照透视和多轨总谱。

## 实际架构与权重

```text
PDF / PNG
  → 无权重的页面、谱表和小节几何切分
  → GLM-OCR 0.9B 基座
  → 一个合并的 GuitarOCR v2 PEFT LoRA adapter
  → 逐小节 M2 事件序列
  → M2 语法与音乐约束；失败时最多三次有界自纠
  → 确定性 M2 → GP5
  → 可选：官方 Guitar Pro 8.1.2.37 回渲染 PRE.pdf
```

不是三个独立模型权重：

| 组件 | 是否有独立权重 | 说明 |
| --- | --- | --- |
| 页面/谱表/小节切分 | 否 | PDF 优先读取矢量谱线、字符框和连续小节号；图片回退到 OpenCV 几何检测 |
| GLM-OCR 基座 | 是，但仓库不上传 | 从 `zai-org/GLM-OCR` 下载到 `tools/models/GLM-OCR` |
| GuitarOCR v2 LoRA | 是，一个文件 | `adapter_model.safetensors` 同时包含视觉塔和语言 Transformer 的 LoRA |
| 旧 CNN | 有，但本入口不加载 | `weights/*.pt` 属于旧管线和低延迟备用流程 |

合并 adapter 共 7,864,320 个可训练参数：

- 视觉塔 LoRA：4,128,768；
- 语言 Transformer LoRA：3,735,552；
- rank 8，alpha 16；
- 权重大小：31,517,928 bytes；
- SHA-256：`CD1F3F230C1C9B84AF20E37A2E49C559109B599C4254C20DDA4F37AC44BBA712`。

PEFT 在加载时根据 `adapter_config.json` 把同一个 safetensors 中的两组矩阵挂到
GLM-OCR 对应视觉层和语言层，不需要分别指定“视觉 LoRA”和“语言 LoRA”。

## 1. 获取分支和 Git LFS 权重

```powershell
git clone --branch agent/guitar-pro-end-to-end `
  git@github.com:LZJU-1/guitarOCR.git
cd guitarOCR
git lfs pull
```

检查以下文件不是 Git LFS 指针文本：

```powershell
Get-Item .\weights\glm_ocr_measure_sequence_v2_lora\adapter_model.safetensors
Get-FileHash -Algorithm SHA256 `
  .\weights\glm_ocr_measure_sequence_v2_lora\adapter_model.safetensors
```

## 2. 创建推理环境

已验证组合：

- Windows 10/11；
- Python 3.11；
- PyTorch 2.7.1 + CUDA 11.8；
- torchvision 0.22.1；
- transformers 5.8.0；
- PEFT 0.18.1；
- safetensors 0.8.0。

```powershell
conda create -n guitarocr-gp python=3.11 -y
conda activate guitarocr-gp

python -m pip install `
  torch==2.7.1 torchvision==0.22.1 `
  --index-url https://download.pytorch.org/whl/cu118

python -m pip install -e ".[glm-ocr]"
```

CPU 可以启动，但逐小节生成会非常慢，正常使用建议 CUDA。

## 3. 下载 GLM-OCR 基座

基座没有提交到本仓库：

```powershell
.\scripts\download_glm_ocr_base.ps1 -Python python
```

等价于从 Hugging Face 下载 `zai-org/GLM-OCR` 到
`tools/models/GLM-OCR`。使用基座和 adapter 时还需遵守对应上游许可证。

## 4. 一条命令生成 GP5

纯 TAB PDF 建议明确传 `-Mode tab`：

```powershell
.\scripts\run_guitarpro_pdf_to_gp5.ps1 `
  -InputPath "D:\scores\song.pdf" `
  -OutputDirectory "D:\scores\song_result" `
  -Mode tab `
  -Python python `
  -Title "Song title"
```

五线谱 + TAB 使用 `-Mode both`，纯五线谱使用 `-Mode notation`。`-Mode auto`
会先做版式分类，但遇到非标准间距时，显式指定模式更稳。

输出目录包含：

| 文件/目录 | 内容 |
| --- | --- |
| `PRE.gp5` | 可编辑的 GP5 |
| `prediction.m2` | 每行一个小节的可读事件序列 |
| `manifest.json` | 版式、小节框、调弦、来源页和输出路径 |
| `recognition.jsonl` | 每次生成、自纠、token 上限和兜底诊断 |
| `measure_crops/` | 实际送入模型的小节图 |
| `overlays/` | 页面切分框，可用于排查漏切/错切 |

断点继续：

```powershell
.\scripts\run_guitarpro_pdf_to_gp5.ps1 `
  -InputPath "D:\scores\song.pdf" `
  -OutputDirectory "D:\scores\song_result" `
  -Mode tab `
  -Resume
```

`-Resume` 只复用同一输出目录中已经通过约束的 `recognition.jsonl` 行。

## 5. 用官方 Guitar Pro 8 回渲染

官方 Guitar Pro 运行时是闭源组件，不在 Git 仓库中。项目当前只验证固定的
Guitar Pro 8.1.2.37 worker。把本地 `guitar-hero-main` 放到仓库根目录后：

```powershell
.\scripts\run_guitarpro_pdf_to_gp5.ps1 `
  -InputPath "D:\scores\song.pdf" `
  -OutputDirectory "D:\scores\song_result" `
  -Mode tab `
  -RenderWithGuitarPro
```

这会额外产生：

- `PRE.pdf`：官方 GP8 回渲染；
- `PRE.layout.json`：官方渲染器返回的 native layout。

也可单独渲染：

```powershell
python -m guitarocr.export.render_gp_to_guitarpro_pdf `
  "D:\scores\song_result\PRE.gp5" `
  "D:\scores\song_result\PRE.pdf" `
  --display-mode tab
```

## 6. 直接使用 Python CLI

```powershell
python -m guitarocr.pipeline.infer_glm_ocr_document `
  "D:\scores\song.pdf" `
  --output "D:\scores\song_result" `
  --mode tab `
  --model ".\tools\models\GLM-OCR" `
  --adapter ".\weights\glm_ocr_measure_sequence_v2_lora" `
  --device cuda `
  --maximum-attempts 3 `
  --max-new-tokens 512 `
  --max-new-tokens-ceiling 2048
```

安装后也可以用 `guitarocr-gp` 代替
`python -m guitarocr.pipeline.infer_glm_ocr_document`。

## 7. 当前独立测试指标与限制

最终 step 27403 adapter 在 3,000 条按曲目来源隔离的 v2 测试小节上：

| 模式 | 样本 | 语法有效 | 核心结构整小节一致 | 节奏整小节一致 | 音符字段整小节一致 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `tab` | 1,000 | 99.9% | 84.9% | 92.0% | 87.9% |
| `both` | 1,000 | 99.6% | 77.4% | 90.9% | 80.7% |
| `notation` | 1,000 | 99.7% | 60.4% | 93.4% | 63.1% |
| 总体 | 3,000 | 99.73% | 74.23% | 92.1% | 77.23% |

`tab` 的对齐音符弦号准确率为 96.53%，品位准确率为 96.04%。这些是小节级
source-disjoint 指标，不等于整首曲完全正确率；当前严格 release gate 尚未通过。

目前较明显的限制：

- 和弦名可以恢复，但和弦图的具体按法点位尚未可靠恢复；
- 中文段落名、标题等自由文本可能乱码或误识别；
- 跨系统连线、复杂导航、反复播放顺序和多轨总谱仍不完整；
- `notation` 中图上不可见的吉他指法不能唯一还原；
- 失败小节最多自纠三次，只有仍无法形成合法 M2 时才写入整小节休止兜底；
- 官方 GP8 回渲染可能重新分页，页码相同不代表系统布局逐像素一致。

## 8. 训练与新数据

数据构建、LoRA 训练、3,000 条测试和 release gate 分别见：

- `docs/GP8_MEASURE_SEQUENCE.md`
- `scripts/build_gp8_measure_sequence_dataset.ps1`
- `scripts/train_glm_ocr_measure_sequence.ps1`
- `scripts/evaluate_glm_ocr_measure_sequence.ps1`
- `configs/glm_ocr_measure_sequence_v2_lora_fp16.yaml`
- `configs/measure_sequence_release_gate.json`

训练、验证、测试必须按原始曲目 SHA 隔离；测试小节和独立验收 PDF 不得回流训练。
