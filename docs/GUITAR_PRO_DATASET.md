# Guitar Pro 8 数据生成

## 当前结论

项目可以用 `guitar-hero-main` 内置的 Guitar Pro 8.1.2.37，在 Windows 后台批量完成：

```text
GP3/GP4/GP5（基准）或 GP5 语料（批量数据集）
  → PyGuitarPro 设置 tab / notation / both 显示模式
  → Guitar Pro 8.1.2.37 导出矢量 PDF
  → native layout JSON
  → 页面 PNG、小节 crop、TNL 语义标签、COCO 版面标签
```

这条链路不依赖鼠标点击或 GUI 坐标。2026-07-14 的本机 v1 构建产出 381 份 PDF、381 份 native layout、1,215 页和 26,103 个小节区域；127 首曲目包含 50 首合成曲和 77 首真实曲，三种显示模式严格配对。Guitar Pro 导出为 Qt 5.15.3 生成的 A4 矢量 PDF，没有页面栅格图片。

## 运行时约束

主仓库不提交 Guitar Pro 二进制、运行时 DLL、注入 DLL或第三方曲谱。默认位置是：

```text
D:\guitarOCR\guitar-hero-main
```

也可以设置：

```powershell
$env:GUITAROCR_GP8_DATAGEN_ROOT = 'D:\path\to\guitar-hero-main'
$env:GUITAROCR_GP8_PYTHON = 'D:\path\to\guitar-hero-main\.venv\Scripts\python.exe'
```

适配器固定接受以下配对，防止硬编码内部地址误用于其他 GP8 构建：

- `GuitarPro.exe`：版本 8.1.2.37，33,815,040 bytes，SHA-256 `F9607B932DD0F0DF6D37603CB548AFEBB39CFA6DF178068C2C8B5C7E1A2F5657`；
- `gt2pdf_inject.dll`：SHA-256 `7959120FF051F46A81C3E68CE3C118F43DA35FA2E5CC654F31071F1E3B6DF639`。

该 worker 通过 DLL 注入和 Guitar Pro 内部函数工作，不是 Arobas Music 的公开 API。它会先结束所有 `GuitarPro.exe` 进程，再启动隐藏 worker；运行前保存 GUI 中未保存的工作。代码静态检查没有发现上传逻辑，但预编译 DLL 是否与源码完全一致无法仅靠静态检查证明。使用者需要自行确认软件许可证、数据权利和本地安全策略，不要关闭杀毒软件来强行运行。

## 轻量环境

数据生成不需要 Paddle、LLM 或训练依赖。若外部项目没有 `.venv`，可以在其目录创建隔离环境并只安装四项数据依赖：

```powershell
cd D:\guitarOCR\guitar-hero-main
python -m uv venv .venv --python 3.12
python -m uv pip install --python .venv\Scripts\python.exe `
  'PyGuitarPro>=0.7' 'PyMuPDF>=1.24' 'Pillow>=10' 'fonttools>=4.63'
```

不要直接执行该外部项目的默认训练依赖组；它会解析 Paddle、CUDA、LLaMA-Factory 等本任务不需要的大依赖。

## 单文件导出

```powershell
python -m guitarocr.export.render_gp_to_guitarpro_pdf `
  D:\scores\song.gp5 `
  D:\scores\song_gp8.pdf `
  --layout-json D:\scores\song_gp8.layout.json
```

输出 PDF 与 layout JSON 同时成功才算完成。layout 的 `bar` 记录含页码、小节索引、谱表类型和毫米坐标；`staff_type=1` 是五线谱，`staff_type=2` 是 TAB。一次 44 小节的实测得到 88 条 bar 记录，完整覆盖索引 0–43。

## 三版式批量数据

主仓库的固定配置为 [configs/guitarpro8_multimode_v1.json](../configs/guitarpro8_multimode_v1.json)，入口是：

```powershell
.\scripts\build_guitarpro_multimode_dataset.ps1 `
  -RealDatasetDir D:\guitarOCR\database\v2\source\gp `
  -WorkDir D:\guitarOCR\database\guitarpro8_multimode_v1
```

主要输出：

```text
guitarpro8_multimode_v1/
├─ tab/                    # 纯 TAB 的 GP5/PDF/layout/标注
├─ notation/               # 纯五线谱
├─ both/                   # 五线谱 + TAB
├─ layout_coco/            # 外部工具原始合并结果，不用于指标
└─ layout_coco_source_disjoint/
   ├─ images/
   ├─ images_mask/
   ├─ annotations/instance_train.json
   ├─ annotations/instance_val.json
   └─ source_disjoint_summary.json
```

v1 的来源隔离划分为 114 首训练曲和 13 首验证曲，交集为 0。验证集按来源族分层：8 首真实曲，加 `balanced / lead / strum / hard / techspan` 各 1 首合成曲。三种版式、同一曲目的全部页面必须跟随同一划分。

80 个真实 GP5 候选中导入了 77 个：两首只有 4/5 小节，被 `min_measures=8` 过滤；一首包含 PyGuitarPro/TNL 尚未支持的 `thumb` note effect，记录在 `real_import_failures.json`。Guitar Pro 导出阶段本身为 381/381 成功。

包含来源隔离 COCO 副本后的完整 v1 本地目录约 1.94 GiB、38,458 个文件；这些都是生成物，不进入 Git。

## 标签能做什么

当前自动标签包括：

- `header_text`、`global_tempo`、`tuning`、`chord_diagram_block` 与 `measure` 版面框；
- 每个小节的 `tab / notation / both` crop；
- 对应的 TNL 小节语义，含节奏、弦/品、和弦、休止、拍号和已覆盖技法；
- 版面阅读顺序和 COCO 检测标签。

它适合训练小节检测器或“整小节图像 → TNL 序列”的小模型。native layout 目前没有给每个数字、音符头、符杆、休止符和技法字形附上语义类别，因此不能直接作为现有单符号 CNN 的逐框监督。要复用当前 CNN 路线，还需在小节框内完成源事件与细粒度图元的对齐，或先用现有检测器生成候选框再进行 Guitar Pro 域微调。

## 自动跨渲染器基准

`build_cross_renderer_benchmark` 已接入同一 worker。它用 PyGuitarPro 把单轨 GP3/GP4/GP5 写成目标 GP5 显示模式，再导出三种 PDF：

```powershell
.\scripts\run_cross_renderer_benchmark.ps1 `
  -SourceCount 12 `
  -Renderers guitarpro `
  -Layouts score_only,tab_only,score_tab `
  -OverwriteRenderings
```

基准样本只从既有 source-disjoint `test` 集选择，并标记 `training_eligible=false`。不要把基准 PDF 加回训练集。

2026-07-14 的 3 首曲目、零 GP8 微调基线为：

- `tab_only`：3/3 完成推理，但真值 122 个小节被预测为 188 个；事件 precision 67.306%、recall 73.636%，核心事件 exact 4.105%，节奏 exact 55.074%，音符 exact 7.526%；
- `score_tab`：0/3 完成，均在第一页的五线谱/TAB 系统配对阶段失败。

这组结果证明自动导出和标注已经可用，但现有 TuxGuitar 域的页面几何与 CNN 权重尚未适配 GP8，不能把“数据生成成功”当作“端到端支持”。
