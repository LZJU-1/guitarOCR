# GuitarOCR GLM-OCR measure-sequence v2 LoRA

This directory contains the final step-27403 PEFT adapter used by
`guitarocr.pipeline.infer_glm_ocr_document`.

The GLM-OCR base model is intentionally not included. Download
`zai-org/GLM-OCR` separately and pass it with `--model`, or place it under
`tools/models/GLM-OCR`.

## Files

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `adapter_model.safetensors` | 31,517,928 | `CD1F3F230C1C9B84AF20E37A2E49C559109B599C4254C20DDA4F37AC44BBA712` |
| `adapter_config.json` | 1,158 | `36FA8FA03C17782217EE42BBF416777E8A48D3107BF6C6656863EFE1D22AE453` |

Git LFS stores the safetensors payload. After cloning, run `git lfs pull`.

## Parameter layout

This is one combined adapter, not two separately loaded LoRA checkpoints:

| Scope | Tensors | Parameters |
| --- | ---: | ---: |
| GLM-OCR visual tower | 240 | 4,128,768 |
| GLM-OCR language Transformer | 192 | 3,735,552 |
| Total | 432 | 7,864,320 |

LoRA rank is 8, alpha is 16, and dropout during training was 0.05.

## Evaluation

`metrics_summary.json` records the final 3,000-sample, source-disjoint v2
evaluation. The strict release gate did not pass; this adapter is an
experimental reproducible baseline, not a claim of perfect transcription.
