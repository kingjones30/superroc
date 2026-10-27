# Models

## Super ROC model cards

One card per model: the exact files (with Hugging Face links), the serve command, measured speed and tool-call results on
Strix Halo, quality numbers where they exist, and known limits.

| Model | Card |
|---|---|
| Qwen3.8-27B on Super ROC (ROCmFP4 and stock 4-bit, DFlash2 drafter) | [card](qwen3.8-27b/SUPERROC.md) |
| Qwen3.8-27B Uncensored, ROCmFP4 STRIX_LEAN with MTP | [card](qwen3.8-27b-uncensored/README.md) |
| Qwen3.8-27B TURBO Fable Cold Fusion Heretic, ROCmFP4 FAST imatrix with MTP | [card](qwen3.8-27b-fable-heretic/README.md) |
| Qwen3.8-Flash-Next on Super ROC (Uncensored ROCmFP4 and stock 4-bit, MTP drafter) | [card](qwen3.8-flash-next/SUPERROC.md) |
| KAT-Coder-V2.5-Dev Abliterated, ROCmFP4 STRIX_LEAN with an 8-bit MTP head | [card](kat-coder-v2.5/README.md) |
| Agents-A1 Abliterated, ROCmFP4 STRIX_LEAN | [card](agents-a1/README.md) |
| Agnes-3.0-Flash-Preview, ROCmFP4 STRIX_LEAN with a Q8_0 MTP sidecar | [card](agnes-3.0-flash/README.md) |
| BTL-4, ROCmFP4 STRIX_LEAN | [card](btl-4/README.md) |
| Nex-N2.5-mini, ROCmFP4 STRIX_LEAN | [card](nex-n2.5-mini/README.md) |
| Ornith-1.5-9B, ROCmFP4 STRIX_LEAN | [card](ornith-1.5-9b/README.md) |
| Ornith-1.5-35B-A3B, ROCmFP4 LEAN v1 with the embedded MTP block | [card](ornith-1.5-35b/README.md) |
| OxCoder-9B, ROCmFP4 STRIX_LEAN with its own MTP block | [card](oxcoder-9b/README.md) |
| MiMo-V2.6-Distill-Qwen-9B, ROCmFP4 STRIX_LEAN | [card](mimo-v2.6-9b/README.md) |
| Ling-3.0-flash-VL, ROCmFP4 STRIX_LEAN with a BF16 vision projector | [card](ling-3.0-flash-vl/README.md) |
| MiniCPM-V-4.7-35B-A3B, ROCmFP4 LEAN with Q6_K trunk | [card](minicpm-v-4.7-35b-a3b/README.md) |
| LFM2.5 (24B-A2B, 8B-A1B, 2.6B, 1.2B-Instruct), ROCmFP4 LEAN | [card](lfm2.5/README.md) |
| Flint 40B-A2.5B, ROCmFP4 COHERENT-PAS | [card](flint/README.md) |
| Nanbeige4.3-33B-Base, Q4_K_M (`llama`, serve-only) | [card](llama/README.md) |

## Models from Gufo v0.19.0

The guides below are Gufo's, with Gufo's measurements.

Linux x86-64 / AMD Strix Halo gfx1151 is the production target. Build with
`nix build`; Python/reference toolchains are development-only. Model weights
are downloaded separately and are not part of the runtime package.

| Model | Weights / supported inputs | Guide |
| --- | --- | --- |
| DeepSeek V4 Flash | Flash 0731 mixed IQ2/Q2/Q8 GGUF; text | [Usage and modes](deepseek-v4-flash/README.md) |
| Qwen3.8 27B | Q4/Q8 GGUF; text/images | [Usage and modes](qwen3.8-27b/README.md) |
| Qwen3.8 Flash-Next | Sharded Q4 GGUF; text/images | [Usage and modes](qwen3.8-flash-next/README.md) |
| Qwen3-ASR 1.7B | BF16 safetensors; audio to text | [Usage](qwen3-asr/README.md) |
| Qwen3-TTS 12Hz 1.7B | BF16 safetensors; text/reference audio to speech | [Voice modes](qwen3-tts/README.md) |
| Qwen-Image-2.1 | BF16 safetensors; generation and image editing | [Usage](qwen-image-2.1/README.md) |
| MiniMax H3 FL2VA | Pinned safetensors; text to audiovisual output | [Presets and usage](minimax-h3/README.md) |

Each model folder contains:

- `README.md`: model card, acquisition, usage and modes.
- `BENCHMARKS.md`: current retained measurements, scope/date and TODO cells.
- `QUALITY.md`: measured quality, reference/scope, replay guarantees and unresolved gaps.
- `EXPERIMENTS.md`: short retained/rejected decisions.
- `artifacts/`: only useful machine-readable results/calibration, when present.

Keep executable fixtures with their tests. New logit/trace dumps and local
profiles belong in ignored top-level `artifacts/`; commit only independent
reference evidence needed by a maintained check. Old experiment records remain
in Git history. See [benchmark methodology](../BENCHMARKS.md).

Quality reports distinguish agreement with an independent model implementation
from consistency between SuperROC execution modes. Greedy speculative decoding
should reproduce AR; sampled speculation can use different random draws.
Seeded replay is checked within the same execution configuration.
For numerical comparisons, lower KL divergence, total variation, relative L2
and word-error rate are better; higher cosine similarity and image PSNR are
better. These bounded checks are not broad capability or perceptual scores.
