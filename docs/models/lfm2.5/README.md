# LFM2.5 (24B-A2B, 8B-A1B, 2.6B, 1.2B-Instruct), ROCmFP4 LEAN

LFM2.5 is a family of compact models covering 24B-A2B and 8B-A1B mixture-of-experts models plus 2.6B and 1.2B-Instruct dense models. These files are ROCmFP4 LEAN quantizations for Strix Halo with the tied embedding kept at higher precision. The 24B-A2B file is built from the vendor's quantization-aware 4-bit file rather than from BF16. None of the four ships a draft head, so all setups run without speculative decoding.

| | |
|---|---|
| Base model | [LiquidAI/LFM2.5-24B-A2B](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B), [LiquidAI/LFM2.5-8B-A1B](https://huggingface.co/LiquidAI/LFM2.5-8B-A1B), [LiquidAI/LFM2.5-2.6B](https://huggingface.co/LiquidAI/LFM2.5-2.6B), [LiquidAI/LFM2.5-1.2B-Instruct](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct) |
| License | other (LFM Open License, see each Hugging Face card) |
| Architecture | `lfm2moe` (24B-A2B: 40 layers, 64 experts top-4; 8B-A1B: 24 layers, 32 experts top-4) and `lfm2` (2.6B: 30 layers; 1.2B-Instruct: 16 layers) |

## Files

| File | Role | Repository |
|---|---|---|
| `LFM2.5-24B-A2B-Q4_0_ROCMFP4_LEAN.gguf` | model | [kingjones777/LFM2.5-24B-A2B-ROCmFP4-GGUF](https://huggingface.co/kingjones777/LFM2.5-24B-A2B-ROCmFP4-GGUF) |
| `LFM2.5-8B-A1B-Q4_0_ROCMFP4_LEAN.gguf` | model | [kingjones777/LFM2.5-8B-A1B-ROCmFP4-LEAN-GGUF](https://huggingface.co/kingjones777/LFM2.5-8B-A1B-ROCmFP4-LEAN-GGUF) |
| `LFM2.5-2.6B-Q4_0_ROCMFP4_LEAN.gguf` | model | [kingjones777/LFM2.5-2.6B-ROCmFP4-LEAN-GGUF](https://huggingface.co/kingjones777/LFM2.5-2.6B-ROCmFP4-LEAN-GGUF) |
| `LFM2.5-1.2B-Instruct-Q4_0_ROCMFP4_LEAN.gguf` | model | [kingjones777/LFM2.5-1.2B-Instruct-ROCmFP4-GGUF](https://huggingface.co/kingjones777/LFM2.5-1.2B-Instruct-ROCmFP4-GGUF) |

## Serve

### LFM2.5-24B-A2B (built from the vendor's QAD 4-bit file)

```sh
GPU_MAX_HW_QUEUES=2 superroc serve --host 127.0.0.1 --port 8080 llm -m /path/to/LFM2.5-24B-A2B-Q4_0_ROCMFP4_LEAN.gguf --served-model-name lfm2.5 --context 32768
```

### LFM2.5-8B-A1B

```sh
GPU_MAX_HW_QUEUES=2 superroc serve --host 127.0.0.1 --port 8080 llm -m /path/to/LFM2.5-8B-A1B-Q4_0_ROCMFP4_LEAN.gguf --served-model-name lfm2.5 --context 32768
```

### LFM2.5-2.6B

```sh
GPU_MAX_HW_QUEUES=2 superroc serve --host 127.0.0.1 --port 8080 llm -m /path/to/LFM2.5-2.6B-Q4_0_ROCMFP4_LEAN.gguf --served-model-name lfm2.5 --context 32768
```

### LFM2.5-1.2B-Instruct

```sh
GPU_MAX_HW_QUEUES=2 superroc serve --host 127.0.0.1 --port 8080 llm -m /path/to/LFM2.5-1.2B-Instruct-Q4_0_ROCMFP4_LEAN.gguf --served-model-name lfm2.5 --context 32768
```

## Measured on Super ROC

AMD Ryzen AI Max+ 395 (Radeon 8060S, gfx1151), 128 GB, ROCm 7.13.0. One request at a time, temperature 0, cold prompt cache, drafter as in the serve command.

### LFM2.5-24B-A2B (built from the vendor's QAD 4-bit file)

| Measurement | Result |
|---|---:|
| Prompt processing, 1518-token prompt | 2459.2 tok/s |
| Generation after that prompt (up to 256 tokens) | 97.8 tok/s |
| Code generation, 6 prompts, up to 256 tokens each, mean (range) | 101.8 tok/s (101.7-101.8) |
| Drafted tokens accepted on those prompts | no drafter |
| Tool calls correct, tool_choice required / auto | 7/7 / 6/7 |

### LFM2.5-8B-A1B

| Measurement | Result |
|---|---:|
| Prompt processing, 1517-token prompt | 4037.2 tok/s |
| Generation after that prompt (up to 256 tokens) | 153.4 tok/s |
| Code generation, 6 prompts, up to 256 tokens each, mean (range) | 158.5 tok/s (158.4-158.6) |
| Drafted tokens accepted on those prompts | no drafter |
| Tool calls correct, tool_choice required / auto | 7/7 / 7/7 |

### LFM2.5-2.6B

| Measurement | Result |
|---|---:|
| Prompt processing, 1518-token prompt | 3317.9 tok/s |
| Generation after that prompt (up to 256 tokens) | 104.9 tok/s |
| Code generation, 6 prompts, up to 256 tokens each, mean (range) | 108.6 tok/s (108.6-108.6) |
| Drafted tokens accepted on those prompts | no drafter |
| Tool calls correct, tool_choice required / auto | 7/7 / 7/7 |

### LFM2.5-1.2B-Instruct

| Measurement | Result |
|---|---:|
| Prompt processing, 1585-token prompt | 6992.6 tok/s |
| Generation after that prompt (up to 256 tokens) | 210.7 tok/s |
| Code generation, 6 prompts, up to 256 tokens each, mean (range) | 223.3 tok/s (222.6-223.9) |
| Drafted tokens accepted on those prompts | no drafter |
| Tool calls correct, tool_choice required / auto | 7/7 / 7/7 |

## Quality

Perplexity 14.2330 for the served 24B-A2B build against 15.3536 for the stock Q4_K_M file (ratio 0.927), with llama.cpp (ROCmFPX build), measured on wiki.test at 512 context over 320 chunks. On a 40-chunk sample the same file reads 14.5655 on the decode path and 14.5154 on batched prefill against 14.5560 for llama.cpp (ROCmFPX build) on the same file and text.

The 2.6B file reads 58.0560 against 58.0778 for llama.cpp (ROCmFPX build) on wiki at 512 context over 40 chunks.

No perplexity is published for the 8B-A1B and 1.2B-Instruct builds; quality evidence in the sources is limited to coherence checks and tensor audits.

## Notes

- None of the four ships a draft head; DSpark drafters exist but run about half as fast as drafter-off, so all setups stay without a drafter.
- The prompt cache works; thinking is returned as a separate reasoning field.
- The 24B-A2B build keeps the tied embedding at Q6_K with experts and attention at 4-bit.
