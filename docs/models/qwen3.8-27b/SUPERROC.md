# Qwen3.8-27B on Super ROC (ROCmFP4 and stock 4-bit, DFlash2 drafter)

Qwen3.8-27B is a dense 27B model with hybrid attention: 64 layers, 16 of them full attention and the rest linear-recurrent layers. This card covers two file sets for Strix Halo: a ROCmFP4 STRIX build quantized with importance-matrix weighting and output plus embedding tensors at Q6_K, and the stock Unsloth UD-Q4_K_XL file. Both setups draft with the same DFlash2 sidecar file.

| | |
|---|---|
| Base model | [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) |
| License | apache-2.0 |
| Architecture | `qwen35`, 27B total / 27B active (dense) |

## Files

| File | Role | Repository |
|---|---|---|
| `Qwen3.8-27B-Q4_0_ROCMFP4_STRIX-imatrix-headQ6K.gguf` | model | [kingjones777/Qwen3.8-27B-ROCmFP4-STRIX-MTP-GGUF](https://huggingface.co/kingjones777/Qwen3.8-27B-ROCmFP4-STRIX-MTP-GGUF) |
| `Qwen3.8-27B-DFlash2-Q4_K_M.gguf` | drafter | [z-lab/Qwen3.8-27B-DFlash2-GGUF](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2-GGUF) |
| `Qwen3.8-27B-UD-Q4_K_XL.gguf` | model | [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) |
| `Qwen3.8-27B-DFlash2-Q4_K_M.gguf` | drafter | [z-lab/Qwen3.8-27B-DFlash2-GGUF](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2-GGUF) |

## Serve

### ROCmFP4 STRIX, imatrix, output/embedding Q6_K

```sh
superroc serve --host 127.0.0.1 --port 8080 llm -m /path/to/Qwen3.8-27B-Q4_0_ROCMFP4_STRIX-imatrix-headQ6K.gguf --served-model-name qwen3.8-27b --context 32768 --think off --speculative dflash2 --dflash-model /path/to/Qwen3.8-27B-DFlash2-Q4_K_M.gguf
```

### Unsloth UD-Q4_K_XL (stock 4-bit GGUF)

```sh
GPU_MAX_HW_QUEUES=2 superroc serve --host 127.0.0.1 --port 8080 llm -m /path/to/Qwen3.8-27B-UD-Q4_K_XL.gguf --served-model-name qwen3.8-27b --context 32768 --think off --speculative dflash2 --dflash-model /path/to/Qwen3.8-27B-DFlash2-Q4_K_M.gguf
```

## Measured on Super ROC

### ROCmFP4 STRIX, imatrix, output/embedding Q6_K

AMD Ryzen AI Max+ 395 (Radeon 8060S, gfx1151), 128 GB, ROCm 7.2.4. One request at a time, temperature 0, cold prompt cache, drafter as in the serve command.

| Measurement | Result |
|---|---:|
| Prompt processing, 1566-token prompt | 513.6 tok/s |
| Generation after that prompt (up to 256 tokens) | 27.6 tok/s |
| Prompt processing, 13384-token prompt | 556.6 tok/s |
| Generation after that prompt | 21.3 tok/s |
| Prompt processing, 26821-token prompt | 534.6 tok/s |
| Generation after that prompt | 21.2 tok/s |
| Code generation, 6 prompts, up to 256 tokens each, mean (range) | 50.6 tok/s (45.4-54.9) |
| Drafted tokens accepted on those prompts | 1106/1578 |
| Tool calls correct, tool_choice required / auto | 7/7 / 7/7 |

### Unsloth UD-Q4_K_XL (stock 4-bit GGUF)

AMD Ryzen AI Max+ 395 (Radeon 8060S, gfx1151), 128 GB, ROCm 7.13.0. One request at a time, temperature 0, cold prompt cache, drafter as in the serve command.

| Measurement | Result |
|---|---:|
| Prompt processing, 1566-token prompt | 545.6 tok/s |
| Generation after that prompt (up to 256 tokens) | 27.2 tok/s |
| Prompt processing, 13384-token prompt | 595.8 tok/s |
| Generation after that prompt | 17.9 tok/s |
| Prompt processing, 26821-token prompt | 568.5 tok/s |
| Generation after that prompt | 19.2 tok/s |
| Code generation, 6 prompts, up to 256 tokens each, mean (range) | 44.3 tok/s (32.2-53.2) |
| Drafted tokens accepted on those prompts | 1132/1634 |
| Tool calls correct, tool_choice required / auto | 7/7 / 7/7 |

## Quality

Measured on the same machine with the same binaries (an earlier Super ROC build):

| File | Perplexity, 6K-token long-text corpus, context 4096 | KL divergence vs BF16, wikitext-2 (16 x 512 tokens) | Top-1 agreement vs BF16 |
|---|---:|---:|---:|
| `Qwen3.8-27B-Q4_0_ROCMFP4_STRIX-imatrix-headQ6K.gguf` | 5.2830 | 0.033983 | 91.078 % |
| `Qwen3.8-27B-UD-Q4_K_XL.gguf` | 5.1989 | 0.006348 | 95.858 % |

Perplexity is from Super ROC's perplexity tool. llama.cpp (ROCmFPX build) scores the same corpus differently (7.4119 and 7.3938
for the two files), so only values from the same tool compare with each other. The ROCmFP4 file is 1.6 % higher in perplexity
than the stock file, and the stock file is closer to BF16. The two setups above were measured on different machines (ROCm
7.2.4 and 7.13.0), so their speeds are not a direct comparison.

## Notes

- Thinking is off in the serve flags. With thinking on, the model writes its reasoning before the answer, and a small token
  budget can leave the answer empty.
- Without a vision projector (`--mmproj`) the model serves text only.
