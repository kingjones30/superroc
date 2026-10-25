# BTL-4, ROCmFP4 STRIX_LEAN

BTL-4 is a 40-layer mixture-of-experts vision model with 256 experts and 8 active per token. This card serves the ROCmFP4 STRIX_LEAN file as text only. The file carries no MTP tensors, so no draft head exists.

| | |
|---|---|
| Base model | [badtheorylabs/BTL-4](https://huggingface.co/badtheorylabs/BTL-4) |
| License | see the Hugging Face card |
| Architecture | `qwen35moe`, 40 layers, 256 experts with 8 active |

## Files

| File | Role | Repository |
|---|---|---|
| `BTL-4-Q4_0_ROCMFP4_STRIX_LEAN.gguf` | model | [kingjones777/BTL-4-ROCmFP4-STRIX_LEAN-GGUF](https://huggingface.co/kingjones777/BTL-4-ROCmFP4-STRIX_LEAN-GGUF) |

## Serve

```sh
superroc serve --host 127.0.0.1 --port 8080 llm -m /path/to/BTL-4-Q4_0_ROCMFP4_STRIX_LEAN.gguf --served-model-name btl-4 --context 131072 --think off
```

## Measured on Super ROC

AMD Ryzen AI Max+ 395 (Radeon 8060S, gfx1151), 128 GB, ROCm 7.2.4. One request at a time, temperature 0, cold prompt cache, drafter as in the serve command.

| Measurement | Result |
|---|---:|
| Prompt processing, 1566-token prompt | 1771.0 tok/s |
| Generation after that prompt (up to 256 tokens) | 80.6 tok/s |
| Code generation, 6 prompts, up to 256 tokens each, mean (range) | 81.4 tok/s (81.2-81.5) |
| Drafted tokens accepted on those prompts | no drafter |
| Tool calls correct, tool_choice required / auto | 7/7 / 7/7 |

## Quality

Perplexity on long6k (n_ctx 4096): 7.31706 on Super ROC against 7.3372 ± 0.31384 on llama.cpp (ROCmFPX build) of the same file.

## Notes

- The base checkpoint ships no MTP tensors, so no speculative flags apply; the drafter-off number is the number.
- Served and measured as text only; image input was not tested on Super ROC.
