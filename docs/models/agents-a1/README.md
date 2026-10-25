# Agents-A1 Abliterated, ROCmFP4 STRIX_LEAN

Agents-A1 is a 40-layer Qwen3.5-35B-A3B mixture-of-experts model. This card serves the abliterated checkpoint as a ROCmFP4 STRIX_LEAN file with a grafted MTP block. No draft head is active, so the numbers below are drafter-off numbers.

| | |
|---|---|
| Base model | [InternScience/Agents-A1](https://huggingface.co/InternScience/Agents-A1); abliterated checkpoint [huihui-ai/Huihui-Agents-A1-abliterated-GGUF](https://huggingface.co/huihui-ai/Huihui-Agents-A1-abliterated-GGUF); MTP block from [SC117/Agents-A1-Uncensored-MTP-APEX-GGUF](https://huggingface.co/SC117/Agents-A1-Uncensored-MTP-APEX-GGUF) |
| License | apache-2.0 |
| Architecture | `qwen35moe`, 35B total / 3B active |

## Files

| File | Role | Repository |
|---|---|---|
| `Agents-A1-Uncensored-MTP-Q4_0_ROCMFP4_STRIX_LEAN.gguf` | model | [kingjones777/Agents-A1-Abliterated-MTP-ROCmFP4-GGUF](https://huggingface.co/kingjones777/Agents-A1-Abliterated-MTP-ROCmFP4-GGUF) |

## Serve

```sh
superroc serve --host 127.0.0.1 --port 8080 llm -m /path/to/Agents-A1-Uncensored-MTP-Q4_0_ROCMFP4_STRIX_LEAN.gguf --served-model-name agents-a1 --context 131072 --think off
```

## Measured on Super ROC

AMD Ryzen AI Max+ 395 (Radeon 8060S, gfx1151), 128 GB, ROCm 7.2.4. One request at a time, temperature 0, cold prompt cache, drafter as in the serve command.

| Measurement | Result |
|---|---:|
| Prompt processing, 1566-token prompt | 1767.7 tok/s |
| Generation after that prompt (up to 256 tokens) | 77.0 tok/s |
| Code generation, 6 prompts, up to 256 tokens each, mean (range) | 78.0 tok/s (78.0-78.0) |
| Drafted tokens accepted on those prompts | no drafter |
| Tool calls correct, tool_choice required / auto | 7/7 / 7/7 |

## Quality

Perplexity on long6k (n_ctx 4096): 8.08751 on Super ROC against 8.1080 ± 0.36500 on llama.cpp (ROCmFPX build) of the same file.

## Notes

- The file's grafted MTP block is stored at 4 bits, and Super ROC refuses to draft with it at load, so the model is served without a drafter.
- With thinking on, the model returns an empty answer on most non-code prompts; the serve flags disable thinking.
- Super ROC serves this model as text only: its vision path takes a projector of width 5120, and this model's projector is 2048 wide.
