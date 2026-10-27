# MiniCPM-V-4.7-35B-A3B, ROCmFP4 LEAN with Q6_K trunk

MiniCPM-V-4.7-35B-A3B is a 35B mixture-of-experts model. These files are ROCmFP4 LEAN quantizations for Strix Halo built with superroc-quant, with the trunk attention and recurrent matrices at Q6_K in the served Q6K-trunk build. This setup serves text only without a drafter.

| | |
|---|---|
| Base model | [openbmb/MiniCPM-V-4.7-35B-A3B](https://huggingface.co/openbmb/MiniCPM-V-4.7-35B-A3B) |
| License | apache-2.0 |
| Architecture | `qwen35moe`, 34.6599B parameters, 733 tensors |

## Files

| File | Role | Repository |
|---|---|---|
| `MiniCPM-V-4.7-35B-A3B-ROCmFP4-LEAN-Q6K-trunk.gguf` | model | [kingjones777/MiniCPM-V-4.7-35B-A3B-ROCmFP4-GGUF](https://huggingface.co/kingjones777/MiniCPM-V-4.7-35B-A3B-ROCmFP4-GGUF) |

## Serve

```sh
GPU_MAX_HW_QUEUES=2 superroc serve --host 127.0.0.1 --port 8080 llm -m /path/to/MiniCPM-V-4.7-35B-A3B-ROCmFP4-LEAN-Q6K-trunk.gguf --served-model-name minicpm-v-4.7-35b-a3b --context 32768 --think off
```

## Measured on Super ROC

AMD Ryzen AI Max+ 395 (Radeon 8060S, gfx1151), 128 GB, ROCm 7.13.0. One request at a time, temperature 0, cold prompt cache, drafter as in the serve command.

| Measurement | Result |
|---|---:|
| Prompt processing, 1566-token prompt | 1876.3 tok/s |
| Generation after that prompt (up to 256 tokens) | 65.2 tok/s |
| Code generation, 6 prompts, up to 256 tokens each, mean (range) | 65.7 tok/s (65.6-65.9) |
| Drafted tokens accepted on those prompts | no drafter |
| Tool calls correct, tool_choice required / auto | 7/7 / 7/7 |

## Quality

Perplexity 7.0038 for the served build against 6.9139 for stock (quality ratio 1.0130 with 95% interval [1.0091, 1.0169]), measured on wiki.test.raw at 512 context over the same 320 chunks.

## Notes

- The served build is the Q6K-trunk variant: its perplexity stays within 2 % of the stock file over the whole 95 % interval, and the plain LEAN variant's does not.
- A DSpark drafter file is in the repository. On Super ROC it drafts exactly (`--speculative dflash2`), but on average it was slower than no drafter for this model, so this setup runs without one.
- Vision was not tested on these builds; serve text only.
- The trunk protection costs 3.5-4.9% of decode against the plain LEAN build.
