# Qwen3.8-27B TURBO Fable Cold Fusion Heretic, ROCmFP4 FAST imatrix with MTP

This file is a ROCmFP4 FAST build of a merged Qwen3.8-27B checkpoint, calibrated with an importance matrix. Attention, feed-forward, linear-attention, and embedding tensors are 4-bit FP4 while the output head is protected at Q6_K. It ships with the model's own Q4_0 draft head and a BF16 vision projector, and this card serves all three together.

| | |
|---|---|
| Base model | [DavidAU/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic-Uncensored-NM-DAU](https://huggingface.co/DavidAU/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic-Uncensored-NM-DAU) (from [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)) |
| License | apache-2.0 |
| Architecture | `qwen3_5`, 27B total / 27B active (dense) |

## Files

| File | Role | Repository |
|---|---|---|
| `Qwen3.8-27B-TURBO-Fable-Cold-Fusion-Heretic-Q4_0-ROCmFP4-FAST-imatrix.gguf` | model | [kingjones777/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-Heretic-ROCmFP4-FAST-imatrix-GGUF](https://huggingface.co/kingjones777/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-Heretic-ROCmFP4-FAST-imatrix-GGUF) |
| `mtp-Qwen3.8-27B-TURBO-Fable-Cold-Fusion-Heretic-Q4_0.gguf` | drafter | [kingjones777/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-Heretic-ROCmFP4-FAST-imatrix-GGUF](https://huggingface.co/kingjones777/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-Heretic-ROCmFP4-FAST-imatrix-GGUF) |
| `mmproj-Qwen3.8-27B-TURBO-Fable-Cold-Fusion-Heretic-BF16.gguf` | vision projector | [kingjones777/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-Heretic-ROCmFP4-FAST-imatrix-GGUF](https://huggingface.co/kingjones777/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-Heretic-ROCmFP4-FAST-imatrix-GGUF) |

## Serve

```sh
superroc serve --host 127.0.0.1 --port 8080 llm -m /path/to/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-Heretic-Q4_0-ROCmFP4-FAST-imatrix.gguf --served-model-name qwen3.8-27b-fable-heretic --context 32768 --think off --speculative mtp --mtp-model /path/to/mtp-Qwen3.8-27B-TURBO-Fable-Cold-Fusion-Heretic-Q4_0.gguf -d 4 --min-draft-tokens 1 --mmproj /path/to/mmproj-Qwen3.8-27B-TURBO-Fable-Cold-Fusion-Heretic-BF16.gguf
```

## Measured on Super ROC

AMD Ryzen AI Max+ 395 (Radeon 8060S, gfx1151), 128 GB, ROCm 7.2.4. One request at a time, temperature 0, cold prompt cache, drafter as in the serve command.

| Measurement | Result |
|---|---:|
| Prompt processing, 1566-token prompt | 498.8 tok/s |
| Generation after that prompt (up to 256 tokens) | 29.1 tok/s |
| Code generation, 6 prompts, up to 256 tokens each, mean (range) | 41.7 tok/s (37.3-43.3) |
| Drafted tokens accepted on those prompts | 920/1295 |
| Tool calls correct, tool_choice required / auto | 7/7 / 7/7 |

## Quality

Perplexity (n_ctx 4096, 2 chunks of 4094 scored tokens): 5.92124 on Super ROC against 5.9535 ± 0.23374 on llama.cpp (ROCmFPX build) of the same file. On held-out WikiText-2 raw (-c 512), the FAST imatrix file reads 6.6577 ± 0.042 against 6.6959 ± 0.042 for the non-imatrix control, both on llama.cpp (ROCmFPX build).

## Notes

- The model has its refusal behavior removed (abliterated); the architecture is unchanged.
- The bundled draft head is the model's own restored MTP block, which predicts its reasoning better than the base-model head.
- Vision is served through the BF16 projector file in the Files table; image input needs at least 1024 image tokens.
