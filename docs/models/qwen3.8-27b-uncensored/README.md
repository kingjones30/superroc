# Qwen3.8-27B Uncensored, ROCmFP4 STRIX_LEAN with MTP

This file is an abliterated build of Qwen3.8-27B, a dense 27B hybrid-attention model with 64 layers. Abliteration removes refusal behavior without adding capability. The card serves the ROCmFP4 STRIX_LEAN file with its Q4_0 MTP draft head.

| | |
|---|---|
| Base model | [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) |
| License | apache-2.0 |
| Architecture | `qwen35`, 27B total / 27B active (dense) |

## Files

| File | Role | Repository |
|---|---|---|
| `Qwen3.8-27B-Uncensored-ROCmFP4-STRIX_LEAN.gguf` | model | [kingjones777/Qwen3.8-27B-Uncensored-ROCmFP4-STRIX-MTP-GGUF](https://huggingface.co/kingjones777/Qwen3.8-27B-Uncensored-ROCmFP4-STRIX-MTP-GGUF) |
| `mtp-Qwen3.8-27B-Q4_0.gguf` | drafter | [kingjones777/Qwen3.8-27B-Uncensored-ROCmFP4-STRIX-MTP-GGUF](https://huggingface.co/kingjones777/Qwen3.8-27B-Uncensored-ROCmFP4-STRIX-MTP-GGUF) |

## Serve

```sh
superroc serve --host 127.0.0.1 --port 8080 llm -m /path/to/Qwen3.8-27B-Uncensored-ROCmFP4-STRIX_LEAN.gguf --served-model-name qwen3.8-27b-uncensored --context 65536 --think off --speculative mtp --mtp-model /path/to/mtp-Qwen3.8-27B-Q4_0.gguf -d 4 --min-draft-tokens 1
```

## Measured on Super ROC

AMD Ryzen AI Max+ 395 (Radeon 8060S, gfx1151), 128 GB, ROCm 7.2.4. One request at a time, temperature 0, cold prompt cache, drafter as in the serve command.

| Measurement | Result |
|---|---:|
| Prompt processing, 1566-token prompt | 509.6 tok/s |
| Generation after that prompt (up to 256 tokens) | 26.3 tok/s |
| Code generation, 6 prompts, up to 256 tokens each, mean (range) | 43.9 tok/s (39.0-50.3) |
| Drafted tokens accepted on those prompts | 1003/1394 |
| Tool calls correct, tool_choice required / auto | 7/7 / 7/7 |

## Quality

Perplexity on long6k (n_ctx 4096): 5.39667 on Super ROC against 5.4187 ± 0.20479 on llama.cpp (ROCmFPX build) of the same file.

## Notes

- The model has its refusal behavior removed (abliterated); the architecture is unchanged.
- The draft depth of 4 is the measured knee of the MTP curve; deeper drafts are slower.
- Use the Q4_0 draft head rather than higher-precision heads; it is faster end to end at equal or better acceptance.
