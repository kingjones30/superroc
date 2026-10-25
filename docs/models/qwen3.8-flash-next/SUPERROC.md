# Qwen3.8-Flash-Next on Super ROC (Uncensored ROCmFP4 and stock 4-bit, MTP drafter)

Qwen3.8-Flash-Next is a 176.9B mixture-of-experts model. This card covers two file sets: an uncensored ROCmFP4 STRIX_LEAN build with importance-matrix weighting (`hc_attn` / `hc_ffn` down and up matrices at Q8_0, `hc` inject and SSM alpha/beta at F32, three shards) and the stock Unsloth UD-Q4_K_XL file in four shards. Both setups draft with a Q8_0 MTP head and run with thinking on at low effort.

| | |
|---|---|
| Base model | [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next); Uncensored variant [orcarouter/Qwen3.8-Flash-Next-Uncensored](https://huggingface.co/orcarouter/Qwen3.8-Flash-Next-Uncensored) |
| License | see the Hugging Face cards |
| Architecture | `qwen4exp`, 176.9B total (mixture-of-experts) |

## Files

| File | Role | Repository |
|---|---|---|
| `Qwen3.8-Flash-Next-Uncensored-Q4_0-ROCmFP4-STRIX_LEAN-imatrix-hcq8-ssmf32-00001-of-00003.gguf (+ 00002, 00003)` | model | [kingjones777/Qwen3.8-Flash-Next-Uncensored-ROCmFP4-STRIX_LEAN-imatrix-GGUF](https://huggingface.co/kingjones777/Qwen3.8-Flash-Next-Uncensored-ROCmFP4-STRIX_LEAN-imatrix-GGUF) |
| `mtp-Qwen3.8-Flash-Next-Q8_0.gguf` | drafter | [kingjones777/Qwen3.8-Flash-Next-MTP-Heads-GGUF](https://huggingface.co/kingjones777/Qwen3.8-Flash-Next-MTP-Heads-GGUF) |
| `UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf` | model | [unsloth/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) |
| `MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf` | drafter | [unsloth/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) |

## Serve

### Uncensored ROCmFP4 STRIX_LEAN imatrix, hc tensors Q8_0, SSM alpha/beta F32 (3 shards)

```sh
superroc serve --host 127.0.0.1 --port 8080 llm -m /path/to/Qwen3.8-Flash-Next-Uncensored-Q4_0-ROCmFP4-STRIX_LEAN-imatrix-hcq8-ssmf32-00001-of-00003.gguf --served-model-name qwen3.8-flash-next --context 32768 --think on --reasoning-effort low --speculative mtp --mtp-model /path/to/mtp-Qwen3.8-Flash-Next-Q8_0.gguf --draft-tokens 3
```

### Unsloth UD-Q4_K_XL (stock 4-bit GGUF, 4 shards) + shared MTP Q8_0

```sh
superroc serve --host 127.0.0.1 --port 8080 llm -m /path/to/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf --served-model-name qwen3.8-flash-next --context 32768 --think on --reasoning-effort low --speculative mtp --mtp-model /path/to/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf
```

## Measured on Super ROC

### Uncensored ROCmFP4 STRIX_LEAN imatrix, hc tensors Q8_0, SSM alpha/beta F32 (3 shards)

AMD Ryzen AI Max+ 395 (Radeon 8060S, gfx1151), 128 GB, ROCm 7.2.4. One request at a time, temperature 0, cold prompt cache, drafter as in the serve command.

| Measurement | Result |
|---|---:|
| Prompt processing, 1594-token prompt | 1261.2 tok/s |
| Generation after that prompt (up to 256 tokens) | 56.1 tok/s |
| Code generation, 6 prompts, up to 256 tokens each, mean (range) | 56.8 tok/s (53.8-60.2) |
| Drafted tokens accepted on those prompts | 1031/1342 |
| Tool calls correct, tool_choice required / auto | 7/7 / 6/7 |

### Unsloth UD-Q4_K_XL (stock 4-bit GGUF, 4 shards) + shared MTP Q8_0

AMD Ryzen AI Max+ 395 (Radeon 8060S, gfx1151), 128 GB, ROCm 7.2.4. One request at a time, temperature 0, cold prompt cache, drafter as in the serve command.

| Measurement | Result |
|---|---:|
| Prompt processing, 1594-token prompt | 1369.1 tok/s |
| Generation after that prompt (up to 256 tokens) | 47.6 tok/s |
| Code generation, 6 prompts, up to 256 tokens each, mean (range) | 59.9 tok/s (50.8-64.7) |
| Drafted tokens accepted on those prompts | 1080/1480 |
| Tool calls correct, tool_choice required / auto | 7/7 / 6/7 |

## Quality

Perplexity on a 6K-token long-text corpus (context 4096), measured with Super ROC on an earlier build: 5.97374 for the
Uncensored file with the `hc` tensors at Q8_0 and F32 and SSM alpha/beta at F32, against 6.18385 for the STRIX_LEAN imatrix
file it was built from (3.4 % lower). No quality measurement for the stock Unsloth file is recorded yet.

## Notes

- The Uncensored variant has its refusal behavior removed; the architecture is unchanged.
- `--draft-tokens 3` is the measured setting for the ROCmFP4 setup; depth 1 was slower and depth 7 no better.
- With thinking on, the model writes its reasoning before the answer; a small token budget can leave the answer empty.
- Without the vision projector the model serves text only.
