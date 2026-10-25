# Agnes-3.0-Flash-Preview, ROCmFP4 STRIX_LEAN with a Q8_0 MTP sidecar

Agnes-3.0-Flash Preview is a 33B dense hybrid model with 72 layers and a 262,144-token context, folded exactly into a standard qwen35 MLP. This card serves the ROCmFP4 STRIX_LEAN file with its language-model head at Q6_K, drafted by a separate Q8_0 MTP head.

| | |
|---|---|
| Base model | [Agnes-AI/Agnes-3.0-Flash](https://huggingface.co/Agnes-AI/Agnes-3.0-Flash) |
| License | apache-2.0 |
| Architecture | `qwen35`, 33B total / 33B active (dense) |

## Files

| File | Role | Repository |
|---|---|---|
| `Agnes-3.0-Flash-Preview-MTP-Q4_0-ROCmFP4-STRIX_LEAN.gguf` | model | [kingjones777/Agnes-3.0-Flash-Preview-MTP-ROCmFP4-GGUF](https://huggingface.co/kingjones777/Agnes-3.0-Flash-Preview-MTP-ROCmFP4-GGUF) |
| `mtp-Agnes-3.0-Flash-Preview-Q8_0.gguf` | drafter | [kingjones777/Agnes-3.0-Flash-Preview-MTP-ROCmFP4-GGUF](https://huggingface.co/kingjones777/Agnes-3.0-Flash-Preview-MTP-ROCmFP4-GGUF) |

## Serve

```sh
superroc serve --host 127.0.0.1 --port 8080 llm -m /path/to/Agnes-3.0-Flash-Preview-MTP-Q4_0-ROCmFP4-STRIX_LEAN.gguf --served-model-name agnes-3.0-flash --context 65536 --think off --speculative mtp --mtp-model /path/to/mtp-Agnes-3.0-Flash-Preview-Q8_0.gguf -d 4 --min-draft-tokens 1
```

## Measured on Super ROC

AMD Ryzen AI Max+ 395 (Radeon 8060S, gfx1151), 128 GB, ROCm 7.2.4. One request at a time, temperature 0, cold prompt cache, drafter as in the serve command.

| Measurement | Result |
|---|---:|
| Prompt processing, 1566-token prompt | 423.0 tok/s |
| Generation after that prompt (up to 256 tokens) | 21.5 tok/s |
| Code generation, 6 prompts, up to 256 tokens each, mean (range) | 29.2 tok/s (23.4-32.0) |
| Drafted tokens accepted on those prompts | 1086/1514 |
| Tool calls correct, tool_choice required / auto | 7/7 / 7/7 |

## Quality

Perplexity on long6k (4096-token context): 6.06492 on Super ROC against 6.0832 ± 0.24495 on llama.cpp (ROCmFPX build) of the same file. The Hugging Face card grades the STRIX_LEAN tier against BF16 logits on held-out wikitext-2 test (-c 2048, 40 chunks): KLD 0.0438 ± 0.0010, same top-1 91.29 %, perplexity 6.4877 ± 0.0792 (1.0250 times BF16), all on llama.cpp (ROCmFPX build).

## Notes

- The draft head is a separate Q8_0 file served at depth 4; the model file alone does not draft.
- With the draft head loaded, image requests fail; for image input, serve without the speculative flags.
- The chat template accepts only xhigh, medium, and low reasoning efforts; there is no high value.
