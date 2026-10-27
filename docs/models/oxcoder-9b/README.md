# OxCoder-9B, ROCmFP4 STRIX_LEAN with its own MTP block

OxCoder-9B is a 9B dense Qwen3.5 hybrid coding model with a 262,144-token context for text and image input. These files are ROCmFP4 STRIX_LEAN quantizations for Strix Halo with the output head at Q6_K and a grafted in-file MTP head from the declared base model. The served file carries 15 grafted MTP tensors and drafts from inside the same file.

| | |
|---|---|
| Base model | [OrionLLM/OxCoder-9B](https://huggingface.co/OrionLLM/OxCoder-9B) |
| License | apache-2.0 |
| Architecture | 32-layer dense Qwen3.5 hybrid, 9,409,813,744 parameters, 262,144-token context |

## Files

| File | Role | Repository |
|---|---|---|
| `OxCoder-9B-MTP-Q4_0-ROCmFP4-STRIX_LEAN.gguf` | model | [kingjones777/OxCoder-9B-MTP-ROCmFP4-GGUF](https://huggingface.co/kingjones777/OxCoder-9B-MTP-ROCmFP4-GGUF) |

## Serve

```sh
SUPERROC_MTP_BORROW_HEAD=1 superroc serve --host 127.0.0.1 --port 8080 llm -m /path/to/OxCoder-9B-MTP-Q4_0-ROCmFP4-STRIX_LEAN.gguf --served-model-name oxcoder-9b --context 65536 --think off --speculative mtp --mtp-model /path/to/OxCoder-9B-MTP-Q4_0-ROCmFP4-STRIX_LEAN.gguf -d 3 --min-draft-tokens 1
```

## Measured on Super ROC

AMD Ryzen AI Max+ 395 (Radeon 8060S, gfx1151), 128 GB, ROCm 7.2.4. One request at a time, temperature 0, cold prompt cache, drafter as in the serve command.

| Measurement | Result |
|---|---:|
| Prompt processing, 1566-token prompt | 1437.7 tok/s |
| Generation after that prompt (up to 256 tokens) | 51.8 tok/s |
| Code generation, 6 prompts, up to 256 tokens each, mean (range) | 76.4 tok/s (67.9-82.5) |
| Drafted tokens accepted on those prompts | 1029/1464 |
| Tool calls correct, tool_choice required / auto | 7/7 / 7/7 |

## Quality

KL divergence 0.0879 ± 0.0034 from BF16 with 89.29% top-1 agreement and perplexity 6.7713 ± 0.0829 (1.0185 times BF16), measured on wikitext-2 test at 2048 context (40 chunks) against BF16 logits computed on the CPU, with llama.cpp (ROCmFPX build), as reported on the Hugging Face card.

## Notes

- Set SUPERROC_MTP_BORROW_HEAD=1: the drafter is the MTP block inside the same file, and this switch makes it use the model's token embedding and output head that are already on the GPU.
- The MTP head is grafted from the declared base model; it was trained for that model, not for OxCoder, and acceptance is measured on the benchmark prompts.
- With the head loaded, image requests fail on this build; for image input, drop the speculative flags and serve with the projector.
- The vision projector is published alongside the file; this setup attaches no projector and serves text only.
