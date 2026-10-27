# Nex-N2.5-mini, ROCmFP4 STRIX_LEAN

Nex-N2.5-mini is a 40-layer Qwen3.5 MoE model with 256 routed experts (8 active) and a 262,144-token context for text and image input. These files are ROCmFP4 quantizations for Strix Halo built with superroc-quant, with the output head kept at Q6_K. The served file uses the STRIX_LEAN tier and carries no MTP head, so it runs without speculative decoding.

| | |
|---|---|
| Base model | [nex-agi/Nex-N2.5-mini](https://huggingface.co/nex-agi/Nex-N2.5-mini) |
| License | apache-2.0 |
| Architecture | Qwen3.5 MoE, 35,107,181,936 parameters, 256 experts / 8 active |

## Files

| File | Role | Repository |
|---|---|---|
| `Nex-N2.5-mini-Q4_0-ROCmFP4-STRIX_LEAN.gguf` | model | [kingjones777/Nex-N2.5-mini-ROCmFP4-GGUF](https://huggingface.co/kingjones777/Nex-N2.5-mini-ROCmFP4-GGUF) |

## Serve

```sh
superroc serve --host 127.0.0.1 --port 8080 llm -m /path/to/Nex-N2.5-mini-Q4_0-ROCmFP4-STRIX_LEAN.gguf --served-model-name nex-n2.5-mini --context 65536 --think off
```

## Measured on Super ROC

AMD Ryzen AI Max+ 395 (Radeon 8060S, gfx1151), 128 GB, ROCm 7.2.4. One request at a time, temperature 0, cold prompt cache, drafter as in the serve command.

| Measurement | Result |
|---|---:|
| Prompt processing, 1566-token prompt | 1753.1 tok/s |
| Generation after that prompt (up to 256 tokens) | 77.1 tok/s |
| Code generation, 6 prompts, up to 256 tokens each, mean (range) | 78.0 tok/s (78.0-78.0) |
| Drafted tokens accepted on those prompts | no drafter |
| Tool calls correct, tool_choice required / auto | 7/7 / 7/7 |

## Quality

Perplexity 6.4740 ± 0.0798 (1.0393 times the BF16 reference) with KL divergence 0.1044 ± 0.0014 from BF16 and 86.66% top-1 agreement, measured on wikitext-2 test at 2048 context (40 chunks of 1,023 scored tokens each) against BF16 logits computed on the CPU, with llama.cpp (ROCmFPX build), as reported on the Hugging Face card.

## Notes

- The checkpoint declares one MTP layer but ships no MTP weights, so there is no draft head and the serve command uses no speculative flags.
- The vision projector is published alongside the file; without it the model serves text only.
- Thinking is controlled with enable_thinking; setting reasoning effort to high or medium leaves the reasoning inside the reply content instead of a separate field.
- The output head is kept at Q6_K while the body uses the STRIX_LEAN 4-bit tier.
