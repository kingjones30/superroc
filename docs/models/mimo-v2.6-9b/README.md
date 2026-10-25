# MiMo-V2.6-Distill-Qwen-9B, ROCmFP4 STRIX_LEAN

MiMo-V2.6-Distill-Qwen-9B is a 9B dense Qwen3.5 model. These files are ROCmFP4 STRIX_LEAN quantizations for Strix Halo with the output head at Q6_K, built without importance weighting. The checkpoint ships no MTP head, so the file serves without speculative decoding.

| | |
|---|---|
| Base model | [XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B) |
| License | mit |
| Architecture | `qwen35`, 8.9538B parameters |

## Files

| File | Role | Repository |
|---|---|---|
| `MiMo-V2.6-Distill-Qwen-9B-Q4_0-ROCmFP4-STRIX_LEAN.gguf` | model | [kingjones777/MiMo-V2.6-Distill-Qwen-9B-ROCmFP4-GGUF](https://huggingface.co/kingjones777/MiMo-V2.6-Distill-Qwen-9B-ROCmFP4-GGUF) |

## Serve

```sh
superroc serve --host 127.0.0.1 --port 8080 llm -m /path/to/MiMo-V2.6-Distill-Qwen-9B-Q4_0-ROCmFP4-STRIX_LEAN.gguf --served-model-name mimo-v2.6-9b --context 32768 --think off
```

## Measured on Super ROC

AMD Ryzen AI Max+ 395 (Radeon 8060S, gfx1151), 128 GB, ROCm 7.2.4. One request at a time, temperature 0, cold prompt cache, drafter as in the serve command.

| Measurement | Result |
|---|---:|
| Prompt processing, 1566-token prompt | 1625.4 tok/s |
| Generation after that prompt (up to 256 tokens) | 42.2 tok/s |
| Code generation, 6 prompts, up to 256 tokens each, mean (range) | 42.5 tok/s (42.5-42.6) |
| Drafted tokens accepted on those prompts | no drafter |
| Tool calls correct, tool_choice required / auto | 6/7 / 5/7 |

## Quality

Perplexity 8.2378 ± 0.079 on wikitext-2 (40 chunks, 4096 context), against 8.4921 ± 0.084 for the BF16 source, with llama.cpp (ROCmFPX build), as reported on the Hugging Face card.

## Notes

- The checkpoint declares one MTP layer but ships no MTP weights, so no speculative mode helps; serve without a drafter.
- On Super ROC the model serves text only; the width-4096 vision projector is refused.
- The STRIX_LEAN tier keeps the output head at Q6_K; the FAST tier without that protection is faster but measures worse on perplexity.
