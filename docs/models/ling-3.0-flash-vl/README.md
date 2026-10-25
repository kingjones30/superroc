# Ling-3.0-flash-VL, ROCmFP4 STRIX_LEAN with a BF16 vision projector

Ling-3.0-flash-VL is a 124B native multimodal mixture-of-experts model (5.5B active) with 512 experts. These files are ROCmFP4 STRIX_LEAN quantizations for Strix Halo with the output head at Q6_K and token embeddings at Q5_K, carrying a multi-token-prediction head grafted from the text sibling. The setup attaches the BF16 vision projector for image input.

| | |
|---|---|
| Base model | [inclusionAI/Ling-3.0-flash-VL](https://huggingface.co/inclusionAI/Ling-3.0-flash-VL) |
| License | mit |
| Architecture | `bailingmoe3`, 124B total / 5.5B active, 43 blocks (42 plus MTP) |

## Files

| File | Role | Repository |
|---|---|---|
| `Ling-3.0-flash-VL-MTP-Q4_0-ROCmFP4-STRIX_LEAN.gguf` | model | [kingjones777/Ling-3.0-flash-VL-MTP-ROCmFP4-GGUF](https://huggingface.co/kingjones777/Ling-3.0-flash-VL-MTP-ROCmFP4-GGUF) |
| `mmproj-Ling-3.0-flash-VL-BF16.gguf` | vision projector | [kingjones777/Ling-3.0-flash-VL-MTP-ROCmFP4-GGUF](https://huggingface.co/kingjones777/Ling-3.0-flash-VL-MTP-ROCmFP4-GGUF) |

## Serve

```sh
LING3_MLA_WMMA2=1 LING3_MLA_TILE=1 LING3_MLA_TILE_MAX=16384 LING3_MOE_V2=4 LING3_MLA_FW=1 superroc serve --host 127.0.0.1 --port 8080 llm -m /path/to/Ling-3.0-flash-VL-MTP-Q4_0-ROCmFP4-STRIX_LEAN.gguf --served-model-name ling-3.0-flash-vl --mmproj /path/to/mmproj-Ling-3.0-flash-VL-BF16.gguf --context 32768
```

## Measured on Super ROC

AMD Ryzen AI Max+ 395 (Radeon 8060S, gfx1151), 128 GB, ROCm 7.13.0. One request at a time, temperature 0, cold prompt cache, drafter as in the serve command.

| Measurement | Result |
|---|---:|
| Prompt processing, 1641-token prompt | 569.2 tok/s |
| Generation after that prompt (up to 256 tokens) | 38.4 tok/s |
| Code generation, 6 prompts, up to 256 tokens each, mean (range) | 37.4 tok/s (37.4-37.4) |
| Drafted tokens accepted on those prompts | no drafter |
| Tool calls correct, tool_choice required / auto | 6/7 / 6/7 |

## Quality

Perplexity 9.2579 ± 0.208 on wikitext-2 test (20 chunks, 2048 context), with llama.cpp (ROCmFPX build), as reported on the Hugging Face card; KL divergence against BF16 logits was not measured for this model.

## Notes

- The file carries an MTP head grafted from the text model; the measured setup is the serve command above, with no speculative flags.
- Images need the projector file alongside the selected tier.
- Thinking is on by default; with a small output limit the whole budget can stay inside the reasoning and leave the answer empty, so allow room or switch thinking off.
