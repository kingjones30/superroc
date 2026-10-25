# Ornith-1.5-9B, ROCmFP4 STRIX_LEAN

The base model is described in its Hugging Face card as a 9B dense Qwen3.5 hybrid with 32 layers. These files are ROCmFP4 STRIX_LEAN quantizations for Strix Halo with Q5_K token embeddings and the output head protected at Q6_K. The checkpoint contains no MTP weights, so the file serves without speculative decoding.

| | |
|---|---|
| Base model | [ornith-ai/Ornith-1.5-9B](https://huggingface.co/ornith-ai/Ornith-1.5-9B) |
| License | apache-2.0 |
| Architecture | `qwen35` dense, 32 layers, hidden 4096, hybrid attention (24 gated-deltanet + 8 full-attention layers) |

## Files

| File | Role | Repository |
|---|---|---|
| `Ornith-1.5-9B-Q4_0_ROCMFP4_STRIX_LEAN.gguf` | model | [kingjones777/Ornith-1.5-9B-ROCmFP4-STRIX-LEAN-GGUF](https://huggingface.co/kingjones777/Ornith-1.5-9B-ROCmFP4-STRIX-LEAN-GGUF) |

## Serve

```sh
superroc serve --host 127.0.0.1 --port 8080 llm -m /path/to/Ornith-1.5-9B-Q4_0_ROCMFP4_STRIX_LEAN.gguf --served-model-name ornith-1.5-9b --context 131072 --think off
```

## Measured on Super ROC

AMD Ryzen AI Max+ 395 (Radeon 8060S, gfx1151), 128 GB, ROCm 7.2.4. One request at a time, temperature 0, cold prompt cache, drafter as in the serve command.

| Measurement | Result |
|---|---:|
| Prompt processing, 1566-token prompt | 1703.1 tok/s |
| Generation after that prompt (up to 256 tokens) | 42.3 tok/s |
| Code generation, 6 prompts, up to 256 tokens each, mean (range) | 42.7 tok/s (42.7-42.7) |
| Drafted tokens accepted on those prompts | no drafter |
| Tool calls correct, tool_choice required / auto | 7/7 / 7/7 |

## Quality

Perplexity 9.71099 on long6k.txt at 4096 context, against 10.0318 ± 0.49546 for the same file on llama.cpp (ROCmFPX build), measured on an earlier Super ROC build.

## Notes

- The file carries no MTP weights, so serve it without a drafter.
- The vision projector is published with the file; the serve command here attaches no projector and serves text only.
- Token embeddings are stored at Q5_K and the output head at Q6_K.
