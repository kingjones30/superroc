# Flint 40B-A2.5B, ROCmFP4 COHERENT-PAS

Flint is a 40B mixture-of-experts coding-agent model (about 2.5B active) with 256 experts and a sliding-window attention stack. These files are ROCmFP4 COHERENT-PAS quantizations for Strix Halo with embedding, output, all attention, layers 0-1 and shared experts at 8-bit. The numbers below were measured with the code in this repository.

| | |
|---|---|
| Base model | [RecursiveSI/flint](https://huggingface.co/RecursiveSI/flint) |
| License | apache-2.0 |
| Architecture | `flint`, 39.8B total / 2.5B active, 49 layers (14 full-attention without positional encoding, 35 sliding-window of 513 tokens) |

## Files

| File | Role | Repository |
|---|---|---|
| `Flint-40B-A2.5B-Q4_0-ROCmFP4-COHERENT-PAS.gguf` | model | [kingjones777/Flint-40B-A2.5B-ROCmFP4-GGUF](https://huggingface.co/kingjones777/Flint-40B-A2.5B-ROCmFP4-GGUF) |

## Serve

```sh
superroc serve --host 127.0.0.1 --port 8080 llm -m /path/to/Flint-40B-A2.5B-Q4_0-ROCmFP4-COHERENT-PAS.gguf --served-model-name flint --context 32768
```

## Measured on Super ROC

AMD Ryzen AI Max+ 395 (Radeon 8060S, gfx1151), 128 GB, ROCm 7.2.4. One request at a time, temperature 0, cold prompt cache, drafter as in the serve command.

| prompt tokens | prompt processing | generation (128 tokens after the prompt) |
|---:|---:|---:|
| 512 | 1119 tok/s | 58.9 tok/s |
| 4096 | 1734 tok/s | 52.7 tok/s |
| 16384 | 1539 tok/s | 41.6 tok/s |

## Quality

| check | this file | BF16 reference |
|---|---:|---:|
| teacher-forced agreement, Python continuation (48 tokens) | 46/48 (95.8 %) | — |
| teacher-forced agreement, chat answer with reasoning (160 tokens) | 150/160 (93.8 %) | — |
| perplexity, Python code (3 × 2048-token windows, 2nd half scored) | 6.57 (token by token) / 6.59 (batched prompt) | 7.48 |
| perplexity, long mixed technical text (2 × 4096-token windows) | 44.97 | 47.04 |

## Notes

- The chat template is embedded; stop on all four stop tokens (end-of-text, user, observation, developer) because the single end-of-text id alone is not enough.
- Constrain tool calls with a grammar limited to the request's tool names and argument keys; unconstrained calls can invent argument names.
- Sampling at temperature 1.0, top-p 0.95, top-k 20 is recommended; set a sensible output limit because a single turn can run to the 8192-token cap.
- The model opens its own reasoning block when it decides to reason.
