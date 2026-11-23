# KAT-Coder-V2.5-Dev Abliterated, ROCmFP4 STRIX_LEAN with an 8-bit MTP head

KAT-Coder-V2.5-Dev is a 35B-A3B mixture-of-experts coding model. This card serves the abliterated checkpoint as a single ROCmFP4 STRIX_LEAN file with its MTP block embedded and re-quantized to 8-bit precision. No separate draft file is needed.

| | |
|---|---|
| Base model | [Kwaipilot/KAT-Coder-V2.5-Dev](https://huggingface.co/Kwaipilot/KAT-Coder-V2.5-Dev); abliterated checkpoint [jakeroxs/KAT-Coder-V2.5-Dev-35B-A3B-MTP-ABLITERATED-GGUF](https://huggingface.co/jakeroxs/KAT-Coder-V2.5-Dev-35B-A3B-MTP-ABLITERATED-GGUF) |
| License | apache-2.0 |
| Architecture | `qwen35moe`, 35.5B total / 3B active |

## Files

| File | Role | Repository |
|---|---|---|
| `KAT-Coder-V2.5-Dev-Abliterated-MTPq8-Q4_0_ROCMFP4_STRIX_LEAN.gguf` | model with embedded MTP | [kingjones777/KAT-Coder-V2.5-Dev-Abliterated-MTP-ROCmFP4-GGUF](https://huggingface.co/kingjones777/KAT-Coder-V2.5-Dev-Abliterated-MTP-ROCmFP4-GGUF) |

## Serve

```sh
superroc serve --host 127.0.0.1 --port 8080 llm -m /path/to/KAT-Coder-V2.5-Dev-Abliterated-MTPq8-Q4_0_ROCMFP4_STRIX_LEAN.gguf --served-model-name kat-coder-v2.5 --context 131072 --think off --speculative mtp -d 3 --min-draft-tokens 1
```

## Speed setting

New in this release: six routed experts per token instead of the file's eight. Add one variable to the serve command.

```sh
SUPERROC_ROUTED_EXPERTS=6 superroc serve --host 127.0.0.1 --port 8080 llm -m /path/to/KAT-Coder-V2.5-Dev-Abliterated-MTPq8-Q4_0_ROCMFP4_STRIX_LEAN.gguf --served-model-name kat-coder-v2.5 --context 131072 --think off --speculative mtp -d 2 --min-draft-tokens 1 --prompt-lookup
```

| Measurement | 8 experts | 6 experts |
|---|---:|---:|
| Generation, 16 prose prompts, 256 tokens each | 98.5 tok/s | 105.4 tok/s (+7.0 %) |
| Prompt processing on those prompts | 2,101.4 tok/s | 2,315.3 tok/s (+10.2 %) |
| Perplexity, 4,096-token held-out text | 6.4023 | 6.5121 (+1.7 %) |

AMD Ryzen AI Max+ 395, 128 GB, ROCm 7.13.0; one build, 8 and 6 experts in alternating runs, temperature 0, the serve command
above.

## Measured on Super Roc

AMD Ryzen AI Max+ 395 (Radeon 8060S, gfx1151), 128 GB, ROCm 7.2.4. One request at a time, temperature 0, cold prompt cache, drafter as in the serve command.

| Measurement | Result |
|---|---:|
| Prompt processing, 1566-token prompt | 1713.1 tok/s |
| Generation after that prompt (up to 256 tokens) | 84.7 tok/s |
| Code generation, 6 prompts, up to 256 tokens each, mean (range) | 107.3 tok/s (102.4-111.0) |
| Drafted tokens accepted on those prompts | 927/1367 |
| Tool calls correct, tool_choice required / auto | 7/7 / 7/7 |

## Quality

Perplexity on long6k (n_ctx 4096): the served MTPq8 file reads 7.57917 on Super Roc against 7.5845 ± 0.33577 on llama.cpp (ROCmFPX build) of the same file, digit-identical to the pre-splice file.

## Notes

- The draft block is embedded in the model file with its compute weights at Q8_0; the original 4-bit draft block cannot draft.
- The drafter pays on code prompts and costs on prose summaries, so set the flags per workload.
- Drafting only engages for greedy requests; the measured numbers use temperature 0.
