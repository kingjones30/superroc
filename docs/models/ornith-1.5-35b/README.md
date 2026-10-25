# Ornith-1.5-35B-A3B, ROCmFP4 LEAN v1 with the embedded MTP block

The base model is described in its Hugging Face card as a 35B mixture-of-experts model with 256 experts. These files are ROCmFP4 LEAN v1 quantizations for Strix Halo built with gufo-quant, with the output and embedding kept at Q6_K and the inline draft block at Q8_0. Drafting uses the MTP block embedded at block 40.

| | |
|---|---|
| Base model | [ornith-ai/Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B) |
| License | apache-2.0 |
| Architecture | `qwen35moe`, 35.5053B parameters, 256 experts / 8 used, about 3B active |

## Files

| File | Role | Repository |
|---|---|---|
| `Ornith-1.5-35B-A3B-Q4_0_ROCMFP4_LEAN-v1.gguf` | model | [kingjones777/Ornith-1.5-35B-A3B-ROCmFP4-STRIX-LEAN-GGUF](https://huggingface.co/kingjones777/Ornith-1.5-35B-A3B-ROCmFP4-STRIX-LEAN-GGUF) |

## Serve

```sh
GPU_MAX_HW_QUEUES=2 superroc serve --host 127.0.0.1 --port 8080 llm -m /path/to/Ornith-1.5-35B-A3B-Q4_0_ROCMFP4_LEAN-v1.gguf --served-model-name ornith-1.5-35b --context 32768 --think off --speculative mtp -d 2 --min-draft-tokens 1
```

## Measured on Super ROC

AMD Ryzen AI Max+ 395 (Radeon 8060S, gfx1151), 128 GB, ROCm 7.13.0. One request at a time, temperature 0, cold prompt cache, drafter as in the serve command.

| Measurement | Result |
|---|---:|
| Prompt processing, 1566-token prompt | 1773.7 tok/s |
| Generation after that prompt (up to 256 tokens) | 74.6 tok/s |
| Code generation, 6 prompts, up to 256 tokens each, mean (range) | 96.4 tok/s (92.6-101.3) |
| Drafted tokens accepted on those prompts | 933/1176 |
| Tool calls correct, tool_choice required / auto | 7/7 / 7/7 |

## Quality

Perplexity 10.2407 on a 96-chunk corpus at 512 context, against 10.0714 for the stock Unsloth UD-Q4_K_XL file (quality ratio 1.01727 with 95% interval [1.01073, 1.02385]), measured on an earlier Super ROC build. KL divergence against a Q8_0 reference is 0.08350 with 85.83 % top-1 agreement (the stock file: 0.02707 and 91.79 %).

## Notes

- The measurement ran with GPU_MAX_HW_QUEUES=2 (a ROCm runtime setting).
- Drafting uses the inline MTP block at block 40 with depth 2 and at least one draft token.
- The setup serves text only; no vision projector is attached.
