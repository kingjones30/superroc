# Super ROC

Super ROC is an LLM inference engine for AMD Strix Halo: Ryzen AI Max+ 395 systems with the Radeon 8060S GPU (`gfx1151`)
and up to 128 GB of unified memory. It serves GGUF models, including 4-bit ROCmFP4 builds, through an OpenAI-compatible
server with MTP and DSpark drafting and a prompt cache that can resume from any earlier conversation state.

Super ROC is a fork of [Gufo](https://github.com/gufo-org/gufo) v0.19.0. Issues, pull requests and forks are welcome.

## What Super ROC adds to Gufo

- **ROCmFP4 / ROCmFPX tensor types** (GGUF types 100 and above) in the loader, the kernels and the serving path, including
  spill-free dense prefill and decode kernels for the 4-bit types.
- **`gufo-quant`**, a standalone GGUF quantizer for the ROC recipes (STRIX, STRIX_LEAN, LEAN, COHERENT, FAST and the G4
  variants) with protect rules, a tensor census, manifests and streaming output. For the ROC recipes its output is byte
  for byte the same as the ROCmFPX `llama-quantize`. See [docs/gufo-quant.md](docs/gufo-quant.md).
- **More model families**: `qwen35` / `qwen35moe` ROC builds (Qwen3.8, KAT-Coder, Ornith, MiniCPM-V, OxCoder, Agnes,
  Nex, BTL-4, MiMo), `bailingmoe3` (Ling-3.0-flash and Ling-3.0-flash-VL), `lfm2` / `lfm2moe` (LFM2.5), `flint`
  (Flint 40B-A2.5B: sliding-window and position-free attention layers with attention sinks) and Muse Glimmer.
- **Drafting**: embedded and sidecar MTP heads with a per-request draft depth and exact verification, the DSpark
  drafter, and prompt-cache snapshots that keep working while a drafter is on.
- **Tool calls** in each model's native format (Qwen, GLM-style, Ling, Flint), constrained by a grammar built from the
  request's tools where the format allows it.
- **Hardening**: HTTP header and body read deadlines and size caps, constant-time API key comparison, checked HIP kernel
  launches, and validation in the GGUF reader, configuration and weight binders.

## Models

Measured with this repository's code on Ryzen AI Max+ 395 systems (128 GB, ROCm 7.2.4 or 7.13.0, named in each card):
one request at a time, temperature 0, cold prompt cache, drafter as in the card's serve command. *Prompt* is prompt
processing on a document question of 1,517 to 1,641 tokens (depending on the tokenizer); *Generation* is the speed of the answer to that prompt (up to
256 tokens); *Code* is the mean generation speed over six coding prompts (up to 256 tokens each). Each card has the
exact files, the serve command, more measurements, quality numbers where they exist, and known limits.

| Model | Size | File (Hugging Face) | Drafter | Prompt tok/s | Generation tok/s | Code tok/s | Card |
|---|---|---|---|---:|---:|---:|---|
| KAT-Coder-V2.5-Dev Abliterated | 35B-A3B MoE | [STRIX_LEAN, MTP block Q8_0](https://huggingface.co/kingjones777/KAT-Coder-V2.5-Dev-Abliterated-MTP-ROCmFP4-GGUF) | embedded MTP | 1,713.1 | 84.7 | 107.3 | [card](docs/models/kat-coder-v2.5/README.md) |
| Ornith-1.5-35B-A3B | 35B-A3B MoE | [LEAN v1](https://huggingface.co/kingjones777/Ornith-1.5-35B-A3B-ROCmFP4-STRIX-LEAN-GGUF) | embedded MTP | 1,773.7 | 74.6 | 96.4 | [card](docs/models/ornith-1.5-35b/README.md) |
| MiniCPM-V-4.7-35B-A3B | 35B-A3B MoE, vision | [LEAN, Q6_K trunk](https://huggingface.co/kingjones777/MiniCPM-V-4.7-35B-A3B-ROCmFP4-GGUF) | none | 1,876.3 | 65.2 | 65.7 | [card](docs/models/minicpm-v-4.7-35b-a3b/README.md) |
| Agents-A1 Abliterated | 35B-A3B MoE | [STRIX_LEAN](https://huggingface.co/kingjones777/Agents-A1-Abliterated-MTP-ROCmFP4-GGUF) | none | 1,767.7 | 77.0 | 78.0 | [card](docs/models/agents-a1/README.md) |
| BTL-4 | MoE (qwen35moe) | [STRIX_LEAN](https://huggingface.co/kingjones777/BTL-4-ROCmFP4-STRIX_LEAN-GGUF) | none | 1,771.0 | 80.6 | 81.4 | [card](docs/models/btl-4/README.md) |
| Nex-N2.5-mini | MoE (qwen35moe) | [STRIX_LEAN](https://huggingface.co/kingjones777/Nex-N2.5-mini-ROCmFP4-GGUF) | none | 1,753.1 | 77.1 | 78.0 | [card](docs/models/nex-n2.5-mini/README.md) |
| Qwen3.8-Flash-Next Uncensored | 177B MoE | [STRIX_LEAN imatrix, hc tensors Q8_0](https://huggingface.co/kingjones777/Qwen3.8-Flash-Next-Uncensored-ROCmFP4-STRIX_LEAN-imatrix-GGUF) | MTP head Q8_0 | 1,261.2 | 56.1 | 56.8 | [card](docs/models/qwen3.8-flash-next/SUPERROC.md) |
| Qwen3.8-Flash-Next | 177B MoE | [Unsloth UD-Q4_K_XL](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) | shared MTP Q8_0 | 1,369.1 | 47.6 | 59.9 | [card](docs/models/qwen3.8-flash-next/SUPERROC.md) |
| Qwen3.8-27B | 27B dense | [STRIX imatrix, head/embedding Q6_K](https://huggingface.co/kingjones777/Qwen3.8-27B-ROCmFP4-STRIX-MTP-GGUF) | DFlash2 | 513.6 | 27.6 | 50.6 | [card](docs/models/qwen3.8-27b/SUPERROC.md) |
| Qwen3.8-27B | 27B dense | [Unsloth UD-Q4_K_XL](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) | DFlash2 | 545.6 | 27.2 | 44.3 | [card](docs/models/qwen3.8-27b/SUPERROC.md) |
| Qwen3.8-27B Uncensored | 27B dense | [STRIX_LEAN](https://huggingface.co/kingjones777/Qwen3.8-27B-Uncensored-ROCmFP4-STRIX-MTP-GGUF) | MTP Q4_0 | 509.6 | 26.3 | 43.9 | [card](docs/models/qwen3.8-27b-uncensored/README.md) |
| Qwen3.8-27B Fable Cold Fusion Heretic | 27B dense, vision | [FAST imatrix](https://huggingface.co/kingjones777/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-Heretic-ROCmFP4-FAST-imatrix-GGUF) | MTP Q4_0 | 498.8 | 29.1 | 41.7 | [card](docs/models/qwen3.8-27b-fable-heretic/README.md) |
| Agnes-3.0-Flash-Preview | 33B dense | [STRIX_LEAN](https://huggingface.co/kingjones777/Agnes-3.0-Flash-Preview-MTP-ROCmFP4-GGUF) | MTP sidecar Q8_0 | 423.0 | 21.5 | 29.2 | [card](docs/models/agnes-3.0-flash/README.md) |
| OxCoder-9B | 9B dense | [STRIX_LEAN](https://huggingface.co/kingjones777/OxCoder-9B-MTP-ROCmFP4-GGUF) | MTP (own block) | 1,437.7 | 51.8 | 76.4 | [card](docs/models/oxcoder-9b/README.md) |
| Ornith-1.5-9B | 9B dense | [STRIX_LEAN](https://huggingface.co/kingjones777/Ornith-1.5-9B-ROCmFP4-STRIX-LEAN-GGUF) | none | 1,703.1 | 42.3 | 42.7 | [card](docs/models/ornith-1.5-9b/README.md) |
| MiMo-V2.6-Distill-Qwen-9B | 9B dense | [STRIX_LEAN](https://huggingface.co/kingjones777/MiMo-V2.6-Distill-Qwen-9B-ROCmFP4-GGUF) | none | 1,625.4 | 42.2 | 42.5 | [card](docs/models/mimo-v2.6-9b/README.md) |
| Ling-3.0-flash-VL | 124B-A5.5B MoE, vision | [STRIX_LEAN](https://huggingface.co/kingjones777/Ling-3.0-flash-VL-MTP-ROCmFP4-GGUF) | none | 569.2 | 38.4 | 37.4 | [card](docs/models/ling-3.0-flash-vl/README.md) |
| LFM2.5-24B-A2B | 24B-A2B MoE | [LEAN (from the QAD 4-bit file)](https://huggingface.co/kingjones777/LFM2.5-24B-A2B-ROCmFP4-GGUF) | none | 2,459.2 | 97.8 | 101.8 | [card](docs/models/lfm2.5/README.md) |
| LFM2.5-8B-A1B | 8B-A1B MoE | [LEAN](https://huggingface.co/kingjones777/LFM2.5-8B-A1B-ROCmFP4-LEAN-GGUF) | none | 4,037.2 | 153.4 | 158.5 | [card](docs/models/lfm2.5/README.md) |
| LFM2.5-2.6B | 2.6B dense | [LEAN](https://huggingface.co/kingjones777/LFM2.5-2.6B-ROCmFP4-LEAN-GGUF) | none | 3,317.9 | 104.9 | 108.6 | [card](docs/models/lfm2.5/README.md) |
| LFM2.5-1.2B-Instruct | 1.2B dense | [LEAN](https://huggingface.co/kingjones777/LFM2.5-1.2B-Instruct-ROCmFP4-GGUF) | none | 6,992.6 | 210.7 | 223.3 | [card](docs/models/lfm2.5/README.md) |
| Flint 40B-A2.5B | 40B-A2.5B MoE | [COHERENT-PAS](https://huggingface.co/kingjones777/Flint-40B-A2.5B-ROCmFP4-GGUF) | none | 1,119 ¹ | 58.9 ¹ | — | [card](docs/models/flint/README.md) |

¹ Flint was measured with its own method: a 512-token prompt and 128 generated tokens. Its card has the 4K and 16K numbers.

Also from Gufo v0.19.0, with Gufo's documentation and measurements (not re-measured here):
[DeepSeek V4 Flash](docs/models/deepseek-v4-flash/README.md), [Qwen3-ASR 1.7B](docs/models/qwen3-asr/README.md),
[Qwen3-TTS 1.7B](docs/models/qwen3-tts/README.md), [Qwen-Image-2.1](docs/models/qwen-image-2.1/README.md) and
[MiniMax H3](docs/models/minimax-h3/README.md). The upstream guides for
[Qwen3.8 27B](docs/models/qwen3.8-27b/README.md) and [Qwen3.8 Flash-Next](docs/models/qwen3.8-flash-next/README.md)
are kept next to the Super ROC cards for those models.

## Build from source

Linux x86-64 on AMD Strix Halo (`gfx1151`) is the supported target.

### With Nix

```sh
nix build
./result/bin/superroc diagnose
./result/bin/superroc serve llm --model /path/to/model.gguf
```

### Without Nix

Install a C++20 compiler, CMake 3.21+, Ninja, pkg-config and the ROCm HIP development libraries (ROCm normally installs
under `/opt/rocm`), then:

```sh
sudo apt install build-essential cmake ninja-build pkg-config \
  libicu-dev libcurl4-openssl-dev libssl-dev libpng-dev libjpeg-dev libwebp-dev ffmpeg

# ROCm libraries, named as AMD's repository ships them.
sudo apt install hipblas-dev hipblaslt-dev rocblas-dev \
  hipcub-dev rocprim-dev rocwmma-dev

cmake --preset release -DCMAKE_INSTALL_PREFIX="$HOME/.local"
cmake --build --preset release --parallel 4
./build/release/superroc diagnose
```

`cmake --install build/release` installs Super ROC, its runtime data and the license notices. The build also creates a
`gufo` link to the same binary, so commands written for Gufo keep working.

## Run

Serve a model (here KAT-Coder-V2.5-Dev Abliterated with its embedded MTP drafter, see its card):

```sh
./build/release/superroc serve --host 127.0.0.1 --port 8080 llm \
  --model /path/to/KAT-Coder-V2.5-Dev-Abliterated-MTPq8-Q4_0_ROCMFP4_STRIX_LEAN.gguf \
  --served-model-name kat-coder --context 131072 --think off --speculative mtp -d 3 --min-draft-tokens 1
```

Then ask it something through the OpenAI-compatible API:

```sh
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "kat-coder", "messages": [{"role": "user", "content": "Write a Python function that reverses a list."}]}'
```

For Open WebUI, VS Code, the OpenAI SDK and Responses-API clients, set the API base URL to `http://localhost:8080/v1`.
Chat Completions and Responses support text, images, function tools, structured output and streaming. The server uses
the model's native context by default; `--context N` sets the context per session and `--max-tokens N` a default
response limit that clients can override (reasoning tokens count toward it). The full API contract is in
[docs/SERVER.md](docs/SERVER.md) and the command line in [docs/CLI.md](docs/CLI.md).

## Known gaps

- Muse Glimmer loads, but its DFlash drafter does not yet, and without it Super ROC generates more slowly than
  llama.cpp (ROCmFPX build) on that model.
- A Qwen3.6-35B-A3B file with its MoE experts stored as Q8_0 does not load on this build.
- Several GPU tests and tools read a model file; they take its path from an environment variable and fail without it.

## Lineage and license

Super ROC began as a fork of Gufo v0.19.0 (https://github.com/gufo-org/gufo). The code is released under the MIT License
in [LICENSE](LICENSE). [NOTICE](NOTICE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) list the code that came from
Gufo, llama.cpp / ggml and other projects, with their licenses. Gufo's original README is kept at
[docs/lineage/GUFO-README.md](docs/lineage/GUFO-README.md). Model weights are not included; each Hugging Face card states
the license of its weights.

## Feedback

Open an issue with the hardware (CPU/GPU, memory, ROCm version), the model file, the exact command and the server log.
Measurements on other Strix Halo systems, new model families and kernel improvements are welcome as pull requests.
