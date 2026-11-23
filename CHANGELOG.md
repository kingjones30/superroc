# Changelog

Notable user-facing changes to Super Roc are recorded here. Releases follow
[Semantic Versioning](https://semver.org/) under the compatibility policy in
[the release guide](docs/RELEASING.md).

## [Unreleased]

### Fixed

- LFM2 family: the prompt cache now keeps the conversation up to the
  generation framing, so a follow-up turn reuses the whole previous prompt
  (a 4,818-token turn now reuses 4,815 tokens; it reused 4,096).

### Added

- LFM2-24B-A2B and LFM2-8B-A1B chat templates (the first LFM2 template's tool
  list and tool-result markers) are recognised and rendered.

## [0.20.0] - 2026-11-23

### Added

- `serve llm --accept-any-model-name`: requests may name any model, as with
  llama-server. Launchers that forward their own model name (Lemonade's
  llama.cpp slot) need it.
- `serve --cors-origin ORIGIN`: send CORS headers for ORIGIN (`*` for any).
- `SUPERROC_REASONING_LOOP_GUARD=<min>:<max>:<repeats>` (for example
  `16:512:3`): an opt-in stop for temperature-0 reasoning loops. While a
  request is still reasoning, the guard fires when the last `repeats` copies
  of one `P`-token block repeat exactly for some `P` in the range, then emits
  the same forced reasoning end a token budget would so the model answers
  instead of looping into the token cap. Unset keeps the fast argmax path
  byte-for-byte; a malformed value fails server startup, each firing logs one
  server line, and the request's completed line carries `loop_guard_fired=0|1`.
- `SUPERROC_REASONING_CLOSE_MESSAGE=<text>` (or `--reasoning-close-message
  TEXT`, which wins): when the reasoning budget or the loop guard forces the
  end of reasoning, the message tokens emit first, then the forced end, then
  the answer. Tokenized once per process with the normal tokenizer; unset
  keeps today's forced end byte for byte.
- `SUPERROC_REASONING_BAN=<word>,<word>,...` (for example
  `Wait,wait,Hmm,hmm,Alternatively,alternatively`): while a request is still
  reasoning, a banned raw argmax falls back to a masked host selection
  (greedy takes the next best, sampled requests renormalise without the
  banned tokens); answers are never affected. Unset, or a ban no token ever
  hits, keeps today's text and decode speed.
- `SUPERROC_CANONICAL_CACHE=1`: a conversation gets the same reply whether the
  server saw it turn by turn or reads the whole history in one request (after
  a restart, an evicted checkpoint, or on another server). A finished request
  leaves only its prompt checkpoints and the next request reads the previous
  reply as prompt; dense Qwen prompt passes take one kernel route at every
  size; Flash-Next prompts are cut and checkpointed only on multiples of 64
  rows with at least 16 rows on both sides, where a cut does not change the
  state, its half-precision projections keep one summation order, and the
  checkpoint before the prompt end is frozen inside the last pass, so the
  switch costs no extra pass there; llama-family models send a one-row
  prompt pass through the batch kernels and read attention keys in one
  sweep; Qwen MoE models needed the cache rule only.
  Measured on 36 agent turns at temperature 0, live against a freshly
  started server, with 3,650 to 4,700 tokens of context: Qwen3.8-27B-UD-Q4_K_XL
  36 of 36 identical replies (34 without the switch), Qwen3.8-27B ROCmFP4 36
  of 36 (31 without), Qwen3.8-Flash-Next ROCmFP4 36 of 36 (29 without),
  Qwen3.8-Flash-Next-UD-Q4_K_XL 36 of 36 (28 without), KAT-Coder-V2.5-Dev
  ROCmFP4 36 of 36 (29 without); with short contexts Nanbeige4.3-33B-Base 36
  of 36 (33 without). See "The same reply for the same
  history" in [the cache guide](docs/KV-CACHE.md).
- `tools/repro/live_vs_cold.py` and `tools/repro/growing_prompt.py` run the
  same-reply checks against Super Roc or any OpenAI-compatible server and
  exit 0 only when every reply matches. `GET /props` reports
  `canonical_cache`, and the request log carries `canonical=1` under the
  switch. See [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).
- Flash-Next ROCmFP4 files whose attention query/key/value tensors use the
  17-byte block layout (recipe overlays do) take the fused projection and
  convolution kernel again: prompt processing 2.6 % faster at 32,000 tokens
  and 3.0 % at 4,000 on a Radeon 8060S, with the same logits bit for bit.
  `LFN_SSM_FUSED_FAST=0` restores the separate kernels.

- The server keeps GPU-registered host memory where the kernel does not move
  it. On an integrated Radeon every move of such a page (memory compaction,
  swap) stops the process's GPU queues until the range is mapped again; with
  the page cache full that cost up to 0.9 s per load and up to 0.5 s on a
  first request. The runtime now starts with `HSA_USERPTR_FOR_PAGED_MEM=0` and
  `GPU_PINNED_MIN_XFER_SIZE=1048576` unless those are already set, and the
  server pins what the runtime registers by itself while a model loads
  (`event=runtime_memory_pinned`). See "Host memory the driver tracks"
  in [the server guide](docs/SERVER.md).

- The server warms up at load: one short greedy generation runs through the
  normal request path before the server listens, so the first request does not
  pay for kernel code, decode graphs, draft buffers and fresh scratch pages.
  It is a raw-text prompt that reuses nothing, leaves no state behind and is
  not counted in the request metrics. `--no-warmup` skips it.

- A dense Qwen model file that carries its own MTP block (Qwen3.8-27B GGUFs
  with a `nextn` block) is its own draft file: `--speculative mtp` no longer
  needs `--mtp-model`, and with no speculative option the server drafts with
  the block (`speculative=mtp-auto` in the load line). `--speculative off`
  serves without a drafter. `/health` reports the mode and the p-min in force.
- The request log carries `prefill_tail_ms`: the draft catch-up and prompt
  snapshot that run after the first token.

- Model recipes: a `superroc-recipe.json` beside the model file carries a tuned
  setup (engine variables, sidecar files such as an int4 route or a tensor
  overlay, and tuning flags). `--recipe auto` loads it from the model's folder,
  `--recipe FILE` names one, `--recipe off` (the default) ignores it. Flags on the
  command line take precedence over the recipe. A recipe is limited to tuning: its
  flags are the speculative-decoding, prompt-lookup, prefill-chunk, thinking and
  sampling options only, so it cannot set the listen address, port, sessions, API
  keys, model, served name, a limit, a timeout, a cache budget, the log level, or
  the trace and disk-cache paths. It cannot set a variable that names a program
  or an output file, a diagnostic switch, or a path through a setting; its files
  must be regular files next to it, for the engine's input-file variables only,
  and the recipe itself at most 1 MiB. A variable the build does not read is
  skipped and reported. `/props` reports what the recipe
  changed. See [docs/recipes.md](docs/recipes.md).
- Responses carry llama.cpp-style `timings` (prompt and predicted token counts,
  milliseconds and tokens per second), so clients and benchmark tools written for
  llama-server read them unchanged.
- OpenAI `logprobs`: chat completions with `top_logprobs` 0–20, streamed beside
  `delta` as OpenAI does; legacy completions with `logprobs` 0–5. A token that is
  part of a multi-byte character is reported as U+FFFD with its exact `bytes`
  (chat) or as `bytes:\xNN` (completions). Requests that do not ask for log
  probabilities do no extra work per token.
- OpenAI `stop` accepts up to 16 strings.
- `/props` reports the serving environment: GPU architecture, ROCm version, kernel
  and whether compute-wave save/restore is available.
- Image input on the Qwen3.5/3.6 MoE family (35B-A3B and 122B-A10B projectors) and
  on the dense 9B and 2048-wide sizes, in addition to Qwen3.8-27B and Flash-Next.
- `SUPERROC_TAIL_CAPTURE=N` takes a second prompt checkpoint N tokens before the
  end of the prompt inside the same prefill pass, so a follow-up that changes only
  the last turn restores from it instead of re-reading the whole prompt. With
  `SUPERROC_CANONICAL_CACHE=1` there is one restore point and N moves it that
  many rows further from the prompt end.
- Streaming responses send a data-chunk heartbeat while a long prefill runs, so
  proxies with idle timeouts keep the connection open.

### Security
- Message text on the Llama, Ling 3 and DeepSeek V4 Flash chat paths can no
  longer become a control token: client text that spells one is read as
  characters, so a message cannot forge a turn or role boundary. Prompts
  without such text tokenize exactly as before.
- The same holds for a response-format description and schema, which the
  server copies into the structured-output instruction: on every chat path
  their text stays characters, while the server's own tool-call delimiters
  stay tokens.
- Streaming errors, image errors and voice websocket errors send fixed texts
  to the client; the cause is written to the server log only.
- JSON-schema intersection charges a work budget for every node it walks
  and every member it compares, so a `$ref` with wide sibling objects is
  rejected in milliseconds instead of costing close to a second each time.
- No CORS headers are sent unless `--cors-origin` is given. Before, every
  response allowed any origin, so any web page could use a local server that
  had no API key.
- A request that fails with an internal error gets the fixed message
  "internal server error"; the cause stays in the server log.
- `/props` reports file names, not absolute paths, for the model and recipe
  files.
- JSON-schema `response_format`: the work spent intersecting `$ref`/`anyOf`
  branches is capped, so a small schema can no longer expand into billions of
  nodes.

### Speed

- MTP on a dense Qwen file. The draft block's K-quant, BF16 or F32 matrices
  are re-encoded as Q4_0 at load, a quarter of the bytes each draft step reads
  (`SUPERROC_MTP_DRAFT_Q4=0` keeps BF16). When the MTP file is the model file
  the draft uses the model's token embedding and output head instead of a
  second copy of both. The draft catches up on the prompt in batches for every
  prompt length (a prompt of up to 257 tokens ran the whole block once per
  token after the first token). A draft ends at the first token the head gives
  less than `SUPERROC_MTP_PMIN` (default 0.40 on a dense block; the probability
  is computed on the device over the draft vocabulary), and without
  `--draft-tokens` a dense block drafts up to 4 tokens per step. The batched catch-up
  scratch holds 1,024 rows and the buffers of the full-block catch-up are
  allocated on first use.

- Prefill on dense ROCmFP4 Qwen3.8 models can run FFN gate and up and attention
  gate, q and qkv through the 4-bit matrix unit. Turn it on with a route weights
  file made for the model file: `SUPERROC_INT4_WEIGHTS=FILE`.
  `SUPERROC_INT4_TENSORS=ffn_gate,ffn_up,...` narrows the tensor list.
- Flash-Next prefill can run the linear-attention query/key/value and gate
  projections and the shared expert's gate and up projections through the
  4-bit matrix unit, with a route weights file made for the model file
  (`SUPERROC_INT4_WEIGHTS=FILE`). On a Radeon 8060S with 7 routed experts a
  24,850-token prompt takes 4.3 % less time and a 3,270-token prompt 3.6 %
  less; perplexity on a 32,768-token text is 6.671 with the route and 6.702
  without. The route is not loaded under `SUPERROC_CANONICAL_CACHE=1`.
- Flash-Next reads mixer down projections stored as ROCmFP4 type 101 beside
  Q8_0 up projections on the fused prompt route and with a decode kernel of
  their own: a token pass without a drafter is 2.7 % faster than with the
  Q8_0 form.
- Qwen3.8 Flash-Next greedy requests with a presence, frequency or repeat
  penalty draft through the small draft head like other greedy requests (the
  target verifies every draft with the same penalties, so the text is the
  same). On a Radeon 8060S with presence penalty 1.5 on 200 GSM8K questions:
  every reply identical, median decode 72.5 tokens per second instead of
  65.3. `LFN_MTP_SUBSET_PENALTIES=0` restores the full head.
- Flash-Next decode switches with bit-identical logits:
  `SUPERROC_FN_PREP_ATTN_SPLIT=1`, `SUPERROC_FN_FUSE_MOE_COMBINE=1`,
  `LFN_VERIFY_ATTN_SHARED=1`; together 1.4 % faster decode with the MTP head
  on a Radeon 8060S. `SUPERROC_FN_MIX_FP4Q8=1` and `SUPERROC_FN_GDN_FP4Q8=1`
  save quantization launches but change the logits slightly.
- `SUPERROC_ROUTED_EXPERTS=N` routes N experts per token in the trunk of a
  mixture-of-experts model; the draft head keeps the file's count.
- `SUPERROC_TENSOR_OVERLAY=FILE` reads the tensors in a small GGUF in place of the
  model file's copies.
- Flash-Next accepts a Q8_0 router and prices a draft row lower when a request
  decodes alone.
- MoE verification reads each expert weight row once for up to 8 rows and runs one
  launch per stage for all rows of a layer.
- Qwen3.8-27B verification shares each KV tile across rows in groups of up to eight
  and reduces every row's score of a token together (bit-identical results).
- FP4 verification rows run at the memory limit, one wave per group.
- MTP: drafts for sampled, penalized and constrained requests; the head's FP4 and
  Q8 matrices stay packed on the GPU; draft attention reads the FP16 K/V; a
  `qwen35moe` model takes a sidecar head with `--mtp-model`.
- `--prompt-lookup` copies draft tokens from the session's own text when it repeats.
- DFlash and DSpark skip prompt rows outside the drafter's window, inject context
  in one batched GEMM, and use a trimmed draft vocabulary for greedy rounds.
- LFM2 and Flint verify draft rows in one batch.
- The prompt checkpoint is taken inside the prefill pass, and the first token is
  published as soon as the pass computes it.
- Short prompt chunks read raw FP4 weights directly.
- Flash-Next takes its intermediate prompt checkpoints where a prefill pass
  already ends. A checkpoint on the 2,048-token grid used to split the pass it
  fell in, and two narrow passes with a capture between them run slower than
  one wide pass: on a Radeon 8060S a 3,252-token prompt prefilled in 1.78 s as
  2,048 + 1,204 rows and prefills in 1.61 s as one pass (2,034 tokens per
  second), a 24,832-token prompt in 13.0 s instead of 13.4 s. A prompt that
  fits in one pass has no intermediate checkpoint; the stable-prefix, tail and
  learned checkpoints are unchanged.
- Flash-Next's last prompt pass takes up to 512 rows beyond the 4,096-row
  chunk (128 before), so a prompt that ends a few hundred rows past a chunk
  has no short closing pass. Late in a long prompt a pass of a few hundred
  rows costs about 0.2 s. On a Radeon 8060S a 24,850-token prompt prefills in
  13.05 s instead of 13.32 s; a 3,270-token prompt is unchanged.

### Changed

- Chat `stop` strings (and Messages `stop_sequences`) are matched in the
  answer, not in the reasoning that precedes it. Before, a stop string that
  the model happened to write while thinking ended the reply with empty
  content. A stop string that contains the reasoning end tag is still matched
  everywhere.

- The reported prefill time (`timings.prompt_ms`, `Server-Timing`,
  `usage.superroc`) ends at the first token. Work that runs after it inside
  the prefill pass (draft catch-up, prompt snapshot) is no longer counted as
  prompt time.
- `SUPERROC_MTP_PMIN` gates on the probability of the drafted token under a
  softmax over the rows the draft head scores, and no longer turns the draft
  vocabulary trim off.

- The previous engine name is gone: its binary links, environment-variable and
  CMake-option fallbacks, Python import paths and stored-name readers are removed.
  Use the `superroc` names.
- `--spec-plain-rates` is removed.
- The upstream project's changelog, its model records for models this project
  does not ship, and its cache design notes are no longer carried. This file now
  records this project's own history; [NOTICE](NOTICE) and
  [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) name the upstream project.

### Fixed

- `--speculative mtp` with `--draft-tokens` above 7 on a Qwen model is refused
  at load. It was accepted, and the first request whose draft reached 8 tokens
  failed with HTTP 400 (`Qwen verification has incompatible sessions or token
  counts`).

- A Flash-Next file or overlay with ROCmFP4 mixer down projections and Q8_0
  up projections read the down projection past its buffer on prompt passes of
  96 rows or more.

- Raw `/v1/completions` adds the BOS token when the GGUF asks for it
  (`tokenizer.ggml.add_bos_token`).
- Ling-3 stop sequences match on the exact text stream.
- Ornith-1.5-122B prompts past 16K tokens: its full attention runs on the WMMA
  prefill kernel instead of the scalar fallback, whose shared memory passed the
  device limit at that depth.
- Flash-Next keeps its FP4 scratch buffers alive across captured graphs when a
  request grows the batch.
- KV casts of planes larger than 2^32 elements.
- A server started with `--prompt-lookup` and no drafter can persist its disk cache.
- The FP4 group scan no longer uses stack scratch.
- An MTP file that carries its own `token_embd` or `output` tensor of another
  size than the model's is refused at load instead of being read past its end.
- The FP4 MoE gate/up kernel launches one block per expert row; expert widths
  that are a multiple of 32 but not of 64 no longer run blocks past the last
  row.
- The DFlash selector checks its head and tail bounds before it launches.
- The Nix package source includes `tools/`, whose sources CMake names, and its
  install check no longer looks for a second binary.
- `THIRD_PARTY_NOTICES.md` names this project's license folder
  (`share/licenses/superroc`) and build option (`SUPERROC_BUILD_TOOLS`).
