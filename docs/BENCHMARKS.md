# Benchmark methodology

Build with `nix build` and measure binaries under `result/bin`.

## Results through Lemonade's benchmark (Strix Halo)

Measured with [Lemonade](https://github.com/lemonade-sdk/lemonade)'s own harness, the command used for the Halogen backend:

```sh
lemonade bench <model> --scenarios context-4k context-32k chat-long-output --ctx-size 40960 --runs 2
```

Every request has temperature 0. Hardware: AMD Ryzen AI Max+ 395 (gfx1151), 128 GB; ROCm 7.13 on the host. Each row is the
mean of 2 runs. The context scenarios generate about 20 tokens, so their decode figure is short-run; chat-long-output
generates 256.

### Qwen3.8-27B, one model file for every engine

File: `Qwen3.8-27B-UD-Q4_K_XL.gguf` (unsloth/Qwen3.8-27B-GGUF). Decode and prefill in tokens per second.

| Engine and setup | Decode chat / 4K / 32K | Prefill 4K / 32K |
| --- | --- | --- |
| llama.cpp b12109 ROCm, default (its automatic MTP drafts) | 23.3 / 27.0 / 29.2 | 376.5 / 323.4 |
| llama.cpp b12109 ROCm, `--spec-type none` | 12.2 / 12.0 / 10.6 | 393.3 / 336.5 |
| llama.cpp b12109 Vulkan, default | 21.5 / 26.3 / 27.2 | 320.1 / 285.6 |
| Super Roc, no speculation | 12.5 / 12.3 / 11.4 | 638.0 / 594.5 |
| Super Roc, MTP from the same file (3-token drafts) | 20.2 / 24.2 / 29.3 | 625.8 / 587.6 |
| Super Roc, recipe (prompt lookup) + DFlash2 draft | 26.2 / 34.3 / 38.9 | 639.5 / 600.6 |

Speculation of the same kind: llama.cpp's MTP decodes faster than Super Roc's MTP row at chat and 4K and equal at 32K;
Super Roc's decode lead comes from the recipe with the DFlash2 draft. Prefill is ahead in every Super Roc row. GSM8K (first
200 questions, chat template, thinking on, temperature 0, lm-evaluation-harness through Lemonade): Super Roc 0.810 with and
without speculation, llama.cpp 0.805.

## Model benchmark driver

`tools/bench/model-bench.py` measures the tables of a model's `BENCHMARKS.md`
over HTTP for SuperROC and for the reference project of its category (llama.cpp
for Qwen; antirez/ds4 for DeepSeek), writes per-table artifacts, and renders the tables
and SVG charts between the `<!-- bench:<id> -->` markers. Workloads and table
layouts are declared in `docs/models/<model>/artifacts/bench.json`; the
`benchmark-model` skill in `.agents/skills` describes the procedure.

Reference runtimes are explicitly selected Nix packages, excluded from SuperROC,
the normal development shell and hosted checks:

| Reference | Package | Executable |
| --- | --- | --- |
| Qwen AR / DFlash2 | `.#llama-cpp-reference` | `llama-server` |
| Flash-Next MTP | `.#llama-cpp-mtp-reference` | `llama-server-mtp` |
| DeepSeek AR / DSpark | `.#ds4-reference` | `ds4-server`, `ds4-bench` |

For example, enter the development shell and add only the needed baseline:

```sh
nix develop
nix shell .#llama-cpp-reference .#llama-cpp-mtp-reference
python3 tools/bench/model-bench.py --model qwen3.8-flash-next --gguf "$MODEL" \
  --mtp "$MTP" run --target reference --table single-mtp
```

DeepSeek's HTTP payload omits pp/tg durations. The driver reads the pinned
server's existing stage timers from its log and checks them against HTTP token
counts. Native `ds4-bench` also supports single-session AR/DSpark, but not
concurrency. The pinned ROCm server disables DSpark when batching, so C>1
DSpark comparisons are **N/A**.

New or refreshed text-model cards use the same pp2048 prose/copying prompts
for single-user d0 and concurrency, with tg128. The driver shares their prompt
generator and records prompt hashes. C1 should cross-check d0; investigate
input, cache, controller and timing differences before attributing a gap to
batching. Short repeated-word corpora remain useful quality probes, but their
rates cannot replace these matched workloads.

For decode comparisons, set `prefill_first: true` in the concurrency table.
Prepare every session with the full prompt and one output token, wait for all
preparations, then time identical tg128 requests with prefix reuse. The driver
pins llama.cpp slots and rejects replay exceeding four prompt tokens. This keeps
peers' long prefills outside llama.cpp's elapsed generation timer; SuperROC reports
active decode time. Preserve a separate fresh AR completion-hash control.

## Direct and serving measurements

`superroc bench` measures the model path. Keep model artifact, prompt length,
generation length, context depth, speculative mode, and sampling controls
fixed when comparing runs. `pp2048` means 2,048 new prompt tokens; context
depth is separate from the amount of new prefill work.

`tools/serving/superroc-serving-bench.py` measures HTTP serving with synchronized
requests. It distinguishes server-stage prefill/decode throughput, scheduler
and client TTFT, token ITL, whole-request throughput, and aggregate concurrency
throughput. C1 serving and a direct single-user benchmark are comparable only
when prompt, cache state, sampling, speculation, and timed scope match.

Model tables report the **sum of individual request decode rates** in each
concurrent group, averaged across measured groups. Prefill and queue time
remain in the latency diagnostics. Pending measurements are **TODO**;
comparisons prevented by missing reference features are **N/A**, with a reason.
Whole-request throughput is not a substitute for decode timings.

`--endpoint-profile openai` supports other OpenAI-compatible servers. Missing
server-stage metrics remain null. `--reference-report` compares completion
hashes against an existing C1 reference; category filters select a focused
subset. Qualify acceptance with a fresh server so prompt-cache hits do not
hide work. See [serving benchmark commands](PERFORMANCE.md) for invocation.

`tools/bench/speculative-corpus.py` compares direct autoregressive and
speculative token traces. Request correctness, acceptance, and end-to-end
speed are separate requirements.

## Quality method (matched-token, per position)

Use the independent references and model-owned limits described in
[testing](TESTING.md). Tokenize once and feed teacher and candidate the same
predetermined history. Compare full next-token distributions before appending
the next evaluation token. Free-running text cannot isolate numerical error
after the first differing choice.

```text
p_t = softmax(teacher_logits)
p_c = softmax(candidate_logits)
KL  = sum_v p_t[v] * (log p_t[v] - log p_c[v])
NLL = -log(p_c[target_token])
```

Record non-finite counts, normalized-logit error, KL, teacher-token
NLL/perplexity, top-1 agreement, and top-k overlap as applicable. Aggregates
are over scored positions. A changed tokenizer, template, suite, or history
changes the comparison identity.

Capability evaluation is separate: [superroc eval](CLI.md#benchmarks-and-evaluations) grades free-running
answers through the OpenAI-compatible server.

## Focused kernel measurements

Build development tools with `cmake --preset gpu-test` and
`cmake --build --preset gpu-test --target superroc-kernel-bench tune_hipblaslt`.
They are omitted from the production package.

`superroc-kernel-bench` exercises production HIP entry points without loading a
model. Correctness sentinels run outside the timed interval:

```sh
./build/gpu-test/superroc-kernel-bench --help
./build/gpu-test/superroc-kernel-bench \
  --case decode-attention --context 4096,8192,12288,16384 \
  --warmup 5 --repetitions 30 --json
```

Standalone microbenchmarks under `tools/bench` and model-specific tool folders
report component correctness alongside timing. Their numbers do not replace
end-to-end model measurements. Collect profiler traces separately and use
[the performance guide](PERFORMANCE.md) to identify the limiting operation.

## Reporting and artifact retention

Record model/engine revisions, artifact hashes, hardware/software fingerprint,
power profile and CPU/GPU temperatures, input identity, context, output count,
sampling, concurrency, cache state, oracle identity, and the exact timed scope.
A numerical mismatch or device failure is not a passing result. Add repetitions
when timing noise or a suspected regression requires them.

Checked-in JSON retains fixtures, model provenance, controller calibration,
and bounded result summaries. Serving summaries preserve aggregate metrics,
per-case completion hashes, and the original report digest. Raw samples,
profiler traces, generated tuning databases, and full-logit captures stay in
ignored `artifacts/`, except for independent reference data needed by a
maintained correctness test.

Serving reports exclude endpoint hosts, prompts, generated text, model paths,
raw token IDs, and timestamps. The canonical machine fingerprint excludes
hostnames, usernames, secrets, timestamps, and private paths.
`superroc diagnose --validate-artifact` validates its supported diagnostic schema;
each model/tool owns the validation of its own result schema.

Downloaded tokenizer files and `.direnv` shell state are local artifacts.
Count tracked source, including HIP translation units, with:

```sh
nix run nixpkgs#cloc -- --vcs=git --force-lang=C++,hip
```
