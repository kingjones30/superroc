# ling3 — Ling-3.0-flash (`bailingmoe3`)

The `bailingmoe3` model package: Ling-3.0-flash and Ling-3.0-flash-VL with ROCmFP4 weights. It serves through `superroc serve`
(see [docs/models/ling-3.0-flash-vl/README.md](../../../docs/models/ling-3.0-flash-vl/README.md)). The scalar CPU reference
below was the first part of the port and is still the oracle for the GPU kernels.

## What this package is

| File | Role |
|---|---|
| `rocmfp4.{hpp,cpp}` | GGML types 100/101 (`Q4_0_ROCMFP4`, `Q4_0_ROCMFP4_FAST`) host decode |
| `config.{hpp,cpp}` | fail-closed `bailingmoe3.*` metadata validator |
| `weights.{hpp,cpp}` | tensor binding for 100/101, Q8_0, Q6_K, F32 |
| `tokenizer.{hpp,cpp}` | GPT-2 byte-level BPE, `pre = bailingmoe2` |
| `chat_template.{hpp,cpp}` | the Bailing-V3 template in C++ (gufo has no Jinja) |
| `cpu_ops.{hpp,cpp}` | scalar operators, every format decoded on the fly |
| `reference.{hpp,cpp}` | the single-token float32 forward — the oracle |

## The one core change this port required

gufo could not open this artifact at all. `GgmlReader` rejects any tensor whose
type `quant::QuantizedBlockElements()` does not recognise, and ROCmFP4 (100/101)
was absent — which is *every weight tensor in the file*. The change is additive
and touches three lines' worth of shared code:

- `src/core/gguf_reader.hpp` — `kQ4_0_ROCMFP4 = 100`, `kQ4_0_ROCMFP4_FAST = 101`
  plus their `ToString` cases.
- `src/core/quant/ggml_dequant.hpp` — both report 32 elements per block.
- `src/core/quant/ggml_dequant.cpp` — 18 and 17 bytes per block.

No existing type changes behaviour. `src/core/` is shared with the other model
packages.

## Layout notes that are easy to get wrong

- **ROCmFP4**: 32 elements per block. Type 100 is `qs[16] + e[2]` (18 B); `e[0]`
  scales elements 0–15 and `e[1]` elements 16–31. Type 101 is `qs[16] + e[1]`
  (17 B), one scale for the whole block. The nibble codebook is
  `0,±1,±2,±3,±4,±6,±8,±10` — E2M1 with the top level retuned 12→10, and index
  8 is the zero code, not a sign bit. Scale bytes are unsigned E4M3, table
  indexed, and anything above `0x7e` decodes to zero.
- **Layer map** comes from `bailingmoe3.attention.head_count_kv`, read verbatim
  rather than inferred from a stride: 0 means KDA, non-zero means gated MLA.
  For this file that is **7 MLA (blocks 5, 11, 17, 23, 29, 35, 41) and 35 KDA**,
  out of 42 trunk blocks plus one MTP block that is bound but never executed.
- **MLA is the absorbed form.** Ling ships `attn_k_b` / `attn_v_b`, not
  `attn_kv_b`, so the fork takes `wk_b && wv_b` rather than the MHA fallback.
  The KV cache is the **576-dim latent (512 + 64 rope) at a single head**; the
  query's non-rotated part is absorbed into the latent space by `wk_b` and the
  value leaves through `wv_b`. The attention scale stays `1/sqrt(192)` — the
  *pre-absorption* key length, not `sqrt(576)`.
- **KDA keeps three conv histories**, one per projection, at
  `(kv_head * n_embd_r_total + qkv * conv_state_size)`. They do *not* hold the
  same values: after one token they carry the Q, K and V projections
  respectively. This is `3 * (kernel-1) * inner` floats, not one history.
- **The KDA scan** is a four-phase delta rule in that order: per-key-channel decay
  `S[i][j] *= exp(g[i])`, then `delta[j] = (v[j] - sum_i S[i][j]k[i]) * beta`,
  then `S[i][j] += k[i] * delta[j]`, then `o[j] = sum_i S[i][j]q[i] * 1/sqrt(D)`.
  The output reads the state **after** the write. State is stored transposed
  (`S[i][j]`, `i` = key axis fastest), which is ggml's layout, not the logical
  one.
- **`ggml_l2_norm` is `x / max(|x|, eps)`** — the `fmaxf` sits *outside* the
  square root, so it is a clamp, not a soft epsilon. And the sum accumulates in a
  `double` whose terms are `xi * xi` rounded to **float** first.
- **The MoE router**: sigmoid (never softmax); `exp_probs_b` biases **selection
  only**, the final weight is the unbiased score; a group score is the sum of the
  **top two** experts in that group; experts are contiguous by group
  (`expert = group * E_g + slot`); weights are sum-normalised with the sum
  clamped at the smallest F16 normal, then multiplied by 2.5, and applied
  **after** the down projection.
- **SwiGLU clamps** are per block and sparse. `swiglu_clamp_exp` is 4.0 on blocks
  **35–41**; `swiglu_clamp_shexp` is 5.0 on 34–39 and 7.0 on 40–41. The clamp is
  `silu(gate)` clamped **above** at L, times `up` clamped **symmetrically**, and
  is skipped entirely when `L <= 1e-6`. Both arrays are read from the file and
  indexed by block; nothing is hardcoded.
- **No BOS.** This artifact carries `add_bos_token = false` and
  `add_eos_token = false`, so encoding a prompt prepends nothing.
- **`token_embd.weight` is Q6_K**, not ROCmFP4. Easy to assume otherwise.
- **The chat template's tool-call markers contain a zero-width space** (U+200B).
  The vocabulary has that form as its own USER_DEFINED token, so the plain ASCII
  spelling does not tokenize to it at all.

## Tests

Everything below runs on a laptop: no model file, no GPU, no `gufo_core`.

```
cmake --build build-lg --target gufo_ling3_semantics   # the four operator tests
cmake --build build-lg --target ling3_step_test        # structural wiring
```

| Test | Compares against | Result |
|---|---|---|
| `tools/ling3_rocmfp4_test` | the fork's `rocmfp4_dequantize_row_*`, dlopen'd from `libggml-base.so`, `memcmp` per float | 200 trials, 0 failures (every legal scale byte 0x00–0x7e plus the reserved 0x7f) |
| `tools/ling3_dequant_parity` | the fork's Q6_K and Q8_0 row decoders | 0 failures |
| `tools/kda_math_test` | `ggml_ssm_conv`, `ggml_l2_norm`, `ggml_gated_delta_net` | 300 / 500 / 400 trials, 0 failures |
| `tools/kda_layer_test` | a transcription of the fork's KDA layer | 60 trials × 6 tokens, 0 failures |
| `tools/mla_layer_test` | a transcription of the fork's absorbed-MLA graph | 200 trials × 5 positions, 0 failures |
| `tools/moe_router_test` | a transcription of `build_moe_ffn` | 3000 router + 20000×6 SwiGLU, 0 failures |
| `tools/ling3_chat_template_test` | marker/format assertions | PASS |
| `tools/ling3_step_test` | — (structural only, see its header) | PASS |

Two real bugs were found by this method, both invisible to a single-token smoke
test: the KDA conv state was shared across Q/K/V instead of per-projection, and
the MLA softmax was indexed by head rather than by position. The technique that
found them — write the fork's semantics a second time, independently, and diff —
is what `kda_layer_test` / `mla_layer_test` / `moe_router_test` exist to make
routine.

End-to-end token equality on the real artifact is checked by
`tools/ling3_ref_main.cpp` against `llama-completion`; see `~/lg/gate_a.sh`.

## Running it

```bash
# greedy decode, printing token ids
./build-lg/src/models/ling3/gufo_ling3_ref <model.gguf> --prompt "hello" --tokens 32

# tokenization only, for a differential test against the fork
./build-lg/src/models/ling3/gufo_ling3_ref <model.gguf> \
    --prompt-file p.txt --tokenize-only

# render the gate-(b) chat prompt with this package's own template
./build-lg/src/models/ling3/gufo_ling3_ref <model.gguf> --render-chat

# per-layer trace, to diff against llama-eval-callback
./build-lg/src/models/ling3/gufo_ling3_ref <model.gguf> \
    --prompt-file p.txt --tokens 1 --trace
```

It is a correctness oracle, not a server: it decodes every weight on the fly and
runs at roughly a second per token on a dozen threads.

## Not done

- **No ROCm session.** The kernels need gufo's `hip::DeviceModel` / `Executor` /
  `Session`, which do not exist for `ling3`. LK's and LX's headers are ready and
  LK's KDA state layout already matches this package's byte for byte, so the
  remaining work is the runtime plumbing, not the semantics.
- **Speculative decoding.** The MTP block is bound and validated but never
  executed; the routing branch rejects `--speculative` for Ling.
- **Non-ASCII tokenization.** `pre = bailingmoe2` is served by the fork's ASCII
  fallback for `\p{L}` and `\p{N}`, and this package follows it — but on
  non-ASCII input the two still land on different pieces. 30 of 31 differential
  cases agree; the one that does not is accented Latin. All the gate prompts are
  ASCII, so this does not affect them.
