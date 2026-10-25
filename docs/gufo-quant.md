# gufo-quant — GGUF quantizer and `kv_swap`

`gufo-quant` is gufo's own GGUF quantizer. It reproduces the ROCmFP4 fork's
`llama-quantize` output for the ROC recipes **byte for byte** (the only added
bytes are `gufo.quant.*` provenance keys), so a file quantized by gufo is a
drop-in for any file the fork produced with the same recipe, imatrix and prune
set. `kv_swap` builds a drop-in that takes the KV/metadata block from one file
and the tensor payloads from another.

Both tools are standalone C++17 programs under `tools/gufo-quant/` — no HIP, no
engine, no external library. Sources: `tools/gufo-quant/gufo_quant_main.cpp`,
`tools/gufo-quant/kv_swap.cpp`, shared reader/writer `tools/gufo-quant/gguf.hpp`,
tensor rules `tools/gufo-quant/rules.hpp`, census `tools/gufo-quant/census.hpp`,
streaming SHA-256 `tools/gufo-quant/sha256.hpp`.

## Build

Through the project CMake (the tools are registered in the top-level
`CMakeLists.txt`; `kv_swap` at the gufo-quant tools block, the two tests under
`if (BUILD_TESTING)`):

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo -DBUILD_TESTING=ON
cmake --build build -j 8 --target gufo-quant kv_swap
```

or standalone, exactly as the shipped 27B/Glimmer files were built:

```sh
g++ -O3 -DNDEBUG -std=c++17 -pthread -I tools/gufo-quant \
    tools/gufo-quant/gufo_quant_main.cpp -o gufo-quant
g++ -O3 -DNDEBUG -std=c++17 -pthread -I tools/gufo-quant \
    tools/gufo-quant/kv_swap.cpp -o kv_swap
```

## gufo-quant

```
gufo-quant [options] in.gguf out.gguf RECIPE [nthreads]
gufo-quant census <file.gguf> [--json]
```

Positional arguments are read in this order: input, output, recipe, thread
count (`gufo_quant_main.cpp:344-347`; `nthreads` defaults to 1 and is clamped to
≥ 1 at `:376`). There is **no `--help`**: an unknown option prints
`unknown option: …` and exits 2 (`:341-342`); with a missing positional it
prints the usage line and exits 2 (`:350-351`).

| flag | meaning | source |
|---|---|---|
| `--allow-requantize` | allow quantizing a tensor whose source type is already quantized (otherwise the source type is kept / the encode throws) | `gufo_quant_main.cpp:286`, used `:705` |
| `--pure` | put **every** quantizable tensor at the recipe's default type; no k-quant mixtures, no per-class special cases | `:287`, applied `:358-375`, `:659` |
| `--dry-run` | header-only: print the per-tensor `src -> dst` type table and total data bytes from dims × bytes-per-block, write nothing, hash nothing | `:288`, `:786-801`, `:844-848` |
| `--leave-output-tensor` | keep `output.weight` in its source type (no quantization of the output head) | `:289`, `:567` |
| `--recipe-file PATH` | load a recipe TOML and **verify** it matches the built-in selection for every class; a mismatch exits 2 | `:290-293`, `:379-435` |
| `--tensor-type RULE=qtype` | force matching tensors to a type; repeatable; unlocks tensors the recipe would leave alone (router, output, norms); RULE is `=exact`, `^prefix`, `suffix$`, a `*` glob, or plain substring (see below) | `:294-306`, `:573-586`, `:646-655` |
| `--tensor-type-trace` | add a `via …` column to the per-tensor table: the rule (with source) or `recipe` each tensor's type came from | `:307`, `:786-801` |
| `--protect FILE` | read `RULE TYPE  # why` lines and apply them as rules BEFORE any `--tensor-type`; repeatable; unknown types and zero-match rules are errors (exit 2) | `:308-311`, `rules.hpp:ParseProtectFile` |
| `--protect-allow-unmatched` | a `--protect` rule matching zero tensors prints a notice instead of failing | `:312`, `:664-686` |
| `--output-tensor-type TYPE` | fork-compatible: type for `output.weight` | `:313-319`, `:627-645` |
| `--token-embedding-type TYPE` | fork-compatible: type for the token embeddings | `:320-326`, `:627-645` |
| `--manifest out.json` | write the receipt JSON (see below); with `--dry-run` the `output` field is null | `:327-330`, `:850-968` |
| `--imatrix FILE` | imatrix-weighted encode (see below) | `:331-334` |
| `--prune-layers LIST` | drop `blk.N` tensors for every N in LIST (commas and/or spaces) | `:335-337`, `:451-527` |
| `--post-pass` | enable the joint scale+code post-pass; **off by default** — on MiMo-9B it lowers the weighted block MSE by construction but raises held-out PPL (7.71 → 8.60), so the default is off | `:339`, `:277-281`, `:440-450` |
| `--no-post-pass` | force the post-pass off (this is the default; the flag exists so a caller can be explicit) | `:340` |

Accepted `--tensor-type` / `--protect` / head-flag values (`TypeNameTable()`,
`rules.hpp`): `F32`, `F16`, `BF16`, `Q4_0`, `Q8_0`, `Q5_1` (v0.8; the encoder
and the recipe's Q5_K shape fallback already knew it, only the CLI table did
not), `Q5_K`, `Q6_K`, `Q4_0_ROCMFP4` (100), `Q4_0_ROCMFP4_FAST` (101),
`G4_Q4` (120), `G4_Q4_FAST` (121) — upper and lower case both accepted.

Float-to-float destinations (v0.8): the only exact conversions are the two
widenings `F16 -> F32` and `BF16 -> F32` (the row arrives dequantized to f32,
so the encoder copies it). Any other float-to-float conversion requested
explicitly (a rule or a head flag — e.g. `=t=BF16`, `=t=F16` on a BF16
source) is rejected in the plan pre-pass on the dry-run AND the real path
with the same message and exit 2, so a dry run never promises a conversion
the real run cannot perform. The recipe's own F16 shape fallback (fork-exact,
e.g. `ple_conv1d` with ne0=4 on real Qwen rosters) still applies: it is
recipe behavior, not a request, and the fork writes F16 there too.

### Tensor-type rules and precedence (v0.5)

A rule is `RULE -> TYPE`. The rule list is the `--protect` rules in file order
followed by the `--tensor-type` rules in command-line order, and the **first
matching rule wins** for each tensor (`rules.hpp:ParseRule`, `Rule::Matches`).

| RULE form | matches | example |
|---|---|---|
| `=name` | exact full-name match (globbing disabled) | `=output.weight` hits ONLY `output.weight` |
| `^prefix` | name starts with `prefix` | `^blk.0.` |
| `suffix$` | name ends with `suffix` | `attn_k.weight$` |
| `a*b*c` | full-name glob (`*` = any run of chars, may be empty, may cross dots) | `blk.*.ssm_alpha.weight`, `*ffn_gate_inp.weight` |
| `plain` | substring match (the pre-v0.5 meaning; kept for compat) | `output.weight` also hits `blk.N.attn_output.weight` |

A `^`/`$` pair may wrap a glob core (redundant — globs already match the full
name — but accepted). `--tensor-type` splits its value at the LAST `=`, so
`=output.weight=Q6_K` parses as rule `=output.weight`, type `Q6_K`; tensor
names never contain `=`, so legacy `name=TYPE` splits exactly as before
(`gufo_quant_main.cpp:294-306`).

Full precedence for one tensor (`:627-662`):

1. the fork head flags (`--output-tensor-type` / `--token-embedding-type`),
   which respect the quantization gate, like the fork;
2. the first matching `--protect` / `--tensor-type` rule, which bypasses the
   gate (the gufo extension the router override relies on);
3. the recipe ladder (`--pure` forces the default type).

`--tensor-type-trace` prints which of these chose each tensor (`via
--output-tensor-type`, `via rule '<text>' <cli|protect:file:line>`, `via
recipe`, `via recipe(pure)`, `via kept`).

### `--protect FILE`

A protect file (e.g. `tools/gufo-quant/protect/qwen35-v0.txt`) holds one `RULE TYPE  # why` rule per line; blank lines and
`#` comments are skipped. The TYPE column accepts the same names as
`--tensor-type`. A malformed line or unknown type exits 2 naming `file:line`,
and a rule matching zero tensors exits 2 unless `--protect-allow-unmatched`
is given (which prints a notice per unmatched rule instead). Match counts are
name matches regardless of precedence: a rule that matches but loses to an
earlier rule still counts as matching. Rules bypass the quantization gate, so
a rule matching a 1-D tensor (norms) or an excluded name (conv1d, router shexp)
WILL quantize it, unlike the fork; keep such rules out of presets.

### `--output-tensor-type` / `--token-embedding-type`

Fork-compatible head flags, with the fork's exact category semantics (verified
against the fork's `llama-quant.cpp`: `tensor_get_category`,
`llama_tensor_get_type`): `--output-tensor-type` applies to OUTPUT
(`output.weight` exactly, plus the fork's NextN mirror
`nextn.shared_head_head`), `--token-embedding-type` to TOKEN_EMBD
(`token_embd.weight` / `per_layer_token_embd.weight` exactly, plus
`nextn.embed_tokens`). Like the fork, the flags apply after the quantization
gate (so `--leave-output-tensor` still keeps `output.weight` at its source
type), before any manual rule, under `--pure` too, and with no shape fallback
(the encode throws on an incompatible `ne0`, as the fork fails). The chosen
types are recorded as `gufo.quant.output_tensor_type` /
`gufo.quant.token_embedding_type` (only when the flag was given).

One deliberate difference: on a tied model (no `output.weight`),
`--output-tensor-type` warns and is a no-op; the fork instead routes it to
`token_embd.weight` through its recipe ladder when `--token-embedding-type`
is unset. The spec'd behavior (warn + no-op) is what ships; the corner is
pinned by `gufo_quant_rules_test` Part 4.

### Recipes

`RecipeNames()` (`:65-78`) accepts: `ROCmFP4`, `LEAN`, `COHERENT`, `FAST`,
`FAST_COHERENT`, `STRIX`, `STRIX_LEAN`, the `_EVEN` aliases `ROCmFP4_EVEN` /
`FAST_EVEN`, and gufo's own `G4_STRIX` / `G4_STRIX_LEAN`. The `_EVEN` names force
`--pure` (`PureRecipes()`, `:79-84`, `:358-375`). Each recipe has a TOML with its
class → type table under `tools/gufo-quant/recipes/` (11 files); the defaults per
recipe come from `RecipeDefault()` (`recipe.hpp:175-186`):

| recipe | GGUF `file_type` | default type | TOML |
|---|---|---|---|
| `ROCmFP4` (100), `LEAN` (101), `COHERENT` (102) | 100, 101, 102 | 100 | `rocmfp4.toml`, `lean.toml`, `coherent.toml` |
| `FAST` | 103 | 101 | `fast.toml` |
| `FAST_COHERENT` | 104 | 101 | `fast_coherent.toml` |
| `STRIX` | 105 | 101, with dual-scale 100 K/V and Q6_K `token_embd` | `strix.toml` |
| `STRIX_LEAN` | 106 | 101 | `strix_lean.toml` |
| `ROCmFP4_EVEN`, `FAST_EVEN` | 100/103, `--pure` | all tensors to the default | `rocmfp4_even.toml`, `fast_even.toml` |
| `G4_STRIX`, `G4_STRIX_LEAN` | written as 105/106, format v2 | 121 (split-scale plane) | `g4_strix.toml`, `g4_strix_lean.toml` |

### `--imatrix`

`--imatrix FILE` loads a **fork-format** imatrix GGUF (`<name>.in_sum2` and
`<name>.counts` tensor pairs) exactly as the fork's `load_imatrix` does —
`LoadImatrix()` at `:104-156`; the per-tensor weight vector is
`sums/counts` per column, and its length must be `ne0 * ne2` (`:719-741`). A
size mismatch throws, except for `token_embd.weight` / `per_layer_token_embd.weight`,
which the fork deliberately leaves unweighted (`:730-740`). A non-finite value
throws (`:149`).

Weighted output records provenance so it is reproducible (`:815-830`):

- `gufo.quant.imatrix.sha256` — sha256 of the exact imatrix file bytes
  (streamed, constant memory: `Sha256File`, `:448`),
- `gufo.quant.post_pass` — `off` or `joint-scale-code`,
- `gufo.quant.imatrix.source` — the path given on the command line.

Every output also carries `gufo.quant.tool` (`gufo-quant 0.8`),
`gufo.quant.recipe` (the name given) and `gufo.quant.format_version` (1, or 2 for
the G4 recipes) — `:803-814`. These keys are ignored by llama.cpp-based loaders. The head flags add `gufo.quant.output_tensor_type` /
`gufo.quant.token_embedding_type` when given (`:831-840`).

### `census` and `--manifest` (v0.5)

`gufo-quant census <file.gguf> [--json]` (`CensusMain`, `:227-250`,
`census.hpp`) prints the header census without touching tensor payloads:
arch/name/basename/finetune, tensor count, the `arch.*` hyperparameters (the
same filter as `~/ls/kit/gguf_census.py`: `block_count|embedding_length|expert
|hc|ssm|feed_forward|head_count$|indexer|ple`), 2-D+ types overall, 2-D types by
family with first-seen shapes, a 1-D type summary, provenance keys
(`gufo.quant.*`, `quantize.*`), total data bytes, element count and BPW
(`8*bytes/elems`, unpadded data bytes). Families are normalized exactly like
the script (`blk.12.` → `blk.N.`, later `.<digits>.` → `.#.`, consumed text not
re-scanned). `--json` emits the same fields as JSON. For a split model, census
each shard separately. The ggml MXFP4 family is sized from its layouts (v0.8:
type 39 MXFP4 17 B/32 elems, 40 NVFP4 18 B/64 elems, 41 Q1_0 36 B/256 elems,
per `qfn_ggml_stubs.h`); id 32 is the generic `T32` (no header in this project
defines it). Remaining unknown-layout types are sized by tensor-offset
differencing against the file size — exact up to alignment slack — so the
census never throws on a well-formed header, and the text names the tensors
it sized that way (`sizes from offset differencing (alignment padding
included): …`).

`--manifest out.json` writes the receipt JSON on every quantize (`:850-968`):
tool version, recipe (+ ftype, `pure`, `dry_run`), threads, wall and hash
seconds, inputs (every shard: path, sha256, bytes), imatrix (path, sha256),
prune list, post-pass mode, head flags, rules applied (rule, source, kind,
type, match counts), the per-family OUTPUT type table, and the output (path,
sha256, bytes, tensor count, data bytes, BPW). Input/output hashes are each
one sequential streaming pass (`Sha256File`, 64 MiB chunks, constant memory),
counted in the same pass — never a whole-file read, never a second pass.
Under `--dry-run` the manifest records input sizes from stat with `sha256`
null plus a `hash_note` saying the dry-run hashed nothing (dry-run reads
headers only: no tensor payload reads, no hashing).

v0.7: the input shard hashes run as concurrent side streams of the encode
(one sequential pass per shard, finished before the manifest is written) and
compress with SHA-NI (`sha256rnds2`/`sha256msg1`/`sha256msg2`) when the CPU
has the SHA extensions — every AMD Zen core, Intel Ice Lake and later — which
is >1.5 GB/s per thread instead of the portable block's ~0.2 GB/s. A single
SHA-256 stream is serial by construction, so this is what keeps a 55 GB
manifest's hash inside the encode's wall. Digests are identical on both paths
(FIPS known-answer vectors run in `gufo_quant_stream_test` Part C on every
ctest); `GQT_SHA256_NO_NI=1` forces the portable block for debugging.

### `--prune-layers`, split inputs, shards

`--prune-layers 64` drops every `blk.64.*` tensor and rewrites every
`block_count` key to one past the highest remaining layer (`:470-502`). If the
prune removes every `nextn.` tensor, the `*nextn_predict_layers` key is dropped
too (`:503-527`) — leaving it set makes the loader require
`blk.<n_layer-1>.nextn.eh_proj.weight` and the file will not load. The served
27B file omits that key.

A split model is passed as its **first shard** (`…-00001-of-00002.gguf`).
`LoadGgufStreaming()` (`gguf.hpp:306-341`) reads each shard header, follows
`split.no` / `split.count` / `split.tensors.count` and computes each shard's own
data base; `FetchTensorData()` (`gguf.hpp:343-357`) seeks to exactly one tensor
at a time, so a 54 GB shard is never loaded whole. The merged single output
drops the three `split.*` keys (`gufo_quant_main.cpp:529-540`), which is what the
fork's `llama-quantize` does.

### Output writer

`WriteGguf()` (`gguf.hpp:467-569`) reproduces the fork's `gguf.cpp` layout:
magic/version/counts, KV in input order with `general.quantization_version=2`
and `general.file_type` replaced in place, the `gufo.quant.*` keys appended,
tensor infos with offsets packed at the file alignment, then each payload padded
to the alignment. Tensor payloads are **streamed straight to disk**; only the
header is buffered (`gguf.hpp:550-568`). The byte layout is unchanged by the
streaming — see the test below.

### Failure behavior, one writer per output, and no leftovers (v0.9.1)

A failed run leaves nothing behind, a run never hangs, and a run leaves no
stray claim file. The quantizer claims the output first: it opens/creates
`<out>.gq-lock`, takes a non-blocking exclusive `flock` on it, and **proves the
descriptor it locked is still the file the path names** (`fstat` on the fd vs
`stat` on the path, dev+ino). If a previous owner unlinked or replaced the path
between the open and the `flock`, the lock is on an orphaned inode, so the tool
closes and retries on a fresh file. A live holder (the `flock` fails) is a
refusal: `error: <out> is being written by pid N`, exit 2, and the holder's file
is left untouched. The model then streams to a per-run unique
`<out>.partial.<pid>` (leftovers of dead runs and the old fixed-name
`<out>.partial` are removed once the claim is held), `fsync`s, and is renamed to
the final name only after a fully successful close; the manifest (when asked)
goes to its own `<manifest>.partial.<pid>` and is renamed in the same sequence,
then each containing directory is `fsync`ed. Every handled failure — write
error, short write, `ENOSPC`, `EFBIG` (including a process file-size limit, which
arrives as a failed write because `SIGXFSZ` is ignored), encode error, input
error, a throwing worker (which wakes the writer instead of hanging it) —
unlinks the partials, joins the fetch thread and the hash threads, and ends in
`return 2` from `main` with one `error: …` line on stderr: no escaping
exception, no exit 134.

When the run ends — after success and after every handled failure — the claim's
destructor **unlinks `<out>.gq-lock` while it still holds the lock, then
closes**. Unlinking under the lock is what keeps it safe: a peer that opened the
old file before the unlink acquires the lock on the now-orphaned inode only
after we release it, then its own `fstat`/`stat` check mismatches and it retries
on the fresh file; a peer that opens after the unlink sees no file (or a fresh
one) and claims cleanly. The path is unlinked only while it still names our
inode, so a peer's file is never removed. A run killed by a signal (`SIGKILL`,
OOM) leaves the file but no lock (the kernel drops it); the next run to that
output takes it over cleanly and removes it at its own exit. So a successful
run leaves no `.partial` and no `.gq-lock`, and a build of several variants
leaves no stray lock beside any output. Pinned by `gufo_quant_failure_test`
(below), whose parts (a)–(h) are described there.

## kv_swap

```
kv_swap <template.gguf> <data.gguf> <out.gguf> [--no-provenance]
```

`kv_swap` writes an output GGUF whose KV/metadata block is the **template's**
and whose tensor payloads are the **data file's**, byte for byte
(`kv_swap.cpp:9-21`, `:23-57`). The two inputs must have the same tensors in the
same order with the same names, dims and types; only the payload bytes may
differ. `--no-provenance` (parsed at `:103`) writes the bare template header with
no `gufo.quant.*` keys.

Validation, all before anything is written:

- tensor count mismatch → exit 4 (`kv_swap.cpp:123-127`),
- name/order mismatch → exit 4 (`:128-136`),
- dims mismatch → exit 4 (`:137-144`),
- type mismatch → exit 4 (`:145-151`),
- data-region length mismatch → exit 4 (`:152-158`),
- the tool then **re-serializes the template header and aborts unless it is
  byte-identical to the template's own header** (`HEADER_VERBATIM=yes`,
  `:162-177`).

Only after all of that does it open the output and stream the data region in
64 MiB chunks (`:180-206`). Without `--no-provenance` it appends
`gufo.quant.tool`, `gufo.quant.kv_template`, `gufo.quant.tensor_source` and
`gufo.quant.format_version` (`:184-190`).

Exit codes: 2 usage / unknown option, 3 cannot open an input, 4 validation
failure, 5 header self-check failed, 6 cannot write, 7 read/write failure.

## The two served-KV drop-ins

Both files are the served file with exactly one thing changed: the tensor
payloads are the imatrix-weighted ones. The KV block — chat template, BOS and
padding tokens, `general.*` — is the served file's.

**Qwen3.8-27B**:

```sh
# 1. imatrix-weighted STRIX of the two-shard BF16, blk.64 (the nextn head) pruned
gufo-quant --imatrix /models/qwen38-27b-bf16/imatrix_unsloth.gguf \
           --prune-layers 64 \
           /models/qwen38-27b-bf16/BF16/Qwen3.8-27B-BF16-00001-of-00002.gguf \
           /models/qwen38-27b-weighted/Qwen3.8-27B-Q4_0_ROCMFP4_STRIX-imatrix.gguf \
           STRIX 16

# 2. the served file's KV block + the weighted payloads
kv_swap /models/Qwen3.8-27B-Q4_0_ROCMFP4_STRIX.gguf \
        /models/qwen38-27b-weighted/Qwen3.8-27B-Q4_0_ROCMFP4_STRIX-imatrix.gguf \
        /models/qwen38-27b-weighted/Qwen3.8-27B-Q4_0_ROCMFP4_STRIX-imatrix-servedkv.gguf
```

**Muse-Glimmer-30B**. The served 103 is `Q4_0_ROCMFP4_FAST_EVEN`; the
BF16 ships as two shards:

```sh
# 1. imatrix-weighted FAST_EVEN
gufo-quant --imatrix /models/muse-a7-quants/muse-agentic-imatrix.gguf \
           /models/muse-glimmer-30b-bf16/BF16/Muse-Glimmer-30B-BF16-00001-of-00002.gguf \
           /models/muse-glimmer-30b-weighted/muse-glimmer-30B-ROCmFP4-FAST-weighted.gguf \
           FAST_EVEN 16

# 2. the served 103's KV block + the weighted payloads
kv_swap /models/muse-glimmer-30b-served/muse-glimmer-30B-ROCmFP4-103.gguf \
        /models/muse-glimmer-30b-weighted/muse-glimmer-30B-ROCmFP4-FAST-weighted.gguf \
        /models/muse-glimmer-30b-weighted/muse-glimmer-30B-ROCmFP4-FAST-weighted-servedkv.gguf
```

The 27B served-KV file is `e64d7a49…`; the Glimmer weighted file is `3b54c059…`
and its served-KV variant `9403dde5…` 

## Tests

Four CTest cases, `cpu` label, registered by the top-level `CMakeLists.txt`:

- **`gufo_quant_stream_test`** (`tests/tools/gufo_quant_stream_test.cpp`) —
  writes a fixed-payload fixture with `WriteGguf`, re-parses it and checks the
  KV order/values, the tensor infos, the per-tensor offsets, the zero padding
  and the data-region bytes, then pins the file's sha256; and runs the built
  `gufo-quant` twice on a deterministic synth source to prove the writer is
  deterministic. The pinned fixture has literal payload bytes and an
  integer-only layout, so its sha256 is toolchain-independent.
- **`gufo_quant_rules_test`** (`tests/tools/gufo_quant_rules_test.cpp`, v0.5) —
  in-process rule matching (exact/anchor/glob/substring, the `output.weight`
  vs `attn_output.weight` case), protect-file parsing (comments, `RULE TYPE
  # why`, unknown-type and malformed-line errors), census helpers (family
  normalization pinned to `gguf_census.py`, per-type byte sizes), and the built
  `gufo-quant` binary on a tiny deterministic BF16 synth: `=exact` vs
  substring, first-match-wins, `--protect` precedence and the zero-match error
  (+ `--protect-allow-unmatched`), the fork head flags (tied-model no-op,
  `--leave-output-tensor` gating), `--tensor-type-trace`,
  `census` text/JSON, and `--manifest` (input/output sha re-verified in-test).
- **`gufo_quant_failure_test`** (`tests/tools/gufo_quant_failure_test.cpp`,
  v0.9.1) — the pool error callback wakes a waiter on the caller's own cv
  (in-process), and the built binary end to end: (a) `RLIMIT_FSIZE` below the
  output size → exit 2 with neither the final path nor the partial left; (b)
  the mid-encode worker-throw case (test-only hooks binary) finishes within the
  bound with exit 2 and no output; (c) a truncated input → exit 2, no output;
  (d) an output path that is a directory → exit 2; (e) a rejected float
  conversion gives the same message and exit code on dry-run and real; (f) a
  successful run leaves no `.partial`; **(g)** a `<out>.gq-lock` left by a
  `SIGKILL`ed run is taken over and removed; **(h)** the loser of a race is
  refused with the writer's pid and leaves the winner's lock file untouched.
  Parts (a)–(f) each also assert **no `.gq-lock`** survives; (a)–(e) fail on the
  v0.7 binary, (b) fails on v0.8, and (g)/(h) fail on v0.9.
- **`kv_swap_test`** (`tests/tools/kv_swap_test.cpp`) — the `kv_swap` success
  case (KV from the template, data from the data file, `HEADER_VERBATIM=yes`,
  the data file's own KV must not leak in), `--no-provenance` byte-for-byte, and
  the five refusal cases above, each asserted to exit non-zero, print the
  specific mismatch and leave no output file.

```sh
ctest -R 'gufo_quant|kv_swap' --output-on-failure -V
```
