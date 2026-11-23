# Model recipes

A model carries its own tuned setup in a `superroc-recipe.json` file next to it
(usually on Hugging Face, beside the GGUF). The fastest setup for a model often
needs settings that are not in the GGUF: engine variables, sidecar files and
matching `serve llm` flags. A launcher that only knows the GGUF gets none of
that; the recipe file ships the tuned setup with the model so any launcher gets
it.

## Use

```sh
superroc serve -i 127.0.0.1 -p 8080 -j 1 llm -m model.gguf --recipe auto
superroc serve -i 127.0.0.1 -p 8080 -j 1 llm -m model.gguf --recipe tuned.json
```

`--recipe off` is the default: every existing command behaves byte-identically
without the flag. `--recipe auto` loads `superroc-recipe.json` from the
directory containing `-m`. `--recipe <path>` loads that file.

## Schema

Schema tag `"superroc-recipe/1"`:

```json
{
  "schema": "superroc-recipe/1",
  "model": "Qwen3.8-Flash-Next-Q4_0-ROCmFP4-STRIX_LEAN-imatrix-hcq8-ssmf32.gguf",
  "env": {"SUPERROC_ROUTED_EXPERTS": "7", "LFN_MTP_P_MIN": "0.5", "SUPERROC_TAIL_CAPTURE": "64"},
  "files": {"SUPERROC_TENSOR_OVERLAY": "Qwen3.8-Flash-Next-f3-overlay.gguf"},
  "file_args": {"--mtp-model": "mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf", "--mmproj": "mmproj-Qwen3.8-Flash-Next-Uncensored-BF16.gguf"},
  "args": ["--speculative", "mtp", "--think", "on", "--reasoning-effort", "low", "-t", "0", "--presence-penalty", "0", "--draft-tokens", "5"],
  "context": 131072,
  "notes": "Flash-Next tuned setup; every name is relative to this file's directory"
}
```

- `model` must equal the basename of `-m`, else the server refuses to start.
- `env` keys must match `^(SUPERROC|LFN|LING3)_[A-Z0-9_]+$`, else the server
  refuses. They are exported with `setenv(overwrite=0)`, so a variable set on
  the command line or in the environment wins over the recipe.
- `env` values are settings, never paths: 1 to 256 letters, digits and
  `_ . , : + -` only, not starting with a dot. Anything else, an empty value
  included, refuses to start.
- `files` keys are the engine's input-file variables, `SUPERROC_INT4_WEIGHTS`
  and `SUPERROC_TENSOR_OVERLAY`. Values are paths relative to the recipe's
  directory (absolute paths and any `..` component refuse to start), must
  name an existing regular file (a directory is not; a symbolic link only
  when it stays inside the recipe's folder), and are exported as environment
  variables with `overwrite=0` holding the absolute path.
- `args` are appended to the `llm` subcommand only for flags not already on
  the command line (the command line wins). A recipe may set these options,
  in either spelling, and no others: `--speculative`, `--draft-policy`,
  `-d`/`--draft-tokens`, `--min-draft-tokens`, `--prompt-lookup`,
  `--prefill-chunk`, `--think`, `--reasoning-effort`, `--reasoning-budget`,
  `--preserve-thinking`, `--chat-template-kwargs`, and the sampling defaults
  `-t`/`--temperature`, `--top-k`, `--top-p`, `--min-p`, `--min-keep`,
  `--repeat-penalty`, `--repeat-last-n`, `--presence-penalty`,
  `--frequency-penalty`, `-s`/`--seed`. Any other flag refuses to start, and
  so does an `args` value that is an absolute path.
- `file_args` maps an `llm` model/file flag to a sidecar file relative to the
  recipe's directory (absolute paths and any `..` component refuse to
  start), which must name an existing regular file. Allowed flags are the
  `llm` sidecar options `--mtp-model`, `--dflash-model`, `--dspark-model`
  and `--mmproj`. At load each value is resolved against the recipe's
  directory and appended as `<flag> <abs path>` unless the flag is already
  on the command line (the command line wins, logged like the other args).
  `-m`/`--model` is not allowed here: the model identity comes from
  the `model` field plus `-m`. The launcher-owned outputs `--cache-disk`
  and `--trace` are refused here (as in `args`).
- `context` becomes `--context` when `--context`/`-c` is absent.
- Unknown top-level keys refuse to start, so typos cannot pass silently.

## What a recipe cannot do

A recipe is downloaded with a model, so it is limited to tuning:

- It cannot set the listen address or port (`SUPERROC_HOST`, `SUPERROC_PORT`
  and the matching flags), sessions, API keys, the model, the served name, or
  the trace and disk-cache paths.
- It cannot set a limit, a timeout, a cache budget or the log level: the
  connection, request, queue and output limits, `--max-tokens`, the request
  and stall timeouts, the cache sizes and `--log-level`/`--verbose` belong to
  the command line. `args` accepts only the options listed above, so an
  option added to the server later is refused until it is listed.
- It cannot set a variable whose value names a program to run or a file the
  engine writes (the diagnostic dump and capture variables, the plan cache,
  the media tool overrides), or the settings of `superroc eval`. Such a key
  refuses to start, in `env` and in `files`.
- It cannot turn on diagnostics: a variable whose name holds the word
  `TRACE`, `DUMP`, `DEBUG`, `PROF`, `PROFILE`, `TIMING`, `VERBOSE` or `CHECK`
  refuses to start, so a recipe cannot make the server print traces, timings
  or prompt tokens.
- An `env` value cannot be a path, and `files` accepts only the input-file
  variables, so no variable can be pointed at a file outside the recipe's
  directory. A sidecar may be a symbolic link, but only one that stays inside
  the recipe's folder; in a Hugging Face cache snapshot, where every file is a
  link into the repository's `blobs` folder, the limit is the repository
  folder. A link that leads anywhere else refuses to start.
- The recipe file itself must be a regular file of at most 1 MiB.
- Only `serve ... llm` reads a recipe.
- A variable this build does not read is skipped and reported, in the startup
  log (`[recipe] ignored NAME (this build does not read it)`) and under
  `ignored` in `GET /props`. A recipe written for a newer engine therefore
  still loads on an older one, and a misspelled name is visible.

## Precedence and startup

The recipe is applied before any `SUPERROC_`/`LFN_`/`LING3_` variable is first read
(the server falls back to `SUPERROC_HOST`/`SUPERROC_PORT` defaults before the
`llm` options are parsed, and several engine readers cache the environment on
first call), via a pre-scan of `argv` at the top of `main()`. The serve path
then re-validates, merges the flags, and logs one line per applied or skipped
item at startup:

```text
[recipe] env SUPERROC_TAIL_CAPTURE=64
[recipe] file SUPERROC_TENSOR_OVERLAY=/models/...-overlay.gguf
[recipe] arg --speculative mtp
[recipe] arg --mtp-model /models/mtp-....gguf
[recipe] kept command-line --context
```

`[recipe] kept env X` means `X` was already set and the recipe left it alone.
`[recipe] ignored X` means this build does not read `X`.

## Introspection

`GET /props` reports the applied recipe:

```json
{"recipe": {"file": "...", "env": {...}, "files": {...}, "file_args": {...}, "args": [...], "ignored": [...]}}
```

With `--recipe off` it reports `"recipe": {}`.

## Generating a recipe

`tools/recipe/command_to_recipe.py` converts a server command line (environment
assignments, binary, `serve ... llm ...` arguments) into a recipe file:

```sh
python3 tools/recipe/command_to_recipe.py serve.line superroc-recipe.json
```

`SUPERROC_`/`LFN_`/`LING3_` assignments become `env`, except the input-file
variables (`SUPERROC_INT4_WEIGHTS`, `SUPERROC_TENSOR_OVERLAY`), which become
`files` with the basename. Variables a recipe cannot set, and variables this
source tree does not read, are dropped and listed on stderr. The `llm`
arguments a recipe may set become `args`; every other flag is dropped with its
value and listed on stderr. `--context` becomes `context`, the file-taking
flags (`--mtp-model`, `--dflash-model`, `--dspark-model`, `--mmproj`) become
`file_args` with the basename, and the model is the basename of `-m`.
`LD_LIBRARY_PATH` and other non-engine assignments are dropped and listed on
stderr, which also lists each `file_args` entry as
`<flag> needs '<basename>' beside the model`.
