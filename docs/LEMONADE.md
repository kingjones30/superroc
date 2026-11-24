# Super Roc in Lemonade

This runs Super Roc models inside a stock Lemonade install — no Lemonade
rebuild, no fork.

## What it does

Lemonade can be pointed at a custom server for its llama.cpp backend
(`llamacpp.rocm_bin`). The installer puts a small shim there in place of
`llama-server`. When Lemonade loads a model, the shim reads the GGUF
header:

- a model family Super Roc runs is served by `superroc serve`;
- anything else — another architecture, embeddings, reranking, a flag the
  shim does not map — is handed to the stock `llama-server` with the
  original arguments, unchanged.

So one Lemonade keeps serving every model it served before; Super Roc
models just run on the faster engine.

## Requirements

- AMD Strix Halo (gfx1151) machine with ROCm installed.
- Lemonade, stock release.
- A `superroc` binary (build below).
- `python3` (the shim is a single stdlib-only script).

## One-step setup (optional)

`tools/lemonade/setup.sh` does everything below in one run, with Lemonade
running:

```sh
tools/lemonade/setup.sh
```

1. Checks the machine: Linux x86-64, an AMD `gfx1151` GPU, the build tools.
2. Builds Super Roc with the `release` preset (skipped with
   `--superroc PATH_TO_BINARY`).
3. Installs it under `/opt/superroc` (`--prefix DIR` for another place); it
   uses sudo for this step only when that folder is not writable.
4. Points the running Lemonade at the launcher and adds the Super Roc model
   list, keeping any model lists already configured.
5. With `--model NAME`, downloads that model through Lemonade.

Every step is safe to repeat. Options:

| Option | Effect |
| --- | --- |
| `--dry-run` | Print the steps; change nothing. |
| `--undo` | Switch Lemonade back to its own llama-server and remove the Super Roc model list. |
| `--lemonade-url URL` | Lemonade server to configure (default `http://localhost:13305`). |
| `--no-lemonade` | Build and install only. |
| `--jobs N` | Parallel build jobs (default 4). |
| `--force` | Continue when no `gfx1151` GPU is found. |

A Lemonade server that requires a key: set `LEMONADE_ADMIN_API_KEY` (or
`LEMONADE_API_KEY`) first. If Lemonade is not running, the script finishes the
build and install and prints the command that completes the setup later.

## Install by hand

Build Super Roc (see the README for toolchain details):

```sh
cmake --preset release
cmake --build --preset release --parallel 4
```

Install the binary and the shim:

```sh
tools/lemonade/install.sh --superroc ./build/release/superroc --prefix /opt/superroc
```

This copies the binary to `/opt/superroc/bin/superroc` and the shim to
`/opt/superroc/lemonade/llama-server`, then prints the next two commands:

```sh
lemonade config set llamacpp.rocm_bin=/opt/superroc/lemonade
```

Add the Super Roc model list, either in Lemonade's `config.json`:

```json
"model_catalogs": [
  {"id": "superroc", "url": "https://raw.githubusercontent.com/kingjones30/superroc/main/docs/lemonade/catalog.json"}
]
```

or with the running server:

```sh
curl -X POST http://localhost:13305/internal/set \
  -H 'Content-Type: application/json' \
  -d '{"model_catalogs": [{"id": "superroc", "url": "https://raw.githubusercontent.com/kingjones30/superroc/main/docs/lemonade/catalog.json"}]}'
```

## Run a model

```sh
lemonade pull LFM2.5-1.2B-Instruct
lemonade chat LFM2.5-1.2B-Instruct
```

Chat, completions and streaming go through Lemonade as usual. Load the
model with the ROCm backend and the shim routes it to Super Roc.

Per-model `superroc serve` options live in
`/opt/superroc/lemonade/superroc-models.json` (GGUF file name → argument
list); edit and reload the model to change them. This file is
operator-trusted configuration: anyone who can edit it chooses serve
options, so the shim drops its listener/model/auth options (`--host`,
`--port`, `--api-key`, `-m` and friends) and warns — those stay under
Lemonade's and the shim's control.

## Switch back

```sh
lemonade config set llamacpp.rocm_bin=builtin
```

or `tools/lemonade/setup.sh --undo`, which also removes the Super Roc model list.

That returns Lemonade to its bundled `llama-server` for every model.

## Notes

A native Super Roc backend for Lemonade is proposed upstream. Until it
lands, this shim route works with stock Lemonade today.

Embeddings and reranking models always use the stock server, as do models
whose GGUF architecture Super Roc does not list.

## Troubleshooting

See what the shim would run, without running it:

```sh
SUPERROC_SHIM_DRYRUN=1 /opt/superroc/lemonade/llama-server -m MODEL.gguf --port 8080
```

It prints one JSON line: `{"mode": "superroc" | "fallback", "reason": ...,
"argv": [...]}`.

Useful environment variables:

- `SUPERROC_BIN` — the `superroc` binary (default: `../bin/superroc`
  relative to the shim, then `PATH`).
- `SUPERROC_FALLBACK_LLAMA_SERVER` — the stock `llama-server` (default:
  the ROCm/Vulkan server in Lemonade's cache, then `PATH`).
- `SUPERROC_EXTRA_ARGS` — extra arguments appended to every
  `superroc serve` command.

`super-roc shim: no stock llama-server for ...` means a model Super Roc
does not run was requested but no stock server was found; let Lemonade
install its llama.cpp backend once, or set
`SUPERROC_FALLBACK_LLAMA_SERVER`.
