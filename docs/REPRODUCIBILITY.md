# Reproducibility

With `SUPERROC_CANONICAL_CACHE=1` a conversation gets the same reply whether
the server saw it turn by turn, through the prompt cache, or reads the whole
history in one request on a freshly started server. Without the switch the
server behaves as before.

This page says what was measured and how to run the same check. How the
switch works, what it costs and what it does not cover is in "The same reply
for the same history" in [the cache guide](KV-CACHE.md).

## What was measured

Temperature 0, seed 1, thinking off, at most 400 tokens per reply, on a
Radeon 8060S. Six agent tasks of six turns each are served turn by turn; then
a freshly started server gets every turn's whole history in one request, and
the two replies are compared.

| Model | Context | Same reply, default | Same reply, switch |
| --- | --- | ---: | ---: |
| Qwen3.8-27B-UD-Q4_K_XL | short | 35 of 36 | 36 of 36 |
| | 4K tokens | 34 of 36 | 36 of 36 |
| Qwen3.8-27B ROCmFP4, DFlash2 draft and prompt lookup | 4K tokens | 31 of 36 | 36 of 36 |
| Qwen3.8-Flash-Next ROCmFP4, 7 routed experts | short | 32 of 36 | 36 of 36 |
| | 4K tokens | 29 of 36 | 36 of 36 |
| Qwen3.8-Flash-Next-UD-Q4_K_XL | 4K tokens | 28 of 36 | 36 of 36 |
| KAT-Coder-V2.5-Dev ROCmFP4 | short | 27 of 36 | 36 of 36 |
| | 4K tokens | 29 of 36 | 36 of 36 |
| Nanbeige4.3-33B-Base Q4_K_M | short | 33 of 36 | 36 of 36 |

A raw prompt that grows by 1 to 12 tokens per request, each reply compared
with a fresh server's: 21 of 21 steps agree on all six files with the switch.
KAT-Coder-V2.5-Dev without the switch: 14 of 21.

## How to check it yourself

Two tools live in the tree:

- [`tools/repro/live_vs_cold.py`](../tools/repro/live_vs_cold.py) runs the 36
  agent turns: turn by turn against one server, then each turn's history
  against a fresh server.
- [`tools/repro/growing_prompt.py`](../tools/repro/growing_prompt.py) runs the
  growing raw prompt, each step against a fresh server.

Both send the same request bodies every time. Both print a summary line,
write a JSON file, and exit 0 when every reply matches and 1 when at least
one differs.

### Against Super Roc

Put the server command in a file, one line. The line may start with
`NAME=VALUE` environment pairs; they are passed to the server.

```
SUPERROC_CANONICAL_CACHE=1 /path/to/superroc serve -i 127.0.0.1 -p 18904 -j 1 llm -m /path/to/model.gguf --served-model-name g --context 32768 -t 0 --presence-penalty 0
```

Then run the tools with that file:

```
python3 tools/repro/live_vs_cold.py --line-file serve.line --port 18904 --model-name g --out live_vs_cold.json
python3 tools/repro/growing_prompt.py --line-file serve.line --port 18904 --model-name g --out growing_prompt.json
```

The tools start and stop the servers themselves, one at a time.
`--server-cmd` takes the same command as one string instead of a file. Run
once with `SUPERROC_CANONICAL_CACHE=1` in the line and once without it, and
compare the two summary lines.

### Against another server

Any OpenAI-compatible server can be tested with the same tools. Give the
commands that start and stop it:

```
python3 tools/repro/live_vs_cold.py --other-server-start "./start-server.sh" --other-server-stop "./stop-server.sh" --port 8080 --model-name some-model --out live_vs_cold.json
```

The start command has to leave the server running in the background; the
tool waits for it to answer before it sends anything. The tool sends no
server flags in this mode, only the request bodies.

## How to read the result

The last line of a run is the summary:

```
REPRO same reply live vs cold: 36 of 36 turns (PASS)
REPRO same reply live vs cold: 21 of 21 steps (PASS)
```

`growing_prompt.py` also prints how many steps reused the cache. The JSON
file holds one row per turn or step: the token counts of both arms, the
first 10 characters of the SHA-256 of each reply, the character where the
two replies first differ, and whether they match. The server logs are
written next to it, one per server: `<out>.srv-live.log` and
`<out>.srv-cold.log`.

## What a server says about itself

- `GET /props` returns `"canonical_cache": true` when the switch is on. It
  says the switch is on. It does not say the loaded model family is covered;
  the cache guide lists the covered families.
- The request log line carries `canonical=1` when the switch is on.
