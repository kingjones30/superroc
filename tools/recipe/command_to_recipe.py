#!/usr/bin/env python3
"""Generate a superroc-recipe/1 JSON file from a server command line.

Usage: command_to_recipe.py LINE_FILE OUT_JSON

LINE_FILE holds one shell command line: leading VAR=value assignments, the
server binary, then `serve ... llm ...` arguments. The recipe it writes carries
the tuned setup next to the model on Hugging Face:

- SUPERROC_/LFN_/LING3_ assignments become `env`, except the input-file
  variables (SUPERROC_INT4_WEIGHTS, SUPERROC_TENSOR_OVERLAY), which become
  `files` with the basename of the path. Variables a recipe cannot set and
  variables this source tree does not read are dropped.
- llm arguments a recipe may set (speculative decoding, prompt lookup,
  prefill chunking, thinking and sampling defaults) become `args`. Every
  other option stays with the command line and is dropped with its value;
  --context becomes `context`; and the file-taking flags
  --mmproj/--dflash-model/--dspark-model/--mtp-model become `file_args` with
  the basename, so the recipe never carries a host path; model is
  basename(-m).
- LD_LIBRARY_PATH and any other non-engine assignment are dropped; each
  dropped name is listed on stderr.
"""

import json
import os
import re
import shlex
import sys

SCHEMA = "superroc-recipe/1"
ENGINE_KEY = re.compile(r"^(SUPERROC|LFN|LING3)_[A-Z0-9_]+$")
# Mirrors src/cli/serve/recipe.cpp: what `files` may set, what a recipe may
# never set, and what an `env` value may look like.
INPUT_FILE_ENV = {"SUPERROC_INT4_WEIGHTS", "SUPERROC_TENSOR_OVERLAY"}
LAUNCHER_ENV = {"SUPERROC_HOST", "SUPERROC_PORT"}
REFUSED_ENV = {
    "LING3_KDA_CAPTURE",
    "SUPERROC_FFMPEG",
    "SUPERROC_FFPROBE",
    "SUPERROC_HIPBLASLT_PLAN_CACHE",
    "SUPERROC_MTP_KV_DUMP",
    "SUPERROC_MTP_TF_DUMP",
    "SUPERROC_MTP_TF_LOGITS",
}
PLAIN_SETTING = re.compile(r"^(?!\.)[A-Za-z0-9_.,:+-]{1,256}$")
# Diagnostic switches (IsDiagnosticEnvKey): whole words of the name.
DIAGNOSTIC_WORDS = {"CHECK", "DEBUG", "DUMP", "PROF", "PROFILE", "TIMING",
                    "TRACE", "VERBOSE"}


def is_diagnostic(key):
    return (key.startswith("LING3_KDA_CAPTURE")
            or any(word in DIAGNOSTIC_WORDS for word in key.split("_")))


def known_env_keys():
    """The variables this source tree reads, or None outside a checkout."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                        "src", "cli", "serve", "recipe_env_keys.inc")
    try:
        with open(path) as handle:
            return set(re.findall(r'"([A-Z0-9_]+)"', handle.read()))
    except OSError:
        return None
# The llm options `args` may set, in both spellings (IsRecipeArgFlag in
# src/cli/serve/recipe.cpp). Anything else on the command line is dropped with
# its value: the model, server identity, secrets, write paths, limits,
# timeouts, cache budgets and logging stay with the command line.
RECIPE_FLAGS = {
    "--chat-template-kwargs", "--draft-policy", "-d", "--draft-tokens",
    "--frequency-penalty", "--min-draft-tokens", "--min-keep", "--min-p",
    "--prefill-chunk", "--presence-penalty", "--preserve-thinking",
    "--prompt-lookup", "--reasoning-budget", "--reasoning-effort",
    "--repeat-last-n", "--repeat-penalty", "-s", "--seed", "--speculative",
    "-t", "--temperature", "--think", "--top-k", "--top-p",
}
# llm options that take a model/file path (from the llm parser in
# src/cli/serve/serve.cpp: --mmproj, --dflash-model, --dspark-model,
# --mtp-model). These become `file_args` with the basename, so the recipe
# never carries a host path.
FILE_FLAGS = {
    "--mmproj", "--dflash-model", "--dspark-model", "--mtp-model",
}
# Looks like a negative number, i.e. a value even though it starts with '-'.
NEGATIVE_NUMBER = re.compile(r"^-\d+(\.\d+)?$")


def is_value(token):
    return not token.startswith("-") or NEGATIVE_NUMBER.match(token) is not None


def split_flag(token):
    """Return (flag, inline_value_or_None) for --flag or --flag=value."""
    if token.startswith("--") and "=" in token:
        flag, _, value = token.partition("=")
        return flag, value
    return token, None


def main():
    if len(sys.argv) != 3:
        print("usage: command_to_recipe.py LINE_FILE OUT_JSON", file=sys.stderr)
        return 2
    line_file, out_json = sys.argv[1], sys.argv[2]
    with open(line_file) as handle:
        text = handle.read().strip()
    line = next((part for part in text.splitlines() if part.strip()), "")
    if not line:
        print(f"command_to_recipe: no command line in {line_file}", file=sys.stderr)
        return 2
    tokens = shlex.split(line)

    assignments = {}
    rest = list(tokens)
    while rest and "=" in rest[0] and not rest[0].startswith("-"):
        key, _, value = rest.pop(0).partition("=")
        assignments[key] = value
    # rest[0] is the server binary; find the llm subcommand after `serve`.
    try:
        llm_at = rest.index("llm")
    except ValueError:
        print("command_to_recipe: no `llm` subcommand on the command line", file=sys.stderr)
        return 2
    llm_args = rest[llm_at + 1:]

    # Model: value of -m/--model on the llm argument list.
    model_path = None
    scan = list(llm_args)
    while scan:
        token = scan.pop(0)
        flag, inline = split_flag(token)
        if flag in ("-m", "--model"):
            model_path = inline if inline is not None else (scan.pop(0) if scan else None)
    if not model_path:
        print("command_to_recipe: no -m/--model on the command line", file=sys.stderr)
        return 2

    # llm flags: lift --context, lift file-taking flags to file_args, keep
    # what a recipe may set and drop the rest.
    args = []
    file_args = {}
    context = None
    dropped_flags = []
    scan = list(llm_args)
    while scan:
        token = scan.pop(0)
        flag, inline = split_flag(token)
        if flag in ("-c", "--context"):
            value = inline if inline is not None else (scan.pop(0) if scan and is_value(scan[0]) else None)
            if value is not None:
                context = int(value)
            continue
        if flag in FILE_FLAGS:
            value = inline if inline is not None else (scan.pop(0) if scan and is_value(scan[0]) else None)
            if value is not None:
                file_args[flag] = os.path.basename(value)
            continue
        if flag not in RECIPE_FLAGS:
            if inline is None and scan and is_value(scan[0]):
                scan.pop(0)
            if flag not in ("-m", "--model") and flag not in dropped_flags:
                dropped_flags.append(flag)
            continue
        args.append(token)
        if token.startswith("-") and inline is None and scan and is_value(scan[0]):
            args.append(scan.pop(0))

    # Environment: engine keys split into env vs files; the rest is dropped.
    # The loader refuses what a recipe cannot set and skips what the engine
    # does not read, so neither is written here.
    known = known_env_keys()
    env, files, dropped = {}, {}, []
    for key, value in assignments.items():
        if not ENGINE_KEY.match(key):
            dropped.append((key, "not an engine variable"))
        elif key in LAUNCHER_ENV or key in REFUSED_ENV or key.startswith("SUPERROC_EVAL_"):
            dropped.append((key, "a recipe cannot set it"))
        elif is_diagnostic(key):
            dropped.append((key, "a diagnostic switch; a recipe cannot set it"))
        elif known is not None and key not in known:
            dropped.append((key, "this source tree does not read it"))
        elif key in INPUT_FILE_ENV:
            files[key] = os.path.basename(value)
        elif not PLAIN_SETTING.match(value):
            dropped.append((key, "its value is not a plain setting"))
        else:
            env[key] = value
    for key, reason in dropped:
        print(f"command_to_recipe: dropped {key} ({reason})", file=sys.stderr)
    for flag in dropped_flags:
        print(f"command_to_recipe: dropped {flag} (the command line owns it)",
              file=sys.stderr)

    recipe = {
        "schema": SCHEMA,
        "model": os.path.basename(model_path),
        "env": env,
        "files": files,
        "file_args": file_args,
        "args": args,
        "notes": f"generated from command line {os.path.basename(line_file)}",
    }
    if context is not None:
        recipe["context"] = context
    for flag, base in sorted(file_args.items()):
        print(f"command_to_recipe: {flag} needs '{base}' beside the model",
              file=sys.stderr)
    with open(out_json, "w") as handle:
        json.dump(recipe, handle, indent=2)
        handle.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
