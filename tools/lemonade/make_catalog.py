#!/usr/bin/env python3
"""Build the Lemonade model catalog and per-model args from library.json.

Input:  docs/lemonade/library.json
        list of {name, repo, file, mmproj, draft, labels, size_gb,
        superroc_args}; mmproj/draft are "REPO:FILE" strings or null.
Output: docs/lemonade/catalog.json
        {name: {"checkpoints": {"main": "repo:file", ...},
                "recipe": "llamacpp", "labels": [...], "size": size_gb}}
        tools/lemonade/superroc-models.json
        {file: superroc_args} for the llama-server shim.
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
LIBRARY = ROOT / "docs" / "lemonade" / "library.json"
CATALOG = ROOT / "docs" / "lemonade" / "catalog.json"
MODEL_ARGS = ROOT / "tools" / "lemonade" / "superroc-models.json"

FIELDS = ("name", "repo", "file", "mmproj", "draft", "labels",
          "size_gb", "superroc_args")


# Lemonade merges a split GGUF (NAME-00001-of-00002.gguf, ...) into one
# variant named NAME, and downloads every part of it.
_SHARD = re.compile(r"^(.+)[-._]\d{5}-of-\d{5}\.gguf$", re.IGNORECASE)


def lemonade_variant(file_name):
    match = _SHARD.match(file_name)
    return match.group(1) if match else file_name


def build(library):
    catalog = {}
    model_args = {}
    for entry in library:
        for field in FIELDS:
            if field not in entry:
                raise SystemExit("library entry missing %r: %r" % (field, entry))
        checkpoints = {"main": "%s:%s" % (entry["repo"],
                                           lemonade_variant(entry["file"]))}
        if entry["mmproj"]:
            checkpoints["mmproj"] = entry["mmproj"]
        if entry["draft"]:
            checkpoints["draft"] = entry["draft"]
        catalog[entry["name"]] = {
            "checkpoints": checkpoints,
            "recipe": "llamacpp",
            "labels": entry["labels"],
            "size": entry["size_gb"],
        }
        model_args[entry["file"]] = entry["superroc_args"]
    return catalog, model_args


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2)
        fh.write("\n")


def main():
    with open(LIBRARY) as fh:
        library = json.load(fh)
    catalog, model_args = build(library)
    write_json(CATALOG, catalog)
    write_json(MODEL_ARGS, model_args)
    print("wrote %s (%d models)" % (CATALOG.relative_to(ROOT), len(catalog)))
    print("wrote %s (%d files)" % (MODEL_ARGS.relative_to(ROOT), len(model_args)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
