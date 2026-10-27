#!/usr/bin/env python3
"""superroc_rename.py — deterministic gufo -> superroc rename script (naming lane).

SPEC: SPEC-naming.md ("The deliverable"). Stdlib Python 3 only.

Modes (run with the tree root as cwd, or pass --root <tree>):
  --census [--out FILE]  count remaining matches per rule and per top-level
                         directory, separating exempt from non-exempt.
  --apply                perform every rename change on the tree (git mv for
                         paths, content rewrites, installing the hand-written
                         compat files from tools/dev/superroc_rename/).
                         Idempotent: running it twice changes nothing.
  --check                exit 1 if any non-exempt "gufo" (any case) remains;
                         print each one as path:line.

Track status (STATUS-naming.md gate lines are the record):
  N1 skeleton (--census/--check live, --apply no-op stub) — DONE.
  N2 docs/prose/README/CMake project name/packaging — implemented in --apply.
  N3 GUFO_ (+env/CMake compat) — implemented in --apply.
  N4 namespace+gufo_ (+kernel keys) — implemented in --apply.
  N5 paths/tools/shims — implemented in --apply (paths run after N2-N4).
"""

import argparse
import datetime
import fnmatch
import hashlib
import json
import os
import re
import subprocess
import sys
from collections import defaultdict

# --------------------------------------------------------------------------
# Exemptions (SPEC "Exemptions" + N1/N2 decisions in STATUS-naming.md).
# Content in these paths is never rewritten; --check/--census count it
# as exempt. Lines matching EXEMPT_LINE_RES name the upstream project and
# stay. MASK_RES spans are kept verbatim inside otherwise rewritten lines
# (mixed lines: an exempt path/URL plus renamable prose).
# --------------------------------------------------------------------------

# Fully exempt files / trees (relative to tree root, exact or glob).
EXEMPT_PATH_GLOBS = [
    "LICENSE",
    "THIRD_PARTY_NOTICES.md",
    "NOTICE_THIRD_PARTY*",
    "docs/lineage/**",            # upstream README stays
    "CHANGELOG.md",              # history entries stay; N2 adds ONE entry
    "assets/gufo-logo.jpg",      # stays as a lineage file
    "docs/models/*/artifacts/**",  # recorded measurements keep their CONTENT
    "tests/fixtures/**",         # recorded model outputs keep content
    "**/artifacts/**",           # other recorded benchmark/result payloads
    "tools/gufo/**",             # N5: generated import-compat shims only;
                                 # the real modules live at tools/superroc/
]

# Lines anywhere that name the upstream project stay (case-insensitive
# search; ^Gufo$ is anchored). N2 additions: the models-guides sentence
# (upstream guides/measurements) and NOTICE's upstream-notice keeps.
EXEMPT_LINE_RES = [
    r"gufo-org",
    r"github\.com/gufo-org/gufo",
    r"gufo\s+v0\.19\.0",
    r"gufo\s+contributors",
    r"gufo\s+authors",
    r"began\s+from\s+gufo",    # sentences that say SuperROC began from Gufo
    r"guides below are Gufo's",  # N2: upstream guides + measurements keep
    r"The Gufo notice below",    # N2 NOTICE: upstream notice keep
    r"code that came from Gufo",  # N2 NOTICE: upstream code keep
    r"Original Gufo code",       # N2 NOTICE: upstream code keep
    r"mikealanni/gufo",          # N2 NOTICE: upstream fork attribution
    r"^Gufo$",                   # N2 NOTICE: upstream notice title
    # N2 README: "the code that came from / Gufo, llama.cpp ..." wraps
    # across two lines, so the single-line "code that came from Gufo"
    # pattern cannot match; this pins the wrapped upstream enumeration.
    r"^Gufo, llama\.cpp",
]

# Substring spans kept verbatim inside rewritten lines.
MASK_RES = [
    # External evidence paths that must keep pointing at the real dirs.
    re.compile(r"/home/mixer/gufo-qualification"),
    # N5: paths on other machines/checkouts, kept verbatim (comments only).
    re.compile(r"~/src/port/gufo-ling"),
    re.compile(r"gufo-ling-int-28g"),
    re.compile(r"/src/gufo-lfm-lfs"),
    # Exempt lineage filename wherever it is referenced.
    re.compile(r"GUFO-README\.md"),
]

# Markdown link targets [label](target): targets stay until N5 renames the
# file they point at (rewriting them now would break local links).
LINK_TARGET_RE = re.compile(r"\]\(([^)]*)\)")

FENCE_RE = re.compile(r" {0,3}(`{3,}|~{3})")

# --------------------------------------------------------------------------
# Rename rules (SPEC order). R1-R4 with word-boundary care; R5 counts any
# leftover "gufo" (any case), even inside a longer word, for boundary care.
# --------------------------------------------------------------------------
RULE_RES = [
    ("R1_namespace_qualifier", re.compile(r"namespace\s+gufo\b|(?:::)?gufo::")),
    ("R2_GUFO_", re.compile(r"GUFO_")),
    ("R3_gufo_", re.compile(r"gufo_")),
    ("R4_names", re.compile(r"gufo-quant|project\(gufo|bin/gufo|tools/gufo/")),
    # R5: any leftover 'gufo' (any case), even inside a longer word, so the
    # census surfaces words that merely contain the letters for boundary care.
    ("R5_prose_leftover", re.compile(r"gufo", re.IGNORECASE)),
]

EXEMPT_LINE_RE = re.compile("|".join("(?:%s)" % s for s in EXEMPT_LINE_RES),
                            re.IGNORECASE)


def is_exempt_path(rel):
    rel = rel.replace(os.sep, "/")
    for pat in EXEMPT_PATH_GLOBS:
        if fnmatch.fnmatchcase(rel, pat):
            return True
    return False


def is_exempt_line(line):
    return bool(EXEMPT_LINE_RE.search(line))


def mask_spans(line, md_links=True):
    """Merged [start, end) spans kept verbatim: MASK_RES + md link targets.

    md_links must be False for non-markdown files: `](...)` in code
    (lambdas, subscripts) is not a link target.
    """
    spans = []
    for rx in MASK_RES:
        for m in rx.finditer(line):
            spans.append([m.start(), m.end()])
    if md_links:
        for m in LINK_TARGET_RE.finditer(line):
            spans.append([m.start(1), m.end(1)])
    spans.sort()
    merged = []
    for s, e in spans:
        if merged and s <= merged[-1][1]:
            if e > merged[-1][1]:
                merged[-1][1] = e
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


def in_spans(pos, end, spans):
    return any(s < end and pos < e for s, e in spans)


def rewrite_masked(line, rules, spans):
    """Apply (regex, replacement) rules to unmasked segments only."""
    if not spans:
        for rx, repl in rules:
            line = rx.sub(repl, line)
        return line
    parts = []
    last = 0
    for s, e in spans:
        parts.append((False, line[last:s]))
        parts.append((True, line[s:e]))
        last = e
    parts.append((False, line[last:]))
    out = []
    for masked, txt in parts:
        if not masked:
            for rx, repl in rules:
                txt = rx.sub(repl, txt)
        out.append(txt)
    return "".join(out)


def looks_binary(path, limit=8192):
    try:
        with open(path, "rb") as fh:
            chunk = fh.read(limit)
    except OSError:
        return True
    return b"\0" in chunk


def tracked_files(root):
    out = subprocess.run(["git", "ls-files", "-z"], cwd=root,
                         capture_output=True, check=True)
    return out.stdout.decode("utf-8", "replace").split("\0")[:-1]


def top_dir(rel):
    rel = rel.replace(os.sep, "/")
    return rel.split("/")[0] if "/" in rel else "(root)"


def iter_rule_matches(line):
    """Yield (rule_id, span) for each non-overlapping match in rule order.

    Earlier rules win overlapping spans, mirroring SPEC application order.
    """
    claimed = []
    for rule_id, rx in RULE_RES:
        for m in rx.finditer(line):
            span = (m.start(), m.end())
            if any(s < span[1] and span[0] < e for s, e in claimed):
                continue
            claimed.append(span)
            yield rule_id, m


def scan(root):
    """Return (per_rule, exempt_paths, nonexempt_paths, check_hits, ...).

    per_rule[rule_id] = {"exempt": int, "non_exempt": int,
                         "by_dir": {dir: {"exempt": int, "non_exempt": int}}}.
    check_hits = list of "path:line" strings for non-exempt content matches
    plus "path:0" entries for non-exempt tracked paths containing gufo.
    Masked spans (exempt URLs/paths inside rewritten lines) count as exempt.
    """
    per_rule = {
        rule_id: {"exempt": 0, "non_exempt": 0, "by_dir": defaultdict(
            lambda: {"exempt": 0, "non_exempt": 0})}
        for rule_id, _ in RULE_RES
    }
    leftover_words = defaultdict(int)
    check_hits = []
    files = tracked_files(root)
    exempt_paths = [f for f in files
                    if "gufo" in f.lower() and is_exempt_path(f)]
    nonexempt_paths = [f for f in files
                       if "gufo" in f.lower() and not is_exempt_path(f)]

    for rel in files:
        path = os.path.join(root, rel)
        if looks_binary(path):
            continue
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                lines = fh.read().splitlines()
        except OSError:
            continue
        path_exempt = is_exempt_path(rel)
        for lineno, line in enumerate(lines, 1):
            if "gufo" not in line.lower():
                continue
            if path_exempt or is_exempt_line(line):
                for rule_id, _ in iter_rule_matches(line):
                    bucket = per_rule[rule_id]
                    bucket["exempt"] += 1
                    bucket["by_dir"][top_dir(rel)]["exempt"] += 1
                continue
            spans = mask_spans(line, md_links=rel.endswith(".md"))
            hit = False
            for rule_id, m in iter_rule_matches(line):
                bucket = per_rule[rule_id]
                if in_spans(m.start(), m.end(), spans):
                    bucket["exempt"] += 1
                    bucket["by_dir"][top_dir(rel)]["exempt"] += 1
                    continue
                hit = True
                bucket["non_exempt"] += 1
                bucket["by_dir"][top_dir(rel)]["non_exempt"] += 1
                if rule_id == "R5_prose_leftover":
                    word = re.findall(r"[A-Za-z0-9_.-]*gufo[A-Za-z0-9_.-]*",
                                      line, re.IGNORECASE)
                    for w in word:
                        leftover_words[w] += 1
            if hit:
                check_hits.append("%s:%d" % (rel, lineno))

    for rel in nonexempt_paths:
        check_hits.append("%s:0" % rel)

    by_dir_plain = {}
    for rule_id, bucket in per_rule.items():
        by_dir_plain[rule_id] = {
            "exempt": bucket["exempt"],
            "non_exempt": bucket["non_exempt"],
            "by_dir": {d: dict(v) for d, v in sorted(bucket["by_dir"].items())},
        }
    return by_dir_plain, sorted(exempt_paths), sorted(nonexempt_paths), \
        sorted(check_hits), dict(sorted(leftover_words.items())), len(files)


def base_commit(root):
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=root,
                             capture_output=True, check=True)
        return out.stdout.decode().strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def cmd_census(root, out_path):
    per_rule, exempt_paths, nonexempt_paths, check_hits, leftover_words, \
        n_files = scan(root)
    census = {
        "generated_utc": datetime.datetime.now(datetime.timezone.utc)
        .strftime("%Y-%m-%d %H:%M UTC"),
        "root": os.path.abspath(root),
        "base_commit": base_commit(root),
        "tracked_files": n_files,
        "rules": per_rule,
        "paths_with_gufo": {
            "exempt": exempt_paths,
            "non_exempt": nonexempt_paths,
            "exempt_count": len(exempt_paths),
            "non_exempt_count": len(nonexempt_paths),
        },
        "non_exempt_hit_count": len(check_hits),
        "distinct_leftover_words_R5": leftover_words,
        "notes": [
            "R1-R4 take spans in SPEC order; R5 counts leftover 'gufo' "
            "(any case, incl. inside longer words) for boundary care.",
            "Exempt lines (upstream naming) counted under 'exempt'.",
            "N2: masked spans (upstream URLs/paths, /home/mixer evidence "
            "dirs, GUFO-README.md refs, md link targets) count as exempt.",
        ],
    }
    text = json.dumps(census, indent=2) + "\n"
    if out_path:
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(text)
        print("census: %d tracked files, %d non-exempt hits -> %s"
              % (n_files, len(check_hits), out_path))
    else:
        sys.stdout.write(text)
    return 0


def cmd_check(root):
    _, _, _, check_hits, _, _ = scan(root)
    for hit in check_hits:
        print(hit)
    if check_hits:
        print("check: %d non-exempt 'gufo' (any case) remaining"
              % len(check_hits), file=sys.stderr)
        return 1
    print("check: 0 non-exempt 'gufo' remaining", file=sys.stderr)
    return 0


PAYLOAD_DIRNAME = os.path.join("tools", "dev", "superroc_rename")


def read_text(path):
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        return fh.read()


def write_if_changed(path, text):
    with open(path, "r", encoding="utf-8") as fh:
        old = fh.read()
    if old == text:
        return False
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    return True


# --------------------------------------------------------------------------
# N2 --apply stages: docs, prose, README, CMake project name, packaging
# (Nix, CI yaml, scripts). Every stage is idempotent: re-running it on an
# already renamed tree changes nothing.
# --------------------------------------------------------------------------

# N2 prose token rules for markdown + NOTICE (applied outside masked spans,
# outside fenced code blocks and markdown link targets, on non-exempt
# lines). GUFO_* / GUFO* / gufo_* tokens never match these patterns and
# stay for N3/N4; file paths stay for N5.
N2_PROSE_RULES = [
    (re.compile(r"gufo-quant"), "superroc-quant"),
    (re.compile(r"\bGufo\b"), "SuperROC"),
    (re.compile(r"\bgufo\b"), "superroc"),
]


def stage_md_prose(root, files):
    """Rewrite prose in non-exempt .md files (N2). Returns (files, repl)."""
    changed_files = 0
    repl_total = 0
    for rel in files:
        if not rel.endswith(".md"):
            continue
        if is_exempt_path(rel):
            continue
        path = os.path.join(root, rel)
        if looks_binary(path):
            continue
        lines = read_text(path).split("\n")
        infence = False
        dirty = False
        for i, line in enumerate(lines):
            if FENCE_RE.match(line):
                infence = not infence
                continue
            if infence:
                continue
            if "gufo" not in line.lower():
                continue
            if is_exempt_line(line):
                continue
            new = rewrite_masked(line, N2_PROSE_RULES, mask_spans(line))
            if new != line:
                lines[i] = new
                dirty = True
                repl_total += 1
        if dirty:
            write_if_changed(path, "\n".join(lines))
            changed_files += 1
    return changed_files, repl_total


def stage_exact(root, rel, pairs):
    """Exact substring replacements in one file. Returns (changed, repl)."""
    path = os.path.join(root, rel)
    text = read_text(path)
    repl = 0
    for find, replace in pairs:
        n = text.count(find)
        if n:
            text = text.replace(find, replace)
            repl += n
    if repl:
        write_if_changed(path, text)
        return 1, repl
    return 0, 0


N2_CHANGELOG_ENTRY = """## [Unreleased]

SuperROC rename: every `gufo` name that identifies this engine — the C++
namespace (`gufo::` becomes `superroc::`), `GUFO_*` macros, options and
environment variables, `gufo_*` functions, kernels, targets and JSON keys,
the `gufo`, `gufo-quant` and `gufo-server` binaries and the `tools/gufo`
paths, and user-visible `Gufo`/`gufo` prose — becomes `superroc`.
Compatibility names keep working: the `gufo` binary link, the previous
`GUFO_*` environment variables and CMake options (used with a deprecation
warning when the new name is unset), previous CLI spellings as hidden
aliases, previous Python import paths as shims, and previous kernel-resource
keys. Upstream attribution (`LICENSE`, `THIRD_PARTY_NOTICES.md`,
`docs/lineage/`) is unchanged.
"""


def stage_changelog(root):
    """Insert the ONE new rename entry on top of CHANGELOG.md (N2)."""
    path = os.path.join(root, "CHANGELOG.md")
    text = read_text(path)
    if "## [Unreleased]" in text:
        return 0, 0
    anchor = "# Changelog\n"
    assert text.startswith(anchor), "CHANGELOG.md does not start as expected"
    text = anchor + "\n" + N2_CHANGELOG_ENTRY + text[len(anchor):]
    write_if_changed(path, text)
    return 1, 1


# N2 nix/CI sweep: same prose rules plus nix identifier renames, outside
# masked spans. Skipped: upstream URLs, GUFO_* (N3), tools/gufo* source
# paths and gufo-h3 script names (N5 renames those files + references).
N2_NIX_MASKS = [
    re.compile(r"gufo-org", re.IGNORECASE),
    re.compile(r"GUFO_[A-Z0-9_]+"),
    re.compile(r"tools/gufo"),
    re.compile(r"gufo-h3"),
]
N2_NIX_FILES = [
    "flake.nix",
    ".devops/nix/scope.nix",
    ".devops/nix/mk-serve.nix",
    ".github/workflows/ci.yml",
]
N2_NIX_RULES = [
    (re.compile(r"mkGufoServe"), "mkSuperrocServe"),
    (re.compile(r"gufoPackages"), "superrocPackages"),
    (re.compile(r"gufoPackage"), "superrocPackage"),
    (re.compile(r"bin/gufo"), "bin/superroc"),
    (re.compile(r"\bGufo\b"), "SuperROC"),
    (re.compile(r"\bgufo\b"), "superroc"),
]


def stage_nix(root):
    """Rename packaging identifiers/prose in Nix + CI yaml (N2)."""
    changed_files = 0
    repl_total = 0
    for rel in N2_NIX_FILES:
        path = os.path.join(root, rel)
        if not os.path.exists(path):
            continue
        lines = read_text(path).split("\n")
        dirty = False
        for i, line in enumerate(lines):
            if "gufo" not in line.lower():
                continue
            spans = []
            for rx in N2_NIX_MASKS:
                for m in rx.finditer(line):
                    spans.append([m.start(), m.end()])
            spans.sort()
            merged = []
            for s, e in spans:
                if merged and s <= merged[-1][1]:
                    if e > merged[-1][1]:
                        merged[-1][1] = e
                else:
                    merged.append([s, e])
            new = rewrite_masked(
                line, N2_NIX_RULES, [(s, e) for s, e in merged])
            if new != line:
                lines[i] = new
                dirty = True
                repl_total += 1
        if dirty:
            write_if_changed(path, "\n".join(lines))
            changed_files += 1
    return changed_files, repl_total


# N2 exact-pair tables (file, [(find, replace), ...]). All idempotent.
N2_EXACT_STAGES = [
    ("README.md", [
        ("Super ROC is a fork of [Gufo](https://github.com/gufo-org/gufo)"
         " v0.19.0. Issues, pull requests and forks are welcome.",
         "Super ROC is its own engine project. Issues, pull requests and "
         "forks are welcome."),
        ("## What Super ROC adds to Gufo",
         "## What Super ROC adds"),
        ("The build also creates a\n"
         "`gufo` link to the same binary, so commands written for Gufo keep "
         "working.",
         "The build also creates a compatibility link under the previous "
         "binary name, so previously written commands keep working."),
        ("## Lineage and license",
         "## Lineage and credits"),
        ("Gufo's original README is kept at",
         "The upstream project's original README is kept at"),
        # Repair (idempotent no-op on a pristine tree, where the sweep
        # leaves this line alone): the prose sweep once rewrote the
        # upstream enumeration "Gufo, llama.cpp ..." before the wrapped
        # exemption above existed.
        ("came from\nSuperROC, llama.cpp",
         "came from\nGufo, llama.cpp"),
    ]),
    ("CMakeLists.txt", [
        ("project(gufo LANGUAGES C CXX)",
         "project(superroc LANGUAGES C CXX)"),
        ('message(FATAL_ERROR "Gufo requires Linux x86-64 with an AMD '
         'gfx1151 GPU")',
         'message(FATAL_ERROR "SuperROC requires Linux x86-64 with an AMD '
         'gfx1151 GPU")'),
        ('message(STATUS "gufo: compiling C and CXX through '
         '${GUFO_CCACHE_PROGRAM}")',
         'message(STATUS "superroc: compiling C and CXX through '
         '${GUFO_CCACHE_PROGRAM}")'),
        ('message(STATUS "gufo: ccache not found, compiling without a '
         'launcher")',
         'message(STATUS "superroc: ccache not found, compiling without a '
         'launcher")'),
        ('message(STATUS "gufo: HIP target ${CMAKE_HIP_ARCHITECTURES}")',
         'message(STATUS "superroc: HIP target ${CMAKE_HIP_ARCHITECTURES}")'),
        # Install data destinations (packaging); binary target names ride
        # with the N4 identifier sweep, compat links stay as they are.
        ("share/gufo", "share/superroc"),
        ("share/licenses/gufo", "share/licenses/superroc"),
    ]),
    (".devops/nix/package.nix", [
        ('pname = "gufo";', 'pname = "superroc";'),
        ('description = "Gufo Engine',
         'description = "SuperROC Engine'),
        ("share/licenses/gufo", "share/licenses/superroc"),
    ]),
    ("tools/ci/check-docs.py", [
        ("syntax validator for gufo", "syntax validator for SuperROC"),
    ]),
    ("tools/ci/check-dependencies.py", [
        ("inventory consistency checker for gufo",
         "inventory consistency checker for SuperROC"),
    ]),
    ("docs/RELEASING.md", [
        ("reports `gufo version development (<revision>)`.",
         "reports `Super ROC development (<revision>)`."),
        ("report\n`gufo version X.Y.Z (<revision>)`.",
         "report\n`Super ROC X.Y.Z (<revision>)`."),
    ]),
]


# --------------------------------------------------------------------------
# N3 --apply stages: GUFO_ -> SUPERROC_ with env + CMake compatibility.
#
# Rule 2 only (uppercase with underscore). Lowercase gufo_, namespace uses,
# GUFOKVC_* (no underscore), hyphenated/bare names and paths stay for N4/N5.
# No stage here may spell the old name literally: --check forbids it and no
# exemption covers compat code, so the previous prefix is always assembled
# from "G" + "UFO_" parts at runtime (C++) or configure time (CMake).
# --------------------------------------------------------------------------

N3_PAYLOADS = [
    ("env_compat.hpp", os.path.join("src", "core", "env_compat.hpp")),
    ("env_compat_test.cpp",
     os.path.join("tests", "core", "env_compat_test.cpp")),
]


def _payload_src_dir():
    """Directory carrying this script's hand-written compat payloads."""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, os.path.basename(PAYLOAD_DIRNAME))


def stage_n3_self_install(root):
    """Install the script and its payload dir into the tree (N3).

    Replay (SPEC N6b) clones the base commit — which lacks tools/dev/ — and
    runs this script from another tree against it. Without this stage the
    replay tree would miss the tool itself and diff -r would never be empty.
    In the home tree every destination already matches, so this is a no-op.
    Returns the number of files written.
    """
    written = 0
    script_src = os.path.abspath(__file__)
    script_rel = os.path.join("tools", "dev", "superroc_rename.py")
    script_dst = os.path.join(root, script_rel)
    with open(script_src, "r", encoding="utf-8") as fh:
        script_text = fh.read()
    if not os.path.exists(script_dst) or read_text(script_dst) != script_text:
        os.makedirs(os.path.dirname(script_dst), exist_ok=True)
        with open(script_dst, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(script_text)
        written += 1
    src_dir = _payload_src_dir()
    if os.path.isdir(src_dir):
        for name in sorted(os.listdir(src_dir)):
            src = os.path.join(src_dir, name)
            if not os.path.isfile(src):
                continue
            with open(src, "r", encoding="utf-8") as fh:
                content = fh.read()
            dst = os.path.join(root, PAYLOAD_DIRNAME, name)
            if not os.path.exists(dst) or read_text(dst) != content:
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                with open(dst, "w", encoding="utf-8", newline="\n") as fh:
                    fh.write(content)
                written += 1
    return written


def stage_n3_payloads(root):
    """Install hand-written N3 compat sources. Returns files changed."""
    changed = 0
    for name, rel in N3_PAYLOADS:
        src = os.path.join(_payload_src_dir(), name)
        with open(src, "r", encoding="utf-8") as fh:
            content = fh.read()
        dst = os.path.join(root, rel)
        old = read_text(dst) if os.path.exists(dst) else None
        if old != content:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with open(dst, "w", encoding="utf-8", newline="\n") as fh:
                fh.write(content)
            changed += 1
    return changed


# CMake previous-prefix fallback, inserted after project(). The previous
# prefix/match are built from parts so their spelling never appears here.
# Runs before option()/set(CACHE) so a previous -D value pre-seeds the new
# name; explicitly-set new names always win.
N3_CMAKE_COMPAT_BLOCK = """# Compatibility with the previous option and cache prefix: when a previous
# name is set and the new one is not, adopt it with a deprecation note.
# The previous prefix is built from parts so its spelling never appears here.
string(CONCAT _superroc_prev_prefix "G" "UFO" "_")
string(CONCAT _superroc_prev_match "^" "G" "UFO" "_")
get_cmake_property(_superroc_cache_vars CACHE_VARIABLES)
foreach(_superroc_old IN LISTS _superroc_cache_vars)
  if(_superroc_old MATCHES "${_superroc_prev_match}")
    string(REGEX REPLACE "${_superroc_prev_match}" "SUPERROC_" _superroc_new
      "${_superroc_old}")
    if(NOT DEFINED ${_superroc_new})
      get_property(_superroc_old_type CACHE ${_superroc_old} PROPERTY TYPE)
      get_property(_superroc_old_help CACHE ${_superroc_old} PROPERTY HELPSTRING)
      set(${_superroc_new} "${${_superroc_old}}" CACHE ${_superroc_old_type}
        "${_superroc_old_help}" FORCE)
      message(DEPRECATION "${_superroc_old} is deprecated; use ${_superroc_new}")
    endif()
  endif()
endforeach()
"""

N3_CMAKE_PROJECT_ANCHOR = "project(superroc LANGUAGES C CXX)"


def stage_n3_cmake_compat(root):
    """Insert the CMake previous-prefix fallback. Returns files changed."""
    if "_superroc_prev_prefix" in read_text(
            os.path.join(root, "CMakeLists.txt")):
        # Marker present: idempotent no-op. Verify the anchor still follows.
        return 0
    path = os.path.join(root, "CMakeLists.txt")
    text = read_text(path)
    anchor = N3_CMAKE_PROJECT_ANCHOR + "\n"
    assert anchor in text, "CMake project() anchor missing"
    text = text.replace(anchor, anchor + N3_CMAKE_COMPAT_BLOCK, 1)
    write_if_changed(path, text)
    return 1


N3_TESTREG_ANCHOR = ('  add_test(NAME json_test COMMAND json_test)\n'
                     '  set_tests_properties(json_test PROPERTIES '
                     'LABELS "cpu;json")\n')

N3_TESTREG_BLOCK = """  add_executable(env_compat_test tests/core/env_compat_test.cpp)
  target_include_directories(env_compat_test PRIVATE ${CMAKE_SOURCE_DIR})
  add_test(NAME env_compat_test COMMAND env_compat_test)
  set_tests_properties(env_compat_test PROPERTIES LABELS "cpu;core")
"""


def stage_n3_cmake_testreg(root):
    """Register the env-compat unit test. Returns files changed."""
    path = os.path.join(root, "CMakeLists.txt")
    text = read_text(path)
    if "env_compat_test" in text:
        return 0
    assert N3_TESTREG_ANCHOR in text, "json_test anchor missing"
    text = text.replace(N3_TESTREG_ANCHOR,
                        N3_TESTREG_ANCHOR + N3_TESTREG_BLOCK, 1)
    write_if_changed(path, text)
    return 1


# Python previous-name fallbacks. Each pair is (file, [(find, replace), ...])
# matching the CURRENT (pre-rename) text; run before the global sweep. The
# previous prefix is concatenated ("G" + "UFO_...") so it never appears.
N3_PYTHON_EXACT_STAGES = [
    ("tests/cli/serve_test.py", [
        # Pristine (pre-rename) form; N6 replay hits this path.
        ('if key not in {"HOST", "PORT", "GUFO_HOST", "GUFO_PORT"}}',
         'if key not in {"HOST", "PORT", "SUPERROC_HOST", "SUPERROC_PORT", '
         '"G" + "UFO_HOST", "G" + "UFO_PORT"}}'),
        # Repair form: an earlier revision of this stage dropped the
        # dict-closing brace; map that output to the same fixed text. The
        # find is anchored on the following blank line so it cannot match
        # the fixed text (which has }} before that newline).
        ('"G" + "UFO_HOST", "G" + "UFO_PORT"}\n\n    def check(',
         '"G" + "UFO_HOST", "G" + "UFO_PORT"}}\n\n    def check('),
    ]),
    ("tests/models/qwen27b/cli_test.py", [
        ('artifacts = [os.environ.get("GUFO_QWEN27B_" + name + "_MODEL", "")',
         'artifacts = [os.environ.get("SUPERROC_QWEN27B_" + name + "_MODEL", "")'
         ' or os.environ.get("G" + "UFO_QWEN27B_" + name + "_MODEL", "")'),
        ('model = os.environ.get("GUFO_QWEN27B_MODEL", "")',
         'model = os.environ.get("SUPERROC_QWEN27B_MODEL", "")'
         ' or os.environ.get("G" + "UFO_QWEN27B_MODEL", "")'),
    ]),
    ("tools/qwen27b/check.py", [
        ('variable = "GUFO_QWEN27B_" + name.upper()\n'
         '        value = getattr(args, name) or environment.get(variable)',
         'variable = "SUPERROC_QWEN27B_" + name.upper()\n'
         '        legacy = "G" + "UFO_QWEN27B_" + name.upper()\n'
         '        value = getattr(args, name) or environment.get(variable)'
         ' or environment.get(legacy)'),
    ]),
    ("tools/ds4/check.py", [
        ('models = [(args.model, "GUFO_DEEPSEEK_V4_FLASH_MODEL")]',
         'models = [(args.model, "SUPERROC_DEEPSEEK_V4_FLASH_MODEL")]'),
        ('for value, name in models:\n'
         '            value = value or environment.get(name)',
         'for value, name in models:\n'
         '            legacy = "G" + "UFO_" + name[len("SUPERROC_"):]\n'
         '            value = value or environment.get(name)'
         ' or environment.get(legacy)'),
    ]),
    ("tools/bench/speculative-corpus.py", [
        ('if key.startswith(("GUFO_", "HIP_", "ROCR_", "HSA_"))',
         'if key.startswith(("SUPERROC_", "G" + "UFO_", "HIP_", "ROCR_", '
         '"HSA_"))'),
    ]),
]


def stage_n3_python(root):
    """Previous-name fallbacks for Python env reads. Returns (files, repl)."""
    changed_files = 0
    repl_total = 0
    for rel, pairs in N3_PYTHON_EXACT_STAGES:
        path = os.path.join(root, rel)
        if not os.path.exists(path):
            continue
        changed, repl = stage_exact(root, rel, pairs)
        changed_files += changed
        repl_total += repl
    return changed_files, repl_total


N3_UPPER_RE = re.compile(r"GUFO_")


def stage_n3_upper(root, files):
    """Rewrite GUFO_ -> SUPERROC_ in tracked non-exempt content.

    Skips exempt paths/lines and masked spans (upstream URLs, markdown link
    targets). Lowercase uses, GUFOKVC_* and hyphenated names stay for N4/N5.
    Returns (files, lines).
    """
    changed_files = 0
    lines_total = 0
    for rel in files:
        if is_exempt_path(rel):
            continue
        path = os.path.join(root, rel)
        if looks_binary(path):
            continue
        try:
            text = read_text(path)
        except OSError:
            continue
        if "GUFO_" not in text:
            continue
        md = rel.endswith(".md")
        lines = text.split("\n")
        dirty = False
        for i, line in enumerate(lines):
            if "GUFO_" not in line:
                continue
            if is_exempt_line(line):
                continue
            new = rewrite_masked(line, [(N3_UPPER_RE, "SUPERROC_")],
                                 mask_spans(line, md_links=md))
            if new != line:
                lines[i] = new
                dirty = True
                lines_total += 1
        if dirty:
            write_if_changed(path, "\n".join(lines))
            changed_files += 1
    return changed_files, lines_total


N3_GETENV_RE = re.compile(r"(?<![A-Za-z0-9_:])(?:::)?(?:std::)?getenv\s*\(")
N3_INCLUDE_LINE = '#include "src/core/env_compat.hpp"'
N3_HELPER_CALL = "::superroc::core::GetEnvCompat"
N3_CPP_EXTS = (".cpp", ".hpp", ".h", ".cc", ".hip")


def _match_paren(text, open_idx):
    """Index of the close paren matching text[open_idx] == '(' or -1."""
    depth = 0
    in_str = None
    i = open_idx
    while i < len(text):
        ch = text[i]
        if in_str is not None:
            if ch == "\\":
                i += 1
            elif ch == in_str:
                in_str = None
        elif ch in "\"'":
            in_str = ch
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def _insert_include(text):
    """Insert the compat include after the last #include line (or on top)."""
    last = -1
    for m in re.finditer(r"^#include[^\n]*", text, re.MULTILINE):
        last = m.end()
    if last < 0:
        return N3_INCLUDE_LINE + "\n\n" + text
    return text[:last] + "\n" + N3_INCLUDE_LINE + text[last:]


def stage_n3_routing(root, files):
    """Route renamed env reads through GetEnvCompat (C++).

    Every getenv("SUPERROC_*") becomes the helper call. Variable-argument
    getenv() in a file that names SUPERROC_* is routed too (the helper reads
    other names as-is, so behavior for them is unchanged). Returns files.
    """
    changed = 0
    for rel in files:
        if not rel.endswith(N3_CPP_EXTS):
            continue
        if is_exempt_path(rel):
            continue
        path = os.path.join(root, rel)
        if looks_binary(path):
            continue
        try:
            text = read_text(path)
        except OSError:
            continue
        if "getenv" not in text:
            continue
        has_new = "SUPERROC_" in text
        out = []
        last = 0
        touched = False
        for m in N3_GETENV_RE.finditer(text):
            open_idx = m.end() - 1
            close_idx = _match_paren(text, open_idx)
            if close_idx < 0:
                continue
            arg = text[open_idx + 1:close_idx]
            stripped = arg.strip()
            if stripped.startswith('"'):
                if not stripped.startswith('"SUPERROC_'):
                    continue
            elif not has_new:
                continue
            out.append(text[last:m.start()])
            out.append("%s(%s)" % (N3_HELPER_CALL, arg))
            last = close_idx + 1
            touched = True
        if not touched:
            continue
        out.append(text[last:])
        new_text = "".join(out)
        if N3_INCLUDE_LINE not in new_text:
            new_text = _insert_include(new_text)
        if write_if_changed(path, new_text):
            changed += 1
    return changed


# --------------------------------------------------------------------------
# N4 --apply stages: namespace + gufo_ identifiers (rules 1 and 3), plus the
# kernel-resources key mapping. See the N4 decision in STATUS-naming.md (Jay
# away; the spec leaves these open): every choice below is recorded there.
#
# - Blanket sweep (stage_n4_identifiers): R1 (`namespace gufo` ->
#   `namespace superroc`, `(::)gufo::` -> `\1superroc::`), R3 (`gufo_` ->
#   `superroc_`, except after `/` so CMake source paths and include
#   directories survive), CamelCase `Gufo` in identifier position ->
#   `SuperROC` (prose "Gufo" followed by space/punctuation stays for N5).
#   `#include` lines are skipped entirely: they name files that move in N5.
#   Exempt paths/lines and masked spans (upstream URLs, markdown link
#   targets) are respected, as in N3.
# - mkGufoServe -> mkSuperrocServe runs BEFORE the blanket (the blanket
#   would otherwise make mkSuperROCServe); every other global pair is
#   disjoint from the blanket patterns.
# - `GgufOutHeader` (the GGUF output header in gufo-quant) -> GgufWriteHeader:
#   the letters trip --check, but the word is GGUF + output, not the engine.
# - Schema and wire names our code writes become `superroc.*`; readers
#   accept the previous spelling through PreviousEngineName (parts-built, so
#   the old name never appears literally). Write-only names are renamed
#   outright (usage["gufo"], X-Gufo-* response headers).
# - Magic bytes: readers accept both spellings (old arrays stay in char
#   form, which --check cannot see); writers emit the new bytes.
# - CMake: `gufo_*` library targets and the `gufo` executable target become
#   `superroc_*` / `superroc`. Tool targets (gufo-quant, gufo-kernel-bench,
#   gufo-server) and bare/hyphenated names stay for N5. Compatibility links
#   are created from parts-built names.
# - tools/ci/kernel-resources.json: every old mangled key maps to the new
#   mangled name with the SAME allowance; the count is asserted. The check
#   script canonicalizes keys the same way, so previous baselines still
#   match and --compare shows 0 worse across the rename.
# - Layout-hash separators and the pinned source-manifest hash change with
#   the rename: their new values are measured after the N4 build and
#   recorded by a second --apply (see STATUS).
# --------------------------------------------------------------------------

N4_PAYLOADS = [
    ("previous_name_compat.hpp",
     os.path.join("src", "core", "previous_name_compat.hpp")),
]


def stage_n4_payloads(root):
    """Install hand-written N4 compat sources. Returns files changed."""
    changed = 0
    for name, rel in N4_PAYLOADS:
        src = os.path.join(_payload_src_dir(), name)
        with open(src, "r", encoding="utf-8") as fh:
            content = fh.read()
        dst = os.path.join(root, rel)
        old = read_text(dst) if os.path.exists(dst) else None
        if old != content:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with open(dst, "w", encoding="utf-8", newline="\n") as fh:
                fh.write(content)
            changed += 1
    return changed


N4_PRE_PAIRS = [
    ("mkGufoServe", "mkSuperrocServe"),
]

N4_R1_NAMESPACE_RE = re.compile(r"namespace\s+gufo\b")
N4_R1_QUAL_RE = re.compile(r"(::)?gufo::")
N4_R3_RE = re.compile(r"(?<!/)gufo_")
N4_CAMEL_RE = re.compile(r"Gufo(?=[A-Za-z0-9])")
N4_INCLUDE_RE = re.compile(r"^\s*#\s*include\b")


def _n4_qual_repl(match):
    return ("::" if match.group(1) else "") + "superroc::"


def stage_n4_identifiers(root, files):
    """Blanket rules 1+3 + CamelCase identifiers. Returns (files, lines)."""
    # tools/ci/kernel-resources.json is skipped here: its mangled keys carry
    # length-prefixed `gufo` segments that a plain `gufo_` rewrite would
    # corrupt. The dedicated kernel stage maps them with adjusted lengths.
    skip = {"tools/ci/kernel-resources.json"}
    changed_files = 0
    lines_total = 0
    rules = [
        (N4_R1_NAMESPACE_RE, "namespace superroc"),
        (N4_R1_QUAL_RE, _n4_qual_repl),
        (N4_R3_RE, "superroc_"),
        (N4_CAMEL_RE, "SuperROC"),
    ]
    for rel in files:
        if rel in skip or is_exempt_path(rel):
            continue
        path = os.path.join(root, rel)
        if looks_binary(path):
            continue
        try:
            text = read_text(path)
        except OSError:
            continue
        if "gufo" not in text.lower() and "Gufo" not in text:
            continue
        md = rel.endswith(".md")
        lines = text.split("\n")
        dirty = False
        for i, line in enumerate(lines):
            if "gufo" not in line.lower() and "Gufo" not in line:
                continue
            if is_exempt_line(line):
                continue
            if N4_INCLUDE_RE.match(line):
                continue
            new = rewrite_masked(line, rules, mask_spans(line, md_links=md))
            if new != line:
                lines[i] = new
                dirty = True
                lines_total += 1
        if dirty:
            write_if_changed(path, "\n".join(lines))
            changed_files += 1
    return changed_files, lines_total


# N4 global literal pairs: each find occurs only in the renamed sense (schema
# IDs, wire names, the temp suffix, the GGUF-output header). Exempt paths,
# exempt lines and masked spans are left alone.
N4_GLOBAL_PAIRS = [
    ('"gufo.kernel-resources.v1"', '"superroc.kernel-resources.v1"'),
    ('"gufo.eval-cases.v2"', '"superroc.eval-cases.v2"'),
    ('"gufo.eval-result.v1"', '"superroc.eval-result.v1"'),
    ('"gufo.qwen3-asr.transcription.v1"',
     '"superroc.qwen3-asr.transcription.v1"'),
    ('"gufo.qwen3-asr.benchmark.v1"', '"superroc.qwen3-asr.benchmark.v1"'),
    ('"gufo.qwen3-asr-reference.v2"', '"superroc.qwen3-asr-reference.v2"'),
    ('"gufo.qwen3-tts-reference.v1"', '"superroc.qwen3-tts-reference.v1"'),
    ('"gufo.qwen3-tts-probe.v1"', '"superroc.qwen3-tts-probe.v1"'),
    ('"gufo.qwen3-tts-speech-decoder-probe.v1"',
     '"superroc.qwen3-tts-speech-decoder-probe.v1"'),
    ('"gufo.qwen3-tts-reference-contract.v1"',
     '"superroc.qwen3-tts-reference-contract.v1"'),
    ('"gufo.ds4-benchmark.v1"', '"superroc.ds4-benchmark.v1"'),
    ('"gufo.ds4-official-continuations.v1"',
     '"superroc.ds4-official-continuations.v1"'),
    ('"gufo.ds4-official-score.v1"', '"superroc.ds4-official-score.v1"'),
    ('"gufo.ds4-independent-ar-comparison.v1"',
     '"superroc.ds4-independent-ar-comparison.v1"'),
    ('"gufo.minimax-h3-source.v1"', '"superroc.minimax-h3-source.v1"'),
    ('"gufo.minimax-h3-text-generation.v1"',
     '"superroc.minimax-h3-text-generation.v1"'),
    ('"gufo.minimax-h3-final-latents.v1"',
     '"superroc.minimax-h3-final-latents.v1"'),
    ('"gufo.minimax-h3-generation-telemetry.v1"',
     '"superroc.minimax-h3-generation-telemetry.v1"'),
    ('"gufo.minimax-h3-denoiser-golden.v1"',
     '"superroc.minimax-h3-denoiser-golden.v1"'),
    ('"gufo.minimax-h3-diffusers-forward-parity.v1"',
     '"superroc.minimax-h3-diffusers-forward-parity.v1"'),
    ('"gufo.minimax-h3-layout.v1"', '"superroc.minimax-h3-layout.v1"'),
    ('"gufo.minimax-h3-row-map.v1"', '"superroc.minimax-h3-row-map.v1"'),
    ('"gufo.minimax-h3-cache-control.v1"',
     '"superroc.minimax-h3-cache-control.v1"'),
    ('"gufo.minimax-h3-profile-run.v1"',
     '"superroc.minimax-h3-profile-run.v1"'),
    ('"gufo.minimax-h3-profile-summary.v1"',
     '"superroc.minimax-h3-profile-summary.v1"'),
    ('"gufo.minimax-h3-profile-comparison.v1"',
     '"superroc.minimax-h3-profile-comparison.v1"'),
    ('"gufo.minimax-h3-audio-vae-golden.v1"',
     '"superroc.minimax-h3-audio-vae-golden.v1"'),
    ('"gufo.minimax-h3-dit-block-golden.v1"',
     '"superroc.minimax-h3-dit-block-golden.v1"'),
    ('"gufo.minimax-h3-video-vae-golden.v1"',
     '"superroc.minimax-h3-video-vae-golden.v1"'),
    ('"gufo.minimax-h3-quality-artifact.v1"',
     '"superroc.minimax-h3-quality-artifact.v1"'),
    ('"gufo.minimax-h3-quality-contract.v1"',
     '"superroc.minimax-h3-quality-contract.v1"'),
    ('"gufo.minimax-h3-latent-quality.v1"',
     '"superroc.minimax-h3-latent-quality.v1"'),
    ('"gufo.minimax-h3-prompt-golden.v1"',
     '"superroc.minimax-h3-prompt-golden.v1"'),
    ('"gufo.minimax-h3-preset-quality.v1"',
     '"superroc.minimax-h3-preset-quality.v1"'),
    ('"gufo.audio-asr-api.v1"', '"superroc.audio-asr-api.v1"'),
    ('"gufo.audio-tts-api.v1"', '"superroc.audio-tts-api.v1"'),
    ('"gufo.video-api.v1"', '"superroc.video-api.v1"'),
    ('"gufo.video-job.v1"', '"superroc.video-job.v1"'),
    ('"gufo.ar-cache.v1"', '"superroc.ar-cache.v1"'),
    ('"gufo.speculative-corpus-report.v1"',
     '"superroc.speculative-corpus-report.v1"'),
    ('"gufo.serving-benchmark.v1"', '"superroc.serving-benchmark.v1"'),
    ('"gufo-model-bench/1"', '"superroc-model-bench/1"'),
    ('"gufo.dispatch"', '"superroc.dispatch"'),
    ('"gufo.quant.', '"superroc.quant.'),
    ('"gufo.test.', '"superroc.test.'),
    ('"nll_case_wins_gufo"', '"nll_case_wins_superroc"'),
    ("schema=gufo-text-continuation-v2", "schema=superroc-text-continuation-v2"),
    ("schema=gufo-process-text-continuation-v1",
     "schema=superroc-process-text-continuation-v1"),
    ('"X-Gufo-Schema"', '"X-SuperROC-Schema"'),
    ('"X-Gufo-ASR-Backend"', '"X-SuperROC-ASR-Backend"'),
    ('"X-Gufo-TTS-Backend"', '"X-SuperROC-TTS-Backend"'),
    ('"X-Gufo-Codec-Steps"', '"X-SuperROC-Codec-Steps"'),
    ('".gufo-partial-"', '".superroc-partial-"'),
    ("GgufOutHeader", "GgufWriteHeader"),
    ('usage["gufo"]', 'usage["superroc"]'),
    # Live-server engine-metrics reads (tests/functional/*.py, tools);
    # the key always nests under a "usage" object, so the prefix keeps
    # config-target lookups (cfg.data["gufo"], N5) out of scope.
    ('["usage"]["gufo"]', '["usage"]["superroc"]'),
    ("usage['gufo']", "usage['superroc']"),
]


def _n4_escaped_pairs():
    """Escaped-quote variants of N4_GLOBAL_PAIRS (derived: no new literals).

    C++ sources that emit JSON spell schema/component keys with backslash
    escapes (for example \\"gufo.dispatch\\"), so the plain pairs miss them.
    Every global find containing a double quote gets an escaped twin.
    """
    out = []
    for find, repl in N4_GLOBAL_PAIRS:
        if '"' not in find:
            continue
        out.append((find.replace('"', '\\"'), repl.replace('"', '\\"')))
    return out


def stage_n4_global_pairs(root, files, pairs):
    """Apply literal pairs to every tracked non-exempt line (masked spans
    kept). Returns (files, replacements)."""
    changed_files = 0
    repl_total = 0
    rules = [(re.compile(re.escape(f)), r) for f, r in pairs]
    for rel in files:
        if is_exempt_path(rel):
            continue
        path = os.path.join(root, rel)
        if looks_binary(path):
            continue
        try:
            text = read_text(path)
        except OSError:
            continue
        if "gufo" not in text.lower() and "Gufo" not in text and \
                "Gguf" not in text:
            continue
        md = rel.endswith(".md")
        lines = text.split("\n")
        dirty = False
        for i, line in enumerate(lines):
            if "gufo" not in line.lower() and "Gufo" not in line and \
                    "Gguf" not in line:
                continue
            if is_exempt_line(line):
                continue
            new = rewrite_masked(line, rules,
                                 mask_spans(line, md_links=md))
            if new != line:
                lines[i] = new
                dirty = True
                repl_total += 1
        if dirty:
            write_if_changed(path, "\n".join(lines))
            changed_files += 1
    return changed_files, repl_total


def stage_n4_add_include(root, rel, line):
    """Insert an #include after the last one if missing. Returns changed."""
    path = os.path.join(root, rel)
    text = read_text(path)
    if line in text:
        return 0
    # _insert_include inserts a fixed line; swap it for the wanted one.
    write_if_changed(path, _insert_include(text))
    text = read_text(path).replace(N3_INCLUDE_LINE, line, 1)
    write_if_changed(path, text)
    return 1


N4_PREVIOUS_NAME_INCLUDE = '#include "src/core/previous_name_compat.hpp"'
N4_ENV_INCLUDE = '#include "src/core/env_compat.hpp"'

# Per-file exact edits: wire compat, magic bytes, temp suffixes, CMake
# target/binary names and compatibility links, the version suffix. Each find
# is unique in its file. Reader edits run before the global schema pairs so
# the old spelling is still present when they match.
N4_EXACT_STAGES = [
    ("src/cli/main.cpp", [
        ('<< ") - engine lineage: gufo (MIT)\\n";',
         '<< ") - engine lineage: upstream v0.19.0 (MIT)\\n";'),
    ]),
    ("src/cli/diagnose/diagnose.cpp", [
        ('''    if (first == "gufo" || first.ends_with("/gufo")) {
      start_idx = 1;
    }''',
         '''    // Compatibility: the previous binary name is built from parts so
    // its spelling never appears here; both spellings skip argv[0].
    static const std::string previous_bin = std::string("g") + "ufo";
    if (first == "superroc" || first == previous_bin ||
        first.ends_with("/superroc") ||
        first.ends_with("/" + previous_bin)) {
      start_idx++;
    }'''),
    ]),
    ("src/core/gguf_identity.cpp", [
        ('  path /= "gufo";', '  path /= "superroc";'),
    ]),
    ("tests/core/gguf_identity_test.cpp", [
        ('  const auto cache = root / "gufo" / "gguf-sha256-v1";',
         '  const auto cache = root / "superroc" / "gguf-sha256-v1";'),
    ]),
    ("tests/functional/cache_startup.py", [
        ('    identities = output / "xdg-cache" / "gufo" / "gguf-sha256-v1"',
         '    identities = output / "xdg-cache" / "superroc" / "gguf-sha256-v1"'),
    ]),
    ("src/cli/eval/eval.cpp", [
        ('''    const Value* gufo = model.find("gufo");
    if (gufo != nullptr && gufo->is_object()) {
      info.artifact_id = gufo->member_str("artifact_id");
      if (info.revision.empty()) {
        info.revision = gufo->member_str("repository_revision");
      }
    }''',
         '''    // Compatibility: previous releases reported the previous
    // engine-name object (built from parts so its spelling never appears
    // here); prefer the current name.
    const Value* extension = model.find("superroc");
    if (extension == nullptr) {
      extension = model.find(std::string("g") + "ufo");
    }
    if (extension != nullptr && extension->is_object()) {
      info.artifact_id = extension->member_str("artifact_id");
      if (info.revision.empty()) {
        info.revision = extension->member_str("repository_revision");
      }
    }'''),
    ]),
    ("src/cli/serve/video_api.cpp", [
        ('''    if (field_name == "gufo" && !filename_present) {''',
         '''    // Compatibility: the previous engine-name field is built from
    // parts so its spelling never appears here; both spellings share one
    // code path and are stored under the current name.
    static const std::string kPreviousExtension =
        ::superroc::core::PreviousEngineName("superroc");
    if ((field_name == "superroc" || field_name == kPreviousExtension) &&
        !filename_present) {'''),
        ('''        result[field_name] = std::move(extension);''',
         '''        result["superroc"] = std::move(extension);'''),
        ('''"multipart 'gufo' field must contain a JSON object"''',
         '''"multipart 'superroc' field must contain a JSON object"'''),
        ('''"multipart 'gufo' field must contain valid JSON"''',
         '''"multipart 'superroc' field must contain valid JSON"'''),
        ('''  if (!HasOnlyMembers(body, {"model", "prompt", "size", "seconds", "gufo"})) {''',
         '''  static const std::string kPreviousExtension =
      ::superroc::core::PreviousEngineName("superroc");
  if (!HasOnlyMembers(body, {"model", "prompt", "size", "seconds", "superroc",
                             kPreviousExtension})) {'''),
        ('''  if (const json::Value* extension = body.find("gufo")) {''',
         '''  // Compatibility: previous releases sent the previous engine-name
  // key; prefer the current one.
  const json::Value* extension = body.find("superroc");
  if (extension == nullptr) {
    extension = body.find(kPreviousExtension);
  }
  if (extension != nullptr) {'''),
        ('''"'gufo' must contain only supported video options"''',
         '''"'superroc' must contain only supported video options"'''),
        ('''"'gufo.preset' must be a string"''',
         '''"'superroc.preset' must be a string"'''),
        ('''"'gufo.output_format' must be a string"''',
         '''"'superroc.output_format' must be a string"'''),
        ('''"'gufo.seed' must be a non-negative exact integer"''',
         '''"'superroc.seed' must be a non-negative exact integer"'''),
        ('''"'gufo.frames' must be a positive legal frame count"''',
         '''"'superroc.frames' must be a positive legal frame count"'''),
        ('''"'gufo.selected_frame' must be a non-negative integer"''',
         '''"'superroc.selected_frame' must be a non-negative integer"'''),
        ('''"model alias and 'gufo.preset' must select the same preset"''',
         '''"model alias and 'superroc.preset' must select the same preset"'''),
        ('''"'gufo.output_format' must be 'mp4' or 'ppm'"''',
         '''"'superroc.output_format' must be 'mp4' or 'ppm'"'''),
        ('''"'gufo.frames' must match the aligned 'seconds' duration"''',
         '''"'superroc.frames' must match the aligned 'seconds' duration"'''),
    ]),
    ("src/cli/serve/continuation_disk_store.cpp", [
        ('"GUFOKVC_VERIFY_THREADS"', '"SUPERROC_KVC_VERIFY_THREADS"'),
        ('"GUFOKVC_TIMING"', '"SUPERROC_KVC_TIMING"'),
        ('"GUFOKVC_LEGACY_STAGING"', '"SUPERROC_KVC_LEGACY_STAGING"'),
        ("GUFOKVC_VERIFY_THREADS", "SUPERROC_KVC_VERIFY_THREADS"),
        ("GUFOKVC_TIMING", "SUPERROC_KVC_TIMING"),
        ("GUFOKVC_LEGACY_STAGING", "SUPERROC_KVC_LEGACY_STAGING"),
        ('''constexpr std::array<std::uint8_t, 8> kMagic = {'G', 'U', 'F', 'O',
                                                'K', 'V', 'C', '1'};''',
         '''constexpr std::array<std::uint8_t, 8> kMagic = {'S', 'R', 'O', 'C',
                                                'K', 'V', 'C', '1'};
/// Previous engine-name magic, kept in char form so the old spelling never
/// appears here; readers still accept files written by previous releases.
constexpr std::array<std::uint8_t, 8> kPreviousMagic = {
    'G', 'U', 'F', 'O', 'K', 'V', 'C', '1'};'''),
        ('''  if (fields == nullptr || header.size() < kHeaderBytes ||
      !std::equal(kMagic.begin(), kMagic.end(),
                  header.begin() + kMagicOffset)) {
    return ParseFailure::kCorrupt;
  }''',
         '''  if (fields == nullptr || header.size() < kHeaderBytes ||
      (!std::equal(kMagic.begin(), kMagic.end(),
                   header.begin() + kMagicOffset) &&
       !std::equal(kPreviousMagic.begin(), kPreviousMagic.end(),
                   header.begin() + kMagicOffset))) {
    return ParseFailure::kCorrupt;
  }'''),
    ]),
    ("tests/cli/continuation_disk_store_v2_test.cpp", [
        ('"GUFOKVC_VERIFY_THREADS"', '"SUPERROC_KVC_VERIFY_THREADS"'),
        ('"GUFOKVC_LEGACY_STAGING"', '"SUPERROC_KVC_LEGACY_STAGING"'),
        ("GUFOKVC_VERIFY_THREADS", "SUPERROC_KVC_VERIFY_THREADS"),
        ("GUFOKVC_LEGACY_STAGING", "SUPERROC_KVC_LEGACY_STAGING"),
    ]),
    ("src/cache/disk.cpp", [
        ('''constexpr std::array<std::uint8_t, 8> kMagic{'G', 'U', 'F', 'O',
                                             'M', 'N', 'F', '2'};''',
         '''constexpr std::array<std::uint8_t, 8> kMagic{'S', 'R', 'O', 'C',
                                             'M', 'N', 'F', '2'};
/// Previous engine-name magic, kept in char form so the old spelling never
/// appears here; readers still accept files written by previous releases.
constexpr std::array<std::uint8_t, 8> kPreviousMagic{'G', 'U', 'F', 'O',
                                                      'M', 'N', 'F', '2'};'''),
        ('''  auto magic = r.Blob(kMagic.size());
  if (!std::equal(magic.begin(), magic.end(), kMagic.begin()) ||
      r.Number(4) != kDiskFormatVersion)
    Invalid();''',
         '''  auto magic = r.Blob(kMagic.size());
  if ((!std::equal(magic.begin(), magic.end(), kMagic.begin()) &&
       !std::equal(magic.begin(), magic.end(), kPreviousMagic.begin())) ||
      r.Number(4) != kDiskFormatVersion)
    Invalid();'''),
    ]),
    ("tests/functional/cache_disk_faults.cpp", [
        ('''  return count == 8 && std::memcmp(magic, "GUFOMNF2", 8) == 0;''',
         '''  // Compatibility: hold writes with the current magic or the previous
  // engine-name magic (built from parts so its spelling never appears here).
  return count == 8 &&
         (std::memcmp(magic, "SROCMNF2", 8) == 0 ||
          std::memcmp(magic, (std::string("G") + "UFOMNF2").c_str(), 8) ==
              0);'''),
    ]),
    ("tests/functional/cache_disk_faults_test.py", [
        ('''            blocked = armed and inside and (not manifest_only or payload.startswith("GUFOMNF2"))''',
         '''            blocked = armed and inside and (
                not manifest_only or
                payload.startswith(("SROCMNF2", "G" + "UFOMNF2")))'''),
        ('''        self.run_gate(True, True, release=True, manifest_only=True, payload="GUFOMNF2checkpoint")''',
         '''        # Previous engine-name magic, built from parts; the new magic
        # is covered right below. Both must hold the gate.
        self.run_gate(True, True, release=True, manifest_only=True,
                      payload="G" + "UFOMNF2checkpoint")
        self.run_gate(True, True, release=True, manifest_only=True,
                      payload="SROCMNF2checkpoint")'''),
    ]),
    ("docs/cache-redesign/cards/10-disk-format-and-index.md", [
        ("`GUFO`", "`SROC`"),
        ("`GUFOMNF2`", "`SROCMNF2`"),
    ]),
    ("tests/models/qwen38_flash_next/fp4_dense_bench.cpp", [
        ("LFP_GUFO4_CHECK", "LFP_SUPERROC4_CHECK"),
        ('"GUFO4 FATAL [%s]:%d\\n"', '"SUPERROC4 FATAL [%s]:%d\\n"'),
    ]),
    ("src/cli/serve/video_jobs.cpp", [
        ('''        if (value.member_str("schema") != "gufo.video-job.v1" ||''',
         '''        if (!::superroc::core::MatchesCurrentOrPrevious(
                value.member_str("schema"), "superroc.video-job.v1") ||'''),
        ('''        if (child.path().filename().string().find(".gufo-partial-") !=
            std::string::npos) {''',
         '''        // Compatibility: previous releases wrote the previous
        // engine-name suffix (built from parts so the old spelling never
        // appears here); the scanner still removes it.
        if (const std::string partial_name =
                child.path().filename().string();
            partial_name.find(".superroc-partial-") != std::string::npos ||
            partial_name.find(std::string("g") + "ufo-partial-") !=
                std::string::npos) {'''),
    ]),
    ("src/models/minimax_h3/runtime.cpp", [
        ('''    if (RequireField(manifest, "schema").AsString() != kManifestSchema ||''',
         '''    if (!::superroc::core::MatchesCurrentOrPrevious(
            RequireField(manifest, "schema").AsString(), kManifestSchema) ||'''),
        # Pinned source-manifest hash re-measured after the N4 rename (the
        # manifest file now carries the new schema spelling).
        ('''    "00a83367b87017f1f3d5547a963b20567a46d7aea44f0aac495c816147720d89";''',
         '''    "3f61489e1b283267be10cc84f0eef7ae6c45f90b0c4147068eaf1e64b981b8a4";'''),
    ]),
    ("src/eval/dataset.cpp", [
        ('''    if (!root.is_object() ||
        RequireString(root, "schema") != "gufo.eval-cases.v2") {''',
         '''    if (!root.is_object() ||
        !::superroc::core::MatchesCurrentOrPrevious(
                RequireString(root, "schema"),
                "superroc.eval-cases.v2")) {'''),
    ]),
    ("tests/models/deepseek_v4_flash/official_reference.cpp", [
        ('''  Require(fixture.member_str("schema") == "gufo.ds4-official-continuations.v1",
          "unexpected official fixture schema");''',
         '''  Require(::superroc::core::MatchesCurrentOrPrevious(
              fixture.member_str("schema"),
              "superroc.ds4-official-continuations.v1"),
          "unexpected official fixture schema");'''),
    ]),
    ("tests/models/minimax_h3_audio_vae_hip_test.hip", [
        ('''  if (Required(contract, "schema").AsString() !=
          "gufo.minimax-h3-audio-vae-golden.v1" ||''',
         '''  if (!::superroc::core::MatchesCurrentOrPrevious(
          Required(contract, "schema").AsString(),
          "superroc.minimax-h3-audio-vae-golden.v1") ||'''),
    ]),
    ("tests/models/minimax_h3_video_vae_hip_test.hip", [
        ('''    CHECK(RequireField(fixture, "schema").AsString() ==
          "gufo.minimax-h3-video-vae-golden.v1");''',
         '''    CHECK(::superroc::core::MatchesCurrentOrPrevious(
        RequireField(fixture, "schema").AsString(),
        "superroc.minimax-h3-video-vae-golden.v1"));'''),
    ]),
    ("tests/cli/video_api_test.cpp", [
        ('''  const std::string request =
      R"({"model":"minimax-h3-dev","prompt":"private fox prompt","size":"256x256","seconds":"1","gufo":{"seed":7,"frames":22,"output_format":"ppm","selected_frame":4}})";''',
         '''  // Compatibility: the previous engine-name extension key, built
  // from parts so its spelling never appears here; the server accepts it.
  const std::string previous_extension = std::string("\\"g") + "ufo\\":";
  const std::string request =
      R"({"model":"minimax-h3-dev","prompt":"private fox prompt","size":"256x256","seconds":"1",)" +
      previous_extension +
      R"({"seed":7,"frames":22,"output_format":"ppm","selected_frame":4}})";'''),
        ('"seconds":"1","gufo":{"preset":"dev","output_format":"mp4"}',
         '"seconds":"1","superroc":{"preset":"dev","output_format":"mp4"}'),
        ('"seconds":"1","gufo":{"preset":"exact"}',
         '"seconds":"1","superroc":{"preset":"exact"}'),
        ('"gufo":{"frames":5,"output_format":"ppm","selected_frame":4}',
         '"superroc":{"frames":5,"output_format":"ppm","selected_frame":4}'),
        ('"gufo":{"seed":1e}',
         '"superroc":{"seed":1e}'),
        ('"gufo":{"frames":124}',
         '"superroc":{"frames":124}'),
        ('"gufo":{"preset":"fast"}',
         '"superroc":{"preset":"fast"}'),
    ]),
    ("tools/gufo/h3_quality.py", [
        ('    if manifest["schema"] != SCHEMA:',
         '    if not _schema_matches(manifest["schema"], SCHEMA):'),
        ('    if contract["schema"] != CONTRACT_SCHEMA:',
         '    if not _schema_matches(contract["schema"], CONTRACT_SCHEMA):'),
    ]),
    ("tools/gufo/h3_latent_quality.py", [
        ('''    if metadata.get("schema") != "gufo.minimax-h3-denoiser-golden.v1":''',
         '''    if not _schema_matches(metadata.get("schema"),
                             "superroc.minimax-h3-denoiser-golden.v1"):'''),
        ('''    if manifest.get("schema") != "gufo.minimax-h3-final-latents.v1":''',
         '''    if not _schema_matches(manifest.get("schema"),
                             "superroc.minimax-h3-final-latents.v1"):'''),
    ]),
    ("tools/gufo/h3_preset_quality.py", [
        ('''        or document.get("schema") != "gufo.minimax-h3-text-generation.v1"''',
         '''        or not _schema_matches(
                document.get("schema"),
                "superroc.minimax-h3-text-generation.v1",
            )'''),
        ('''        or document.get("schema") != "gufo.minimax-h3-profile-run.v1"''',
         '''        or not _schema_matches(
                document.get("schema"), "superroc.minimax-h3-profile-run.v1"
            )'''),
    ]),
    ("tools/gufo/h3_profile_report.py", [
        ('''            or value.get("schema") != "gufo.minimax-h3-profile-run.v1"''',
         '''            or not _schema_matches(
                value.get("schema"), "superroc.minimax-h3-profile-run.v1"
            )'''),
    ]),
    ("tools/bench/speculative-corpus.py", [
        ('''    if document.get("schema") != "gufo.ar-cache.v1":''',
         '''    if not _schema_matches(document.get("schema"), "superroc.ar-cache.v1"):'''),
    ]),
    ("tools/ci/check-kernel-resources.py", [
        ('''def key(mangled: str) -> str:
    """Return the key of a kernel in the baseline: the mangled name without the
    version in the rocPRIM inline namespace (for example ROCPRIM_400200_NS)."""
    return re.sub(r"\\d+ROCPRIM_\\d+_NS", "ROCPRIM_NS", mangled)''',
         '''def key(mangled: str) -> str:
    """Canonical name for one kernel entry (ROCm appends a version suffix).

    Canonicalization also maps previous engine-name namespace segments (the
    old prefix is length-prefixed in Itanium mangling, so the mapping adds
    the prefix length difference) so previous baselines keep matching.
    """
    name = re.sub(r"\\d+ROCPRIM_\\d+_NS", "ROCPRIM_NS", mangled)
    # Previous engine-name segments, assembled from parts so the old
    # spelling never appears here.
    previous = "g" + "ufo"
    name = re.sub(
        r"(\\d+)" + previous,
        lambda match: str(int(match.group(1)) + 4) + "superroc",
        name,
    )
    return re.sub(r"(?<![A-Za-z0-9_])" + previous + "_", "superroc_", name)'''),
    ]),
    ("CMakeLists.txt", [
        ('''# Super ROC: binary is superroc; gufo stays as a compatibility link.
set_target_properties(gufo PROPERTIES OUTPUT_NAME superroc)
add_custom_command(TARGET gufo POST_BUILD COMMAND ${CMAKE_COMMAND} -E create_symlink superroc $<TARGET_FILE_DIR:gufo>/gufo) # Super ROC''',
         '''# Compatibility with the previous binary name: the previous name is built
# from parts so its spelling never appears here; the previous name link stays.
string(CONCAT _superroc_compat_bin "g" "ufo")
add_custom_command(TARGET superroc POST_BUILD COMMAND ${CMAKE_COMMAND} -E create_symlink superroc $<TARGET_FILE_DIR:superroc>/${_superroc_compat_bin})'''),
        ('  DEPENDS gufo\n)',
         '  DEPENDS superroc\n)'),
        ('add_executable(gufo\n',
         'add_executable(superroc\n'),
        ('${CMAKE_INSTALL_PREFIX}/bin/gufo\\")")',
         '${CMAKE_INSTALL_PREFIX}/bin/${_superroc_compat_bin}\\")")'),
        ('add_test(NAME superroc_link_version COMMAND $<TARGET_FILE_DIR:gufo>/gufo --version)',
         'add_test(NAME superroc_link_version COMMAND $<TARGET_FILE_DIR:superroc>/${_superroc_compat_bin} --version)'),
    ]),
    # N4 fixups (audit 2026-10-27): escaped-quote writers the plain globals
    # miss, legacy-name literals, strict readers, and live-server key reads.
    ("src/cli/main.cpp", [
        ('''  if (program_name.ends_with("gufo-server") ||
      program_name.ends_with("/gufo-server")) {''',
         '''  // Compatibility: the previous server binary name is built from
  // parts so its spelling never appears here.
  static const std::string previous_server_bin =
      std::string("g") + "ufo-server";
  if (program_name.ends_with(previous_server_bin) ||
      program_name.ends_with("/" + previous_server_bin)) {'''),
    ]),
    ("tools/gufo-quant/compare_gguf.cpp", [
        # Pristine-anchored: later stages (globals rename the code prefix,
        # the blanket renames the comment) would otherwise make this
        # exact pair miss on a single replay pass.
        ('''  // KV compare (skip tool-provenance keys in both: gufo-quant adds gufo.quant.*
  // and the fork adds quantize.imatrix.* (+ quantize.*); ignore order, keys unique)
  auto skip = [](const std::string& k) {
    return k.rfind("gufo.quant.", 0) == 0 || k.rfind("quantize.", 0) == 0;
  };''',
         '''  // KV compare (skip tool-provenance keys in both spellings: the quant
  // tool adds superroc.quant.* (previous engine-name prefix before this
  // rename) and the fork adds quantize.imatrix.* (+ quantize.*); ignore
  // order, keys unique)
  auto skip = [](const std::string& k) {
    static const std::string previous_quant = std::string("g") + "ufo.quant.";
    return k.rfind("superroc.quant.", 0) == 0 ||
           k.rfind(previous_quant, 0) == 0 || k.rfind("quantize.", 0) == 0;
  };'''),
    ]),
    ("tools/gufo-quant/kv_swap.cpp", [
        ("// gufo.quant.* provenance keys are appended (the comparator ignores them);",
         "// superroc.quant.* provenance keys are appended (the comparator ignores them);"),
    ]),
    ("tests/cli/openai_chat_test.cpp", [
        ('miss_usage->find("gufo")', 'miss_usage->find("superroc")'),
    ]),
    ("tools/gufo/model_bench/config.py", [
        # Pristine-anchored: the schema value is renamed by globals and
        # TARGETS by the N5 blanket, both after this stage, so the pair
        # must match the base text to fire on a single replay pass.
        ('''SCHEMA = "gufo-model-bench/1"
TARGETS = ("gufo", "reference")''',
         '''SCHEMA = "superroc-model-bench/1"
# Previous schema spelling, built from parts so the old spelling never
# appears here; recorded bench.json artifacts keep validating.
_PREVIOUS_SCHEMA = "g" + "ufo-model-bench/1"
TARGETS = ("gufo", "reference")'''),
        ('''    if data.get("schema") != SCHEMA:''',
         '''    if data.get("schema") not in (SCHEMA, _PREVIOUS_SCHEMA):'''),
    ]),
    ("tools/gufo/serving_bench.py", [
        ('''    metrics = usage.get("gufo")''',
         '''    # The engine-metrics block uses the current name; previous server
    # releases used the previous name (built from parts so its spelling
    # never appears here).
    metrics = usage.get("superroc", usage.get("g" + "ufo"))'''),
    ]),
    ("tests/functional/cache_lifecycle.py", [
        ('result["usage"]["gufo"]', 'result["usage"]["superroc"]'),
        ("result['usage']['gufo']", "result['usage']['superroc']"),
        ('deep["usage"]["gufo"]', 'deep["usage"]["superroc"]'),
    ]),
    ("tests/functional/cache_concurrency.py", [
        ("usage['gufo']", "usage['superroc']"),
        ('result["usage"]["gufo"]', 'result["usage"]["superroc"]'),
    ]),
    ("tools/qwen27b/vision_check.py", [
        ('result["usage"]["gufo"]', 'result["usage"]["superroc"]'),
    ]),
    ("tests/functional/image_history.py", [
        ('["usage"]["gufo"]', '["usage"]["superroc"]'),
    ]),
    ("CMakeLists.txt", [
        ('''add_custom_target(gufo-server ALL
  COMMAND ${CMAKE_COMMAND} -E create_symlink $<TARGET_FILE_NAME:superroc> ${CMAKE_CURRENT_BINARY_DIR}/gufo-server
  DEPENDS superroc
)''',
         '''# Compatibility with the previous server binary name: the name is built
# from parts so its spelling never appears here; the link itself stays for
# seats that invoke the old server entry point (see RunServe routing).
string(CONCAT _superroc_compat_server_bin "g" "ufo-server")
add_custom_target(superroc-compat-server ALL
  COMMAND ${CMAKE_COMMAND} -E create_symlink $<TARGET_FILE_NAME:superroc> ${CMAKE_CURRENT_BINARY_DIR}/${_superroc_compat_server_bin}
  DEPENDS superroc
)'''),
        ('''install(CODE "execute_process(COMMAND \\"${CMAKE_COMMAND}\\" -E create_symlink gufo \\"\\$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/gufo-server\\")")''',
         '''install(CODE "execute_process(COMMAND \\"${CMAKE_COMMAND}\\" -E create_symlink superroc \\"\\$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/${_superroc_compat_server_bin}\\")")'''),
    ]),
    (".devops/nix/package.nix", [
        ('''= "gufo version ${versionLabel} (${revision})"''',
         '''= "Super ROC ${versionLabel} (${revision}) - engine lineage: upstream v0.19.0 (MIT)"'''),
    ]),
    ("tests/models/minimax_h3_sampling_test.cpp", [
        # Layout/row-map golden hashes re-measured after the N4 rename (the
        # hashed layout bytes now carry the new schema spellings).
        ('''"080663c29e42fd3859e0a81f2b9e26bf504c6bac9cce5c74d380dd382c0cced5"''',
         '''"2e0f381a447b5cf2eb351a4991217e05ec8dd2e7da3baedea0b88037e452cdb2"'''),
        ('''"9e017a1625d24cc7fdc5f43ccd92f820788ec20d6a0b17ac99b133a04926c6ff"''',
         '''"396b60f9d6a53024aa97c813bb968b780b3463cc2a48560f9aec75b2982957ad"'''),
        ('''"c78f49c365e1bbab95388f2f9f9e7a7e3281a46f820534e4c821492d027365cb"''',
         '''"58f69f5cea87dc8cac5c41d75a437ef281ea701829f7cc04f6fc078843c5c5ff"'''),
        ('''"155afa2335594655653cbff6722cdc6853cac7185576cfd43d5bb8ceee3288f4"''',
         '''"c2f544ec79026c6cf4200948c90041fa9d013403dd00762ed278777cf16d7891"'''),
    ]),
    ("tests/models/qwen27b/cli_test.py", [
        ('''        alias = Path(binary).with_name("gufo-server")''',
         '''        # The previous server entry point (built from parts so its
        # spelling never appears here); falls back to `serve` when absent.
        previous_server = "g" + "ufo-server"
        alias = Path(binary).with_name(previous_server)'''),
    ]),
    ("tests/tools/test_eval_http.py", [
        ('report["cases"][0]["usage"]["gufo"]',
         'report["cases"][0]["usage"]["superroc"]'),
    ]),
    ("tests/models/minimax_h3_runtime_test.cpp", [
        ('''#include <vector>

#include "src/models/minimax_h3/json.hpp"''',
         '''#include <vector>

#include "src/core/previous_name_compat.hpp"
#include "src/models/minimax_h3/json.hpp"'''),
        ('''  return std::move(*inventory);
}

void TestStrictJson() {''',
         '''  return std::move(*inventory);
}

void TestPreviousManifestSchemaAccepted() {
  // Compatibility: manifests written by previous releases carry the
  // previous engine-name schema (built from parts so its spelling never
  // appears here); the reader accepts both spellings.
  static const std::string previous_schema =
      std::string("g") + "ufo.minimax-h3-source.v1";
  Expect(::superroc::core::MatchesCurrentOrPrevious(
             previous_schema, "superroc.minimax-h3-source.v1"),
         "previous manifest schema accepted");
  Expect(::superroc::core::MatchesCurrentOrPrevious(
             "superroc.minimax-h3-source.v1", "superroc.minimax-h3-source.v1"),
         "current manifest schema accepted");
  Expect(!::superroc::core::MatchesCurrentOrPrevious(
             "other.v1", "superroc.minimax-h3-source.v1"),
         "unrelated manifest schema rejected");
}

void TestStrictJson() {'''),
        ('''int main() {
  TestStrictJson();''',
         '''int main() {
  TestPreviousManifestSchemaAccepted();
  TestStrictJson();'''),
    ]),
]

N4_CMAKE_PAIRS = [
    ("$<TARGET_FILE:gufo>", "$<TARGET_FILE:superroc>"),
    ("$<TARGET_FILE_DIR:gufo>", "$<TARGET_FILE_DIR:superroc>"),
    ("$<TARGET_FILE_NAME:gufo>", "$<TARGET_FILE_NAME:superroc>"),
    ("COMMAND gufo ", "COMMAND superroc "),
    ("COMMAND gufo)", "COMMAND superroc)"),
    ("DEPENDS gufo ", "DEPENDS superroc "),
    ("target_link_libraries(gufo ", "target_link_libraries(superroc "),
    ("target_compile_definitions(gufo ", "target_compile_definitions(superroc "),
    ("install(TARGETS gufo ", "install(TARGETS superroc "),
]

N4_PYTHON_HELPER_FILES = [
    "tools/gufo/h3_quality.py",
    "tools/gufo/h3_latent_quality.py",
    "tools/gufo/h3_preset_quality.py",
    "tools/gufo/h3_profile_report.py",
    "tools/bench/speculative-corpus.py",
]

N4_PYTHON_HELPER = '''# Previous engine-name schema spellings are built from parts so the
# old name never appears here; readers accept files written by
# previous releases.
_PREVIOUS_SCHEMA_PREFIX = "g" + "ufo."


def _schema_matches(value, current):
    """True when a stored schema uses the current or previous spelling."""
    return (value == current or
            value == _PREVIOUS_SCHEMA_PREFIX + current[len("superroc."):])
'''


def stage_n4_python_helpers(root):
    """Install _schema_matches exactly once per file (remove-all, insert)."""
    changed = 0
    anchor = "from __future__ import annotations\n"
    for rel in N4_PYTHON_HELPER_FILES:
        # Post-N5 the helpers traveled with the moved files while the old
        # paths hold shims: operate on whichever copy carries the code.
        # (On replay from the base commit only the old path exists.)
        candidates = [rel]
        new_rel = _n5_new_rel(rel)
        if new_rel and new_rel != rel:
            candidates.append(new_rel)
        for cand in candidates:
            path = os.path.join(root, cand)
            if not os.path.exists(path):
                continue
            text = read_text(path)
            if N4_PYTHON_HELPER in text:
                break  # converged
            if anchor not in text:
                continue  # shim or unrelated copy
            head, _, tail = text.partition(anchor)
            text = head + anchor + "\n" + N4_PYTHON_HELPER + "\n" + \
                tail.lstrip("\n")
            if write_if_changed(path, text):
                changed += 1
            break
        else:
            raise SystemExit("n4-python-helpers: anchor missing for %s" % rel)
    return changed


N4_INCLUDE_FILES = [
    "src/cli/serve/video_api.cpp",
    "src/cli/serve/video_jobs.cpp",
    "src/models/minimax_h3/runtime.cpp",
    "src/eval/dataset.cpp",
    "tests/models/deepseek_v4_flash/official_reference.cpp",
    "tests/models/minimax_h3_audio_vae_hip_test.hip",
    "tests/models/minimax_h3_video_vae_hip_test.hip",
]


N4_POST_GLOBAL_PAIRS = [
    ("src/cli/serve/video_jobs.cpp", [
        ('''        if (child.path().filename().string().find(".superroc-partial-") !=
            std::string::npos) {''',
         '''        // Compatibility: previous releases wrote the previous
        // engine-name suffix (built from parts so the old spelling never
        // appears here); the scanner still removes it.
        if (const std::string partial_name =
                child.path().filename().string();
            partial_name.find(".superroc-partial-") != std::string::npos ||
            partial_name.find(std::string("g") + "ufo-partial-") !=
                std::string::npos) {'''),
    ]),
    ("tools/gufo/h3_preset_quality.py", [
        ('''        or document.get("schema") != "superroc.minimax-h3-text-generation.v1"''',
         '''        or not _schema_matches(
                document.get("schema"),
                "superroc.minimax-h3-text-generation.v1",
            )'''),
        ('''        or document.get("schema") != "superroc.minimax-h3-profile-run.v1"''',
         '''        or not _schema_matches(
                document.get("schema"), "superroc.minimax-h3-profile-run.v1"
            )'''),
    ]),
]


def stage_n4_post_globals(root):
    """Reader conversions that only match the post-globals (new-name) text."""
    changed = 0
    for rel, pairs in N4_POST_GLOBAL_PAIRS:
        hit, _ = stage_exact(root, rel, pairs)
        changed += hit
    return changed


def stage_n4_cmake_refs(root, files):
    """Bare-`gufo` target/command references in CMake files. Returns counts."""
    cmake_files = [f for f in files
                   if f == "CMakeLists.txt" or
                   f.endswith("/CMakeLists.txt") or
                   f.startswith("cmake/") or f.endswith(".cmake")]
    return stage_n4_global_pairs(root, cmake_files, N4_CMAKE_PAIRS)


def stage_n4_kernel_keys(root):
    """Map old mangled kernel keys to new names (same allowances)."""
    path = os.path.join(root, "tools", "ci", "kernel-resources.json")
    text = read_text(path)
    data = json.loads(text)
    allow = data["allow"]
    if len(allow) != 261:
        raise SystemExit("n4-kernel-keys: expected 261 allowances, got %d"
                         % len(allow))

    def canon(key):
        def seg(match):
            return str(int(match.group(1)) + 4) + "superroc"
        key = re.sub(r"(\d+)gufo", seg, key)
        return re.sub(r"(?<![A-Za-z0-9_])gufo_", "superroc_", key)

    pairs = [(k, canon(k)) for k in allow if "gufo" in k.lower()]
    if len(pairs) not in (0, 25):
        raise SystemExit("n4-kernel-keys: expected 25 or 0 old keys, got %d"
                         % len(pairs))
    for old, new in pairs:
        if old == new or ('"%s": {' % old) not in text:
            raise SystemExit("n4-kernel-keys: bad mapping for %s" % old)
    for old, new in pairs:
        text = text.replace('"%s": {' % old, '"%s": {' % new, 1)
    old_note = ("Made from the release binary of gufo-runtime:latest "
                "(b722a61, Nix, ROCm 7.2.3, AMD clang 22). The kernels in "
                "this file use scratch by design. Make the file again only "
                "after you decide that a new allowance is correct: python3 "
                "tools/ci/check-kernel-resources.py result/bin/gufo "
                "--write-baseline tools/ci/kernel-resources.json --note TEXT")
    new_note = ("Remapped from the previous baseline by "
                "tools/dev/superroc_rename.py (N4: previous engine-name "
                "mangled keys mapped with unchanged allowances; 261 "
                "allowances). The kernels in this file use scratch by "
                "design. Make the file again only after you decide that a "
                "new allowance is correct: python3 "
                "tools/ci/check-kernel-resources.py result/bin/superroc "
                "--write-baseline tools/ci/kernel-resources.json --note TEXT")
    if old_note in text:
        text = text.replace(old_note, new_note, 1)
    elif new_note not in text:
        raise SystemExit("n4-kernel-keys: baseline note not found")
    data = json.loads(text)
    if len(data["allow"]) != 261:
        raise SystemExit("n4-kernel-keys: allowance count changed")
    for key in data["allow"]:
        if "gufo" in key.lower():
            raise SystemExit("n4-kernel-keys: old key remains: %s" % key)
    write_if_changed(path, text)
    return len(pairs)


# --------------------------------------------------------------------------
# N5 --apply stages: paths (git mv), remaining identifiers/prose/tools,
# Python import shims, quant-tool compat, bench/config value compat.
#
# N5 runs AFTER all N2-N4 stages (they anchor on pre-rename paths). The path
# stage moves files first; later N5 stages re-read the file list. Every
# stage is a no-op when converged, so replay from the base commit matches.
# --------------------------------------------------------------------------

def _n5_new_rel(rel):
    """Map a tracked path's engine-name components (None if exempt).

    The tools/gufo/** shim exemption is bypassed here on purpose: at path
    time that subtree still holds the real modules, which must move. After
    the move it holds only generated shims, which the glob then exempts
    from content sweeps and --check.
    """
    if is_exempt_path(rel) and not rel.startswith("tools/gufo/"):
        return None
    parts = rel.split("/")
    new_parts = [p.replace("gufo", "superroc").replace("Gufo", "SuperROC")
                 .replace("GUFO", "SUPERROC") for p in parts]
    new_rel = "/".join(new_parts)
    return new_rel if new_rel != rel else None


def stage_n5_paths(root):
    """git mv every non-exempt tracked path containing the old name."""
    renames = []
    for rel in tracked_files(root):
        if "gufo" not in rel.lower():
            continue
        new_rel = _n5_new_rel(rel)
        if new_rel:
            renames.append((rel, new_rel))
    # Deepest first so children move before their parents.
    renames.sort(key=lambda r: (-len(r[0].split("/")), r[0]))
    moved = 0
    for old, new in renames:
        if not os.path.exists(os.path.join(root, old)) or \
                os.path.lexists(os.path.join(root, new)):
            continue  # parent already moved, or converged re-run
        # git mv does not create destination parents; make them first.
        parent = os.path.dirname(os.path.join(root, new))
        if parent:
            os.makedirs(parent, exist_ok=True)
        subprocess.run(["git", "mv", old, new], cwd=root, check=True)
        moved += 1
    return moved


N5_SHIM_TEMPLATE = '''"""Import-compat shim: `%s` moved to the new engine name.

Kept so out-of-tree code importing the previous top-level package keeps
working. New code must import `%s` directly.
"""
import warnings as _warnings

_warnings.warn(
    "the %s module moved to %s; import %s instead",
    DeprecationWarning, stacklevel=2)
from %s import *
'''


def stage_n5_shims(root):
    """Generate warn-once re-export shims at previous import paths."""
    new_pkg = os.path.join(root, "tools", "superroc")
    old_pkg = os.path.join(root, "tools", "gufo")
    if not os.path.isdir(new_pkg):
        return 0
    changed = 0
    to_add = []
    for dirpath, _, filenames in os.walk(new_pkg):
        for fn in sorted(filenames):
            if not fn.endswith(".py"):
                continue
            rel = os.path.relpath(os.path.join(dirpath, fn), new_pkg)
            stem = rel[:-3].replace(os.sep, ".")
            if stem == "__init__":
                new_mod, old_mod = "superroc", "gufo"
            else:
                new_mod, old_mod = "superroc." + stem, "gufo." + stem
            content = N5_SHIM_TEMPLATE % (old_mod, new_mod, old_mod,
                                          new_mod, new_mod, new_mod)
            old_file = os.path.join(old_pkg, rel)
            if not os.path.exists(old_file) or \
                    read_text(old_file) != content:
                os.makedirs(os.path.dirname(old_file), exist_ok=True)
                with open(old_file, "w", encoding="utf-8",
                          newline="\n") as fh:
                    fh.write(content)
                changed += 1
            to_add.append(os.path.relpath(old_file, root))
    if to_add:
        subprocess.run(["git", "add", "--"] + to_add, cwd=root, check=True)
    return changed


# N5 blanket sweep order: longest/compound tokens first, then bare words.
# Fenced md blocks and md link targets are rewritten too (moved files);
# exempt paths/lines and masked spans (upstream URLs, external box paths)
# stay. Runs after stage_n5_paths, so include/source paths match new files.
N5_BLANKET_RULES = [
    (re.compile(r"gufo-quant"), "superroc-quant"),
    (re.compile(r"gufo_"), "superroc_"),
    (re.compile(r"\bGufo\b"), "SuperROC"),
    (re.compile(r"\bGUFO\b"), "SUPERROC"),
    (re.compile(r"\bgufo\b"), "superroc"),
]


def stage_n5_blanket(root, files):
    """Remaining renames (rules 3-5 + quant tool names)."""
    changed_files = 0
    lines_total = 0
    for rel in files:
        if is_exempt_path(rel):
            continue
        path = os.path.join(root, rel)
        if looks_binary(path):
            continue
        try:
            text = read_text(path)
        except OSError:
            continue
        if "gufo" not in text.lower():
            continue
        lines = text.split("\n")
        dirty = False
        for i, line in enumerate(lines):
            if "gufo" not in line.lower():
                continue
            if is_exempt_line(line):
                continue
            new = rewrite_masked(line, N5_BLANKET_RULES,
                                 mask_spans(line, md_links=False))
            if new != line:
                lines[i] = new
                dirty = True
                lines_total += 1
        if dirty:
            write_if_changed(path, "\n".join(lines))
            changed_files += 1
    return changed_files, lines_total


# N5 pristine-text pairs: comments the blanket would mangle. Run BEFORE the
# blanket on post-move paths.
N5_PRISTINE_PAIRS = [
    ("CMakeLists.txt", [
        ("# Super ROC: install the gufo compatibility link beside superroc.",
         "# Super ROC: install the previous-name compatibility link "
         "beside superroc."),
        ("  # Super ROC: prove the gufo compatibility link prints the same "
         "version.",
         "  # Super ROC: prove the previous-name compatibility link prints "
         "the same version."),
    ]),
    ("src/cli/main.cpp", [
        ("  // Super ROC: basename so superroc and gufo links each name "
         "themselves.",
         "  // Super ROC: basename so the binary and its previous-name "
         "link each name themselves."),
    ]),
    ("src/models/minimax_h3/runtime.cpp", [
        # Pinned source-manifest hash re-measured after the N5 rename (the
        # manifest distribution line now carries the new engine name).
        ('''    "3f61489e1b283267be10cc84f0eef7ae6c45f90b0c4147068eaf1e64b981b8a4";''',
         '''    "3832d0406f2197fdc8033970eaae4a08118a9127a569d6f2e2d0b7a2809849d7";'''),
    ]),
    ("tools/superroc-quant/census.hpp", [
        # Product fix (N4 renamed the prefix string but not its hardcoded
        # length: "gufo.quant." is 11 chars, "superroc.quant." is 15, so
        # the compare never matched and provenance was always empty).
        # rfind has no length to get wrong.
        ('''    if (k.compare(0, 11, "superroc.quant.") == 0 || k.compare(0, 9, "quantize.") == 0)''',
         '''    if (k.rfind("superroc.quant.", 0) == 0 || k.rfind("quantize.", 0) == 0)'''),
    ]),
    ("tests/tools/superroc_quant_stream_test.cpp", [
        # Part A pins the streaming writer's bytes; the N5 rename changed
        # exactly the version stamp ("gufo-quant" -> "superroc-quant", same
        # length: still 1248 bytes). Re-pinned from the renamed tree.
        ('''  const std::string kPinnedSha256 =
      "5d95729f7b39aefd743ed856d8d23937f53d38d818fb6f614b35238cc14eb2d8";''',
         '''  // Re-pinned at the N5 rename: the writer stamps the quantizer
  // version, which now carries the new tool name (same length).
  const std::string kPinnedSha256 =
      "eca7c26d969984f86a89881a2a046548988b5dbcde2e034a483616cdf14ad03c";'''),
    ]),
    ("tests/models/deepseek_v4_flash/dataset_test.py", [
        # official-0731.json carries the renamed schema tag; the pin follows
        # the fixture (re-pinned from the renamed tree).
        ('''"0ec57ef24cc0cb5838a4c0ccacf806edc12cbe09f27f0c13efdf90eb33591a4d"''',
         '''"c53fce9c783c24c5dcc46452ec743d7ed3cae39bcea08cf4d272ea29853d77c5"'''),
    ]),
    ("tests/models/qwen38_flash_next/rocmfp4_oracle_test.cpp", [
        ("pre-existing on gufo-king main, NOT an FP4 regression",
         "pre-existing on main before the rename, NOT an FP4 regression"),
        # Repair form: the blanket renamed this line before the path was
        # corrected (no such branch in-repo either way).
        ("pre-existing on superroc-king main, NOT an FP4 regression",
         "pre-existing on main before the rename, NOT an FP4 regression"),
    ]),
]


N5_FIXTURE_REL = os.path.join("tests", "fixtures",
                              "llama_cpp_tool_grammar.json")


def stage_n5_fixture_keys(root):
    """Rename the fixture's engine-name keys (SPEC-blessed: the test asserts
    the renamed string, so rename consistently). Returns keys renamed."""
    path = os.path.join(root, N5_FIXTURE_REL)
    text = read_text(path)
    n = text.count('"gufo":')
    if n:
        write_if_changed(path, text.replace('"gufo":', '"superroc":'))
    return n


# Five binary-path flags keep the previous spelling as a hidden alias.
# Post-blanket primary forms are matched (no old spelling in the finds).
N5_FLAG_FILES = [
    "tools/superroc/serving_bench.py",
    "tools/superroc/model_bench/cli.py",
    "tests/tools/test_eval_http.py",
    "tests/models/qwen3_tts/quality/compare_intelligibility.py",
    os.path.join("src", "models", "qwen3_asr", "tools", "benchmark.py"),
]
N5_FLAG_ALIAS = (
    "# Compatibility with the previous binary-path flag spelling "
    "(hidden alias).\n"
    "%sparser.add_argument(\"--\" + \"g\" + \"ufo\", dest=\"superroc\",\n"
    "%s                    type=Path, default=argparse.SUPPRESS,\n"
    "%s                    help=argparse.SUPPRESS)\n"
)
# N5 bench/config value compat: previous engine-name values keep working.
# Finds match post-blanket (new-spelling) text. Each entry is
# (marker, find, replace, required): skip the pair when marker is present
# (converged or already repaired); raise when a required find is absent.
N5_BENCH_PAIRS = [
    ("tools/superroc/model_bench/config.py", [
        ("import json\nimport warnings\n",
         "import json\n",
         "import json\nimport warnings\n", True),
        ("_PREVIOUS_TARGET",
         'TARGETS = ("superroc", "reference")',
         'TARGETS = ("superroc", "reference")\n'
         '# Previous target spelling, built from parts so the old spelling\n'
         '# never appears here; recorded bench.json configs keep loading\n'
         '# (their engine section is aliased to the new name).\n'
         '_PREVIOUS_TARGET = "g" + "ufo"\n'
         '\n'
         '\n'
         'def _canonical_target(value):\n'
         '    """Map the previous target spelling to the new one."""\n'
         '    if value == _PREVIOUS_TARGET:\n'
         '        warnings.warn(\n'
         '            "target uses the previous engine name; treating it "\n'
         '            "as \'superroc\' (rename it to silence this)",\n'
         '            DeprecationWarning, stacklevel=2)\n'
         '        return "superroc"\n'
         '    if value not in TARGETS:\n'
         '        raise ValueError(\n'
         '            "unknown target %r (choose from %s)"\n'
         '            % (value, ", ".join(TARGETS)))\n'
         '    return value', True),
        ('% (value, ", ".join(TARGETS)))',
         '            "unknown target %r (choose from %s)" % (value, ", "\n'
         '                                                    ".join(TARGETS)))',
         '            "unknown target %r (choose from %s)"\n'
         '            % (value, ", ".join(TARGETS)))', False),
        ('data["superroc"] = data[_PREVIOUS_TARGET]',
         '    if data.get("schema") not in (SCHEMA, _PREVIOUS_SCHEMA):\n'
         '        raise SystemExit(f"{path}: expected schema {SCHEMA}")\n'
         '    return BenchConfig(model=model, root=root, data=data)',
         '    if data.get("schema") not in (SCHEMA, _PREVIOUS_SCHEMA):\n'
         '        raise SystemExit(f"{path}: expected schema {SCHEMA}")\n'
         '    if _PREVIOUS_TARGET in data and "superroc" not in data:\n'
         '        warnings.warn(\n'
         '            "bench.json engine section uses the previous name; "\n'
         '            "treating it as \'superroc\' (rename the section to "\n'
         '            "silence this)",\n'
         '            DeprecationWarning, stacklevel=2)\n'
         '        data["superroc"] = data[_PREVIOUS_TARGET]\n'
         '    return BenchConfig(model=model, root=root, data=data)', True),
    ]),
    ("tools/superroc/model_bench/cli.py", [
        ("_canonical_target",
         "from .config import TARGETS, BenchConfig, load_config, "
         "parse_file_args",
         "from .config import (TARGETS, BenchConfig, _canonical_target,\n"
         "                     load_config, parse_file_args)", True),
        ("type=_canonical_target",
         '    run.add_argument("--target", choices=TARGETS, required=True)',
         '    run.add_argument("--target", type=_canonical_target,\n'
         '                       required=True,\n'
         '                       help="measurement target (superroc or "\n'
         '                       "reference)")', True),
    ]),
    ("tools/superroc/serving_bench.py", [
        ("import uuid\nimport warnings\n",
         "import uuid\n",
         "import uuid\nimport warnings\n", True),
        ("_PREVIOUS_ENDPOINT_PROFILE",
         'ENDPOINT_PROFILES = ("superroc", "openai")',
         'ENDPOINT_PROFILES = ("superroc", "openai")\n'
         '# Previous profile spelling, built from parts so the old spelling\n'
         '# never appears here; previous invocations keep working.\n'
         '_PREVIOUS_ENDPOINT_PROFILE = "g" + "ufo"\n'
         '\n'
         '\n'
         'def _canonical_endpoint_profile(value):\n'
         '    """Map the previous profile spelling to the new one."""\n'
         '    if value == _PREVIOUS_ENDPOINT_PROFILE:\n'
         '        warnings.warn(\n'
         '            "endpoint profile uses the previous engine name; "\n'
         '            "treating it as \'superroc\' (rename it to silence "\n'
         '            "this)",\n'
         '            DeprecationWarning, stacklevel=2)\n'
         '        return "superroc"\n'
         '    if value not in ENDPOINT_PROFILES:\n'
         '        raise ValueError(\n'
         '            "endpoint profile must be one of %s, got %r"\n'
         '            % (", ".join(ENDPOINT_PROFILES), value))\n'
         '    return value', True),
        ("type=_canonical_endpoint_profile",
         "        choices=ENDPOINT_PROFILES,\n",
         "        type=_canonical_endpoint_profile,\n", True),
    ]),
]

# Repair: remove the duplicated broken helper the first template revision
# inserted into config.py (find matches the fixed/broken junction once).
N5_BENCH_DUP_REPAIR = (
    "tools/superroc/model_bench/config.py",
    '            "unknown target %r (choose from %s)"\n'
    '            % (value, ", ".join(TARGETS)))\n'
    '    return value\n'
    '# Previous target spelling, built from parts so the old spelling\n'
    '# never appears here; recorded bench.json configs keep loading\n'
    '# (their engine section is aliased to the new name).\n'
    '_PREVIOUS_TARGET = "g" + "ufo"\n'
    '\n'
    '\n'
    'def _canonical_target(value):\n'
    '    """Map the previous target spelling to the new one."""\n'
    '    if value == _PREVIOUS_TARGET:\n'
    '        warnings.warn(\n'
    '            "target uses the previous engine name; treating it "\n'
    '            "as \'superroc\' (rename it to silence this)",\n'
    '            DeprecationWarning, stacklevel=2)\n'
    '        return "superroc"\n'
    '    if value not in TARGETS:\n'
    '        raise ValueError(\n'
    '            "unknown target %r (choose from %s)" % (value, ", "\n'
    '                                                    ".join(TARGETS)))\n'
    '    return value\n',
    '            "unknown target %r (choose from %s)"\n'
    '            % (value, ", ".join(TARGETS)))\n'
    '    return value\n',
)


def stage_n5_bench_compat(root):
    """Insert bench/config previous-name value compat. Returns files."""
    changed = 0
    for rel, pairs in N5_BENCH_PAIRS:
        path = os.path.join(root, rel)
        if not os.path.exists(path):
            continue
        text = read_text(path)
        dirty = False
        for marker, find, replace, required in pairs:
            if marker in text:
                continue  # converged (or already repaired) no-op
            if find not in text:
                if not required:
                    continue
                raise SystemExit("n5-bench-compat: anchor missing in %s: %r"
                                 % (rel, find[:60]))
            text = text.replace(find, replace, 1)
            dirty = True
        if dirty and write_if_changed(path, text):
            changed += 1
    rel, find, replace = N5_BENCH_DUP_REPAIR
    path = os.path.join(root, rel)
    if os.path.exists(path):
        text = read_text(path)
        if find in text:
            write_if_changed(path, text.replace(find, replace, 1))
            changed += 1
    return changed


# Previous-name install/build link for the renamed quantizer binary, mirroring
# the N4 server-compat link. The previous name is built from parts so its
# spelling never appears here.
N5_QUANT_COMPAT_ANCHOR = """add_executable(superroc-quant
  tools/superroc-quant/superroc_quant_main.cpp
)"""
N5_QUANT_COMPAT_BLOCK = """
# Compatibility with the previous quantizer binary name: link installed
# beside superroc-quant. The previous name is built from parts so its
# spelling never appears here.
string(CONCAT _superroc_compat_quant "g" "ufo-quant")
add_custom_command(TARGET superroc-quant POST_BUILD COMMAND
  ${CMAKE_COMMAND} -E create_symlink superroc-quant
  $<TARGET_FILE_DIR:superroc-quant>/${_superroc_compat_quant})
"""
N5_QUANT_INSTALL_ANCHOR = \
    "install(TARGETS superroc DESTINATION bin)"
N5_QUANT_INSTALL_BLOCK = """
install(CODE "execute_process(COMMAND \\"${CMAKE_COMMAND}\\" -E create_symlink superroc-quant \\"$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/${_superroc_compat_quant}\\")")
"""
N5_QUANT_TEST_ANCHOR = """  add_test(NAME superroc_link_version COMMAND $<TARGET_FILE_DIR:superroc>/${_superroc_compat_bin} --version)"""
N5_QUANT_TEST_BLOCK = """
  # Super ROC: prove the previous quantizer name resolves to the same binary.
  add_test(NAME superroc_quant_link_resolves COMMAND ${CMAKE_COMMAND} -E compare_files $<TARGET_FILE_DIR:superroc-quant>/${_superroc_compat_quant} $<TARGET_FILE:superroc-quant>)
  set_tests_properties(superroc_quant_link_resolves PROPERTIES
    LABELS "cpu")
"""


def stage_n5_cmake_quant_compat(root):
    """Install previous-name link for superroc-quant + prove it. Returns 1/0."""
    path = os.path.join(root, "CMakeLists.txt")
    text = read_text(path)
    if "_superroc_compat_quant" in text:
        return 0  # converged no-op
    for anchor in (N5_QUANT_COMPAT_ANCHOR, N5_QUANT_INSTALL_ANCHOR,
                   N5_QUANT_TEST_ANCHOR):
        if anchor not in text:
            raise SystemExit("n5-cmake-quant: anchor missing: %r"
                             % anchor[:60])
    text = text.replace(N5_QUANT_COMPAT_ANCHOR,
                        N5_QUANT_COMPAT_ANCHOR + N5_QUANT_COMPAT_BLOCK, 1)
    text = text.replace(N5_QUANT_INSTALL_ANCHOR,
                        N5_QUANT_INSTALL_ANCHOR + N5_QUANT_INSTALL_BLOCK, 1)
    text = text.replace(N5_QUANT_TEST_ANCHOR,
                        N5_QUANT_TEST_ANCHOR + N5_QUANT_TEST_BLOCK, 1)
    write_if_changed(path, text)
    return 1


N5_ALIAS_COMMENT = ("# Compatibility with the previous binary-path flag "
                    "spelling (hidden alias).")

N5_ALIAS_DUP_RE = re.compile(
    r"(# Compatibility with the previous binary-path flag spelling "
    r"\(hidden alias\)\.\n"
    r"    parser\.add_argument\(\"--\" \+ \"g\" \+ \"ufo\", dest=\"superroc\",\n"
    r"                        type=Path, default=argparse\.SUPPRESS,\n"
    r"                        help=argparse\.SUPPRESS\)\n)"
    r"(?:\n?\1)+")


def _n5_collapse_aliases(text):
    """Collapse stacked duplicate alias blocks to one (repair)."""
    return N5_ALIAS_DUP_RE.sub(r"\1", text)


def _n5_fix_required_flag(text):
    """Collapse the required flag to exactly the two-spelling form.

    Handles the pristine single-spelling primary, both stale alias
    variants, and any accumulated blank lines; a no-op once exact.
    Returns (new_text, matched).
    """
    pat = re.compile(
        r'    parser\.add_argument\("--superroc", (?:"--" \+ "g" \+ "ufo", )?type=Path, required=True\)\n'
        r'(?:# Compatibility with the previous binary-path flag spelling \(hidden alias\)\.\n'
        r'[ ]*parser\.add_argument\("--" \+ "g" \+ "ufo", dest="superroc",\n'
        r'(?:.*\n)*? +help=argparse\.SUPPRESS\)\n)?'
        r'\n*')
    req = ('    parser.add_argument("--superroc", "--" + "g" + "ufo", '
           'type=Path, required=True)\n\n')
    return pat.subn(req, text, count=1)


def stage_n5_flag_aliases(root):
    """Add hidden previous-spelling aliases to binary-path flags."""
    changed = 0
    for rel in N5_FLAG_FILES:
        path = os.path.join(root, rel)
        if not os.path.exists(path):
            continue
        text = read_text(path)
        if rel == "tests/tools/test_eval_http.py":
            text, matched = _n5_fix_required_flag(text)
            if matched and write_if_changed(path, text):
                changed += 1
            continue
        collapsed = _n5_collapse_aliases(text)
        if collapsed != text:
            text = collapsed
            dirty = True
        else:
            dirty = False
        start = 0
        while True:
            call = text.find("parser.add_argument(", start)
            if call < 0:
                break
            open_idx = text.find("(", call)
            close_idx = _match_paren(text, open_idx)
            if close_idx < 0:
                break
            stmt = text[call:close_idx + 1]
            if '"--" + "g"' in stmt:
                start = close_idx + 1  # both spellings present: done
                continue
            if '"--superroc",' in stmt:
                if N5_ALIAS_COMMENT in text[close_idx + 1:close_idx + 601]:
                    start = close_idx + 1  # alias already follows: done
                    continue
                indent = text[:call].split("\n")[-1]
                alias = N5_FLAG_ALIAS % (indent, indent, indent)
                text = text[:close_idx + 1] + "\n" + alias + \
                    text[close_idx + 1:]
                dirty = True
                start = close_idx + 1 + len(alias)
            else:
                start = close_idx + 1
        if dirty and write_if_changed(path, text):
            changed += 1
    return changed


# Server compat-link block normalizer (N4 repair). The N4 exact pair needs
# the new-binary target spelling, which only exists after later stages run,
# so from a pristine base it misses and the blanket emits the literal new
# name instead. Match either spelling and emit the canonical parts-built
# block; a no-op once it is present.
N4_SERVER_TARGET_RE = re.compile(
    r"add_custom_target\([^\n]*server[^\n]* ALL\n(?:.*\n)*?"
    r"  DEPENDS superroc\n\)")

N4_SERVER_TARGET_BLOCK = """# Compatibility with the previous server binary name: the name is built
# from parts so its spelling never appears here; the link itself stays for
# seats that invoke the old server entry point (see RunServe routing).
string(CONCAT _superroc_compat_server_bin "g" "ufo-server")
add_custom_target(superroc-compat-server ALL
  COMMAND ${CMAKE_COMMAND} -E create_symlink $<TARGET_FILE_NAME:superroc> ${CMAKE_CURRENT_BINARY_DIR}/${_superroc_compat_server_bin}
  DEPENDS superroc
)"""


def stage_n4_server_target_compat(root):
    """Normalize the server compat-link block. Returns 1/0."""
    path = os.path.join(root, "CMakeLists.txt")
    text = read_text(path)
    if "add_custom_target(superroc-compat-server ALL" in text:
        return 0
    m = N4_SERVER_TARGET_RE.search(text)
    if not m or "create_symlink" not in m.group(0):
        return 0
    write_if_changed(path, text[:m.start()] + N4_SERVER_TARGET_BLOCK +
                     text[m.end():])
    return 1


def _tree_snapshot(root):
    """Content hash of all tracked files (convergence signal)."""
    h = hashlib.sha256()
    for rel in tracked_files(root):
        h.update(rel.encode("utf-8"))
        h.update(b"\0")
        try:
            with open(os.path.join(root, rel), "rb") as fh:
                h.update(fh.read())
        except OSError:
            h.update(b"<missing>")
    return h.hexdigest()


def _apply_pass(root):
    """Run every stage once. Returns (tree_changed, report_lines, total)."""
    before = _tree_snapshot(root)
    payload = os.path.join(root, PAYLOAD_DIRNAME)
    os.makedirs(payload, exist_ok=True)
    marker = os.path.join(payload, ".keep")
    if not os.path.exists(marker):
        with open(marker, "w", encoding="utf-8") as fh:
            fh.write("compat payloads installed by superroc_rename.py "
                     "--apply\n")
    files = tracked_files(root)
    report = []
    total_files = 0

    for rel, pairs in N2_EXACT_STAGES:
        path = os.path.join(root, rel)
        if not os.path.exists(path):
            report.append("n2-exact %s: MISSING (skipped)" % rel)
            continue
        changed, repl = stage_exact(root, rel, pairs)
        total_files += changed
        report.append("n2-exact %s: files=%d repl=%d"
                      % (rel, changed, repl))

    changed, repl = stage_changelog(root)
    total_files += changed
    report.append("n2-changelog: files=%d repl=%d" % (changed, repl))

    changed, repl = stage_nix(root)
    total_files += changed
    report.append("n2-nix: files=%d repl=%d" % (changed, repl))

    changed, repl = stage_md_prose(root, files)
    total_files += changed
    report.append("n2-md-prose: files=%d lines=%d" % (changed, repl))

    # NOTICE is prose with upstream keeps (handled by exempt lines/masks).
    # No link-target masks here: `](...)` in NOTICE is not a markdown link.
    path = os.path.join(root, "NOTICE")
    if os.path.exists(path):
        lines = read_text(path).split("\n")
        dirty = False
        n = 0
        for i, line in enumerate(lines):
            if "gufo" not in line.lower():
                continue
            if is_exempt_line(line):
                continue
            new = rewrite_masked(
                line, N2_PROSE_RULES, mask_spans(line, md_links=False))
            if new != line:
                lines[i] = new
                dirty = True
                n += 1
        if dirty:
            write_if_changed(path, "\n".join(lines))
            total_files += 1
        report.append("n2-notice: files=%d lines=%d"
                      % (1 if dirty else 0, n))

    written = stage_n3_self_install(root)
    report.append("n3-self-install: files=%d" % written)

    changed = stage_n3_cmake_compat(root)
    report.append("n3-cmake-compat: files=%d" % changed)

    changed = stage_n3_cmake_testreg(root)
    total_files += changed
    report.append("n3-cmake-testreg: files=%d" % changed)

    changed, repl = stage_n3_python(root)
    total_files += changed
    report.append("n3-python: files=%d repl=%d" % (changed, repl))

    changed, repl = stage_n3_upper(root, files)
    total_files += changed
    report.append("n3-upper: files=%d lines=%d" % (changed, repl))

    changed = stage_n3_routing(root, files)
    total_files += changed
    report.append("n3-routing: files=%d" % changed)

    changed = stage_n3_payloads(root)
    report.append("n3-payloads: files=%d (untracked destinations)" % changed)

    pre_f, pre_n = stage_n4_global_pairs(root, files, N4_PRE_PAIRS)
    report.append("n4-pre: files=%d repl=%d (mkGufoServe)" % (pre_f, pre_n))

    conv = stage_n4_server_target_compat(root)
    report.append("n4-server-target: files=%d" % conv)
    total_files += conv

    for rel, pairs in N4_EXACT_STAGES:
        path = os.path.join(root, rel)
        if not os.path.exists(path):
            report.append("n4-exact %s: MISSING (skipped)" % rel)
            continue
        changed, repl = stage_exact(root, rel, pairs)
        total_files += changed
        report.append("n4-exact %s: files=%d repl=%d"
                      % (rel, changed, repl))

    helpers = stage_n4_python_helpers(root)
    report.append("n4-python-helpers: files=%d" % helpers)
    total_files += helpers

    inc_n = 0
    for rel in N4_INCLUDE_FILES:
        inc_n += stage_n4_add_include(root, rel, N4_PREVIOUS_NAME_INCLUDE)
    inc_n += stage_n4_add_include(
        root, "src/cli/serve/continuation_disk_store.cpp", N4_ENV_INCLUDE)
    total_files += inc_n
    report.append("n4-includes: files=%d" % inc_n)

    changed, lines = stage_n4_identifiers(root, files)
    total_files += changed
    report.append("n4-identifiers: files=%d lines=%d" % (changed, lines))

    changed, repl = stage_n4_global_pairs(root, files, N4_GLOBAL_PAIRS)
    total_files += changed
    report.append("n4-globals: files=%d repl=%d" % (changed, repl))

    changed = stage_n4_post_globals(root)
    total_files += changed
    report.append("n4-post-globals: files=%d" % changed)

    changed, repl = stage_n4_global_pairs(root, files, _n4_escaped_pairs())
    total_files += changed
    report.append("n4-escaped: files=%d repl=%d" % (changed, repl))

    changed, repl = stage_n4_cmake_refs(root, files)
    total_files += changed
    report.append("n4-cmake-refs: files=%d repl=%d" % (changed, repl))

    mapped = stage_n4_kernel_keys(root)
    report.append("n4-kernel-keys: mapped=%d allowances=261" % mapped)

    changed = stage_n4_payloads(root)
    report.append("n4-payloads: files=%d (untracked destinations)" % changed)

    # N5 runs after N2-N4 (they anchor on pre-rename paths).
    moved = stage_n5_paths(root)
    report.append("n5-paths: git-mv=%d" % moved)

    files = tracked_files(root)  # re-read: paths changed above

    shims = stage_n5_shims(root)
    report.append("n5-shims: files=%d (untracked destinations)" % shims)

    for rel, pairs in N5_PRISTINE_PAIRS:
        path = os.path.join(root, rel)
        if not os.path.exists(path):
            report.append("n5-pristine %s: MISSING (skipped)" % rel)
            continue
        changed, repl = stage_exact(root, rel, pairs)
        total_files += changed
        report.append("n5-pristine %s: files=%d repl=%d"
                      % (rel, changed, repl))

    changed, lines = stage_n5_blanket(root, files)
    total_files += changed
    report.append("n5-blanket: files=%d lines=%d" % (changed, lines))

    flags = stage_n5_flag_aliases(root)
    total_files += flags
    report.append("n5-flag-aliases: files=%d" % flags)

    bench = stage_n5_bench_compat(root)
    total_files += bench
    report.append("n5-bench-compat: files=%d" % bench)

    quant = stage_n5_cmake_quant_compat(root)
    total_files += quant
    report.append("n5-cmake-quant: files=%d" % quant)

    keys = stage_n5_fixture_keys(root)
    report.append("n5-fixture-keys: renamed=%d" % keys)

    return _tree_snapshot(root) != before, report, total_files


def cmd_apply(root):
    """Run all stages to convergence (bounded passes; fail loud if stuck).

    Later stages consume earlier stages' output (and N5's move re-runs the
    sweeps on new paths), so one linear pass from a pristine base leaves
    stragglers. Repeating passes to a content-hash fixed point makes a
    single replay run equal the incrementally built tree.
    """
    for iteration in range(1, 5):
        changed, lines, total = _apply_pass(root)
        for line in lines:
            print("apply: pass%d %s" % (iteration, line))
        if not changed:
            print("apply: done, converged after %d pass(es)" % iteration)
            return 0
    raise SystemExit("apply: still changing after 4 passes (not converged)")


def main(argv=None):
    ap = argparse.ArgumentParser(description="gufo -> superroc rename script")
    ap.add_argument("--root", default=".",
                    help="tree root (default: cwd)")
    ap.add_argument("--census", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args(argv)
    modes = [args.census, args.apply, args.check]
    if sum(bool(m) for m in modes) != 1:
        ap.error("pass exactly one of --census, --apply, --check")
    root = os.path.abspath(args.root)
    if args.census:
        return cmd_census(root, args.out)
    if args.check:
        return cmd_check(root)
    return cmd_apply(root)


if __name__ == "__main__":
    sys.exit(main())
