#!/bin/bash
# One-step Super Roc setup for a stock Lemonade install (see docs/LEMONADE.md).
#
#   tools/lemonade/setup.sh [options]
#
# 1. Checks this machine: Linux x86-64, an AMD gfx1151 (Strix Halo) GPU and
#    the build tools.
# 2. Builds Super Roc (`cmake --preset release`), unless --superroc names a
#    binary you already have.
# 3. Installs the binary and the Lemonade launcher under PREFIX (install.sh).
#    Uses sudo for that step only when PREFIX is not writable.
# 4. Points the running Lemonade server at the launcher and adds the Super Roc
#    model list. Model lists you already configured are kept.
# 5. With --model NAME, downloads that model through Lemonade.
#
# Run it again at any time: every step is safe to repeat. --dry-run prints the
# steps without changing anything. --undo switches Lemonade back to its own
# llama-server and removes the Super Roc model list.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
CATALOG_URL="https://raw.githubusercontent.com/kingjones30/superroc/main/docs/lemonade/catalog.json"

SUPERROC=""
PREFIX="/opt/superroc"
JOBS=4
LEMONADE_URL="${LEMONADE_URL:-http://localhost:13305}"
MODEL=""
DRY_RUN=0
UNDO=0
NO_LEMONADE=0
FORCE=0

usage() {
    cat <<'EOF'
usage: tools/lemonade/setup.sh [options]

  --superroc PATH      use this superroc binary instead of building one
  --prefix DIR         install location (default /opt/superroc)
  --jobs N             parallel build jobs (default 4)
  --lemonade-url URL   Lemonade server (default $LEMONADE_URL or http://localhost:13305)
  --model NAME         download this Super Roc model through Lemonade at the end
  --no-lemonade        build and install only; do not change Lemonade
  --undo               switch Lemonade back to its own llama-server
  --force              continue when no gfx1151 GPU is found
  --dry-run            print the steps, change nothing
  -h, --help           this help

Lemonade admin access: set LEMONADE_ADMIN_API_KEY (or LEMONADE_API_KEY) when
your Lemonade server requires a key.
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --superroc) SUPERROC="${2:?--superroc needs a path}"; shift 2 ;;
        --prefix) PREFIX="${2:?--prefix needs a directory}"; shift 2 ;;
        --jobs) JOBS="${2:?--jobs needs a number}"; shift 2 ;;
        --lemonade-url) LEMONADE_URL="${2:?--lemonade-url needs a URL}"; shift 2 ;;
        --model) MODEL="${2:?--model needs a name}"; shift 2 ;;
        --no-lemonade) NO_LEMONADE=1; shift ;;
        --undo) UNDO=1; shift ;;
        --force) FORCE=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "setup.sh: unknown argument: $1 (see --help)" >&2; exit 2 ;;
    esac
done

case "$JOBS" in ''|*[!0-9]*) echo "setup.sh: --jobs must be a number" >&2; exit 2 ;; esac
case "$PREFIX" in /*) ;; *) PREFIX="$(pwd)/$PREFIX" ;; esac
PREFIX="${PREFIX%/}"
LEMONADE_URL="${LEMONADE_URL%/}"

step() { printf '\n==> %s\n' "$*"; }
note() { printf '    %s\n' "$*"; }
die() { printf 'setup.sh: %s\n' "$*" >&2; exit 1; }
run() {
    printf '    + %s\n' "$*"
    if [ "$DRY_RUN" -eq 0 ]; then "$@"; fi
}

AUTH=()
KEY="${LEMONADE_ADMIN_API_KEY:-${LEMONADE_API_KEY:-}}"
if [ -n "$KEY" ]; then AUTH=(-H "Authorization: Bearer $KEY"); fi
api() { curl -fsS --max-time 30 ${AUTH[@]+"${AUTH[@]}"} "$@"; }

# Reads Lemonade's config (JSON on stdin) and prints the /internal/set body.
# mode=install: rocm_bin -> the launcher, add the Super Roc catalog.
# mode=undo:    rocm_bin -> builtin, drop the Super Roc catalog.
# mode=check:   exit 0 when the config already matches the install.
lemonade_body() {
    python3 -c '
import json, sys
mode, catalog_url, launcher = sys.argv[1:4]
cfg = json.load(sys.stdin)
llamacpp = cfg.get("llamacpp") if isinstance(cfg.get("llamacpp"), dict) else {}
rocm_bin = llamacpp.get("rocm_bin", cfg.get("llamacpp.rocm_bin", "builtin"))
cats = [c for c in (cfg.get("model_catalogs") or [])
        if not (isinstance(c, dict) and c.get("id") == "superroc")]
if mode == "check":
    ours = any(isinstance(c, dict) and c.get("id") == "superroc" and c.get("url") == catalog_url
               for c in (cfg.get("model_catalogs") or []))
    sys.exit(0 if rocm_bin == launcher and ours else 1)
if mode == "previous":
    print(rocm_bin)
    print(llamacpp.get("backend", "auto"))
    sys.exit(0)
if mode == "install":
    cats.append({"id": "superroc", "url": catalog_url})
    body = {"llamacpp": {"rocm_bin": launcher}, "model_catalogs": cats}
else:
    body = {"llamacpp": {"rocm_bin": "builtin"}, "model_catalogs": cats}
print(json.dumps(body))
' "$@"
}

configure_lemonade() {
    local mode="$1" launcher="$PREFIX/lemonade" cfg body info prev backend
    command -v curl >/dev/null || die "curl is required to configure Lemonade"
    if ! cfg="$(api "$LEMONADE_URL/internal/config" 2>/dev/null)"; then
        printf '\nsetup.sh: no Lemonade server answered at %s.\n' "$LEMONADE_URL" >&2
        printf 'Start Lemonade, then run this again' >&2
        if [ "$mode" = install ]; then
            printf ' (the build and install are done; this skips them):\n' >&2
            printf '  %s --superroc %s --prefix %s --lemonade-url %s\n' \
                "$0" "$PREFIX/bin/superroc" "$PREFIX" "$LEMONADE_URL" >&2
        else
            printf ':\n  %s --undo --lemonade-url %s\n' "$0" "$LEMONADE_URL" >&2
        fi
        printf 'A server that needs a key: set LEMONADE_ADMIN_API_KEY.\n' >&2
        exit 3
    fi
    info="$(printf '%s' "$cfg" | lemonade_body previous "$CATALOG_URL" "$launcher" 2>/dev/null)" ||
        die "$LEMONADE_URL/internal/config did not return Lemonade's settings (is that a Lemonade server?)"
    prev="${info%%$'\n'*}"
    backend="${info#*$'\n'}"
    body="$(printf '%s' "$cfg" | lemonade_body "$mode" "$CATALOG_URL" "$launcher")"
    if [ "$mode" = install ] && [ "$prev" != builtin ] && [ "$prev" != "$launcher" ]; then
        note "Lemonade's llama.cpp ROCm launcher was $prev; it is replaced."
        note "To restore it later: lemonade config set llamacpp.rocm_bin=$prev"
    fi
    note "POST $LEMONADE_URL/internal/set $body"
    if [ "$DRY_RUN" -eq 1 ]; then return 0; fi
    api -X POST -H 'Content-Type: application/json' --data-binary "$body" \
        "$LEMONADE_URL/internal/set" >/dev/null
    cfg="$(api "$LEMONADE_URL/internal/config")"
    if [ "$mode" = install ]; then
        printf '%s' "$cfg" | lemonade_body check "$CATALOG_URL" "$launcher" ||
            die "Lemonade did not keep the new settings; see docs/LEMONADE.md"
        note "Lemonade now starts $launcher/llama-server for llama.cpp ROCm models."
        case "$backend" in
            auto|rocm) ;;
            *) note "Lemonade's llama.cpp backend is '$backend'. Super Roc runs on the ROCm backend:"
               note "  lemonade config set llamacpp.backend=rocm" ;;
        esac
    else
        note "Lemonade uses its own llama-server again."
    fi
}

if [ "$UNDO" -eq 1 ]; then
    step "Switching Lemonade back to its own llama-server"
    configure_lemonade undo
    if [ -d "$PREFIX" ]; then
        note "Installed files are left in $PREFIX; delete that folder to remove them."
    fi
    exit 0
fi

step "Checking this machine"
[ "$(uname -s)" = Linux ] || die "Super Roc runs on Linux x86-64 (this is $(uname -s))"
[ "$(uname -m)" = x86_64 ] || die "Super Roc runs on Linux x86-64 (this is $(uname -m))"
command -v python3 >/dev/null || die "python3 is required (the Lemonade launcher is a Python script)"
gpu=0
if command -v rocminfo >/dev/null && rocminfo 2>/dev/null | grep -q 'gfx1151'; then
    gpu=1
elif grep -qs 'gfx_target_version 110501' /sys/class/kfd/kfd/topology/nodes/*/properties; then
    gpu=1
fi
if [ "$gpu" -eq 1 ]; then
    note "AMD gfx1151 (Strix Halo) GPU found."
elif [ "$FORCE" -eq 1 ] || [ "$DRY_RUN" -eq 1 ]; then
    note "WARNING: no gfx1151 GPU found; continuing."
else
    die "no AMD gfx1151 (Strix Halo) GPU found. Super Roc targets that GPU; --force continues anyway."
fi

if [ -n "$SUPERROC" ]; then
    [ -f "$SUPERROC" ] || die "--superroc $SUPERROC is not a file"
    BIN="$(cd "$(dirname "$SUPERROC")" && pwd)/$(basename "$SUPERROC")"
    step "Using the Super Roc binary $BIN"
else
    step "Building Super Roc (release preset, $JOBS jobs)"
    missing=()
    command -v cmake >/dev/null || missing+=(cmake)
    command -v ninja >/dev/null || missing+=(ninja-build)
    command -v hipcc >/dev/null || [ -x "${ROCM_PATH:-/opt/rocm}/bin/hipcc" ] || missing+=("ROCm (hipcc)")
    if [ "${#missing[@]}" -gt 0 ]; then
        die "missing build tools: ${missing[*]}. Install the packages listed in README.md (Build from source), then run this again."
    fi
    note "The first build takes a while; later runs only rebuild what changed."
    (cd "$ROOT" && run cmake --preset release && run cmake --build --preset release --parallel "$JOBS")
    BIN="$ROOT/build/release/superroc"
fi
if [ "$DRY_RUN" -eq 0 ]; then
    note "$("$BIN" --version 2>&1 | head -1)"
fi

step "Installing into $PREFIX"
probe="$PREFIX"
while [ ! -e "$probe" ]; do probe="$(dirname "$probe")"; done
SUDO=()
for d in "$probe" "$PREFIX/bin" "$PREFIX/lemonade"; do
    if [ -e "$d" ] && [ ! -w "$d" ]; then SUDO=(sudo); fi
done
if [ "${#SUDO[@]}" -gt 0 ]; then
    command -v sudo >/dev/null ||
        die "$PREFIX is not writable and sudo is missing; pick one you own, for example --prefix $HOME/.local/superroc"
    note "$PREFIX is not writable by $(id -un); using sudo for this step."
fi
run ${SUDO[@]+"${SUDO[@]}"} "$HERE/install.sh" --superroc "$BIN" --prefix "$PREFIX" --no-hints

if [ "$NO_LEMONADE" -eq 1 ]; then
    step "Done (Lemonade not changed)"
    note "Point Lemonade at Super Roc later with:"
    note "  $0 --superroc $PREFIX/bin/superroc --prefix $PREFIX"
    exit 0
fi

step "Configuring Lemonade at $LEMONADE_URL"
configure_lemonade install

LEMONADE_CLI=()
if command -v lemonade >/dev/null; then
    hostport="${LEMONADE_URL#*://}"; hostport="${hostport%%/*}"
    host="${hostport%:*}"; port="${hostport##*:}"
    [ "$port" = "$hostport" ] && port=13305
    LEMONADE_CLI=(lemonade --host "$host" --port "$port")
fi

if [ -n "$MODEL" ]; then
    step "Downloading $MODEL through Lemonade"
    if [ "${#LEMONADE_CLI[@]}" -gt 0 ]; then
        run "${LEMONADE_CLI[@]}" pull "$MODEL"
    else
        note "The lemonade command is not on PATH; run: lemonade pull $MODEL"
    fi
fi

step "Done"
try="${MODEL:-LFM2.5-1.2B-Instruct}"
if [ -z "$MODEL" ]; then
    note "Download a small model first:  lemonade pull $try"
fi
note "Chat with it:                  lemonade chat $try"
note "Models Super Roc does not run still go to Lemonade's own llama-server."
undo="$0 --undo"
[ "$LEMONADE_URL" = http://localhost:13305 ] || undo="$undo --lemonade-url $LEMONADE_URL"
note "Switch back at any time:       $undo"
