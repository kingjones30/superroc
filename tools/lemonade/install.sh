#!/bin/bash
# Install Super Roc for use from stock Lemonade (see docs/LEMONADE.md).
#
#   install.sh --superroc PATH_TO_BINARY [--prefix /opt/superroc] [--no-hints]
#
# Copies the binary to PREFIX/bin/superroc and the llama-server shim to
# PREFIX/lemonade/, then prints the two Lemonade commands to run (not with
# --no-hints; tools/lemonade/setup.sh runs them itself).
# No sudo: run it as a user who can write to PREFIX.
set -eu

SUPERROC=""
PREFIX="/opt/superroc"
HINTS=1
HERE="$(cd "$(dirname "$0")" && pwd)"

while [ $# -gt 0 ]; do
    case "$1" in
        --superroc) SUPERROC="$2"; shift 2 ;;
        --prefix) PREFIX="$2"; shift 2 ;;
        --no-hints) HINTS=0; shift ;;
        -h|--help)
            echo "usage: install.sh --superroc PATH_TO_BINARY [--prefix /opt/superroc] [--no-hints]"
            exit 0 ;;
        *) echo "install.sh: unknown argument: $1" >&2; exit 2 ;;
    esac
done

if [ -z "$SUPERROC" ] || [ ! -f "$SUPERROC" ]; then
    echo "install.sh: --superroc PATH_TO_BINARY is required and must be a file" >&2
    exit 2
fi

mkdir -p "$PREFIX/bin" "$PREFIX/lemonade"
# Re-running with the installed binary itself keeps it in place.
if [ ! "$SUPERROC" -ef "$PREFIX/bin/superroc" ]; then
    install -m 0755 "$SUPERROC" "$PREFIX/bin/superroc"
fi
install -m 0755 "$HERE/llama-server" "$PREFIX/lemonade/llama-server"
install -m 0644 "$HERE/superroc-models.json" "$PREFIX/lemonade/superroc-models.json"

echo "Installed:"
echo "  $PREFIX/bin/superroc"
echo "  $PREFIX/lemonade/llama-server"
echo "  $PREFIX/lemonade/superroc-models.json"
[ "$HINTS" -eq 1 ] || exit 0
echo
echo "Now point Lemonade at the shim and add the Super Roc model list:"
echo "  lemonade config set llamacpp.rocm_bin=$PREFIX/lemonade"
echo "  curl -X POST http://localhost:13305/internal/set -H 'Content-Type: application/json' -d '{\"model_catalogs\": [{\"id\": \"superroc\", \"url\": \"https://raw.githubusercontent.com/kingjones30/superroc/main/docs/lemonade/catalog.json\"}]}'"
