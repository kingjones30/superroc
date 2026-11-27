#!/bin/bash
# package-linux.sh BINARY VERSION OUT_DIR
# Builds the Linux x86_64 binary package that is attached to a GitHub release:
#   superroc-<VERSION>-linux-x86_64-rocm7.2-gfx1151.tar.gz  (bin/superroc + licenses + INSTALL.md)
#   SHA256SUMS                                               (checksum of the tarball)
# BINARY is a Release build of this tree (`cmake --build build --target superroc`) whose `--version`
# reports VERSION. Run it on the build machine; upload with:
#   gh release upload v<VERSION> OUT_DIR/superroc-<VERSION>-linux-x86_64-rocm7.2-gfx1151.tar.gz OUT_DIR/SHA256SUMS
set -euo pipefail
bin=${1:?binary}; ver=${2:?version}; out=${3:?out dir}
here=$(cd "$(dirname "$0")" && pwd); root=$(cd "$here/../.." && pwd)
"$bin" --version 2>&1 | head -1 | grep -q "Super Roc $ver " || { echo "binary does not report Super Roc $ver: $("$bin" --version 2>&1 | head -1)" >&2; exit 1; }
name="superroc-$ver-linux-x86_64-rocm7.2-gfx1151"
stage=$(mktemp -d); trap 'rm -rf "$stage"' EXIT
mkdir -p "$stage/$name/bin"
cp "$bin" "$stage/$name/bin/superroc"
strip --strip-unneeded "$stage/$name/bin/superroc"
"$stage/$name/bin/superroc" --version > /dev/null 2>&1 || { echo "stripped binary does not start" >&2; exit 1; }
cp "$root/LICENSE" "$root/NOTICE" "$root/THIRD_PARTY_NOTICES.md" "$stage/$name/"
sed "s/@VERSION@/$ver/g" "$here/INSTALL.md.in" > "$stage/$name/INSTALL.md"
mkdir -p "$out"
tar -C "$stage" --owner=0 --group=0 --numeric-owner -czf "$out/$name.tar.gz" "$name"
(cd "$out" && sha256sum "$name.tar.gz" > SHA256SUMS)
ls -la "$out/$name.tar.gz"; cat "$out/SHA256SUMS"
