#!/bin/bash
# package-linux.sh STAGE_DIR OUT_DIR VERSION [LABEL]
# Packages a build-linux.sh install (STAGE_DIR/usr/local) as the archive that is
# attached to a GitHub release:
#   superroc-<LABEL>-linux-x86_64-rocm7.2-gfx1151.tar.gz
#   SHA256SUMS
# VERSION is what `superroc --version` must report: X.Y.Z for a release, or
# `development` for a nightly build. LABEL names the archive (default VERSION).
set -euo pipefail
stage=${1:?stage dir}; out=${2:?out dir}; version=${3:?version}; label=${4:-$3}
here=$(cd "$(dirname "$0")" && pwd)
prefix="$stage/usr/local"
[ -x "$prefix/bin/superroc" ] || { echo "no $prefix/bin/superroc" >&2; exit 1; }
reported=$("$prefix/bin/superroc" --version 2>&1 | head -1)
case "$reported" in
  "Super Roc $version "*) ;;
  *) echo "the binary reports '$reported', not Super Roc $version" >&2; exit 1 ;;
esac
revision=$(printf '%s\n' "$reported" | sed -n 's/^Super Roc [^ ]* (\([0-9a-f]*\)).*/\1/p')

name="superroc-$label-linux-x86_64-rocm7.2-gfx1151"
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
mkdir -p "$tmp/$name"
cp -a "$prefix/bin" "$prefix/share" "$tmp/$name/"
strip --strip-unneeded "$tmp/$name/bin/superroc"
"$tmp/$name/bin/superroc" --version > /dev/null
cp "$prefix/share/licenses/superroc/LICENSE" "$prefix/share/licenses/superroc/NOTICE" \
  "$prefix/share/licenses/superroc/THIRD_PARTY_NOTICES.md" "$tmp/$name/"
sed -e "s/@VERSION@/$version/g" -e "s/@NAME@/$name/g" -e "s/@REVISION@/${revision:-unknown}/g" \
  "$here/INSTALL.md.in" > "$tmp/$name/INSTALL.md"
mkdir -p "$out"
tar -C "$tmp" --owner=0 --group=0 --numeric-owner --sort=name -czf "$out/$name.tar.gz" "$name"
(cd "$out" && sha256sum "$name.tar.gz" > SHA256SUMS)
ls -l "$out/$name.tar.gz"
cat "$out/SHA256SUMS"
