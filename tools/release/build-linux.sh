#!/bin/bash
# build-linux.sh SOURCE_DIR BUILD_DIR STAGE_DIR [VERSION]
# Builds the Linux x86_64 gfx1151 binary the way release packages are built and
# installs it under STAGE_DIR/usr/local (DESTDIR install, prefix /usr/local).
# VERSION (X.Y.Z) sets the release identity; without it the binary reports
# `Super Roc development (<revision>)`, as nightly builds do. The build uses the
# `release` preset; file names in diagnostics and debug information are mapped
# to `superroc/...`, so the binary carries no paths of the build machine. The
# installed binary looks for the ROCm libraries in /opt/rocm/lib, the standard
# ROCm location; LD_LIBRARY_PATH still takes precedence.
# ROCM_PATH (default /opt/rocm) selects ROCm; JOBS sets the build parallelism;
# CCACHE=1 compiles through ccache.
set -euo pipefail
src=$(cd "${1:?source dir}" && pwd)
build=${2:?build dir}
stage=${3:?stage dir}
version=${4:-}
mkdir -p "$build" "$stage"
build=$(cd "$build" && pwd)
stage=$(cd "$stage" && pwd)
case "$build/" in "$src"/*) echo "build dir must be outside the source dir" >&2; exit 1 ;; esac

rocm=${ROCM_PATH:-/opt/rocm}
[ -x "$rocm/lib/llvm/bin/clang++" ] || { echo "no ROCm HIP compiler at $rocm/lib/llvm/bin/clang++ (set ROCM_PATH)" >&2; exit 1; }
export ROCM_PATH=$rocm

map="-ffile-prefix-map=$src=superroc -ffile-prefix-map=$build=superroc/build"
args=(-S "$src" -B "$build" --preset release -DSUPERROC_RELOCATABLE=ON -DCMAKE_INSTALL_PREFIX=/usr/local
  "-DCMAKE_HIP_COMPILER=$rocm/lib/llvm/bin/clang++" "-DCMAKE_PREFIX_PATH=$rocm"
  "-DCMAKE_INSTALL_RPATH=/opt/rocm/lib"
  "-DCMAKE_C_FLAGS=$map" "-DCMAKE_CXX_FLAGS=$map" "-DCMAKE_HIP_FLAGS=$map")
[ -n "$version" ] && args+=("-DSUPERROC_RELEASE_VERSION=$version")
[ "${CCACHE:-0}" = 1 ] && args+=(-DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache
  -DCMAKE_HIP_COMPILER_LAUNCHER=ccache)
cmake "${args[@]}"
cmake --build "$build" -j "${JOBS:-$(nproc)}"
rm -rf "${stage:?}/usr"
DESTDIR="$stage" cmake --install "$build"

bin="$stage/usr/local/bin/superroc"
if strings -n 8 "$bin" | grep -qF -e "$src" -e "$build"; then
  echo "the binary still names the build machine's source or build directory:" >&2
  strings -n 8 "$bin" | grep -F -e "$src" -e "$build" | sort | uniq -c | sort -rn | head -5 >&2
  exit 1
fi
"$bin" --version
