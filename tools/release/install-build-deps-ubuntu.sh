#!/bin/bash
# install-build-deps-ubuntu.sh
# Installs what a release build needs on Ubuntu 24.04: the compiler toolchain,
# the system libraries the binary links, and ROCm 7.2.4 (HIP, hipBLAS, rocBLAS,
# hipBLASLt, hipCUB, rocWMMA) from AMD's package repository. Run as root or with
# sudo available. No GPU is needed to build.
set -euo pipefail
rocm_version=${ROCM_VERSION:-7.2.4}
sudo=""
[ "$(id -u)" -eq 0 ] || sudo=sudo
export DEBIAN_FRONTEND=noninteractive

$sudo apt-get update
$sudo apt-get install -y --no-install-recommends \
  build-essential ca-certificates ccache cmake curl git gnupg ninja-build pkg-config python3 \
  libcurl4-openssl-dev libicu-dev libjpeg-turbo8-dev libpng-dev libssl-dev libwebp-dev zlib1g-dev binutils

$sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://repo.radeon.com/rocm/rocm.gpg.key | gpg --dearmor | $sudo tee /etc/apt/keyrings/rocm.gpg > /dev/null
echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/$rocm_version noble main" |
  $sudo tee /etc/apt/sources.list.d/rocm.list > /dev/null
printf 'Package: *\nPin: release o=repo.radeon.com\nPin-Priority: 600\n' | $sudo tee /etc/apt/preferences.d/rocm-pin-600 > /dev/null
$sudo apt-get update
$sudo apt-get install -y --no-install-recommends \
  rocm-llvm rocm-device-libs hipcc hip-dev hip-runtime-amd comgr hsa-rocr-dev \
  hipblas-dev rocblas-dev hipblaslt-dev hipcub-dev rocprim-dev rocwmma-dev
$sudo apt-get clean
