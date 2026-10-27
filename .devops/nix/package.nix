{
  lib,
  stdenv,
  cmake,
  ninja,
  pkg-config,
  makeWrapper,
  icu,
  curl,
  libpng,
  libjpeg,
  libwebp,
  openssl,
  ffmpeg-headless,
  rocmPackages,
  releaseVersion ? null,
  revision,
}:

let
  # Keep SVM available for managed model/state allocations, but prevent
  # compaction of ROCr's queue save areas and HIP's runtime-owned buffers from
  # suspending inference.
  rocmRuntime = rocmPackages.rocm-runtime.overrideAttrs (old: {
    patches = (old.patches or [ ]) ++ [ ./patches/rocr-fixed-cwsr.patch ];
  });
  versionLabel = if releaseVersion == null then "development" else releaseVersion;
  packageVersion =
    if releaseVersion == null then "0-unstable-${revision}" else releaseVersion;
  sourceRoot = ../..;
  productionSource = lib.cleanSourceWith {
    src = sourceRoot;
    filter =
      path: _type:
      let
        root = toString sourceRoot;
        pathString = toString path;
        relativePath = lib.removePrefix "${root}/" pathString;
      in
      pathString == root
      || builtins.elem relativePath [ "CMakeLists.txt" "LICENSE" "NOTICE" "THIRD_PARTY_NOTICES.md" ]
      || relativePath == "licenses"
      || lib.hasPrefix "licenses/" relativePath
      || relativePath == "cmake"
      || lib.hasPrefix "cmake/" relativePath
      || relativePath == "src"
      || lib.hasPrefix "src/" relativePath
      || relativePath == "tests"
      || relativePath == "tests/models"
      || relativePath == "tests/models/deepseek_v4_flash"
      || relativePath == "tests/models/deepseek_v4_flash/fixtures"
      || relativePath == "tests/models/deepseek_v4_flash/fixtures/antirez-ds4.json";
  };
in
stdenv.mkDerivation {
  pname = "superroc";
  version = packageVersion;
  src = productionSource;

  nativeBuildInputs = [
    cmake
    ninja
    pkg-config
    makeWrapper
    rocmPackages.clr
  ];

  buildInputs = [
    icu
    curl
    libpng
    libjpeg
    libwebp
    openssl
    ffmpeg-headless
    rocmPackages.clr
    rocmPackages.hipblas
    rocmPackages.hipblaslt
    rocmPackages.hipcub
    rocmPackages.rocprim
    rocmPackages.rocwmma
    rocmPackages.rocblas
  ];

  cmakeFlags = [
    "-DCMAKE_BUILD_TYPE=RelWithDebInfo"
    "-DBUILD_TESTING=OFF"
    "-DSUPERROC_RELEASE_VERSION=${versionLabel}"
    "-DSUPERROC_REVISION=${revision}"
    "-DSUPERROC_FFMPEG_EXECUTABLE=${ffmpeg-headless}/bin/ffmpeg"
    "-DSUPERROC_FFPROBE_EXECUTABLE=${ffmpeg-headless}/bin/ffprobe"
    "-DCMAKE_HIP_COMPILER=${rocmPackages.llvm.clang}/bin/clang"
  ];

  doInstallCheck = true;
  postFixup = ''
    # HIP loads ROCr transitively. Select the patched ABI-compatible runtime
    # without rebuilding the unchanged HIP/BLAS compiler and kernel libraries.
    wrapProgram "$out/bin/superroc" \
      --prefix LD_LIBRARY_PATH : "${rocmRuntime}/lib"
  '';
  installCheckPhase = ''
    runHook preInstallCheck

    test "$($out/bin/superroc --version)" = "Super ROC ${versionLabel} (${revision}) - engine lineage: upstream v0.19.0 (MIT)"
    $out/bin/superroc --help >/dev/null
    test "$($out/bin/superroc-server --version)" = "Super ROC ${versionLabel} (${revision}) - engine lineage: upstream v0.19.0 (MIT)"
    $out/bin/superroc-server --help >/dev/null
    test ! -e $out/bin/superroc-kernel-bench
    test -f $out/share/licenses/superroc/LICENSE
    test -f $out/share/licenses/superroc/third-party/LICENSE.ds4

    runHook postInstallCheck
  '';

  passthru = {
    inherit releaseVersion revision rocmPackages rocmRuntime;
    toolchain = {
      targetPlatform = "x86_64-linux";
      targetGpu = "gfx1151";
      cxxCompiler = stdenv.cc.name;
      rocmVersion = rocmPackages.clr.version;
      hipClangVersion = rocmPackages.llvm.clang.version;
    };
  };

  meta = with lib; {
    description = "SuperROC Engine — local inference runtime for AMD Strix Halo (gfx1151 GPU)";
    homepage = "https://github.com/gufo-org/gufo";
    license = licenses.mit;
    platforms = [ "x86_64-linux" ];
    mainProgram = "superroc";
  };
}
