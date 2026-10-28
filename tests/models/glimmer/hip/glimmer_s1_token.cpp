// GLIMMER S1 decode at a real context. One GGUF, read in place.
// usage: glimmer_s1_token <gguf> [context]
// With no context, runs 2048, 8192 and 32768. Position 0. The long score
// span is glimmer_s1_test; this job proves the KV allocation and the step.
// Exit 0 pass, 1 the step failed, 2 usage, 77 skipped (no file named).

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>

#include "src/core/gguf_reader.hpp"
#include "src/models/qwen/hip/executor.hpp"

namespace {

int RunOne(const char* path, std::uint32_t context) {
  try {
    std::string err;
    auto opened = superroc::core::GgufReader::OpenFile(path, &err);
    if (opened == nullptr) {
      std::cerr << "SKIP glimmer_s1_token: " << path << " unreadable: " << err
                << '\n';
      return 77;
    }
    std::shared_ptr<const superroc::core::GgufReader> reader(std::move(opened));

    superroc::hip::QwenExecutionPolicy policy;
    policy.kv_cache_storage = superroc::hip::QwenKvCacheStorage::kFp32;
    policy.fuse_qk_norm_rope_kv = false;
    auto executor = superroc::hip::QwenGpuExecutor::CreateFromGguf(
        reader, &err, context, policy);
    if (executor == nullptr) {
      std::cerr << "FAIL load context=" << context << " " << err << '\n';
      return 1;
    }
    const auto& config = executor->GetConfig();
    if (!config.attention_only || config.architecture != "muse-glimmer") {
      std::cerr << "FAIL not muse-glimmer attention-only\n";
      return 1;
    }
    if (executor->GetMaxContext() < context) {
      std::cerr << "FAIL context " << executor->GetMaxContext() << " < "
                << context << '\n';
      return 1;
    }
    const auto bos = executor->GetTokenizer().GetBosTokenId();
    if (bos >= config.vocab_size) {
      std::cerr << "FAIL bos out of range\n";
      return 1;
    }
    const auto next = executor->ForwardToken(bos, 0, true);
    std::cout << path << " S1 token bos=" << bos << " next=" << next
              << " vocab=" << config.vocab_size << " context=" << context
              << " fp32_kv=1 fuse_qk_norm_rope_kv=0\n";
    std::cout << "S1 token PASS context=" << context << '\n';
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "FAIL context=" << context << " " << ex.what() << '\n';
    return 1;
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc > 3) {
    std::cerr << "usage: glimmer_s1_token <gguf> [context]\n";
    return 2;
  }
  // The muse-glimmer file is named on the command line or, under ctest, by
  // GLIMMER_S1_GGUF. A file that is neither named nor present skips (ctest
  // reports Skipped, exit 77) instead of failing, so a machine without the
  // weights still builds green.
  const char* configured = argc >= 2 ? argv[1] : std::getenv("GLIMMER_S1_GGUF");
  if (configured == nullptr) {
    std::cerr << "SKIP glimmer_s1_token: GLIMMER_S1_GGUF is not set\n";
    return 77;
  }
  const std::string path = configured;
  if (argc == 3) {
    return RunOne(path.c_str(),
                  static_cast<std::uint32_t>(std::stoul(argv[2])));
  }
  const std::uint32_t contexts[] = {2048U, 8192U, 32768U};
  for (const std::uint32_t context : contexts) {
    const int rc = RunOne(path.c_str(), context);
    if (rc != 0) {
      return rc;
    }
  }
  std::cout << "S1 token contexts PASS\n";
  return 0;
}
