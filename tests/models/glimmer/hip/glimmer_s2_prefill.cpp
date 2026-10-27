// GLIMMER S2 prefill. One GGUF, read in place. Context 512.
// A one-token chunk must match ForwardToken. A 32-token chunk must return
// an in-vocab token. Exit 0 pass, 1 fail, 2 usage, 77 skipped (no file).

#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/qwen/hip/executor.hpp"

int main(int argc, char** argv) {
  if (argc > 2) {
    std::cerr << "usage: glimmer_s2_prefill <gguf>\n";
    return 2;
  }
  // The muse-glimmer file is named on the command line or, under ctest, by
  // GLIMMER_S2_GGUF. A file that is neither named nor present skips (ctest
  // reports Skipped, exit 77) instead of failing, so a machine without the
  // weights still builds green.
  const char* configured = argc == 2 ? argv[1] : std::getenv("GLIMMER_S2_GGUF");
  if (configured == nullptr) {
    std::cerr << "SKIP glimmer_s2_prefill: GLIMMER_S2_GGUF is not set\n";
    return 77;
  }
  const std::string path = configured;
  try {
    std::string err;
    auto opened = superroc::core::GgufReader::OpenFile(path, &err);
    if (opened == nullptr) {
      std::cerr << "SKIP glimmer_s2_prefill: " << path << " unreadable: " << err
                << '\n';
      return 77;
    }
    std::shared_ptr<const superroc::core::GgufReader> reader(std::move(opened));
    superroc::hip::QwenExecutionPolicy policy;
    policy.kv_cache_storage = superroc::hip::QwenKvCacheStorage::kFp32;
    policy.fuse_qk_norm_rope_kv = false;
    constexpr std::uint32_t kContext = 512;
    auto executor = superroc::hip::QwenGpuExecutor::CreateFromGguf(
        reader, &err, kContext, policy);
    if (executor == nullptr) {
      std::cerr << "FAIL load " << err << '\n';
      return 1;
    }
    const auto& config = executor->GetConfig();
    const auto bos = executor->GetTokenizer().GetBosTokenId();
    if (bos >= config.vocab_size) {
      std::cerr << "FAIL bos out of range\n";
      return 1;
    }
    const superroc::tokenization::TokenId one[] = {bos};
    const auto prefill_one = executor->ForwardPromptBatch(one, 0, true);
    executor->Reset();
    const auto decode_one = executor->ForwardToken(bos, 0, true);
    std::cout << path << " S2 one prefill=" << prefill_one
              << " decode=" << decode_one << " context=" << kContext << '\n';
    if (prefill_one != decode_one) {
      std::cerr << "FAIL one-token prefill disagrees with decode\n";
      return 1;
    }

    executor->Reset();
    std::vector<superroc::tokenization::TokenId> tokens(32);
    for (std::size_t i = 0; i < tokens.size(); ++i) {
      tokens[i] = static_cast<superroc::tokenization::TokenId>(
          (bos + static_cast<superroc::tokenization::TokenId>(i)) %
          config.vocab_size);
    }
    const auto next = executor->ForwardPromptBatch(tokens, 0, true);
    std::cout << "S2 prefill32 next=" << next << " vocab=" << config.vocab_size
              << '\n';
    if (next >= config.vocab_size) {
      std::cerr << "FAIL prefill token out of range\n";
      return 1;
    }
    std::cout << "S2 prefill PASS\n";
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "FAIL " << ex.what() << '\n';
    return 1;
  }
}
