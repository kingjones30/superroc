// ling3_trace_decode.cpp — LS3 PHASE6: a minimal decode runner for kernel
// tracing. Loads the model on the ROCm backend, batch-prefills a whole prompt
// file, then runs EXACTLY `ntok` greedy Step tokens (the server's decode
// path). A 2 s sleep after prefill puts an unambiguous gap in the kernel
// trace so the analysis script can split prefill from decode kernels.
// Usage: superroc_ling3_trace_decode <model.gguf> <prompt-file> [ntok=16]
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

#include "src/models/ling3/engine.hpp"

namespace l3 = ::superroc::models::ling3;

int main(int argc, char** argv) {
  std::setbuf(stdout, nullptr);
  if (argc < 3) {
    std::printf("usage: %s <model> <prompt-file> [ntok]\n", argv[0]);
    return 64;
  }
  const int ntok = argc > 3 ? std::atoi(argv[3]) : 16;
  l3::ModelOptions opt;
  opt.max_context = 65536;
  opt.backend = l3::BackendKind::kRocm;
  opt.prefill_batch = 1;
  opt.stream_experts = false;
  std::string err;
  std::printf("loading %s ...\n", argv[1]);
  auto model = l3::Model::Load(argv[1], opt, &err);
  if (!model) {
    std::printf("LOAD FAIL: %s\n", err.c_str());
    return 2;
  }
  void* state = model->NewState(&err);
  if (state == nullptr) {
    std::printf("STATE FAIL: %s\n", err.c_str());
    return 2;
  }
  const std::size_t V = model->config().vocab_size;

  std::string text;
  {
    FILE* f = std::fopen(argv[2], "rb");
    if (f == nullptr) {
      std::printf("PROMPT FAIL\n");
      return 2;
    }
    char buf[65536];
    std::size_t n;
    while ((n = std::fread(buf, 1, sizeof(buf), f)) > 0) text.append(buf, n);
    std::fclose(f);
  }
  std::vector<std::int32_t> ids = model->tokenizer().Encode(text);
  if (ids.size() > 60000) ids.resize(60000);
  std::printf("prompt_tokens=%zu ntok=%d\n", ids.size(), ntok);

  const auto t0 = std::chrono::steady_clock::now();
  std::vector<float> pre_logits(V);
  for (std::size_t off = 0; off < ids.size(); off += 512) {
    const std::size_t take = ids.size() - off < 512 ? ids.size() - off : 512;
    if (!model->StepBatch(ids.data() + off, static_cast<int>(take), state,
                          std::span<float>(pre_logits), &err)) {
      std::printf("PREFILL FAIL at %zu: %s\n", off, err.c_str());
      return 3;
    }
  }
  const auto t1 = std::chrono::steady_clock::now();
  std::printf("prefill %.2f s; sleeping 2 s (trace gap)\n",
              std::chrono::duration<double>(t1 - t0).count());
  std::this_thread::sleep_for(std::chrono::seconds(2));

  std::vector<float> logits(V);
  const auto t2 = std::chrono::steady_clock::now();
  std::int32_t next = 0;
  for (int i = 0; i < ntok; ++i) {
    if (!model->Step(next, state, std::span<float>(logits), &err)) {
      std::printf("STEP FAIL at %d: %s\n", i, err.c_str());
      return 4;
    }
    std::size_t best = 0;
    for (std::size_t j = 1; j < V; ++j) {
      if (logits[j] > logits[best]) best = j;
    }
    next = static_cast<std::int32_t>(best);
  }
  const auto t3 = std::chrono::steady_clock::now();
  std::printf("decode %d tokens in %.3f s = %.2f ms/token (%.2f tok/s)\n", ntok,
              std::chrono::duration<double>(t3 - t2).count(),
              std::chrono::duration<double, std::milli>(t3 - t2).count() / ntok,
              ntok / std::chrono::duration<double>(t3 - t2).count());
  model->DestroyState(state);
  return 0;
}
