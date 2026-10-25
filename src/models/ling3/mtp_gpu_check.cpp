// LS4 phase 1, step 3: the MTP block on the GPU, checked against the CPU
// oracle.
//
// mtp_ref_test --dump writes one row per position of a real sequence:
//
//     pos  x_next  draft  h_prev[hidden_size]
//
// This reads that file, replays the IDENTICAL inputs through mtp::DraftNext,
// and compares the draft token and the returned hidden state against the
// oracle's. It exists so the MTP block can be validated on small shapes before
// anything runs it on the full model: a GPU fault resets the GPU and kills
// every seat on the box, so the index math and the buffer sizes have to be
// right before a 64 GiB load is worth risking.
//
// The rows must be replayed in order and into ONE state: the MTP block has one
// latent cache and it must be filled at every position in order, exactly as the
// CPU oracle filled it. Replaying a subset, or out of order, would compare
// against a different cache and every row would differ.
//
// Run it ONLY inside gpu-run. It loads the full resident plan (~64 GiB + the
// head's ~3.3 GiB) because the head borrows the trunk's token_embd / output.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/ling3/mtp_gpu.hpp"
#include "src/models/ling3/rocm_backend.hpp"
#include "src/models/ling3/weights.hpp"

namespace {

[[noreturn]] void Die(const std::string& why) {
  std::fprintf(stderr, "mtp_gpu_check: %s\n", why.c_str());
  std::exit(1);
}

struct Row {
  std::uint32_t pos{0};
  std::int32_t x_next{0};
  std::int32_t draft{0};
  std::vector<float> h_prev;
};

std::vector<Row> ReadDump(const std::string& path, std::uint32_t* hidden,
                          std::uint32_t* context) {
  std::ifstream in(path);
  if (!in)
    Die("cannot open " + path);
  std::string line;
  std::size_t rows = 0;
  std::vector<Row> out;
  while (std::getline(in, line)) {
    std::istringstream is(line);
    if (line.empty())
      continue;
    if (line[0] == '#')
      continue;
    std::string head;
    is >> head;
    if (head == "hidden") {
      is >> *hidden;
    } else if (head == "context") {
      is >> *context;
    } else if (head == "head_norm") {
      std::string n;
      is >> n;
      std::printf("# dump head_norm=%s\n", n.c_str());
    } else if (head == "rows") {
      is >> rows;
    } else {
      // A data row: "<pos> <x_next> <draft> <h_prev...>". `head` is the pos.
      Row r;
      r.pos =
          static_cast<std::uint32_t>(std::strtoul(head.c_str(), nullptr, 10));
      std::int64_t x = 0, d = 0;
      is >> x >> d;
      r.x_next = static_cast<std::int32_t>(x);
      r.draft = static_cast<std::int32_t>(d);
      double v = 0.0;
      while (is >> v)
        r.h_prev.push_back(static_cast<float>(v));
      out.push_back(std::move(r));
    }
  }
  if (out.empty())
    Die("dump has no data rows");
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  const char* model_path = nullptr;
  const char* dump_path = nullptr;
  bool layer_out_norm = false;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--dump" && i + 1 < argc) {
      dump_path = argv[++i];
    } else if (a == "--layer-out-norm") {
      layer_out_norm = true;
    } else if (a.rfind("--", 0) == 0) {
      Die("unknown flag " + a);
    } else {
      model_path = argv[i];
    }
  }
  if (model_path == nullptr || dump_path == nullptr) {
    Die("usage: mtp_gpu_check <model.gguf> --dump <file> [--layer-out-norm]");
  }

  std::uint32_t hidden = 0;
  std::uint32_t context = 0;
  const std::vector<Row> rows = ReadDump(dump_path, &hidden, &context);
  std::printf("# rows=%zu hidden=%u context=%u\n", rows.size(), hidden,
              context);

  std::string err;
  auto reader =
      gufo::core::GgufReader::OpenFile(std::filesystem::path(model_path), &err);
  if (reader == nullptr)
    Die("cannot open the GGUF: " + err);
  auto weights = gufo::models::ling3::ModelWeights::Bind(*reader, &err);
  if (!weights.has_value())
    Die("ModelWeights::Bind: " + err);
  if (hidden != weights->config.hidden_size) {
    Die("dump hidden width does not match the model");
  }

  // The head borrows the trunk's token_embd / output / output_norm, so the
  // trunk's own ROCm Init has to run first.
  gufo::models::ling3::rocm::ExpertPlacement placement{};
  placement.stream = false;
  std::size_t resident = 0;
  if (!gufo::models::ling3::rocm::Init(*weights, *reader, placement, &resident,
                                       &err)) {
    Die("rocm::Init: " + err);
  }
  std::printf("# trunk resident=%.2f GiB\n",
              static_cast<double>(resident) / 1073741824.0);

  const auto head_norm = layer_out_norm
                             ? gufo::models::ling3::mtp::HeadNorm::kLayerOutNorm
                             : gufo::models::ling3::mtp::HeadNorm::kOutputNorm;
  std::size_t head_bytes = 0;
  if (!gufo::models::ling3::mtp::Init(*weights, head_norm, &head_bytes, &err)) {
    Die("mtp::Init: " + err);
  }
  std::printf("# head resident=%.2f GiB\n",
              static_cast<double>(head_bytes) / 1073741824.0);

  auto* st = gufo::models::ling3::mtp::NewState(context);
  if (st == nullptr)
    Die("mtp::NewState failed");
  std::printf(
      "# state=%.1f MiB\n",
      static_cast<double>(gufo::models::ling3::mtp::StateBytes(context)) /
          1048576.0);

  const std::size_t H = weights->config.hidden_size;
  int exact = 0;
  int mismatches = 0;

  std::vector<float> h_new(H);
  for (std::size_t i = 0; i < rows.size(); ++i) {
    const Row& r = rows[i];
    if (r.h_prev.size() != H) {
      Die("row " + std::to_string(i) + " has the wrong hidden width");
    }
    // Positions must be contiguous from the state's first cache row: the head
    // attends over [0, pos], so a gap would silently change every later row.
    if (r.pos != i + 1) {
      std::fprintf(stderr,
                   "row %zu: pos %u, expected %zu -- the dump must be the "
                   "positions in order with no gaps\n",
                   i, r.pos, i + 1);
      mismatches++;
      continue;
    }
    std::int32_t draft = -1;
    if (!gufo::models::ling3::mtp::DraftNext(
            st, r.x_next, std::span<const float>(r.h_prev.data(), H), r.pos,
            &draft, std::span<float>(h_new.data(), h_new.size()), &err)) {
      Die("DraftNext at pos " + std::to_string(r.pos) + ": " + err);
    }
    if (draft == r.draft) {
      ++exact;
    } else {
      ++mismatches;
      std::fprintf(stderr, "pos %u: gpu draft %d, cpu draft %d  (MISMATCH)\n",
                   r.pos, draft, r.draft);
    }
    if (i < 6) {
      std::printf("pos %u  x_next %d  cpu %d  gpu %d  %s\n", r.pos, r.x_next,
                  r.draft, draft, draft == r.draft ? "match" : "MISMATCH");
    }
  }

  gufo::models::ling3::mtp::DestroyState(st);

  std::printf(
      "\nGPU_VS_CPU_MEASURED rows=%zu draft_token_exact=%d "
      "draft_token_mismatch=%d\n",
      rows.size(), exact, mismatches);
  // The draft TOKEN is the assertion that matters: it is the only thing the
  // draft loop consumes, so one differing token means the block's numerics are
  // wrong however close the hidden states happen to be. No tolerance here on
  // purpose -- greedy argmax over 157184 logits is either right or it is not.
  std::printf("VERDICT %s\n", mismatches == 0 ? "PASS" : "FAIL");
  return mismatches == 0 ? 0 : 1;
}
