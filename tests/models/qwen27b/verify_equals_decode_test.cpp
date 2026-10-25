// sq16: the verify path must compute the same tokens as the decode path.
// For verify widths 1-8, N positions run through ForwardToken (decode) and
// ForwardVerificationChunk (verify) from the same snapshot; every row must
// agree on argmax and be bit-identical on logits. Widths 9-16 run as a
// two-session ForwardVerificationBatch (one item is capped at 8 rows); the
// ForwardTokenBatch path serve uses is covered with N sessions against
// single ForwardToken references. With GUFO_VERIFY_DEBUG=1 the first
// differing per-stage snapshot is also reported.
// Forced modes (SPEC2 S2-1): GUFO_FP4_FORCE_REFUSE=1 declines every FP4
// GEMV intercept (decode and verify alike); GUFO_VERIFY_DISCARD_WARMUP=1
// drops the FP4 warm mark after load so every ShapeOk fails. Both must give
// verify == decode, bit for bit, with no exception.
// GUFO_VERIFY_SMOKE=1 runs the ticket-smoke subset (widths 1,2,8,9, forms
// at 9, batch N=2); default runs everything.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/core/rocm_fp4/gufo_fp4_gemv_q8.hpp"
#include "src/models/qwen/hip/detail/verify_debug.hpp"
#include "src/models/qwen/hip/executor.hpp"
#include "src/models/qwen/hip/fp4_launch.hpp"
#include "src/testing/compare/logit_comparator.hpp"

namespace {
using Token = gufo::tokenization::TokenId;
using Executor = gufo::hip::QwenGpuExecutor;
using Model = gufo::hip::QwenGpuModel;
using Stage = gufo::hip::detail::VerifyDebugStage;

constexpr int kSkipped = 77;
constexpr std::uint32_t kPromptSize = 24;
constexpr std::size_t kSuffix = 16;

constexpr std::string_view kText =
    "Virtual memory lets an operating system give each process its own address "
    "space. Pages map virtual addresses to physical memory. A page fault "
    "occurs when the requested page is unavailable and the operating system "
    "must act. Caches hide latency by keeping copies close to where the "
    "processor needs them most. Branch predictors guess which way a branch "
    "goes before it resolves. Pipelines overlap fetch, decode and execute "
    "across many instructions at once.";

bool EnvOn(const char* name) {
  const char* env = std::getenv(name);
  return env != nullptr && env[0] == '1';
}

void Expect(bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

bool ByteEqual(std::span<const float> a, std::span<const float> b) {
  return a.size() == b.size() &&
         std::memcmp(a.data(), b.data(), a.size_bytes()) == 0;
}

std::size_t Argmax(std::span<const float> values) {
  return static_cast<std::size_t>(
      std::max_element(values.begin(), values.end()) - values.begin());
}

struct TopEntry {
  std::size_t id;
  float value;
};

std::vector<TopEntry> TopK(std::span<const float> values, std::size_t k) {
  std::vector<std::size_t> ids(values.size());
  std::iota(ids.begin(), ids.end(), 0);
  std::partial_sort(
      ids.begin(), ids.begin() + k, ids.end(),
      [&](std::size_t a, std::size_t b) { return values[a] > values[b]; });
  std::vector<TopEntry> out;
  for (std::size_t i = 0; i < k; ++i) {
    out.push_back({ids[i], values[ids[i]]});
  }
  return out;
}

void PrintTop5(const char* label, std::span<const float> values) {
  const auto top = TopK(values, 5);
  std::cout << label << " top5:";
  for (const auto& entry : top) {
    std::cout << " " << entry.id << "=" << entry.value;
  }
  std::cout << " margin=" << (top[0].value - top[1].value) << "\n";
}

struct StageDiff {
  bool compared{false};
  bool equal{false};
  std::string tag;
  std::uint32_t layer{0};
  float max_abs{0.0F};
  std::size_t max_idx{0};
  float decode_value{0.0F};
  float verify_value{0.0F};
};

// First stage (in dump order) whose row differs between one decode run and
// one verify row. Tags present on only one side are skipped.
StageDiff FirstStageDiff(const std::vector<Stage>& decode,
                         const std::vector<Stage>& verify,
                         std::size_t verify_row, std::size_t time_step_rank) {
  StageDiff diff;
  std::size_t vi = 0;
  for (const auto& d : decode) {
    if (d.row_count != 1) {
      continue;
    }
    const Stage* v = nullptr;
    for (; vi < verify.size(); ++vi) {
      bool match = verify[vi].tag == d.tag && verify[vi].layer == d.layer;
      if (!match && d.tag == "alpha" && verify[vi].tag == "alpha_pack" &&
          verify[vi].layer == d.layer) {
        match = true;  // packed ssm controls: alpha is the first half
      }
      if (match) {
        v = &verify[vi];
        ++vi;
        break;
      }
    }
    if (v == nullptr) {
      continue;
    }
    diff.compared = true;
    std::span<const float> dd(d.data.data(), d.data.size());
    std::span<const float> vv;
    if (d.tag == "alpha" && v->tag == "alpha_pack") {
      const std::size_t stride = v->floats_per_row;
      vv = std::span<const float>(v->data.data() + verify_row * stride,
                                  time_step_rank);
    } else {
      if (v->floats_per_row != d.floats_per_row || verify_row >= v->row_count) {
        continue;
      }
      vv = std::span<const float>(
          v->data.data() + verify_row * v->floats_per_row, d.floats_per_row);
    }
    const std::size_t n = std::min(dd.size(), vv.size());
    float max_abs = 0.0F;
    std::size_t max_idx = 0;
    for (std::size_t i = 0; i < n; ++i) {
      const float delta = std::fabs(dd[i] - vv[i]);
      if (!(delta <= max_abs)) {
        max_abs = delta;
        max_idx = i;
      }
    }
    const bool equal =
        (n == dd.size() && n == vv.size() && max_abs == 0.0F &&
         std::memcmp(dd.data(), vv.data(), n * sizeof(float)) == 0);
    if (!equal) {
      diff.equal = false;
      diff.tag = d.tag;
      diff.layer = d.layer;
      diff.max_abs = max_abs;
      diff.max_idx = max_idx;
      diff.decode_value = dd[max_idx];
      diff.verify_value = vv[max_idx];
      return diff;
    }
  }
  // Packed beta: compare against the second half of alpha_pack.
  for (const auto& d : decode) {
    if (d.tag != "beta" || d.row_count != 1) {
      continue;
    }
    for (const auto& v : verify) {
      if (v.tag != "alpha_pack" || v.layer != d.layer) {
        continue;
      }
      const std::size_t stride = v.floats_per_row;
      std::span<const float> vv(
          v.data.data() + verify_row * stride + time_step_rank, time_step_rank);
      std::span<const float> dd(d.data.data(), d.data.size());
      float max_abs = 0.0F;
      std::size_t max_idx = 0;
      for (std::size_t i = 0; i < time_step_rank; ++i) {
        const float delta = std::fabs(dd[i] - vv[i]);
        if (!(delta <= max_abs)) {
          max_abs = delta;
          max_idx = i;
        }
      }
      if (max_abs != 0.0F) {
        diff.tag = "beta";
        diff.layer = d.layer;
        diff.max_abs = max_abs;
        diff.max_idx = max_idx;
        diff.decode_value = dd[max_idx];
        diff.verify_value = vv[max_idx];
        return diff;
      }
    }
  }
  diff.equal = true;
  return diff;
}

// Compares one decode row against one verify row; prints the mismatch detail
// and returns false on any difference (argmax, GPU token, or logit bits).
bool CheckRow(std::size_t width, std::size_t row, std::span<const float> dd,
              std::span<const float> vv, Token decode_pred, Token verify_pred,
              const std::vector<Stage>& decode_stages,
              const std::vector<Stage>& verify_stages, std::size_t verify_row,
              std::size_t time_step_rank, bool debug_on) {
  const bool same_argmax = Argmax(dd) == Argmax(vv);
  const bool same_gpu = decode_pred == verify_pred;
  const bool same_bits = ByteEqual(dd, vv);
  if (same_argmax && same_gpu && same_bits) {
    return true;
  }
  const auto cmp = gufo::testing::CompareLogits(dd, vv);
  std::cout << "width=" << width << " row=" << row
            << " argmax_decode=" << Argmax(dd)
            << " argmax_verify=" << Argmax(vv) << " gpu_decode=" << decode_pred
            << " gpu_verify=" << verify_pred << " max_abs=" << cmp.max_abs_diff
            << " max_idx=" << cmp.first_mismatch_idx << "\n";
  PrintTop5("  decode", dd);
  PrintTop5("  verify", vv);
  if (debug_on && !decode_stages.empty() && !verify_stages.empty()) {
    const auto stage = FirstStageDiff(decode_stages, verify_stages, verify_row,
                                      time_step_rank);
    if (stage.compared && !stage.equal) {
      std::cout << "  first stage: layer=" << stage.layer
                << " tag=" << stage.tag << " max_abs=" << stage.max_abs
                << " idx=" << stage.max_idx << " decode=" << stage.decode_value
                << " verify=" << stage.verify_value << "\n";
    } else if (!stage.compared) {
      std::cout << "  first stage: no common tags\n";
    } else {
      std::cout << "  first stage: all common stages bit-equal\n";
    }
  }
  return false;
}

}  // namespace

int main(int argc, const char* const* argv) {
  try {
    const char* model = argc > 1 ? argv[1] : std::getenv("GUFO_VERIFY_MODEL");
    if (model == nullptr) {
      std::cout << "Set GUFO_VERIFY_MODEL for verify_equals_decode.\n";
      return kSkipped;
    }
    const bool smoke = EnvOn("GUFO_VERIFY_SMOKE");
    const bool force_refuse = EnvOn("GUFO_FP4_FORCE_REFUSE");
    const bool discard_warmup = EnvOn("GUFO_VERIFY_DISCARD_WARMUP");
    std::string error;
    auto owner = gufo::core::GgufReader::OpenFile(model, &error);
    Expect(owner != nullptr, error);
    // One model, up to four sessions: widths 9-16 need two executors (one
    // verification item is capped at 8 rows) and the token-batch section
    // needs N.
    auto shared = Model::CreateFromGguf(
        std::shared_ptr<const gufo::core::GgufReader>(std::move(owner)),
        &error);
    Expect(shared != nullptr, error);
    const std::size_t kSessions = 4;
    // Serve sizes one slot set per session stream; the test needs one per
    // executor or the later Creates refuse their attach (and every FP4
    // launch on those streams then fails).
    Expect(gufo_fp4_slot_sets_configure(static_cast<int>(kSessions)) ==
               GUFO_FP4_OK,
           "FP4 slot sets configure refused");
    std::cout << "slot sets=" << gufo_fp4_slot_sets_count() << "\n";
    std::vector<std::unique_ptr<Executor>> exec;
    for (std::size_t i = 0; i < kSessions; ++i) {
      auto e = Executor::Create(shared, &error, 512);
      Expect(e != nullptr, error);
      exec.push_back(std::move(e));
    }
    const auto& config = exec[0]->GetConfig();
    std::cout << "model hidden=" << config.hidden_size
              << " layers=" << config.num_layers
              << " vocab=" << config.vocab_size
              << " tsr=" << config.ssm_time_step_rank << "\n";
    std::cout << "mode refuse=" << (force_refuse ? 1 : 0)
              << " discard=" << (discard_warmup ? 1 : 0)
              << " smoke=" << (smoke ? 1 : 0) << "\n";
    if (discard_warmup) {
      // After every Create (Create re-warms): every RocmFp4ShapeOk now
      // fails, so decode and verify both serve their fp32 fallbacks.
      gufo::hip::DiscardRocmFp4Warmup();
      std::cout << "warmup discarded (forced shape fallback)\n";
    }
    if (force_refuse) {
      std::cout << "force-refuse on (engine declines every FP4 GEMV "
                   "intercept)\n";
    }

    auto tokens = exec[0]->GetTokenizer().Encode(std::string(kText));
    Expect(tokens.size() >= kPromptSize + kSuffix,
           "fixture text tokenizes too short");
    tokens.resize(kPromptSize + kSuffix);
    exec[0]->Reset();
    (void)exec[0]->ForwardPromptBatch(
        std::span<const Token>(tokens).first(kPromptSize));
    auto snapshot = exec[0]->SaveSnapshot(kPromptSize);
    Expect(snapshot != nullptr, "snapshot at prompt end");

    const bool debug_on = gufo::hip::detail::VerifyDebugEnabled();
    std::cout << "stage snapshots: " << (debug_on ? "on" : "off") << "\n";
    std::vector<std::size_t> widths;
    if (smoke) {
      widths = {1, 2, 8, 9};
    } else {
      for (std::size_t w = 1; w <= kSuffix; ++w) {
        widths.push_back(w);
      }
    }
    bool failed = false;
    for (const std::size_t width : widths) {
      // Decode path: one ForwardToken per position from the same snapshot.
      exec[0]->RestoreSnapshot(*snapshot);
      std::vector<std::vector<float>> decode_logits;
      std::vector<std::vector<Stage>> decode_stages;
      std::vector<Token> decode_pred;
      for (std::size_t i = 0; i < width; ++i) {
        exec[0]->ClearVerifyDebug();
        const Token pred =
            exec[0]->ForwardToken(tokens[kPromptSize + i],
                                  kPromptSize + static_cast<std::uint32_t>(i));
        decode_pred.push_back(pred);
        const auto logits = exec[0]->CopyLastLogits();
        decode_logits.emplace_back(logits.begin(), logits.end());
        const auto& stages = exec[0]->GetVerifyDebug();
        decode_stages.emplace_back(stages.begin(), stages.end());
      }
      // Verify path: one chunk over the same positions and state (two
      // sessions past width 8; item B repeats the same prefix tokens at
      // the same positions, so every row has a decode reference).
      std::vector<Token> verify_pred_all;
      std::vector<std::vector<float>> verify_logits;
      std::vector<Stage> verify_stages;
      if (width <= 8) {
        exec[0]->RestoreSnapshot(*snapshot);
        exec[0]->ClearVerifyDebug();
        const auto verify_pred = exec[0]->ForwardVerificationChunk(
            std::span<const Token>(tokens).subspan(kPromptSize, width),
            kPromptSize, true);
        Expect(verify_pred.size() == width, "verification width mismatch");
        verify_pred_all = verify_pred;
        for (std::size_t i = 0; i < width; ++i) {
          const auto logits = exec[0]->CopyVerificationLogits(i);
          verify_logits.emplace_back(logits.begin(), logits.end());
        }
        const auto& stages = exec[0]->GetVerifyDebug();
        verify_stages.assign(stages.begin(), stages.end());
      } else {
        const std::size_t w1 = width / 2;
        const std::size_t w2 = width - w1;
        exec[0]->RestoreSnapshot(*snapshot);
        exec[1]->RestoreSnapshot(*snapshot);
        exec[0]->ClearVerifyDebug();
        exec[1]->ClearVerifyDebug();
        const gufo::hip::QwenGpuVerificationItem items[2] = {
            {exec[0].get(),
             std::span<const Token>(tokens).subspan(kPromptSize, w1),
             kPromptSize, true},
            {exec[1].get(),
             std::span<const Token>(tokens).subspan(kPromptSize, w2),
             kPromptSize, true},
        };
        const auto preds =
            Executor::ForwardVerificationBatch(std::span(items, 2));
        Expect(preds.size() == 2, "two-session batch prediction count");
        Expect(preds[0].size() == w1 && preds[1].size() == w2,
               "two-session batch widths");
        for (std::size_t i = 0; i < w1; ++i) {
          verify_pred_all.push_back(preds[0][i]);
          const auto logits = exec[0]->CopyVerificationLogits(i);
          verify_logits.emplace_back(logits.begin(), logits.end());
        }
        for (std::size_t j = 0; j < w2; ++j) {
          verify_pred_all.push_back(preds[1][j]);
          const auto logits = exec[1]->CopyVerificationLogits(j);
          verify_logits.emplace_back(logits.begin(), logits.end());
        }
        const auto& stages = exec[0]->GetVerifyDebug();
        verify_stages.assign(stages.begin(), stages.end());
      }

      bool width_ok = true;
      for (std::size_t i = 0; i < width; ++i) {
        // Item-B row j repeats decode row j (same token, same position);
        // its coordinator row is offset by w1.
        std::size_t decode_row = i;
        std::size_t verify_row = i;
        if (width > 8 && i >= width / 2) {
          decode_row = i - width / 2;
        }
        width_ok =
            CheckRow(width, i, decode_logits[decode_row], verify_logits[i],
                     decode_pred[decode_row], verify_pred_all[i],
                     decode_stages[decode_row], verify_stages, verify_row,
                     config.ssm_time_step_rank, debug_on) &&
            width_ok;
      }
      std::cout << "width=" << width << " rows=" << width
                << " exact=" << (width_ok ? 1 : 0) << "\n";
      failed = failed || !width_ok;
    }

    // S2-2: the >8-row integer path in both forms (<=8-row verify_m chunks
    // vs the per-row loop) must be bit-identical; report both timings
    // (min of 3 runs each).
    const std::vector<std::size_t> form_widths =
        smoke ? std::vector<std::size_t>{9}
              : std::vector<std::size_t>{9, 12, 16};
    for (const std::size_t width : form_widths) {
      const std::size_t w1 = width / 2;
      const std::size_t w2 = width - w1;
      std::vector<std::vector<float>> chunked_logits;
      std::vector<std::vector<float>> perrow_logits;
      double chunked_ms = 0.0;
      double perrow_ms = 0.0;
      for (int form = 0; form < 2; ++form) {
        if (form == 1) {
          (void)::setenv("GUFO_VERIFY_NO_CHUNK", "1", 1);
        }
        double best = 0.0;
        for (int rep = 0; rep < 3; ++rep) {
          exec[0]->RestoreSnapshot(*snapshot);
          exec[1]->RestoreSnapshot(*snapshot);
          const gufo::hip::QwenGpuVerificationItem items[2] = {
              {exec[0].get(),
               std::span<const Token>(tokens).subspan(kPromptSize, w1),
               kPromptSize, true},
              {exec[1].get(),
               std::span<const Token>(tokens).subspan(kPromptSize, w2),
               kPromptSize, true},
          };
          const auto t0 = std::chrono::steady_clock::now();
          (void)Executor::ForwardVerificationBatch(std::span(items, 2));
          const auto t1 = std::chrono::steady_clock::now();
          const double ms =
              std::chrono::duration<double, std::milli>(t1 - t0).count();
          if (rep == 0 || ms < best) {
            best = ms;
          }
          if (rep == 0) {
            auto& sink = (form == 0) ? chunked_logits : perrow_logits;
            for (std::size_t i = 0; i < w1; ++i) {
              const auto logits = exec[0]->CopyVerificationLogits(i);
              sink.emplace_back(logits.begin(), logits.end());
            }
            for (std::size_t j = 0; j < w2; ++j) {
              const auto logits = exec[1]->CopyVerificationLogits(j);
              sink.emplace_back(logits.begin(), logits.end());
            }
          }
        }
        if (form == 0) {
          chunked_ms = best;
        } else {
          perrow_ms = best;
          (void)::unsetenv("GUFO_VERIFY_NO_CHUNK");
        }
      }
      bool forms_ok = chunked_logits.size() == perrow_logits.size();
      float max_abs = 0.0F;
      if (forms_ok) {
        for (std::size_t i = 0; i < chunked_logits.size(); ++i) {
          if (!ByteEqual(chunked_logits[i], perrow_logits[i])) {
            forms_ok = false;
            const auto cmp = gufo::testing::CompareLogits(chunked_logits[i],
                                                          perrow_logits[i]);
            max_abs = cmp.max_abs_diff;
            std::cout << "forms width=" << width << " row=" << i
                      << " max_abs=" << max_abs << "\n";
          }
        }
      }
      std::cout << "forms width=" << width << " chunked_ms=" << chunked_ms
                << " perrow_ms=" << perrow_ms << " equal=" << (forms_ok ? 1 : 0)
                << "\n";
      failed = failed || !forms_ok;
    }

    // S2-4: the ForwardTokenBatch path serve uses, N sessions against
    // single ForwardToken references from identical per-session states.
    const std::size_t batch_n = smoke ? 2 : 4;
    {
      std::vector<std::unique_ptr<gufo::hip::QwenGpuSnapshot>> snaps;
      for (std::size_t i = 0; i < batch_n; ++i) {
        exec[i]->RestoreSnapshot(*snapshot);
        for (std::size_t j = 0; j < i; ++j) {
          (void)exec[i]->ForwardToken(
              tokens[kPromptSize + j],
              kPromptSize + static_cast<std::uint32_t>(j));
        }
        snaps.push_back(
            exec[i]->SaveSnapshot(kPromptSize + static_cast<std::uint32_t>(i)));
        Expect(snaps.back() != nullptr, "batch session snapshot");
      }
      std::vector<gufo::hip::QwenGpuBatchItem> items;
      for (std::size_t i = 0; i < batch_n; ++i) {
        items.push_back({exec[i].get(), tokens[kPromptSize + i],
                         kPromptSize + static_cast<std::uint32_t>(i)});
      }
      const auto batch_pred = Executor::ForwardTokenBatch(items);
      Expect(batch_pred.size() == batch_n, "token batch size");
      std::vector<std::vector<float>> batch_logits;
      for (std::size_t i = 0; i < batch_n; ++i) {
        const auto logits = exec[i]->CopyLastLogits();
        batch_logits.emplace_back(logits.begin(), logits.end());
      }
      bool batch_ok = true;
      for (std::size_t i = 0; i < batch_n; ++i) {
        exec[i]->RestoreSnapshot(*snaps[i]);
        const Token ref =
            exec[i]->ForwardToken(tokens[kPromptSize + i],
                                  kPromptSize + static_cast<std::uint32_t>(i));
        const auto logits = exec[i]->CopyLastLogits();
        const bool same =
            (ref == batch_pred[i]) && ByteEqual(logits, batch_logits[i]);
        if (!same) {
          batch_ok = false;
          const auto cmp = gufo::testing::CompareLogits(
              std::span<const float>(logits.data(), logits.size()),
              batch_logits[i]);
          std::cout << "tokenbatch row=" << i << " gpu_single=" << ref
                    << " gpu_batch=" << batch_pred[i]
                    << " max_abs=" << cmp.max_abs_diff << "\n";
        }
      }
      std::cout << "tokenbatch N=" << batch_n << " exact=" << (batch_ok ? 1 : 0)
                << "\n";
      failed = failed || !batch_ok;
    }

    if (failed) {
      throw std::runtime_error("verify path differs from decode path");
    }
    std::cout << "verify_equals_decode passed.\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << "\n";
    return 1;
  }
}
