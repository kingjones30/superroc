// FRZA SPEC2: host-only test (no GPU) for the Qwen FP4 F16 prefill gate.
// File mix -> route, before and after the patch, plus the env tri-state.
// Neighbour style: tests/models/qwen/gemm_route_test.cpp (Check + abort).
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "src/models/qwen/qwen_fp4_f16_gate.hpp"

namespace {

using superroc::core::GgmlType;
using superroc::models::qwen::DecideQwenFp4F16Prefill;
using superroc::models::qwen::kQwenFp4F16PrefillForce;
using superroc::models::qwen::kQwenFp4F16PrefillGated;
using superroc::models::qwen::kQwenFp4F16PrefillOff;
using superroc::models::qwen::ParseQwenFp4F16PrefillMode;
using superroc::models::qwen::QwenFp4F16Decision;
using superroc::models::qwen::QwenFp4F16LayerTypes;
using superroc::models::qwen::QwenFp4F16Route;

constexpr GgmlType kT100 = GgmlType::kQ4_0_ROCMFP4;
constexpr GgmlType kT101 = GgmlType::kQ4_0_ROCMFP4_FAST;
constexpr GgmlType kKQ =
    GgmlType::kQ4_K;  // stock K-quant stand-in (any non-FP4)

int g_checks = 0;

void Check(bool condition, const char* message) {
  ++g_checks;
  if (!condition) {
    std::cerr << "Qwen FP4 F16 gate test failure: " << message << '\n';
    std::abort();
  }
}

void CheckRoute(QwenFp4F16Route got, QwenFp4F16Route want, const char* what) {
  ++g_checks;
  if (got != want) {
    std::cerr << "Qwen FP4 F16 gate test failure: route for " << what << " is "
              << (got == QwenFp4F16Route::kF16 ? "F16" : "legacy") << ", want "
              << (want == QwenFp4F16Route::kF16 ? "F16" : "legacy") << '\n';
    std::abort();
  }
}

// Four layers, alternating kinds (layer 0 SSM), every checked tensor = fill.
std::vector<QwenFp4F16LayerTypes> Mix(GgmlType fill) {
  std::vector<QwenFp4F16LayerTypes> layers(4);
  for (std::size_t i = 0; i < layers.size(); ++i) {
    layers[i].is_full_attention = (i % 2 == 1);
    layers[i].ffn_gate = layers[i].ffn_up = layers[i].ffn_down = fill;
    layers[i].attn_q = layers[i].attn_k = layers[i].attn_v =
        layers[i].attn_output = fill;
    layers[i].attn_qkv = layers[i].attn_gate = layers[i].ssm_alpha =
        layers[i].ssm_beta = layers[i].ssm_out = fill;
  }
  return layers;
}

QwenFp4F16Decision Decide(int mode,
                          const std::vector<QwenFp4F16LayerTypes>& layers) {
  return DecideQwenFp4F16Prefill(mode, layers);
}

void TestEnvParse() {
  static_assert(ParseQwenFp4F16PrefillMode(nullptr) == kQwenFp4F16PrefillGated,
                "unset is the gated rule");
  static_assert(ParseQwenFp4F16PrefillMode("1") == kQwenFp4F16PrefillGated,
                "1 is the gated rule");
  static_assert(ParseQwenFp4F16PrefillMode("0") == kQwenFp4F16PrefillOff,
                "0 is the kill switch");
  static_assert(ParseQwenFp4F16PrefillMode("2") == kQwenFp4F16PrefillForce,
                "2 forces the tag route");
  Check(ParseQwenFp4F16PrefillMode("") == kQwenFp4F16PrefillGated,
        "empty env is the gated rule");
  Check(ParseQwenFp4F16PrefillMode("00") == kQwenFp4F16PrefillOff,
        "leading 0 stays off (today's off condition)");
  Check(ParseQwenFp4F16PrefillMode("3") == kQwenFp4F16PrefillGated,
        "unknown value fails closed");
  Check(ParseQwenFp4F16PrefillMode("2x") == kQwenFp4F16PrefillGated,
        "near-2 fails closed");
}

void TestNamedMixes() {
  // COHERENT: 100 everywhere -> old route under the gate (ffn_up, layer 0
  // refuses first), F16 under force, old route under the kill switch.
  auto coherent = Mix(kT100);
  QwenFp4F16Decision d = Decide(kQwenFp4F16PrefillGated, coherent);
  Check(d.has_fp4, "COHERENT has FP4");
  CheckRoute(d.route, QwenFp4F16Route::kLegacy, "COHERENT gated");
  Check(d.refused_by_gate, "COHERENT refused_by_gate");
  Check(std::strcmp(d.family, "ffn_up") == 0 && d.layer == 0,
        "COHERENT refusal names ffn_up layer 0");
  CheckRoute(Decide(kQwenFp4F16PrefillForce, coherent).route,
             QwenFp4F16Route::kF16, "COHERENT force");
  d = Decide(kQwenFp4F16PrefillOff, coherent);
  CheckRoute(d.route, QwenFp4F16Route::kLegacy, "COHERENT off");
  Check(!d.refused_by_gate, "COHERENT off is the kill switch, not a refusal");

  // FAST: all 101 -> F16 (type 101 never refuses).
  auto fast = Mix(kT101);
  d = Decide(kQwenFp4F16PrefillGated, fast);
  Check(d.has_fp4 && !d.refused_by_gate, "FAST has FP4, no refusal");
  CheckRoute(d.route, QwenFp4F16Route::kF16, "FAST gated");
  CheckRoute(Decide(kQwenFp4F16PrefillForce, fast).route, QwenFp4F16Route::kF16,
             "FAST force");
  CheckRoute(Decide(kQwenFp4F16PrefillOff, fast).route,
             QwenFp4F16Route::kLegacy, "FAST off");

  // Winner (STRIX): 100 on attn_qkv only, 101 elsewhere -> F16 (kStore cell).
  auto winner = Mix(kT101);
  for (auto& layer : winner) {
    if (!layer.is_full_attention)
      layer.attn_qkv = kT100;
  }
  d = Decide(kQwenFp4F16PrefillGated, winner);
  Check(d.has_fp4 && !d.refused_by_gate, "winner has FP4, no refusal");
  CheckRoute(d.route, QwenFp4F16Route::kF16, "winner gated");

  // Bisect ssm_out: 100 on ssm_out only -> old route (kResidual epilogue).
  auto bisect_ssm = Mix(kT101);
  for (auto& layer : bisect_ssm) {
    if (!layer.is_full_attention)
      layer.ssm_out = kT100;
  }
  d = Decide(kQwenFp4F16PrefillGated, bisect_ssm);
  CheckRoute(d.route, QwenFp4F16Route::kLegacy, "bisect ssm_out gated");
  Check(d.refused_by_gate && std::strcmp(d.family, "ssm_out") == 0 &&
            d.layer == 0,
        "bisect ssm_out names ssm_out layer 0");
  CheckRoute(Decide(kQwenFp4F16PrefillForce, bisect_ssm).route,
             QwenFp4F16Route::kF16, "bisect ssm_out force");

  // Bisect attn_gate: 100 on attn_gate only -> F16 (kStore cell).
  auto bisect_gate = Mix(kT101);
  for (auto& layer : bisect_gate) {
    if (!layer.is_full_attention)
      layer.attn_gate = kT100;
  }
  d = Decide(kQwenFp4F16PrefillGated, bisect_gate);
  CheckRoute(d.route, QwenFp4F16Route::kF16, "bisect attn_gate gated");

  // Bisect attn_k/attn_v: 100 there only -> F16 (kStore cells).
  auto bisect_kv = Mix(kT101);
  for (auto& layer : bisect_kv) {
    if (layer.is_full_attention)
      layer.attn_k = layer.attn_v = kT100;
  }
  d = Decide(kQwenFp4F16PrefillGated, bisect_kv);
  CheckRoute(d.route, QwenFp4F16Route::kF16, "bisect attn_k/v gated");

  // Stock K-quant file: no FP4 -> the caller's old logic verbatim at every
  // mode (the engine's pre-existing branch, untouched by this patch).
  auto stock = Mix(kKQ);
  for (int mode : {kQwenFp4F16PrefillOff, kQwenFp4F16PrefillGated,
                   kQwenFp4F16PrefillForce}) {
    d = Decide(mode, stock);
    Check(!d.has_fp4 && !d.refused_by_gate, "stock has no FP4, no refusal");
    CheckRoute(d.route, QwenFp4F16Route::kLegacy, "stock mode");
  }
}

void TestSingleFamilySweep() {
  // 100 on exactly one family (101 elsewhere): the four fused-epilogue
  // families refuse, the eight plain-store families take F16. This sweep is
  // the call-site verification: ffn_up=kSwiGLU; ffn_down/ssm_out/attn_output
  // =kResidual; the rest=kStore (dual declines FP4, so ffn_gate stays single).
  struct Family {
    const char* name;
    bool full_only;  // checked on full-attention layers only
    bool ssm_only;   // checked on SSM layers only
    bool expect_f16;
  };
  const Family families[] = {
      {"ffn_gate", false, false, true},    {"ffn_up", false, false, false},
      {"ffn_down", false, false, false},   {"attn_q", true, false, true},
      {"attn_k", true, false, true},       {"attn_v", true, false, true},
      {"attn_output", true, false, false}, {"attn_qkv", false, true, true},
      {"attn_gate", false, true, true},    {"ssm_alpha", false, true, true},
      {"ssm_beta", false, true, true},     {"ssm_out", false, true, false},
  };
  for (const auto& fam : families) {
    auto mix = Mix(kT101);
    for (std::size_t i = 0; i < mix.size(); ++i) {
      auto& layer = mix[i];
      const bool full = layer.is_full_attention;
      if (fam.full_only && !full)
        continue;
      if (fam.ssm_only && full)
        continue;
      const std::string name = fam.name;
      if (name == "ffn_gate") {
        layer.ffn_gate = kT100;
      } else if (name == "ffn_up") {
        layer.ffn_up = kT100;
      } else if (name == "ffn_down") {
        layer.ffn_down = kT100;
      } else if (name == "attn_q") {
        layer.attn_q = kT100;
      } else if (name == "attn_k") {
        layer.attn_k = kT100;
      } else if (name == "attn_v") {
        layer.attn_v = kT100;
      } else if (name == "attn_output") {
        layer.attn_output = kT100;
      } else if (name == "attn_qkv") {
        layer.attn_qkv = kT100;
      } else if (name == "attn_gate") {
        layer.attn_gate = kT100;
      } else if (name == "ssm_alpha") {
        layer.ssm_alpha = kT100;
      } else if (name == "ssm_beta") {
        layer.ssm_beta = kT100;
      } else if (name == "ssm_out") {
        layer.ssm_out = kT100;
      }
    }
    QwenFp4F16Decision d = Decide(kQwenFp4F16PrefillGated, mix);
    Check(d.has_fp4, fam.name);
    CheckRoute(
        d.route,
        fam.expect_f16 ? QwenFp4F16Route::kF16 : QwenFp4F16Route::kLegacy,
        fam.name);
    Check(d.refused_by_gate != fam.expect_f16, fam.name);
    if (!fam.expect_f16) {
      Check(std::strcmp(d.family, fam.name) == 0, fam.name);
      Check(d.layer == (fam.full_only ? 1 : 0), fam.name);
    }
    // Force overrides every refusal; the kill switch overrides every F16.
    CheckRoute(Decide(kQwenFp4F16PrefillForce, mix).route,
               QwenFp4F16Route::kF16, fam.name);
    CheckRoute(Decide(kQwenFp4F16PrefillOff, mix).route,
               QwenFp4F16Route::kLegacy, fam.name);
  }
}

void TestEdgeCases() {
  // A type-100 tensor in a slot the layer kind never runs (stale full-attn
  // slot on an SSM layer) must not refuse: same checked set as the engine.
  auto mix = Mix(kT101);
  mix[0].attn_output = kT100;  // layer 0 is SSM; attn_output unchecked there
  QwenFp4F16Decision d = Decide(kQwenFp4F16PrefillGated, mix);
  CheckRoute(d.route, QwenFp4F16Route::kF16, "unchecked slot");
  Check(!d.refused_by_gate, "unchecked slot no refusal");
  // Empty file: no layers, no FP4, legacy branch.
  std::vector<QwenFp4F16LayerTypes> empty;
  d = Decide(kQwenFp4F16PrefillGated, empty);
  Check(!d.has_fp4, "empty has no FP4");
  CheckRoute(d.route, QwenFp4F16Route::kLegacy, "empty");
}

}  // namespace

int main() {
  TestEnvParse();
  TestNamedMixes();
  TestSingleFamilySweep();
  TestEdgeCases();
  std::cout << "Qwen FP4 F16 gate tests passed (" << g_checks << " checks).\n";
  return 0;
}
