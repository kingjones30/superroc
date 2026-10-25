// KAT: bind a synthetic qwen35moe whose experts are BF16 and prove ExpertSlice
// uses a k·2 stride. A null slice used to reuse the previous expert's scratch.
// Also refuses expert_count > 256 and expert_used_count > 8 at bind, and
// refuses a gate tensor whose element count matches but whose dimensions do
// not.

#include <fcntl.h>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>

#include <csignal>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/qwen/moe/forward_moe.hpp"

namespace {

void Expect(bool ok, const std::string& msg) {
  if (!ok) {
    std::cerr << "FAIL " << msg << "\n";
    std::exit(1);
  }
}

void PutU32(std::vector<std::uint8_t>& o, std::uint32_t v) {
  const auto* p = reinterpret_cast<const std::uint8_t*>(&v);
  o.insert(o.end(), p, p + 4);
}

void PutU64(std::vector<std::uint8_t>& o, std::uint64_t v) {
  const auto* p = reinterpret_cast<const std::uint8_t*>(&v);
  o.insert(o.end(), p, p + 8);
}

void PutStr(std::vector<std::uint8_t>& o, const std::string& s) {
  PutU64(o, s.size());
  o.insert(o.end(), s.begin(), s.end());
}

struct Tensor {
  std::string name;
  std::vector<std::uint64_t> dims;
  std::uint32_t type = 0;  // ggml: 0 F32, 1 F16, 30 BF16
  std::vector<std::uint8_t> raw;
};

struct Kv {
  std::string key;
  enum Kind { kU32, kF32, kStr } kind;
  std::uint32_t u = 0;
  float f = 0;
  std::string s;
};

void AddU32(std::vector<Kv>& kv, const std::string& k, std::uint32_t v) {
  kv.push_back(Kv{k, Kv::kU32, v, 0, {}});
}

void AddF32(std::vector<Kv>& kv, const std::string& k, float v) {
  kv.push_back(Kv{k, Kv::kF32, 0, v, {}});
}

void AddStr(std::vector<Kv>& kv, const std::string& k, const std::string& v) {
  kv.push_back(Kv{k, Kv::kStr, 0, 0, v});
}

std::uint64_t NElem(const std::vector<std::uint64_t>& d) {
  std::uint64_t n = 1;
  for (auto v : d) {
    n *= v;
  }
  return n;
}

void AddF32Tensor(std::vector<Tensor>& ts, const std::string& name,
                  std::vector<std::uint64_t> dims) {
  Tensor t;
  t.name = name;
  t.dims = std::move(dims);
  t.type = 0;
  t.raw.assign(static_cast<std::size_t>(NElem(t.dims)) * 4, 0);
  ts.push_back(std::move(t));
}

void AddBf16Tensor(std::vector<Tensor>& ts, const std::string& name,
                   std::vector<std::uint64_t> dims, std::uint16_t bits) {
  Tensor t;
  t.name = name;
  t.dims = std::move(dims);
  t.type = 30;
  const auto n = static_cast<std::size_t>(NElem(t.dims));
  t.raw.resize(n * 2);
  for (std::size_t i = 0; i < n; ++i) {
    std::memcpy(t.raw.data() + i * 2, &bits, 2);
  }
  ts.push_back(std::move(t));
}

// Gate/up layout is [hidden, ff, n_expert], expert e packed contiguously.
void AddBf16Experts(std::vector<Tensor>& ts, const std::string& name,
                    std::uint64_t k, std::uint64_t rows, std::uint64_t n_expert,
                    std::uint16_t expert0, std::uint16_t expert1) {
  Tensor t;
  t.name = name;
  t.dims = {k, rows, n_expert};
  t.type = 30;
  const auto per = static_cast<std::size_t>(k * rows);
  t.raw.resize(per * static_cast<std::size_t>(n_expert) * 2);
  for (std::uint64_t e = 0; e < n_expert; ++e) {
    const std::uint16_t bits = e == 0 ? expert0 : expert1;
    for (std::size_t i = 0; i < per; ++i) {
      std::memcpy(t.raw.data() + (static_cast<std::size_t>(e) * per + i) * 2,
                  &bits, 2);
    }
  }
  ts.push_back(std::move(t));
}

std::vector<std::uint8_t> WriteGguf(const std::vector<Kv>& kv,
                                    const std::vector<Tensor>& ts) {
  constexpr std::uint32_t kAlign = 32;
  std::vector<std::uint8_t> o;
  o.push_back('G');
  o.push_back('G');
  o.push_back('U');
  o.push_back('F');
  PutU32(o, 3);
  PutU64(o, ts.size());
  PutU64(o, kv.size());
  for (const auto& item : kv) {
    PutStr(o, item.key);
    if (item.kind == Kv::kStr) {
      PutU32(o, 8);
      PutStr(o, item.s);
    } else if (item.kind == Kv::kF32) {
      PutU32(o, 6);
      const auto* p = reinterpret_cast<const std::uint8_t*>(&item.f);
      o.insert(o.end(), p, p + 4);
    } else {
      PutU32(o, 4);
      PutU32(o, item.u);
    }
  }
  std::uint64_t run = 0;
  std::vector<std::uint64_t> offs(ts.size());
  for (std::size_t i = 0; i < ts.size(); ++i) {
    offs[i] = run;
    const auto nb = static_cast<std::uint64_t>(ts[i].raw.size());
    run += (nb + kAlign - 1) / kAlign * kAlign;
  }
  for (std::size_t i = 0; i < ts.size(); ++i) {
    PutStr(o, ts[i].name);
    PutU32(o, static_cast<std::uint32_t>(ts[i].dims.size()));
    for (auto d : ts[i].dims) {
      PutU64(o, d);
    }
    PutU32(o, ts[i].type);
    PutU64(o, offs[i]);
  }
  while (o.size() % kAlign) {
    o.push_back(0);
  }
  for (const auto& t : ts) {
    o.insert(o.end(), t.raw.begin(), t.raw.end());
    while (o.size() % kAlign) {
      o.push_back(0);
    }
  }
  return o;
}

void BaseKv(std::vector<Kv>& kv, std::uint32_t n_expert, std::uint32_t n_used,
            std::uint32_t ff, std::uint32_t shexp) {
  AddStr(kv, "general.architecture", "qwen35moe");
  AddU32(kv, "general.alignment", 32);
  AddU32(kv, "qwen35moe.block_count", 4);
  AddU32(kv, "qwen35moe.embedding_length", 32);
  AddU32(kv, "qwen35moe.expert_count", n_expert);
  AddU32(kv, "qwen35moe.expert_used_count", n_used);
  AddU32(kv, "qwen35moe.expert_feed_forward_length", ff);
  AddU32(kv, "qwen35moe.expert_shared_feed_forward_length", shexp);
  AddU32(kv, "qwen35moe.attention.head_count", 2);
  AddU32(kv, "qwen35moe.attention.head_count_kv", 1);
  AddU32(kv, "qwen35moe.attention.key_length", 16);
  AddU32(kv, "qwen35moe.context_length", 128);
  AddU32(kv, "qwen35moe.full_attention_interval", 4);
  AddU32(kv, "qwen35moe.rope.dimension_count", 4);
  AddF32(kv, "qwen35moe.rope.freq_base", 10000.0F);
  AddU32(kv, "qwen35moe.ssm.conv_kernel", 4);
  AddU32(kv, "qwen35moe.ssm.state_size", 128);
  AddU32(kv, "qwen35moe.ssm.group_count", 1);
  AddU32(kv, "qwen35moe.ssm.time_step_rank", 1);
  AddU32(kv, "qwen35moe.ssm.inner_size", 128);
}

std::string BindFile(const std::vector<std::uint8_t>& bytes, const char* path) {
  const int fd = open(path, O_CREAT | O_TRUNC | O_WRONLY, 0644);
  Expect(fd >= 0, "open synth");
  std::size_t off = 0;
  while (off < bytes.size()) {
    const ssize_t n = write(fd, bytes.data() + off, bytes.size() - off);
    Expect(n > 0, "write synth");
    off += static_cast<std::size_t>(n);
  }
  close(fd);
  std::string error;
  auto reader = gufo::core::GgufReader::OpenFile(path, &error);
  if (reader == nullptr) {
    return "open: " + error;
  }
  auto weights = gufo::models::QwenModelWeights::LoadFromGguf(*reader, &error);
  if (!weights.has_value()) {
    return error;
  }
  return "";
}

void AddTrunk(std::vector<Tensor>& ts, bool swap_gate_dims) {
  constexpr std::uint64_t H = 32;
  constexpr std::uint64_t FF = 8;
  constexpr std::uint64_t SH = 8;
  constexpr std::uint64_t E = 2;
  constexpr std::uint64_t QKV = 384;  // 2*1*128 + 128
  constexpr std::uint64_t INNER = 128;
  constexpr std::uint64_t ATTN = 32;  // 2 heads * 16
  constexpr std::uint64_t KV = 16;
  const std::uint16_t one = 0x3f80;  // BF16 1.0
  const std::uint16_t zero = 0;
  AddF32Tensor(ts, "token_embd.weight", {H, 8});
  AddF32Tensor(ts, "output_norm.weight", {H});
  AddF32Tensor(ts, "output.weight", {H, 8});
  for (int layer = 0; layer < 4; ++layer) {
    const std::string p = "blk." + std::to_string(layer) + ".";
    AddF32Tensor(ts, p + "attn_norm.weight", {H});
    AddF32Tensor(ts, p + "post_attention_norm.weight", {H});
    const bool attn = ((layer + 1) % 4) == 0;
    if (attn) {
      AddF32Tensor(ts, p + "attn_q.weight", {H, 2 * ATTN});
      AddF32Tensor(ts, p + "attn_k.weight", {H, KV});
      AddF32Tensor(ts, p + "attn_v.weight", {H, KV});
      AddF32Tensor(ts, p + "attn_output.weight", {ATTN, H});
      AddF32Tensor(ts, p + "attn_q_norm.weight", {16});
      AddF32Tensor(ts, p + "attn_k_norm.weight", {16});
    } else {
      AddF32Tensor(ts, p + "attn_qkv.weight", {H, QKV});
      AddF32Tensor(ts, p + "attn_gate.weight", {H, INNER});
      AddF32Tensor(ts, p + "ssm_a", {1});
      AddF32Tensor(ts, p + "ssm_conv1d.weight", {QKV * 4});
      AddF32Tensor(ts, p + "ssm_dt.bias", {1});
      AddF32Tensor(ts, p + "ssm_alpha.weight", {H, 1});
      AddF32Tensor(ts, p + "ssm_beta.weight", {H, 1});
      AddF32Tensor(ts, p + "ssm_norm.weight", {128});
      AddF32Tensor(ts, p + "ssm_out.weight", {INNER, H});
    }
    AddF32Tensor(ts, p + "ffn_gate_inp.weight", {H, E});
    if (swap_gate_dims) {
      // Same element count as [H, FF, E], wrong axis order.
      AddBf16Tensor(ts, p + "ffn_gate_exps.weight", {FF, H, E}, one);
    } else {
      AddBf16Experts(ts, p + "ffn_gate_exps.weight", H, FF, E, one, zero);
    }
    AddBf16Tensor(ts, p + "ffn_up_exps.weight", {H, FF, E}, one);
    AddBf16Tensor(ts, p + "ffn_down_exps.weight", {FF, H, E}, one);
    AddF32Tensor(ts, p + "ffn_gate_inp_shexp.weight", {H});
    AddBf16Tensor(ts, p + "ffn_gate_shexp.weight", {H, SH}, one);
    AddBf16Tensor(ts, p + "ffn_up_shexp.weight", {H, SH}, one);
    AddBf16Tensor(ts, p + "ffn_down_shexp.weight", {SH, H}, one);
  }
}

}  // namespace

int main() {
  using gufo::core::GgmlType;
  using gufo::models::KatExpertRowBytes;
  using gufo::models::KatExpertSlice;

  Expect(KatExpertRowBytes(GgmlType::kF32, 32) == 128, "F32 stride k*4");
  Expect(KatExpertRowBytes(GgmlType::kF16, 32) == 64, "F16 stride k*2");
  Expect(KatExpertRowBytes(GgmlType::kBF16, 32) == 64, "BF16 stride k*2");
  Expect(KatExpertRowBytes(GgmlType::kQ8_0, 31) == 0,
         "misaligned quant stride is 0");

  // A 0 stride must abort, not hand back a null slice.
  const pid_t child = fork();
  Expect(child >= 0, "fork");
  if (child == 0) {
    struct rlimit lim{};
    lim.rlim_cur = 0;
    lim.rlim_max = 0;
    setrlimit(RLIMIT_CORE, &lim);
    const int sink = open("/dev/null", O_WRONLY);
    if (sink >= 0) {
      dup2(sink, 2);
    }
    gufo::models::QwenTensorRef empty;
    empty.type = GgmlType::kBF16;
    empty.num_elements = 64;
    empty.data = nullptr;
    KatExpertSlice(empty, 0, 4, 8);
    _exit(0);
  }
  int status = 0;
  Expect(waitpid(child, &status, 0) == child, "wait");
  Expect(WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT,
         "null BF16 slice aborts");

  const char* cap_path = "/tmp/lkat-bf16-cap.gguf";
  const char* used_path = "/tmp/lkat-bf16-used.gguf";
  const char* legal_path = "/tmp/lkat-bf16-legal.gguf";
  const char* ok_path = "/tmp/lkat-bf16-ok.gguf";
  const char* swap_path = "/tmp/lkat-bf16-swap.gguf";

  {
    std::vector<Kv> kv;
    BaseKv(kv, 300, 8, 8, 8);
    const auto err = BindFile(WriteGguf(kv, {}), cap_path);
    Expect(err.find("expert_count > 256") != std::string::npos,
           "cap 300: " + err);
  }
  {
    std::vector<Kv> kv;
    BaseKv(kv, 16, 9, 8, 8);
    const auto err = BindFile(WriteGguf(kv, {}), used_path);
    Expect(err.find("expert_used_count > 8") != std::string::npos,
           "cap used 9: " + err);
  }
  {
    std::vector<Kv> kv;
    BaseKv(kv, 256, 8, 8, 8);
    const auto err = BindFile(WriteGguf(kv, {}), legal_path);
    Expect(err.find("refuses expert_count") == std::string::npos,
           "256/8 must not trip the cap: " + err);
    Expect(err.find("Missing required Qwen tensor") != std::string::npos,
           "256/8 fails later on a missing tensor: " + err);
  }

  std::string ok_error;
  {
    std::vector<Kv> kv;
    std::vector<Tensor> ts;
    BaseKv(kv, 2, 1, 8, 8);
    AddTrunk(ts, false);
    const int fd = open(ok_path, O_CREAT | O_TRUNC | O_WRONLY, 0644);
    Expect(fd >= 0, "open ok");
    const auto bytes = WriteGguf(kv, ts);
    std::size_t off = 0;
    while (off < bytes.size()) {
      const ssize_t n = write(fd, bytes.data() + off, bytes.size() - off);
      Expect(n > 0, "write ok");
      off += static_cast<std::size_t>(n);
    }
    close(fd);
    auto reader = gufo::core::GgufReader::OpenFile(ok_path, &ok_error);
    Expect(reader != nullptr, "open ok gguf: " + ok_error);
    // Audit J7 (2026-10-14): a qwen35moe trunk whose experts are not ROCmFP4
    // is refused at bind (the GPU expert path reads ROCmFP4 only), so the BF16
    // slice stride is checked on a hand-made ref instead of a bound layer.
    auto weights =
        gufo::models::QwenModelWeights::LoadFromGguf(*reader, &ok_error);
    Expect(
        !weights.has_value() &&
            ok_error.find("trunk experts must be ROCmFP4") != std::string::npos,
        "bf16 trunk experts refused at bind: " + ok_error);
    const auto* gate = reader->FindTensor("blk.0.ffn_gate_exps.weight");
    Expect(gate != nullptr && gate->dimensions.size() == 3 &&
               gate->dimensions[0] == 32 && gate->dimensions[1] == 8 &&
               gate->dimensions[2] == 2,
           "synthetic gate dims");
    std::vector<std::uint16_t> slab(2 * 8 * 32, 0x3f80);  // two BF16 experts
    gufo::models::QwenTensorRef ref;
    ref.type = GgmlType::kBF16;
    ref.data = slab.data();
    ref.num_elements = slab.size();
    const auto s0 = KatExpertSlice(ref, 0, 8, 32);
    const auto s1 = KatExpertSlice(ref, 1, 8, 32);
    const auto delta = static_cast<const std::uint8_t*>(s1.data) -
                       static_cast<const std::uint8_t*>(s0.data);
    Expect(delta == 8 * 32 * 2, "expert 1 is one BF16 slab past expert 0");
    Expect(s0.data != s1.data, "experts do not alias");
    std::cout << "KAT_BF16 bind=refused(J7) delta_bytes=" << delta << "\n";
  }

  {
    std::vector<Kv> kv;
    std::vector<Tensor> ts;
    BaseKv(kv, 2, 1, 8, 8);
    AddTrunk(ts, true);
    const auto err = BindFile(WriteGguf(kv, ts), swap_path);
    Expect(err.find("dimension mismatch") != std::string::npos &&
               err.find("ffn_gate_exps") != std::string::npos,
           "swapped dims refused: " + err);
  }

  unlink(cap_path);
  unlink(used_path);
  unlink(legal_path);
  unlink(ok_path);
  unlink(swap_path);
  std::cout << "KAT_BF16 PASS stride_f32=128 stride_f16=64 stride_bf16=64 "
               "null_slice=abort cap300=refused used9=refused legal256=allowed "
               "bf16_experts=refused dims_swapped=refused\n";
  return 0;
}
