// LVV: greedy describe through the ling3 engine, plus the text-only check.
// Usage:
//   lvv_describe selfcheck
//   lvv_describe plan <model.gguf> <w> <h>
//   lvv_describe run <model.gguf> --tokens N --image <w> <h> <emb.bin> <label>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <span>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/core/session_mode.hpp"
#include "src/models/ling3/engine.hpp"
#include "src/models/ling3/vision_prompt.hpp"

namespace l3 = superroc::models::ling3;
namespace vis = superroc::models::ling3::vision;

namespace {

std::uint64_t HashFloats(std::span<const float> xs) {
  std::uint64_t h = 14695981039346656037ull;
  for (float f : xs) {
    std::uint32_t u = 0;
    std::memcpy(&u, &f, 4);
    h ^= u;
    h *= 1099511628211ull;
  }
  return h;
}

std::vector<float> ReadFloats(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  in.seekg(0, std::ios::end);
  const auto n = static_cast<std::size_t>(in.tellg());
  in.seekg(0);
  std::vector<float> out(n / sizeof(float));
  in.read(reinterpret_cast<char*>(out.data()),
          static_cast<std::streamsize>(out.size() * sizeof(float)));
  if (!in)
    out.clear();
  return out;
}

std::string Piece(const l3::Model& model, std::int32_t id) {
  std::string s = model.TokenText(id);
  std::string o;
  for (unsigned char c : s) {
    if (c == '\n')
      o += "\\n";
    else if (c == '\r')
      o += "\\r";
    else
      o.push_back(static_cast<char>(c));
  }
  return o;
}

struct Gen {
  std::string text;
  std::uint64_t logit_hash{0};
  std::int32_t first{0};
  bool ok{false};
};

Gen Greedy(l3::Session& session, l3::Model& model, int ntok) {
  Gen g;
  g.logit_hash = HashFloats(session.Logits());
  g.first = session.Argmax();
  std::int32_t tok = g.first;
  for (int i = 0; i < ntok; ++i) {
    if (model.IsStopToken(tok))
      break;
    g.text += Piece(model, tok);
    std::string err;
    if (!session.Evaluate(tok, &err)) {
      std::fprintf(stderr, "EVAL FAIL %s\n", err.c_str());
      return g;
    }
    tok = session.Argmax();
  }
  g.ok = true;
  return g;
}

bool PrefillIds(l3::Session& session, std::span<const std::int32_t> ids,
                std::string* err) {
  for (std::size_t off = 0; off < ids.size();) {
    const std::size_t take = std::min<std::size_t>(512, ids.size() - off);
    if (!session.EvaluateBatch(ids.subspan(off, take), err))
      return false;
    off += take;
  }
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  std::setbuf(stdout, nullptr);
  if (argc < 2) {
    std::fprintf(stderr, "usage: lvv_describe selfcheck|plan|run ...\n");
    return 2;
  }
  const std::string cmd = argv[1];
  if (cmd == "selfcheck") {
    std::string err;
    if (!vis::RopeSelfCheck(&err)) {
      std::fprintf(stderr, "SELFCHECK FAIL %s\n", err.c_str());
      return 1;
    }
    std::printf("SELFCHECK ok\n");
    return 0;
  }
  if (argc < 3)
    return 2;
  const std::string model_path = argv[2];

  if (cmd == "plan") {
    if (argc < 5)
      return 2;
    const int w = std::atoi(argv[3]);
    const int h = std::atoi(argv[4]);
    std::string err;
    auto reader = superroc::core::GgufReader::OpenFile(model_path, &err);
    if (!reader) {
      std::fprintf(stderr, "OPEN FAIL %s\n", err.c_str());
      return 1;
    }
    auto tok = l3::LingTokenizer::FromGguf(*reader, &err);
    if (!tok) {
      std::fprintf(stderr, "TOK FAIL %s\n", err.c_str());
      return 1;
    }
    const int n = (w / 32) * (h / 32);
    std::vector<float> emb(static_cast<std::size_t>(n) * 2560, 0.0f);
    vis::PreparedPrompt prep;
    if (!vis::Prepare(*tok, "Describe the image in one sentence.", true, w, h,
                      emb, &prep, &err)) {
      std::fprintf(stderr, "PLAN FAIL %s\n", err.c_str());
      return 1;
    }
    std::printf("PLAN tokens=%zu images=%zu next_logical=%d\n",
                prep.tokens.size(), prep.image_phys.size(), prep.next_logical);
    if (!prep.image_phys.empty()) {
      const int p = prep.image_phys.front();
      const int q = prep.image_phys.back();
      std::printf(
          "PLAN first_phys=%d t=%d h=%d w=%d last_phys=%d t=%d h=%d w=%d\n", p,
          prep.rope[static_cast<std::size_t>(p) * 3],
          prep.rope[static_cast<std::size_t>(p) * 3 + 1],
          prep.rope[static_cast<std::size_t>(p) * 3 + 2], q,
          prep.rope[static_cast<std::size_t>(q) * 3],
          prep.rope[static_cast<std::size_t>(q) * 3 + 1],
          prep.rope[static_cast<std::size_t>(q) * 3 + 2]);
    }
    return 0;
  }

  if (cmd != "run")
    return 2;
  int ntok = 24;
  struct Img {
    int w;
    int h;
    std::string emb;
    std::string label;
  };
  std::vector<Img> images;
  bool do_text = false;
  for (int i = 3; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--tokens" && i + 1 < argc) {
      ntok = std::atoi(argv[++i]);
    } else if (a == "--text") {
      do_text = true;
    } else if (a == "--image" && i + 4 < argc) {
      Img im;
      im.w = std::atoi(argv[++i]);
      im.h = std::atoi(argv[++i]);
      im.emb = argv[++i];
      im.label = argv[++i];
      images.push_back(im);
    } else {
      std::fprintf(stderr, "bad arg %s\n", a.c_str());
      return 2;
    }
  }

  l3::ModelOptions opt;
  opt.max_context = 2048;
  opt.backend = l3::BackendKind::kRocm;
  opt.prefill_batch = 1;
  opt.stream_experts = false;
  std::string err;
  std::printf("LOAD %s\n", model_path.c_str());
  auto model = l3::Model::Load(model_path, opt, &err);
  if (!model) {
    std::fprintf(stderr, "LOAD FAIL %s\n", err.c_str());
    return 1;
  }

  if (do_text) {
    const std::vector<std::int32_t> ids =
        model->tokenizer().Encode("The capital of France is", false, true);
    auto s1 = model->CreateSession(superroc::core::SessionMode::kAutoregressive,
                                   2048, &err);
    if (!s1 || !PrefillIds(*s1, ids, &err)) {
      std::fprintf(stderr, "TEXT PREFILL FAIL %s\n", err.c_str());
      return 1;
    }
    Gen a = Greedy(*s1, *model, ntok);
    std::printf("TEXT unarmed hash=%016llx first=%d ok=%d text=%s\n",
                static_cast<unsigned long long>(a.logit_hash), a.first,
                static_cast<int>(a.ok), a.text.c_str());
    s1.reset();
    vis::Clear();
    auto ident = vis::IdentityOf(ids);
    auto s2 = model->CreateSession(superroc::core::SessionMode::kAutoregressive,
                                   2048, &err);
    if (!s2 || !s2->PrefillVision(ident, &err)) {
      std::fprintf(stderr, "TEXT IDENT FAIL %s\n", err.c_str());
      return 1;
    }
    Gen b = Greedy(*s2, *model, ntok);
    std::printf("TEXT identity hash=%016llx first=%d ok=%d text=%s\n",
                static_cast<unsigned long long>(b.logit_hash), b.first,
                static_cast<int>(b.ok), b.text.c_str());
    const bool same = a.ok && b.ok && a.logit_hash == b.logit_hash &&
                      a.text == b.text && a.first == b.first;
    std::printf("TEXT_ONLY %s\n", same ? "UNCHANGED" : "CHANGED");
    s2.reset();
    vis::Clear();
  }

  for (const Img& im : images) {
    const std::vector<float> emb = ReadFloats(im.emb);
    vis::PreparedPrompt prep;
    if (!vis::Prepare(model->tokenizer(), "Describe the image in one sentence.",
                      true, im.w, im.h, emb, &prep, &err)) {
      std::fprintf(stderr, "PREP FAIL %s %s\n", im.label.c_str(), err.c_str());
      return 1;
    }
    std::printf("IMAGE %s tokens=%zu grid=%zu next_logical=%d\n",
                im.label.c_str(), prep.tokens.size(), prep.image_phys.size(),
                prep.next_logical);
    if (!prep.image_phys.empty()) {
      const int p = prep.image_phys.front();
      std::printf("IMAGE %s mrope0 t=%d h=%d w=%d\n", im.label.c_str(),
                  prep.rope[static_cast<std::size_t>(p) * 3],
                  prep.rope[static_cast<std::size_t>(p) * 3 + 1],
                  prep.rope[static_cast<std::size_t>(p) * 3 + 2]);
    }
    auto session = model->CreateSession(
        superroc::core::SessionMode::kAutoregressive, 2048, &err);
    if (!session || !session->PrefillVision(prep, &err)) {
      std::fprintf(stderr, "IMAGE PREFILL FAIL %s %s\n", im.label.c_str(),
                   err.c_str());
      return 1;
    }
    Gen g = Greedy(*session, *model, ntok);
    std::printf("SUPERROC %s ok=%d first=%d text=%s\n", im.label.c_str(),
                static_cast<int>(g.ok), g.first, g.text.c_str());
    session.reset();
    vis::Clear();
  }
  return 0;
}
