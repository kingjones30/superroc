// gufo-quant v0.9 — gufo's own GGUF quantizer (product track, RV20 stages 0+1,
// LQP SPEC1: exact/anchored tensor rules, --protect, fork head flags, census,
// --manifest; LQP SPEC2: persistent pool + chunk tasks across tensors,
// streaming header-first writer, side-stream hashing (SHA-NI where the CPU
// has it), instant --dry-run; LQP SPEC3: failure-proof output (partial file
// + fsync + rename, exit 2 on every handled failure, no hangs), exact F32
// widenings, lossy float overrides rejected, census MXFP4-family sizes).
//
// v0.9 (LQP SPEC4; RV-LQP3 F1/F2/F3/F4/F6/F7): one publish sequence for the
// model AND its manifest (model partial -> fsync; manifest partial -> fsync;
// rename model; rename manifest; fsync the directories), one writer per
// output (a flock-held <out>.gq-lock claim; unique <out>.partial.<pid>), a
// worker-failure test path that is compiled ONLY into the test target
// (GUFO_QUANT_TEST_HOOKS), an RAII joiner for the hash threads, and the
// corrected pool.hpp comment.
//
// v0.9.1 (LQP SPEC5): the per-output <out>.gq-lock claim file is REMOVED when
// the run ends -- after success and after every handled failure -- so a build
// of seven variants leaves no stray lock beside the output, and a failed run
// leaves nothing at all. OutputClaim::Acquire proves the fd it locked is still
// the file the path names (fstat vs stat, dev+ino) and retries on a fresh file
// if a previous owner unlinked it; the destructor unlinks the path while it
// still holds the lock, then closes. See the OutputClaim comment. The tool
// version string is now "gufo-quant 0.9.1".
//
// Makes the same GGUF files the fork's llama-quantize makes for the seven ROC
// recipes (ftypes 100..106), byte for byte (except the added `gufo.quant.*` KV
// keys). See SPEC.md / RV20-DESIGN.md.
//
// Usage:
//   gufo-quant [options] in.gguf out.gguf <RECIPE> [nthreads]
//     RECIPE in {ROCmFP4, LEAN, COHERENT, FAST, FAST_COHERENT, STRIX,
//                STRIX_LEAN}  (+ *_EVEN alias -> --pure)
//   gufo-quant census <file.gguf> [--json]
//   --allow-requantize  requantize already-quantized tensors
//   --pure              no k-quant mixtures, all tensors to default type
//   --tensor-type RULE=qtype   override (repeatable; RULE is =exact, ^prefix,
//                       suffix$, glob with *, or plain substring)
//   --tensor-type-trace print the rule (or recipe) each tensor's type came from
//   --protect FILE      RULE TYPE lines applied before --tensor-type (repeatable)
//   --protect-allow-unmatched  warn instead of failing on protect rules that
//                       match zero tensors
//   --output-tensor-type TYPE  fork-compatible: type for output.weight
//   --token-embedding-type TYPE  fork-compatible: type for token_embd.weight
//   --manifest out.json write the receipt JSON (inputs/output sha256, rules,
//                       per-family table, wall time)
//   --dry-run           print sizes without writing (sizes only: no encode,
//                       no reads past the headers; the table is identical to
//                       a real run's)
//   --time              print per-stage seconds to stderr (header read, input
//                       hash, encode wall/cpu/max-tensor, output write)

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <exception>
#include <fcntl.h>
#include <fstream>
#include <sys/file.h>
#include <iostream>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <thread>
#include <sstream>
#include <algorithm>
#include <unistd.h>
#include <unordered_map>
#include <vector>

#include "census.hpp"
#include "g4_encode.hpp"
#include "gguf.hpp"
#include "pool.hpp"
#include "recipe.hpp"
#include "rocmfp4_encode.hpp"
#include "rules.hpp"
#include "sha256.hpp"
#include "toml.hpp"
#include "vendored_quantize.hpp"

namespace gqt = gufo::quant_tool;
using gqt::Type;
using gqt::Recipe;

// v0.5: the accepted --tensor-type values live in rules.hpp's TypeNameTable
// (same table; shared with the CTest suite).
static const std::map<std::string, Type>& TypeNames() {
  return gqt::TypeNameTable();
}
static const std::map<std::string, Recipe>& RecipeNames() {
  static const std::map<std::string, Recipe> m = {
      {"ROCmFP4", Recipe::kRocmFp4}, {"LEAN", Recipe::kLean},
      {"COHERENT", Recipe::kCoherent}, {"FAST", Recipe::kFast},
      {"FAST_COHERENT", Recipe::kFastCoherent}, {"STRIX", Recipe::kStrix},
      {"STRIX_LEAN", Recipe::kStrixLean},
      // EVEN recipes = base recipe + --pure (fork quantize.cpp sets params.pure
      // for Q4_0_ROCMFP4_EVEN / Q4_0_ROCMFP4_FAST_EVEN).
      {"ROCmFP4_EVEN", Recipe::kRocmFp4}, {"FAST_EVEN", Recipe::kFast},
      // gufo-native G4 (split-scale-plane) recipes.
      {"G4_STRIX", Recipe::kG4Strix}, {"G4_STRIX_LEAN", Recipe::kG4StrixLean},
  };
  return m;
}
static const std::map<std::string, Recipe>& PureRecipes() {
  static const std::map<std::string, Recipe> m = {
      {"ROCmFP4_EVEN", Recipe::kRocmFp4}, {"FAST_EVEN", Recipe::kFast},
  };
  return m;
}

inline float Bf16ToF32(std::uint16_t h) {
  std::uint32_t u = (std::uint32_t)h << 16;
  float f; std::memcpy(&f, &u, 4); return f;
}

// v0.5: SHA-256 lives in sha256.hpp (one-shot gqt::Sha256Hex, same bytes in ->
// same hex out, plus the streaming form --manifest uses for 100 GB inputs).
inline std::string Sha256Hex(const std::vector<std::uint8_t>& data) {
  return gqt::Sha256Hex(data);
}

// imatrix store: tensor name (base, no .in_sum2/.counts) -> per-column weights.
using ImatrixMap = std::unordered_map<std::string, std::vector<float>>;

// Load a fork imatrix GGUF, replicating the fork's load_imatrix (quantize.cpp)
// exactly: tensors `<name>.in_sum2` (sums of squared activations) and
// `<name>.counts`; value[j*ne0+i] = sums / count (or 1.0 when count <= 0). The
// per-tensor length must be ne[0]*ne[2] to be used on a quantized tensor.
static ImatrixMap LoadImatrix(const std::string& path, std::string* sha_out) {
  // Reuse the full GGUF loader (throws if not a GGUF).
  gqt::GgufFile f = gqt::LoadGguf(path);
  // v0.6: hash the bytes the loader already read (f.whole is the exact file
  // image) instead of a second full-file pass; same bytes in, same hex out.
  if (sha_out) *sha_out = gqt::Sha256Hex(f.whole);
  // Collect in_sum2/counts pairs.
  std::map<std::string, std::pair<const gqt::GgufTensor*, const gqt::GgufTensor*>> pairs;
  for (auto& t : f.tensors) {
    std::string nm = t.name;
    const std::string suff_sum{".in_sum2"}, suff_cnt{".counts"};
    if (nm.size() > suff_sum.size() &&
        nm.compare(nm.size() - suff_sum.size(), suff_sum.size(), suff_sum) == 0) {
      nm.resize(nm.size() - suff_sum.size());
      pairs[nm].first = &t;
    } else if (nm.size() > suff_cnt.size() &&
               nm.compare(nm.size() - suff_cnt.size(), suff_cnt.size(), suff_cnt) == 0) {
      nm.resize(nm.size() - suff_cnt.size());
      pairs[nm].second = &t;
    }
  }
  ImatrixMap out;
  for (auto& [name, sc] : pairs) {
    const gqt::GgufTensor* sums = sc.first;
    const gqt::GgufTensor* counts = sc.second;
    if (!sums || !counts)
      throw std::runtime_error("imatrix: mismatched sums/counts for " + name);
    const std::uint64_t ne0 = sums->dims[0];
    const std::uint64_t ne1 = sums->dims.size() > 1 ? sums->dims[1] : 1;
    const std::uint64_t n = ne0 * ne1;
    if (sums->type != (Type)0 || counts->type != (Type)0)  // F32
      throw std::runtime_error("imatrix: " + name + " not F32");
    std::vector<float> e(n);
    const float* sd = (const float*)sums->raw.data();
    const float* cd = (const float*)counts->raw.data();
    for (std::uint64_t j = 0; j < ne1; ++j) {
      const float count = cd[j];
      if (count > 0.0f) {
        for (std::uint64_t i = 0; i < ne0; ++i) e[j * ne0 + i] = sd[j * ne0 + i] / count;
      } else {
        for (std::uint64_t i = 0; i < ne0; ++i) e[j * ne0 + i] = 1.0f;
      }
    }
    // SPEC3B hardening (fork llama-quant.cpp:1326-1333): the fork throws on any
    // non-finite per-column value after load. Mirror it so a bad imatrix file is
    // rejected identically instead of silently weighting differently (RV29 D1).
    for (float v : e) {
      if (!std::isfinite(v))
        throw std::runtime_error("imatrix contains non-finite value for " + name);
    }
    out[name] = std::move(e);
  }
  if (out.empty()) throw std::runtime_error("imatrix: no entries loaded from " + path);
  std::cerr << "imatrix: loaded " << out.size() << " entries from " << path << "\n";
  return out;
}

static void EncodeRow(Type t, const float* x, std::uint8_t* out, std::int64_t k,
                      const float* qw = nullptr, bool post_pass = false) {
  switch (t) {
    case (Type)100: gqt::QuantizeRowRocmFp4Ex(x, (gqt::block_rocmfp4*)out, k, qw, post_pass); break;
    case (Type)101: gqt::QuantizeRowRocmFp4FastEx(x, (gqt::block_rocmfp4_fast*)out, k, qw, post_pass); break;
    case (Type)120: gqt::EncodeRowG4Dual(x, out, k); break;
    case (Type)121: gqt::EncodeRowG4Fast(x, out, k); break;
    case (Type)2: ::quantize_row_q4_0_ref(x, (::block_q4_0*)out, k); break;
    case (Type)7: ::quantize_row_q5_1_ref(x, (::block_q5_1*)out, k); break;
    case (Type)8: ::quantize_row_q8_0_ref(x, (::block_q8_0*)out, k); break;
    // SPEC4 (weighted k-quants): when a per-row qw is present, the fork's
    // quantize_qX_K dispatches to the STRUCTURALLY DIFFERENT _impl rowwise
    // algorithm (make_qkx3_quants + make_qp_quants with prepared
    // qw·sqrt(σ²+x²) weights). With qw==nullptr the unweighted _ref is used,
    // which is exactly the fork's no-imatrix (and token_embd) path. This keeps
    // the default path byte-identical to v0.2 / fork-unweighted.
    case (Type)13:
      if (qw) { ::quantize_row_q5_K_impl(x, (::block_q5_K*)out, k, qw); break; }
      ::quantize_row_q5_K_ref(x, (::block_q5_K*)out, k); break;
    case (Type)14:
      if (qw) { ::quantize_row_q6_K_impl(x, (::block_q6_K*)out, k, qw); break; }
      ::quantize_row_q6_K_ref(x, (::block_q6_K*)out, k); break;
    // F16 pass-through: used for the fork's catch-all shape fallback (a tensor
    // whose ne0 is not divisible by any quant block size, e.g. blk.N.ple_conv1d
    // with ne0=4 -> F16). Same layout/rounding as a quantitative F16 tensor.
    case (Type)1: {
      std::uint16_t* o16 = (std::uint16_t*)out;
      for (std::int64_t i = 0; i < k; ++i) o16[i] = ::GGML_FP32_TO_FP16(x[i]);
      break;
    }
    // v0.8 (F4): F32 destination. The row arrives dequantized to f32, so this
    // is the EXACT widening for F16/BF16 sources (a memcpy); F32 sources are
    // same-type keeps and never reach the encoder. Every other float-to-float
    // request the tool cannot do exactly is rejected in the plan pre-pass.
    case (Type)0: {
      std::memcpy(out, x, (std::size_t)k * 4);
      break;
    }
    default: throw std::runtime_error("EncodeRow: unsupported type " + std::to_string(t));
  }
}

/// Dequantize source rows [r0, r1) of one tensor (ne0 inner dim) to f32 in
/// `out` (which holds (r1-r0)*ne0 floats). v0.6's ranged form of the old
/// whole-tensor DequantizeToF32: identical per-element arithmetic, so every
/// element is still written exactly once and chunk splits are byte-identical.
/// The plan pre-pass checks the source type first and throws this same
/// message for an unsupported source (as the old caller did).
static void DequantizeRowsToF32(const std::uint8_t* raw, Type src,
                                std::uint64_t ne0, std::uint64_t r0,
                                std::uint64_t r1, float* out) {
  const std::uint64_t base = r0 * ne0;
  if (src == (Type)0) {
    for (std::uint64_t r = r0; r < r1; ++r)
      for (std::uint64_t i = 0; i < ne0; ++i) {
        std::uint64_t k = r * ne0 + i;
        std::memcpy(&out[k - base], &raw[k * 4], 4);
      }
  } else if (src == (Type)1) {
    for (std::uint64_t r = r0; r < r1; ++r)
      for (std::uint64_t i = 0; i < ne0; ++i) {
        std::uint64_t k = r * ne0 + i;
        std::uint16_t h; std::memcpy(&h, &raw[k * 2], 2);
        out[k - base] = ::GGML_FP16_TO_FP32(h);
      }
  } else if (src == (Type)30) {
    for (std::uint64_t r = r0; r < r1; ++r)
      for (std::uint64_t i = 0; i < ne0; ++i) {
        std::uint64_t k = r * ne0 + i;
        std::uint16_t h; std::memcpy(&h, &raw[k * 2], 2);
        out[k - base] = Bf16ToF32(h);
      }
  } else {
    throw std::runtime_error("source tensor type " + std::to_string((int)src) +
                             " not supported (need F32/F16/BF16 source)");
  }
}

static const char* kToolVersion = "gufo-quant 0.9.1";

// ---- v0.9 (SPEC4): publish + one-writer helpers -------------------------

// A test-only hook. When GUFO_QUANT_TEST_HOOKS is defined (ONLY the separate
// `gufo-quant-testhooks` target defines it -- the shipped `gufo-quant` target
// never does), a chunk task can be made to throw in the middle of the encode
// by setting GUFO_QUANT_TEST_FAIL_CHUNK=1, exercising the real
// worker-throw -> writer-wake -> unlink -> exit-2 path (RV-LQP3 F5). In the
// shipped tool the macro expands to nothing and the code does not exist.
#ifdef GUFO_QUANT_TEST_HOOKS
#define GUFO_QUANT_CHUNK_HOOK()                                              \
  do {                                                                      \
    const char* gq_hook = std::getenv("GUFO_QUANT_TEST_FAIL_CHUNK");        \
    if (gq_hook && *gq_hook && *gq_hook != '0')                              \
      throw std::runtime_error("test hook: injected mid-encode chunk fail"); \
  } while (0)
#else
#define GUFO_QUANT_CHUNK_HOOK() ((void)0)
#endif

// The directory part of a path ("." when the path has no slash).
static std::string DirName(const std::string& p) {
  size_t s = p.find_last_of('/');
  return (s == std::string::npos) ? std::string(".") : p.substr(0, s);
}

// fsync a path's data (open O_RDONLY) -- throws on any failure. Used for the
// model partial and the manifest partial before either is renamed.
static void FsyncPath(const std::string& path, const char* what) {
  int fd = ::open(path.c_str(), O_RDONLY);
  if (fd < 0)
    throw std::runtime_error(std::string("cannot reopen ") + what + " for fsync: " +
                             path + ": " + std::strerror(errno));
  if (::fsync(fd) != 0) {
    int e = errno;
    ::close(fd);
    throw std::runtime_error(std::string("fsync failed: ") + what + ": " + path +
                             ": " + std::strerror(e));
  }
  ::close(fd);
}

// fsync a directory once (after a rename) -- throws on failure. Durability of
// the rename itself (RV-LQP3 F3).
static void FsyncDirOnce(const std::string& dir) {
  int dfd = ::open(dir.c_str(), O_RDONLY | O_DIRECTORY);
  if (dfd < 0)
    throw std::runtime_error("cannot open directory for fsync: " + dir + ": " +
                             std::strerror(errno));
  if (::fsync(dfd) != 0) {
    int e = errno;
    ::close(dfd);
    throw std::runtime_error("fsync failed: directory " + dir + ": " +
                             std::strerror(e));
  }
  ::close(dfd);
}

// RAII holder for the per-output write claim (RV-LQP3 F2; lock removal LQP
// SPEC5). The claim is a flock(LOCK_EX) on `<out>.gq-lock`, held for the whole
// run: exactly one writer per output. v0.9.1 also REMOVES the lock file at
// every exit so a run leaves no leftovers, without reopening the race F2
// closed. Two ideas make that safe:
//
//   * Acquire() opens/creates the path, takes the lock, then proves the fd it
//     locked is still the file the path names (fstat on the fd vs stat on the
//     path, comparing dev+ino). If a previous owner unlinked or replaced the
//     path while we were between open() and flock(), we hold the lock on an
//     orphaned inode -- not on the path -- so we close and start over on a
//     fresh file. (Without this check the removal below could be defeated.)
//   * The destructor unlinks the path WHILE still holding the lock, then
//     closes. Unlinking under the lock is what keeps it safe: a peer that
//     opened the old file before our unlink acquires the lock on the now
//     orphaned inode only after we release it, then its own fstat/stat check
//     mismatches and it retries on the fresh file; a peer that opens after our
//     unlink sees no file (or a fresh one) and claims cleanly. We unlink only
//     when the path still names our inode, so we never delete a peer's file.
//
// The lock is dropped only by close() (or by the kernel if the process dies).
// A run killed by a signal therefore leaves the file but no lock; the next
// Acquire() takes it over cleanly and that run removes it at its own exit.
struct OutputClaim {
  int fd = -1;
  std::string path_;  // the lock path we own; empty until we hold the claim
  OutputClaim() = default;
  OutputClaim(const OutputClaim&) = delete;
  OutputClaim& operator=(const OutputClaim&) = delete;
  ~OutputClaim() { Release(); }  // unlink while locked, then close

  // Remove our lock file and drop the lock. Idempotent; never throws.
  void Release() {
    if (fd < 0) return;
    if (!path_.empty()) {
      // Unlink only if the path still names the inode we locked: if a peer has
      // already replaced the file (our unlink already ran, or someone raced a
      // fresh one in), the path is not ours to remove.
      struct stat fst, pst;
      if (::fstat(fd, &fst) == 0 && ::stat(path_.c_str(), &pst) == 0 &&
          fst.st_dev == pst.st_dev && fst.st_ino == pst.st_ino)
        ::unlink(path_.c_str());
    }
    ::close(fd);  // drops the flock (the last close of this open file)
    fd = -1;
    path_.clear();
  }

  // Returns "" on success, or an error message (no "error: " prefix) when the
  // output is already being written by a live process.
  std::string Acquire(const std::string& out) {
    std::string lock = out + ".gq-lock";
    int lfd = -1;
    // Bounded retry: each pass either proves we own the path's lock or finds
    // our lock orphaned (a peer unlinked the path between our open and flock)
    // and starts over. The bound can only be hit under pathological churn.
    for (int attempt = 0; attempt < 1000; ++attempt) {
      lfd = ::open(lock.c_str(), O_CREAT | O_RDWR | O_CLOEXEC, 0644);
      if (lfd < 0)
        return "cannot open claim file " + lock + ": " + std::strerror(errno);
      if (::flock(lfd, LOCK_EX | LOCK_NB) != 0) {
        int e = errno;
        std::string holder;
        { std::ifstream hf(lock); hf >> holder; }
        ::close(lfd);
        if (e == EWOULDBLOCK)
          return out + " is being written by pid " +
                 (holder.empty() ? std::string("?") : holder);
        return "cannot lock claim file " + lock + ": " + std::strerror(e);
      }
      // We hold the lock: prove the fd still IS the file the path names. If a
      // previous owner unlinked/replaced the path after our open, we hold an
      // orphaned inode and must retry on a fresh file (else the removal below
      // would delete a file we do not own, or leave ours behind).
      struct stat fst, pst;
      if (::fstat(lfd, &fst) == 0 && ::stat(lock.c_str(), &pst) == 0 &&
          fst.st_dev == pst.st_dev && fst.st_ino == pst.st_ino)
        break;
      ::close(lfd);  // orphaned: drop it and retry on the (fresh) path
      lfd = -1;
    }
    if (lfd < 0)
      return "cannot claim " + lock + ": the path kept changing under us";
    // We hold the exclusive claim: no live writer for `out` can exist now, so
    // remove every leftover partial -- v0.8's fixed-name `<out>.partial` and
    // any `<out>.partial.<deadpid>` from a killed run. (A `<out>.partial.<pid>`
    // whose process is still alive but holds no claim is a refusal.)
    std::string dir = DirName(out);
    std::string base = out.substr(dir == "." ? 0 : dir.size() + 1);
    const std::string prefix = base + ".partial";
    std::vector<std::string> stale;
    if (DIR* d = ::opendir(dir.c_str())) {
      while (struct dirent* de = ::readdir(d)) {
        std::string n = de->d_name;
        if (n == prefix) { stale.push_back(n); continue; }  // old fixed name
        if (n.rfind(prefix + ".", 0) != 0) continue;
        std::string suf = n.substr(prefix.size() + 1);
        if (suf.find_first_not_of("0123456789") != std::string::npos) continue;
        long pid = std::strtol(suf.c_str(), nullptr, 10);
        if (pid > 0 && (::kill((pid_t)pid, 0) == 0 || errno != ESRCH)) {
          ::close(lfd);
          return out + " is being written by pid " + suf;
        }
        stale.push_back(n);
      }
      ::closedir(d);
    }
    for (auto& n : stale) ::unlink((dir == "." ? n : dir + "/" + n).c_str());
    // Record our pid for the next run's diagnostic message.
    if (::ftruncate(lfd, 0) == 0) {
      std::string pid = std::to_string((long)::getpid());
      ssize_t wr = ::write(lfd, pid.data(), pid.size());
      (void)wr;
      ::fsync(lfd);
    }
    fd = lfd;
    path_ = lock;
    return "";
  }
};

// RAII joiner for the input-hash threads (RV-LQP3 F6): the destructor joins
// whatever a failure path left, so no early return can leave a joinable
// thread behind (std::terminate). It never rethrows -- reporting is the
// explicit Join()'s job, so the destructor is safe during unwinding.
class HashThreadJoiner {
 public:
  ~HashThreadJoiner() {
    for (auto& th : threads_)
      if (th.joinable()) th.join();
  }
  void Add(std::thread t) { threads_.push_back(std::move(t)); }
  std::vector<std::exception_ptr>& Errors() { return ex_; }
  void Join() {  // idempotent: join, then rethrow the first stored error
    for (auto& th : threads_)
      if (th.joinable()) th.join();
    threads_.clear();
    for (auto& e : ex_)
      if (e) std::rethrow_exception(e);
  }
 private:
  std::vector<std::thread> threads_;
  std::vector<std::exception_ptr> ex_;
};


// `gufo-quant census <file.gguf> [--json]`: header census (SPEC1 item 4).
int CensusMain(const std::vector<std::string>& args) {
  std::string file;
  bool json = false;
  for (size_t i = 1; i < args.size(); ++i) {
    if (args[i] == "--json") {
      json = true;
    } else if (!args[i].empty() && args[i][0] == '-') {
      std::cerr << "unknown census option: " << args[i] << "\n";
      return 2;
    } else if (file.empty()) {
      file = args[i];
    } else {
      std::cerr << "census takes one file\n";
      return 2;
    }
  }
  if (file.empty()) {
    std::cerr << "usage: gufo-quant census <file.gguf> [--json]\n";
    return 2;
  }
  gqt::GgufFile f;
  try {
    f = gqt::LoadCensusFile(file);
  } catch (const std::exception& e) {
    std::cerr << "gufo-quant census: " << e.what() << "\n";
    return 2;
  }
  gqt::Census c = gqt::ComputeCensus(f);
  if (json) {
    std::cout << gqt::CensusJson(c, file);
  } else {
    std::string base = file;
    auto slash = base.find_last_of("/\\");
    if (slash != std::string::npos) base = base.substr(slash + 1);
    std::cout << gqt::CensusText(c, base);
  }
  return 0;
}

int main(int argc, char** argv) {
  // v0.8 (F1): a file-size limit (RLIMIT_FSIZE, the gpu-run 8 GiB cap) must
  // arrive as a failed write -- handled, exit 2 -- instead of killing the
  // process with SIGXFSZ before any cleanup runs.
  std::signal(SIGXFSZ, SIG_IGN);
  const auto t_start = std::chrono::steady_clock::now();
  std::vector<std::string> args(argv + 1, argv + argc);
  if (!args.empty() && args[0] == "census") return CensusMain(args);
  bool allow_requantize = false, pure = false, dry_run = false, leave_output = false;
  bool g4_format2 = false;  // G4 files carry gufo.quant.format_version = 2
  std::vector<std::pair<std::string, Type>> overrides;
  std::string in, out, recipe_name, recipe_file;
  std::string imatrix_file;  // SPEC3: fork imatrix GGUF (weighted ROC encode)
  std::string prune_layers;  // SPEC5: comma/space list of block layers to drop
  // v0.5 (LQP SPEC1): --protect files, the fork head flags, trace, manifest.
  std::vector<std::string> protect_files;
  bool protect_allow_unmatched = false;
  bool has_output_type = false, has_token_type = false;
  Type output_tensor_type = 0, token_embedding_type = 0;
  bool tensor_type_trace = false;
  bool show_time = false;
  std::string manifest_file;
  // SPEC3 §2 joint scale+code post-pass. Default OFF: on MiMo-9B the post-pass
  // lowers per-block weighted MSE by construction but DEGRADES held-out PPL
  // (7.71 -> 8.60), while the imatrix weighting alone beats the fork. The flag
  // remains for the user who wants the weighted-MSE-optimal variant.
  bool post_pass = false;
  int nthread = 1;

  for (size_t i = 0; i < args.size(); ++i) {
    const std::string& a = args[i];
    if (a == "--allow-requantize") allow_requantize = true;
    else if (a == "--pure") pure = true;
    else if (a == "--dry-run") dry_run = true;
    else if (a == "--leave-output-tensor") leave_output = true;
    else if (a == "--recipe-file") {
      if (++i >= args.size()) { std::cerr << "missing --recipe-file value\n"; return 2; }
      recipe_file = args[i];
    }
    else if (a == "--tensor-type") {
      if (++i >= args.size()) { std::cerr << "missing --tensor-type value\n"; return 2; }
      const std::string v = args[i];
      // Split at the LAST '=': v0.5 RULEs may start with '=' (exact match),
      // and tensor names never contain '=', so legacy `name=TYPE` (one '=')
      // splits identically to the old first-'=' split. An empty pattern
      // (`=Q8_0`) keeps its old meaning: substring "" matches every tensor.
      auto eq = v.rfind('=');
      if (eq == std::string::npos) { std::cerr << "bad --tensor-type " << v << "\n"; return 2; }
      auto it = TypeNames().find(v.substr(eq + 1));
      if (it == TypeNames().end()) { std::cerr << "unknown type " << v.substr(eq + 1) << "\n"; return 2; }
      overrides.emplace_back(v.substr(0, eq), it->second);
    }
    else if (a == "--tensor-type-trace") tensor_type_trace = true;
    else if (a == "--protect") {
      if (++i >= args.size()) { std::cerr << "missing --protect value\n"; return 2; }
      protect_files.push_back(args[i]);
    }
    else if (a == "--protect-allow-unmatched") protect_allow_unmatched = true;
    else if (a == "--output-tensor-type") {
      if (++i >= args.size()) { std::cerr << "missing --output-tensor-type value\n"; return 2; }
      auto it = TypeNames().find(args[i]);
      if (it == TypeNames().end()) { std::cerr << "unknown type " << args[i] << "\n"; return 2; }
      output_tensor_type = it->second;
      has_output_type = true;
    }
    else if (a == "--token-embedding-type") {
      if (++i >= args.size()) { std::cerr << "missing --token-embedding-type value\n"; return 2; }
      auto it = TypeNames().find(args[i]);
      if (it == TypeNames().end()) { std::cerr << "unknown type " << args[i] << "\n"; return 2; }
      token_embedding_type = it->second;
      has_token_type = true;
    }
    else if (a == "--manifest") {
      if (++i >= args.size()) { std::cerr << "missing --manifest value\n"; return 2; }
      manifest_file = args[i];
    }
    else if (a == "--time") show_time = true;
    else if (a == "--imatrix") {
      if (++i >= args.size()) { std::cerr << "missing --imatrix value\n"; return 2; }
      imatrix_file = args[i];
    }
    else if (a == "--prune-layers") {
      if (++i >= args.size()) { std::cerr << "missing --prune-layers value\n"; return 2; }
      prune_layers = args[i];
    }
    else if (a == "--post-pass") post_pass = true;
    else if (a == "--no-post-pass") post_pass = false;
    else if (!a.empty() && a[0] == '-' && a != "-") {
      std::cerr << "unknown option: " << a << "\n"; return 2;
    } else {
      if (in.empty()) in = a;
      else if (out.empty()) out = a;
      else if (recipe_name.empty()) recipe_name = a;
      else nthread = std::atoi(a.c_str());
    }
  }
  if (in.empty() || out.empty() || recipe_name.empty()) {
    std::cerr << "usage: gufo-quant [opts] in.gguf out.gguf RECIPE [nthreads]\n"; return 2;
  }
  Recipe recipe = Recipe::kRocmFp4;
  // First-class EVEN names force --pure (fork: Q4_0_ROCMFP4_EVEN/FAST_EVEN).
  // Handles both the exact "ROCmFP4_EVEN"/"FAST_EVEN" and any legacy
  // "<recipe>_EVEN" alias (which is rejected unless it is one of the EVEN pair
  // or resolves to a known recipe).
  auto pr = PureRecipes().find(recipe_name);
  if (pr != PureRecipes().end()) {
    pure = true;
    recipe = pr->second;
  } else {
    // legacy "<recipe>_EVEN" alias -> pure=true + base recipe
    auto pos = recipe_name.rfind("_EVEN");
    if (pos != std::string::npos && pos + 5 == recipe_name.size()) {
      std::string base = recipe_name.substr(0, pos);
      auto b = RecipeNames().find(base);
      if (b != RecipeNames().end()) { pure = true; recipe = b->second; }
      else { std::cerr << "unknown EVEN base recipe: " << base << "\n"; return 2; }
    } else {
      auto rc = RecipeNames().find(recipe_name);
      if (rc == RecipeNames().end()) { std::cerr << "unknown recipe: " << recipe_name << "\n"; return 2; }
      recipe = rc->second;
    }
  }
  if (nthread < 1) nthread = 1;
  if (gqt::IsG4(recipe)) g4_format2 = true;

  // Optional TOML recipe file: load the class->type table and verify it matches
  // the (byte-exact, proven) built-in selection for every category. Recipes are
  // shipped as data in tools/gufo-quant/recipes/*.toml; this both consumes that
  // data and guards against drift.
  if (!recipe_file.empty()) {
    auto tbl = gqt::ParseToml(recipe_file);
    auto get_int = [&](const std::string& k) -> std::int64_t {
      auto it = tbl.find("classes." + k);
      if (it == tbl.end()) return -1;
      try { return std::stoll(it->second); } catch (...) { return -1; }
    };
    // Map every TOML class with a concrete integer type.
    std::map<gqt::Cat, Type> from_toml;
    auto assign = [&](gqt::Cat c, const std::string& key) {
      auto v = get_int(key);
      if (v != -1) from_toml[c] = (Type)v;
    };
    assign(gqt::Cat::kOutput, "output");
    assign(gqt::Cat::kTokenEmbd, "token_embd");
    assign(gqt::Cat::kAttnQkv, "attn_qkv");
    assign(gqt::Cat::kAttnKvB, "attn_kv_b");
    assign(gqt::Cat::kAttnV, "attn_v");
    assign(gqt::Cat::kAttnK, "attn_k");
    assign(gqt::Cat::kAttnQ, "attn_q");
    assign(gqt::Cat::kAttnOut, "attn_out");
    assign(gqt::Cat::kFfnUp, "ffn_up");
    assign(gqt::Cat::kFfnGate, "ffn_gate");
    assign(gqt::Cat::kOther, "other");
    // For class -> type classes with NO layer-position predicate, the built-in
    // per-tensor result equals the TOML class type. Probe each with an
    // arbitrary representative tensor and compare. (Predicate classes —
    // ffn_down, output-vs-embd head, ffn_gate use_more_bits — are pruned below.)
    std::map<gqt::Cat, std::string> probe_name = {
      {gqt::Cat::kAttnQkv, "blk.3.attn_qkv.weight"},
      {gqt::Cat::kAttnV, "blk.3.attn_v.weight"},
      {gqt::Cat::kAttnKvB, "blk.3.attn_kv_b.weight"},
      {gqt::Cat::kAttnK, "blk.3.attn_k.weight"},
      {gqt::Cat::kAttnQ, "blk.3.attn_q.weight"},
      {gqt::Cat::kAttnOut, "blk.3.attn_output.weight"},
      {gqt::Cat::kFfnUp, "blk.3.ffn_up.weight"},
      {gqt::Cat::kOther, "blk.3.custom.weight"},
    };
    int mism = 0;
    for (auto& [cat, nm] : probe_name) {
      auto it = from_toml.find(cat);
      if (it == from_toml.end()) continue;  // not in TOML
      Type builtin = gqt::SelectType(recipe, nm, cat, gqt::RecipeDefault(recipe),
                                     /*ne0*/256, /*n_layer*/24, false, false);
      if (it->second != builtin) {
        std::cerr << "[recipe-file] mismatch " << gqt::CatName(cat) << ": toml="
                  << gqt::TypeName(it->second) << " builtin=" << gqt::TypeName(builtin) << "\n";
        ++mism;
      }
    }
    if (mism) { std::cerr << "recipe-file does not match built-in table\n"; return 2; }
    std::cerr << "recipe-file OK: " << recipe_file << " matches built-in table\n";
  }

  // v0.8 (F1): every failure below -- input errors, plan errors, pipeline
  // errors -- ends in `return 2` with one stderr line. No escaping exception
  // (no exit 134), no joinable thread left behind (the quantize branch joins
  // everything before any throw can propagate past it).
  try {
  // ---- load ----
  gqt::GgufFile f = gqt::LoadGgufStreaming(in);

  // SPEC3: an explicit --imatrix enables the fork-verbatim weighted ROC encode
  // (types 100/101); the joint post-pass (§2) refines it unless disabled.
  ImatrixMap imatrix;
  std::string imatrix_sha;
  if (!imatrix_file.empty()) {
    imatrix = LoadImatrix(imatrix_file, &imatrix_sha);
    // post_pass stays whatever the CLI set: default true (spec: fused §1+§2),
    // --no-post-pass disables it. Do NOT force it here (that broke isolation).
    std::cerr << "imatrix.sha256=" << imatrix_sha << " post_pass=" << (post_pass ? 1 : 0) << "\n";
  }

  // SPEC5: parse --prune-layers (comma and/or space separated integers).
  std::vector<int> prune_list;
  if (!prune_layers.empty()) {
    std::stringstream ss(prune_layers);
    std::string tok;
    while (ss >> tok) {
      std::string sub;
      std::stringstream ss2(tok);
      while (std::getline(ss2, sub, ',')) {
        if (sub.empty()) continue;
        try { prune_list.push_back(std::stoi(sub)); }
        catch (...) { std::cerr << "bad --prune-layers value: " << sub << "\n"; return 2; }
      }
    }
    std::sort(prune_list.begin(), prune_list.end());
    prune_list.erase(std::unique(prune_list.begin(), prune_list.end()), prune_list.end());
    std::cerr << "prune: layers ["; for (int x : prune_list) std::cerr << x << " "; std::cerr << "]\n";
  }
  if (!prune_list.empty()) {
    auto layer_of_name = [](const std::string& name) -> int {
      auto p = name.find("blk.");
      if (p == std::string::npos) return -1;
      size_t q = p + 4; long v = 0; bool any = false;
      while (q < name.size() && name[q] >= '0' && name[q] <= '9') { v = v * 10 + (name[q] - '0'); any = true; ++q; }
      return any ? (int)v : -1;
    };
    std::vector<gqt::GgufTensor> kept;
    kept.reserve(f.tensors.size());
    int dropped = 0;
    for (auto& t : f.tensors) {
      int l = layer_of_name(t.name);
      if (l >= 0 && std::find(prune_list.begin(), prune_list.end(), l) != prune_list.end()) {
        ++dropped; continue;
      }
      kept.push_back(std::move(t));
    }
    std::cerr << "prune: dropped " << dropped << " tensors\n";
    f.tensors = std::move(kept);
    int max_blk = 0;
    for (auto& t : f.tensors) {
      int l = layer_of_name(t.name);
      if (l + 1 > max_blk) max_blk = l + 1;
    }
    if (max_blk > 0) {
      for (auto& [k, v] : f.kv) {
        if (k.find("block_count") != std::string::npos && v.scalar_bytes.size() >= 4) {
          std::uint32_t x = (std::uint32_t)max_blk;
          std::memcpy(&v.scalar_bytes[0], &x, 4);
        }
      }
    }
    // The fork's prune rewrites block_count and leaves nextn_predict_layers.
    // On Qwen3.5 that leftover 1 makes the loader treat blk.<n_layer-1> as an
    // MTP block and require blk.<n_layer-1>.nextn.eh_proj, which was pruned
    // with the old last layer. King's served file omits the key (default 0).
    // Drop it only when the prune removed every nextn tensor.
    bool any_nextn = false;
    for (const auto& t : f.tensors) {
      if (t.name.find("nextn.") != std::string::npos) any_nextn = true;
    }
    if (!any_nextn) {
      std::vector<std::pair<std::string, gqt::KvValue>> kv2;
      kv2.reserve(f.kv.size());
      for (auto& [k, v] : f.kv) {
        const std::string tail = "nextn_predict_layers";
        if (k.size() >= tail.size() &&
            k.compare(k.size() - tail.size(), tail.size(), tail) == 0) {
          std::cerr << "prune: drop KV " << k
                    << " (no nextn tensors left)\n";
          continue;
        }
        kv2.push_back({k, std::move(v)});
      }
      f.kv = std::move(kv2);
    }
  }

  // SPEC5: when the input was split, the merged single output must drop the
  // split.* KVs (the fork's llama-quantize calls gguf_remove_key on them). On a
  // single file they are absent; harmless no-op otherwise.
  {
    std::vector<std::pair<std::string, gqt::KvValue>> kv2;
    kv2.reserve(f.kv.size());
    for (auto& [k, v] : f.kv) {
      if (k == "split.no" || k == "split.count" || k == "split.tensors.count") continue;
      kv2.push_back({k, v});
    }
    f.kv = std::move(kv2);
  }

  int n_layer = 32;
  for (auto& [k, v] : f.kv) {
    if (k.find("block_count") != std::string::npos && v.scalar_bytes.size() >= 4) {
      std::uint32_t bc = 0; std::memcpy(&bc, v.scalar_bytes.data(), 4);
      n_layer = (int)bc;
      break;
    }
  }
  // Sort tensors like the fork's weight_name_comparer: layer (blk.<N>., -1 for
  // non-blk) ascending, then full name lexicographic. This fixes output order.
  auto layer_of = [](const std::string& name) -> int {
    constexpr const char* pat = "blk.";
    auto p = name.find(pat);
    if (p == std::string::npos) return -1;
    size_t q = p + 4; long v = 0; bool any = false;
    while (q < name.size() && name[q] >= '0' && name[q] <= '9') { v = v * 10 + (name[q] - '0'); any = true; q++; if (q - (p+4) > 9) break; }
    return any ? (int)v : -1;
  };
  std::stable_sort(f.tensors.begin(), f.tensors.end(),
    [&](const gqt::GgufTensor& a, const gqt::GgufTensor& b) {
      int la = layer_of(a.name), lb = layer_of(b.name);
      if (la != lb) return la < lb;
      return a.name < b.name;
    });

  bool quantize_output = !leave_output;
  bool tied = true;
  for (auto& t : f.tensors) if (t.name == "output.weight") { tied = false; break; }
  bool has_expert = false;
  for (auto& t : f.tensors) if (t.dims.size() >= 3 && t.dims[2] > 1) { has_expert = true; break; }

  // v0.5 rule list: --protect rules (file order) BEFORE --tensor-type rules
  // (command-line order); the first matching rule wins for each tensor.
  std::vector<gqt::Rule> rules;
  for (auto& pf : protect_files) {
    try {
      auto pr = gqt::ParseProtectFile(pf);
      rules.insert(rules.end(), pr.begin(), pr.end());
    } catch (const std::exception& e) {
      std::cerr << "error: " << e.what() << "\n";
      return 2;
    }
  }
  for (auto& [pat, ov] : overrides) rules.push_back(gqt::ParseRule(pat, ov, "cli"));
  std::vector<std::uint64_t> rule_matches(rules.size(), 0);

  // Fork head flags (SPEC1 item 3): --output-tensor-type applies to OUTPUT
  // (exact `output.weight`, plus the fork's NextN mirror
  // `nextn.shared_head_head`), --token-embedding-type to TOKEN_EMBD (exact
  // `token_embd.weight` / `per_layer_token_embd.weight`, plus
  // `nextn.embed_tokens`) — the same categories recipe.hpp ports from the
  // fork. On a tied model (no output.weight) --output-tensor-type is a no-op
  // with a warning (SPEC1; the fork instead routes it to token_embd through
  // its ladder — see docs/gufo-quant.md).
  if (has_output_type && tied) {
    std::cerr << "warning: --output-tensor-type is a no-op on this model "
                 "(tied embeddings: no output.weight)\n";
  }

  Type default_type = gqt::RecipeDefault(recipe);
  std::vector<Type> dst_types(f.tensors.size());
  std::vector<bool> allows(f.tensors.size());
  std::vector<std::string> via(f.tensors.size());  // --tensor-type-trace text

  // Explicit --tensor-type overrides may target tensors that are otherwise
  // gated off (e.g. the router ffn_gate_inp.weight -> Q8_0, RV20 §3.4). Track
  // which tensors were overridden so they bypass the quantization gate.
  std::vector<bool> tf_overridden(f.tensors.size(), false);
  for (size_t i = 0; i < f.tensors.size(); ++i) {
    auto& t = f.tensors[i];
    std::string name = t.name;
    gqt::Cat cat = gqt::Categorize(name);
    std::uint64_t ne0 = t.dims.empty() ? 0 : t.dims[0];
    // Fork rule: ggml_n_dims(t) < 2 -> never quantize (norms, ssm_a, router
    // shexp, ... are stored with trailing-1 dims and read as 1-D). ggml_n_dims
    // counts dims up through the last dimension > 1.
    int ndims = 0;
    if (!t.dims.empty()) {
      int last_non1 = -1;
      for (int di = 0; di < (int)t.dims.size(); ++di) if (t.dims[di] > 1) last_non1 = di;
      ndims = last_non1 + 1;
    }
    bool allow = gqt::TensorAllowsQuantization(name, quantize_output) && ndims >= 2;
    // Precedence (docs/gufo-quant.md): fork head flags (gated, like the fork)
    // first, then the first matching rule (which bypasses the gate: the gufo
    // extension the router override relies on), then the recipe ladder.
    bool head_hit = false;
    Type dst = t.type;
    if (allow) {
      if (has_output_type && !tied && cat == gqt::Cat::kOutput) {
        dst = output_tensor_type;
        head_hit = true;
        via[i] = "via --output-tensor-type";
      } else if (has_token_type && cat == gqt::Cat::kTokenEmbd) {
        dst = token_embedding_type;
        head_hit = true;
        via[i] = "via --token-embedding-type";
      }
    }
    if (head_hit) {
      allows[i] = true;
      dst_types[i] = dst;
      tf_overridden[i] = true;
      continue;
    }
    bool overridden = false;
    for (auto& r : rules)
      if (r.Matches(name)) { dst = r.type; overridden = true; via[i] = "via rule '" + r.text + "' " + r.source; break; }
    tf_overridden[i] = overridden;
    if (overridden) {
      // A manual override always wins over the standard ladder; it also unlocks
      // tensors the recipe normally leaves alone (router, output, norms...).
      allows[i] = true;
      dst_types[i] = dst;
    } else {
      allows[i] = allow;
      if (!allow) { dst_types[i] = t.type; via[i] = "via kept"; }
      else if (pure) { dst_types[i] = default_type; via[i] = "via recipe(pure)"; }
      else { dst_types[i] = gqt::SelectType(recipe, name, cat, default_type, ne0, n_layer, has_expert, tied); via[i] = "via recipe"; }
    }
  }

  // Match counts per rule (name matches, regardless of precedence: a rule that
  // matches but loses to an earlier rule still counts as matching). A
  // --protect rule matching zero tensors is an error unless
  // --protect-allow-unmatched; CLI rules keep the old silent behavior.
  for (size_t r = 0; r < rules.size(); ++r)
    for (auto& t : f.tensors)
      if (rules[r].Matches(t.name)) rule_matches[r]++;
  {
    bool fail = false;
    for (size_t r = 0; r < rules.size(); ++r) {
      if (rule_matches[r] != 0) continue;
      if (rules[r].source.compare(0, 8, "protect:") != 0) continue;
      if (protect_allow_unmatched) {
        std::cerr << "protect: rule '" << rules[r].text << "' (" << rules[r].source
                  << ") matched 0 tensors (allowed by --protect-allow-unmatched)\n";
      } else {
        std::cerr << "error: --protect rule '" << rules[r].text << "' ("
                  << rules[r].source << ") matched 0 tensors\n";
        fail = true;
      }
    }
    if (fail) return 2;
  }

  // ---- v0.6 plan pre-pass (serial, tensor order) ----
  // encode_one's per-tensor decisions (kept vs encode, source support,
  // imatrix lookup, output sizes) run here, in the same order and with the
  // same throws, so the header -- and hence every output offset -- is known
  // before the first byte is encoded. Throw order within a tensor matches
  // v0.5 exactly: kept-checks, source support, RowSize, imatrix size.
  using Clock = std::chrono::steady_clock;
  auto tlog = [&](const std::string& stage, double s) {
    if (!show_time) return;
    char b[160];
    std::snprintf(b, sizeof b, "[time] %s=%.2fs", stage.c_str(), s);
    std::cerr << b << "\n";
  };
  struct TensorPlan {
    bool encode = false;  // false => kept (output payload = source raw)
    std::uint64_t ne0 = 0, nrows = 0, rowb = 0, out_size = 0;
    std::uint64_t ne1 = 1, ne2 = 1;
    const float* irow = nullptr;  // imatrix weights (ne0*ne2), or null
  };
  std::vector<TensorPlan> plans(f.tensors.size());
  for (size_t i = 0; i < f.tensors.size(); ++i) {
    auto& t = f.tensors[i];
    TensorPlan p;
    Type dt = dst_types[i];
    if (dt == t.type || !allows[i]) { p.out_size = t.size_bytes; plans[i] = p; continue; }
    std::uint64_t ne0 = t.dims[0], nrows = 1;
    for (size_t d = 1; d < t.dims.size(); ++d) nrows *= t.dims[d];
    if (nrows == 0) { p.out_size = t.size_bytes; plans[i] = p; continue; }
    if (!(t.type == (Type)0 || t.type == (Type)1 || t.type == (Type)30)) {
      // Same rethrow the old encode_one catch performed for an unsupported
      // source (e.g. quantized input without --allow-requantize).
      if (!allow_requantize)
        throw std::runtime_error("source tensor type " + std::to_string((int)t.type) +
                                 " not supported (need F32/F16/BF16 source)");
      p.out_size = t.size_bytes; plans[i] = p; continue;
    }
    // v0.8 (F4): a float-to-float request the tool cannot do exactly is
    // rejected here -- on the dry-run AND the real path (this pre-pass runs
    // for both, so the message and the exit-2 are identical). The two exact
    // widenings (F16/BF16 -> F32) encode; the recipe's own F16 shape fallback
    // (fork-exact, e.g. ple_conv1d with ne0=4 on real Qwen rosters) still
    // applies; an explicitly requested lossy float conversion (rules, head
    // flags) does not. Same-type keeps continued above; the source is float
    // (checked just above), so only the destination needs testing.
    {
      const bool dst_float = (dt == (Type)0 || dt == (Type)1 || dt == (Type)30);
      const bool widening =
          (dt == (Type)0 && (t.type == (Type)1 || t.type == (Type)30));
      const bool recipe_selected = via[i].compare(0, 10, "via recipe") == 0;
      if (dst_float && !widening && !recipe_selected)
        throw std::runtime_error("unsupported float conversion " +
                                 gqt::TypeName(t.type) + " -> " +
                                 gqt::TypeName(dt) + " for " + t.name +
                                 " (only the exact F16 -> F32 and BF16 -> F32 "
                                 "widenings are supported)");
    }
    p.encode = true;
    p.ne0 = ne0; p.nrows = nrows;
    p.ne1 = t.dims.size() > 1 ? t.dims[1] : 1;
    p.ne2 = t.dims.size() > 2 ? t.dims[2] : 1;
    p.rowb = gqt::RowSize(dt, ne0);
    // SPEC3: look up this tensor's imatrix row(s). The fork requires the
    // imatrix entry length to be ne[0]*ne[2]; expert i03 (over ne[2]) uses the
    // weight vector e[i03*ne0 ..] applied to the ne[1] rows of that expert.
    if (!imatrix.empty()) {
      auto it = imatrix.find(t.name);
      if (it != imatrix.end()) {
        if (it->second.size() == (size_t)(ne0 * p.ne2)) {
          p.irow = it->second.data();
        } else {
          // SPEC3B hardening (fork llama-quant.cpp:1605-1615): a size-mismatched
          // imatrix entry is a serious error — most of the model would silently
          // quantize unweighted. The fork throws, except token_embd which always
          // triggers the mismatch and is deliberately ignored (stays unweighted).
          const bool is_tok = t.name == "token_embd.weight" ||
                              t.name == "per_layer_token_embd.weight";
          if (!is_tok)
            throw std::runtime_error("imatrix size " +
                std::to_string(it->second.size()) + " is different from tensor size " +
                std::to_string((size_t)(ne0 * p.ne2)) + " for " + t.name);
          // token_embd: irow stays null -> unweighted, matching the fork.
        }
      }
    }
    p.out_size = p.rowb * nrows;
    plans[i] = p;
  }
  std::vector<std::uint64_t> out_sizes(f.tensors.size());
  std::uint64_t total = 0;
  for (size_t i = 0; i < f.tensors.size(); ++i) {
    out_sizes[i] = plans[i].out_size;
    total += out_sizes[i];
  }
  const auto t_load_end = Clock::now();
  tlog("header_read", std::chrono::duration<double>(t_load_end - t_start).count());

  // ---- report table (identical format in dry-run and real runs) ----
  // Sizes come from the pre-pass; the pipeline writes exactly out_sizes[i]
  // bytes per tensor, so this table matches a real run's byte for byte.
  auto print_table = [&]() {
    for (size_t i = 0; i < f.tensors.size(); ++i) {
      if (tensor_type_trace) {
        std::printf("%-40s %-8s -> %-8s  %8llu -> %8llu bytes  %s\n",
                    f.tensors[i].name.c_str(), gqt::TypeName(f.tensors[i].type).c_str(),
                    gqt::TypeName(dst_types[i]).c_str(),
                    (unsigned long long)f.tensors[i].size_bytes,
                    (unsigned long long)out_sizes[i], via[i].c_str());
      } else {
        std::printf("%-40s %-8s -> %-8s  %8llu -> %8llu bytes\n",
                    f.tensors[i].name.c_str(), gqt::TypeName(f.tensors[i].type).c_str(),
                    gqt::TypeName(dst_types[i]).c_str(),
                    (unsigned long long)f.tensors[i].size_bytes,
                    (unsigned long long)out_sizes[i]);
      }
    }
    std::printf("total data bytes: %llu\n", (unsigned long long)total);
  };

  // ---- write ----
  std::vector<gqt::ExtraKv> extra;
  { gqt::ExtraKv e; e.key = "gufo.quant.tool"; e.v.type = gqt::ValType::kString;
    e.v.str = kToolVersion; extra.push_back(e); }
  { gqt::ExtraKv e; e.key = "gufo.quant.recipe"; e.v.type = gqt::ValType::kString;
    e.v.str = recipe_name; extra.push_back(e); }
  { gqt::ExtraKv e; e.key = "gufo.quant.format_version"; e.v.type = gqt::ValType::kUint32;
    // G4 files (split-scale-plane types 120/121) are format v2; classical ROC
    // recipes keep v1. Per SPEC2 item 3.
    std::uint32_t x = g4_format2 ? 2u : 1u;
    e.v.scalar_bytes.assign((std::uint8_t*)&x, (std::uint8_t*)&x + 4);
    extra.push_back(e); }
  // SPEC3: imatrix provenance (exact bytes) + encoder settings, so any weighted
  // output is reproducible. Only written when --imatrix was given.
  if (!imatrix_sha.empty()) {
    { gqt::ExtraKv e; e.key = "gufo.quant.imatrix.sha256"; e.v.type = gqt::ValType::kString;
      e.v.str = imatrix_sha; extra.push_back(e); }
    { gqt::ExtraKv e; e.key = "gufo.quant.post_pass"; e.v.type = gqt::ValType::kString;
      e.v.str = post_pass ? "joint-scale-code" : "off"; extra.push_back(e); }
    { gqt::ExtraKv e; e.key = "gufo.quant.imatrix.source";
      e.v.type = gqt::ValType::kString;
      e.v.str = imatrix_file; extra.push_back(e); }
  }
  // v0.5: the fork head flags are recorded so the receipt is reproducible.
  // Only written when the flag was given (like the imatrix keys above).
  if (has_output_type) {
    gqt::ExtraKv e; e.key = "gufo.quant.output_tensor_type";
    e.v.type = gqt::ValType::kString;
    e.v.str = gqt::TypeName(output_tensor_type); extra.push_back(e);
  }
  if (has_token_type) {
    gqt::ExtraKv e; e.key = "gufo.quant.token_embedding_type";
    e.v.type = gqt::ValType::kString;
    e.v.str = gqt::TypeName(token_embedding_type); extra.push_back(e);
  }

  // G4 recipes write file_type = their STRIX base (105/106); the g4 recipe is
  // recorded in gufo.quant.recipe + format_version 2.
  std::uint32_t out_ftype = (std::uint32_t)recipe;
  if (g4_format2) out_ftype = (recipe == gqt::Recipe::kG4Strix) ? 105u : 106u;

  if (!dry_run) {
    // The writer opens `out` while the fetch thread still reads the input,
    // so out must not be the input (v0.5 read everything first and got away
    // with it; streaming would corrupt the input mid-read).
    for (auto& p : f.shard_paths) {
      bool same = (p == out);
      if (!same) {
        struct stat sa, sb;
        same = (!stat(p.c_str(), &sa) && !stat(out.c_str(), &sb) &&
                sa.st_dev == sb.st_dev && sa.st_ino == sb.st_ino);
      }
      if (same) { std::cerr << "error: output is an input shard: " << out << "\n"; return 2; }
    }
    // v0.8 (F1): fail fast when the output path is a directory (the rename
    // below could never publish over it); exit 2, nothing created.
    {
      struct stat dost;
      if (!stat(out.c_str(), &dost) && S_ISDIR(dost.st_mode)) {
        std::cerr << "error: output is a directory: " << out << "\n";
        return 2;
      }
    }
  }

  // ---- input hashes (side stream, --manifest only) ----
  // One thread per shard, each a sequential pass over its own shard: the
  // shards hash in parallel with each other and with the encode, instead of
  // v0.5's serial post-pass. Same bytes hashed, same hexes out.
  struct HashedFile { std::string path, sha; std::uint64_t bytes = 0; };
  std::vector<HashedFile> inputs(f.shard_paths.size());
  std::vector<double> hash_wall(f.shard_paths.size(), 0.0);
  // v0.9 (F6): the threads live in an RAII joiner so every return path joins
  // them (a bare vector + lambda relied on each branch remembering to join).
  HashThreadJoiner hashes;
  hashes.Errors().resize(f.shard_paths.size());
  // A dry-run is header-only: sizes come from the pre-pass and stat, never
  // from a content read, so no input hashing even with --manifest (the
  // manifest records sizes with sha256 null and says so; see hash_note).
  if (!manifest_file.empty() && !dry_run) {
    for (size_t s = 0; s < f.shard_paths.size(); ++s) {
      inputs[s].path = f.shard_paths[s];
      hashes.Add(std::thread([&, s]() {
        try {
          auto h0 = Clock::now();
          inputs[s].sha = gqt::Sha256File(f.shard_paths[s], &inputs[s].bytes);
          hash_wall[s] = std::chrono::duration<double>(Clock::now() - h0).count();
        } catch (...) {
          hashes.Errors()[s] = std::current_exception();
        }
      }));
    }
  }
  if (!manifest_file.empty() && dry_run) {
    for (size_t s = 0; s < f.shard_paths.size(); ++s) {
      inputs[s].path = f.shard_paths[s];
      struct stat stb;
      if (!stat(f.shard_paths[s].c_str(), &stb))
        inputs[s].bytes = (std::uint64_t)stb.st_size;
    }
  }
  HashedFile outh;
  outh.path = out;

  // The receipt JSON (SPEC1 item 5). Built once per run, before any publish,
  // so the manifest is durable before either final name appears.
  auto BuildManifestJson = [&]() -> std::string {
    const double wall_s =
        std::chrono::duration<double>(Clock::now() - t_start).count();
    double hash_s = 0.0;
    for (double w : hash_wall) hash_s = std::max(hash_s, w);
    tlog("input_hash", hash_s);
    // Per-family OUTPUT type table (all tensors, incl. kept 1-D at source).
    std::map<std::string, std::map<std::string, std::uint64_t>> fam;
    std::uint64_t out_elems = 0;
    for (size_t i = 0; i < f.tensors.size(); ++i) {
      fam[gqt::CensusFamily(f.tensors[i].name)][gqt::TypeName(dst_types[i])]++;
      std::uint64_t e = 1;
      for (auto d : f.tensors[i].dims) e *= d;
      out_elems += e;
    }
    double out_bpw = 0.0;
    if (!dry_run && out_elems) out_bpw = 8.0 * (double)total / (double)out_elems;
    std::ostringstream mf;
    mf << "{\n";
    mf << "  \"tool\": \"" << kToolVersion << "\",\n";
    mf << "  \"recipe\": \"" << gqt::JsonEscape(recipe_name) << "\",\n";
    mf << "  \"recipe_ftype\": " << out_ftype << ",\n";
    mf << "  \"pure\": " << (pure ? "true" : "false") << ",\n";
    mf << "  \"dry_run\": " << (dry_run ? "true" : "false") << ",\n";
    mf << "  \"threads\": " << nthread << ",\n";
    {
      char b[64];
      std::snprintf(b, sizeof b, "  \"wall_seconds\": %.2f,\n", wall_s);
      mf << b;
      std::snprintf(b, sizeof b, "  \"hash_seconds\": %.2f,\n", hash_s);
      mf << b;
    }
    mf << "  \"inputs\": [";
    for (size_t i = 0; i < inputs.size(); ++i) {
      if (i) mf << ",";
      mf << "\n    {\"path\": \"" << gqt::JsonEscape(inputs[i].path)
         << "\", \"sha256\": ";
      if (inputs[i].sha.empty())
        mf << "null";
      else
        mf << "\"" << inputs[i].sha << "\"";
      mf << ", \"bytes\": " << inputs[i].bytes << "}";
    }
    mf << (inputs.empty() ? "],\n" : "\n  ],\n");
    if (dry_run)
      mf << "  \"hash_note\": \"dry-run is header-only: input sizes from stat, "
            "sha256 not computed\",\n";
    if (imatrix_file.empty()) {
      mf << "  \"imatrix\": null,\n";
    } else {
      mf << "  \"imatrix\": {\"path\": \"" << gqt::JsonEscape(imatrix_file)
         << "\", \"sha256\": \"" << imatrix_sha << "\"},\n";
    }
    mf << "  \"prune_layers\": [";
    for (size_t i = 0; i < prune_list.size(); ++i) {
      if (i) mf << ", ";
      mf << prune_list[i];
    }
    mf << "],\n";
    mf << "  \"post_pass\": \"" << (post_pass ? "joint-scale-code" : "off") << "\",\n";
    mf << "  \"head_flags\": {\"output_tensor_type\": "
       << (has_output_type ? ("\"" + gqt::TypeName(output_tensor_type) + "\"") : "null")
       << ", \"token_embedding_type\": "
       << (has_token_type ? ("\"" + gqt::TypeName(token_embedding_type) + "\"") : "null")
       << "},\n";
    mf << "  \"rules\": [";
    for (size_t i = 0; i < rules.size(); ++i) {
      if (i) mf << ",";
      mf << "\n    {\"rule\": \"" << gqt::JsonEscape(rules[i].text)
         << "\", \"source\": \"" << gqt::JsonEscape(rules[i].source)
         << "\", \"kind\": \"" << gqt::RuleKindName(rules[i].kind)
         << "\", \"type\": \"" << gqt::TypeName(rules[i].type)
         << "\", \"matches\": " << rule_matches[i] << "}";
    }
    mf << (rules.empty() ? "],\n" : "\n  ],\n");
    mf << "  \"families\": {";
    {
      bool first = true;
      for (auto& [fn, tm] : fam) {
        if (!first) mf << ",";
        first = false;
        mf << "\n    \"" << gqt::JsonEscape(fn) << "\": {";
        bool f2 = true;
        for (auto& [tn, n] : tm) {
          if (!f2) mf << ",";
          f2 = false;
          mf << "\"" << gqt::JsonEscape(tn) << "\": " << n;
        }
        mf << "}";
      }
    }
    mf << (fam.empty() ? "},\n" : "\n  },\n");
    if (dry_run) {
      mf << "  \"output\": null,\n";
    } else {
      mf << "  \"output\": {\"path\": \"" << gqt::JsonEscape(out)
         << "\", \"sha256\": \"" << outh.sha << "\", \"bytes\": " << outh.bytes
         << ", \"tensors\": " << f.tensors.size() << ", \"data_bytes\": " << total;
      char b[64];
      std::snprintf(b, sizeof b, ", \"bpw\": %.4f},\n", out_bpw);
      mf << b;
    }
    mf << "  \"tensors\": " << f.tensors.size() << "\n";
    mf << "}\n";
    return mf.str();
  };

  if (dry_run) {
    // Sizes only: no tensor reads past the headers, no encode. The table is
    // identical to a real run's (same pre-pass, same printf).
    print_table();
    std::cerr << "dry-run: nothing written\n";
    // v0.9 (SPEC4 item 1, F4): a dry-run manifest is published atomically too
    // (partial -> fsync -> rename -> dir fsync), and a failure is `error:`+2.
    if (!manifest_file.empty()) {
      try {
        std::string js = BuildManifestJson();
        const std::string mpart =
            manifest_file + ".partial." + std::to_string((long)::getpid());
        {
          std::ofstream mfo(mpart, std::ios::binary);
          if (!mfo)
            throw std::runtime_error("cannot write manifest partial " + mpart +
                                     ": " + std::strerror(errno));
          mfo << js;
          mfo.flush();
          mfo.close();
          if (!mfo)
            throw std::runtime_error("cannot write manifest partial " + mpart);
        }
        FsyncPath(mpart, "manifest partial");
        if (::rename(mpart.c_str(), manifest_file.c_str()) != 0)
          throw std::runtime_error("cannot rename " + mpart + " to " +
                                   manifest_file + ": " + std::strerror(errno));
        FsyncDirOnce(DirName(manifest_file));
        std::cerr << "wrote " << manifest_file << "\n";
      } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 2;
      } catch (...) {
        std::cerr << "error: unknown failure\n";
        return 2;
      }
    }
  } else {
    // ---- streaming pipeline: header first, then fetch/encode/write ----
    // One fetch thread reads tensors in index order under a byte budget; the
    // persistent pool steals fixed chunk tasks across tensor boundaries; this
    // thread writes each payload in index order as its encode completes (the
    // output hash feeds as a side stream of the write). Chunk boundaries are
    // a pure function of shape (kChunkElems), never of thread count, and
    // every chunk writes a disjoint range -- bytes are identical for any
    // scheduling (threads 1/8/32 provably identical, as v0.2 was).
    //
    // v0.9 (SPEC4 item 2, F2): one writer per output. Claim the output first
    // with a flock-held `<out>.gq-lock`; a live claim is a refusal. The model
    // goes to a per-run unique `<out>.partial.<pid>` and the manifest (when
    // asked) to `<manifest>.partial.<pid>`; leftovers of dead runs and the old
    // fixed-name `<out>.partial` are removed by the claim. v0.9.1 (SPEC5): the
    // claim file itself is removed when this scope ends -- after success and
    // after every handled failure (the `return 2`s below) -- by ~OutputClaim.
    OutputClaim claim;
    {
      std::string claim_err = claim.Acquire(out);
      if (!claim_err.empty()) {
        std::cerr << "error: " << claim_err << "\n";
        return 2;
      }
    }
    const std::string pid_s = std::to_string((long)::getpid());
    const std::string partial = out + ".partial." + pid_s;
    std::string manifest_partial;
    if (!manifest_file.empty())
      manifest_partial = manifest_file + ".partial." + pid_s;
    bool pipe_failed = false;
    bool model_published = false;  // set once the model rename succeeds
    std::string pipe_msg;
    try {
    gqt::GgufOutHeader hdr =
        gqt::BuildGgufHeader(f, dst_types, out_sizes, out_ftype, extra);
    gqt::Sha256 out_hash_state;
    gqt::Sha256* out_hash = manifest_file.empty() ? nullptr : &out_hash_state;
    const auto t_write0 = Clock::now();
    gqt::GgufStreamWriter writer(partial, hdr.bytes, f.alignment, out_hash);

    constexpr std::uint64_t kChunkElems = 1 << 20;
    constexpr std::uint64_t kFetchBudget = 8ull << 30;
    struct Job {
      TensorPlan plan;
      std::uint8_t* payload = nullptr;  // into payloads[i]; null when kept
      std::uint8_t pad_byte = 0;        // valid target for 0-byte payloads
      std::atomic<std::uint64_t> left{0};
      bool done = false;  // under mu
      Clock::time_point t_enq, t_done;
    };
    std::vector<Job> jobs(f.tensors.size());
    std::vector<std::vector<std::uint8_t>> payloads(f.tensors.size());
    std::mutex mu;
    std::condition_variable cv;
    bool pipe_error = false;
    std::uint64_t inflight = 0;  // raw+payload bytes held (under mu)
    std::atomic<std::uint64_t> encode_cpu_ns{0};
    gqt::ThreadPool pool(nthread);
    // v0.8 (F2): a throwing chunk task wakes this thread's pipeline wait --
    // without this the writer slept forever when the throwing tensor was the
    // last one in flight (the pool's own done_cv_ is a different variable).
    pool.SetErrorCallback([&]() {
      std::lock_guard<std::mutex> lk(mu);
      pipe_error = true;
      cv.notify_all();
    });
    const auto t_enc0 = Clock::now();
    double max_tensor_s = 0.0;

    std::exception_ptr fetch_ex;
    std::thread fetch_thr([&]() {
      try {
        for (size_t i = 0; i < f.tensors.size(); ++i) {
          const std::uint64_t need =
              f.tensors[i].size_bytes + (plans[i].encode ? plans[i].out_size : 0);
          {
            std::unique_lock<std::mutex> lk(mu);
            // Always admit when nothing is in flight (one tensor may exceed
            // the budget); otherwise wait for the writer to drain.
            cv.wait(lk, [&]() {
              return pipe_error || pool.error() || inflight == 0 ||
                     inflight + need <= kFetchBudget;
            });
            if (pipe_error || pool.error()) return;
            inflight += need;
          }
          gqt::FetchTensorData(f, i);
          Job& jb = jobs[i];
          jb.plan = plans[i];
          jb.t_enq = Clock::now();
          if (!plans[i].encode) {
            std::lock_guard<std::mutex> lk(mu);
            jb.t_done = jb.t_enq;
            jb.done = true;
            cv.notify_all();
            continue;
          }
          payloads[i].resize(plans[i].out_size);
          jb.payload = payloads[i].empty() ? &jb.pad_byte : payloads[i].data();
          const std::uint64_t ne0 = plans[i].ne0, nrows = plans[i].nrows;
          std::uint64_t rpc = ne0 ? kChunkElems / ne0 : 1;
          if (rpc < 1) rpc = 1;
          const std::uint64_t nchunks = (nrows + rpc - 1) / rpc;
          jb.left.store(nchunks, std::memory_order_relaxed);
          for (std::uint64_t c = 0; c < nchunks; ++c) {
            const std::uint64_t r0 = c * rpc;
            const std::uint64_t r1 = std::min(r0 + rpc, nrows);
            pool.Submit([&, i, r0, r1]() {
              GUFO_QUANT_CHUNK_HOOK();  // test-only; absent in the shipped tool
              thread_local std::vector<float> tmp;  // reused f32 scratch
              const auto c0 = Clock::now();
              Job& wj = jobs[i];
              const TensorPlan& p = wj.plan;
              const std::uint8_t* raw = f.tensors[i].raw.data();
              const std::uint64_t needf = (r1 - r0) * p.ne0;
              if (tmp.size() < needf) tmp.resize(needf);
              DequantizeRowsToF32(raw, f.tensors[i].type, p.ne0, r0, r1,
                                  tmp.data());
              Type dt = dst_types[i];
              for (std::uint64_t r = r0; r < r1; ++r) {
                // Per-row weight pointer (weight vector advances every ne1
                // rows), verbatim the old row_weights helper.
                const float* qw = nullptr;
                if (p.irow) {
                  std::uint64_t expert = (p.ne1 > 0) ? (r / p.ne1) : 0;
                  qw = p.irow + expert * p.ne0;
                }
                EncodeRow(dt, tmp.data() + (r - r0) * p.ne0,
                          wj.payload + r * p.rowb, (std::int64_t)p.ne0, qw,
                          post_pass);
              }
              encode_cpu_ns.fetch_add(
                  (std::uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
                      Clock::now() - c0).count(),
                  std::memory_order_relaxed);
              if (wj.left.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                std::lock_guard<std::mutex> lk(mu);
                wj.t_done = Clock::now();
                wj.done = true;
                cv.notify_all();
              }
            });
          }
        }
      } catch (...) {
        std::lock_guard<std::mutex> lk(mu);
        if (!fetch_ex) fetch_ex = std::current_exception();
        pipe_error = true;
        cv.notify_all();
      }
    });

    // Writer: this thread. Waits for each tensor in index order, writes it,
    // frees its buffers (the SPEC5 bound), and moves on.
    bool write_ok = true;
    std::string write_err;
    for (size_t i = 0; i < f.tensors.size(); ++i) {
      {
        std::unique_lock<std::mutex> lk(mu);
        cv.wait(lk, [&]() { return jobs[i].done || pipe_error || pool.error(); });
      }
      if (!jobs[i].done) { write_ok = false; break; }
      auto& t = f.tensors[i];
      // v0.8 (F1): a write failure stops the pipeline like any other error:
      // record it, wake the fetch thread, and break to the join below (the
      // partial is unlinked after the join; the final name is never touched).
      try {
        if (plans[i].encode) {
          writer.WritePayload(payloads[i].empty() ? nullptr : payloads[i].data(),
                              payloads[i].size());
        } else {
          writer.WritePayload(t.raw.empty() ? nullptr : t.raw.data(), t.raw.size());
        }
      } catch (const std::exception& e) {
        write_err = e.what();
        write_ok = false;
        { std::lock_guard<std::mutex> lk(mu); pipe_error = true; }
        cv.notify_all();
        break;
      } catch (...) {
        write_err = "write failed: " + partial;
        write_ok = false;
        { std::lock_guard<std::mutex> lk(mu); pipe_error = true; }
        cv.notify_all();
        break;
      }
      {
        std::lock_guard<std::mutex> lk(mu);
        inflight -= t.size_bytes + (plans[i].encode ? payloads[i].size() : 0);
        double ts =
            std::chrono::duration<double>(jobs[i].t_done - jobs[i].t_enq).count();
        if (ts > max_tensor_s) max_tensor_s = ts;
      }
      t.raw.clear(); t.raw.shrink_to_fit(); t.raw_loaded = false;
      payloads[i].clear(); payloads[i].shrink_to_fit();
      cv.notify_all();
    }
    const auto t_write1 = Clock::now();
    fetch_thr.join();
    pool.Shutdown();
    if (!write_ok || pipe_error || pool.error()) {
      // v0.8 (F1): no output behind a failure. The final name was never
      // touched (the writer used the partial); capture the first error's
      // message and throw it to the branch catch, which unlinks the partial,
      // joins the hash threads, and returns 2. Nothing propagates past this
      // branch with a joinable thread alive.
      try { writer.Close(); } catch (...) {}
      std::string msg = write_err;
      if (msg.empty() && fetch_ex) {
        try {
          std::rethrow_exception(fetch_ex);
        } catch (const std::exception& e) {
          msg = e.what();
        } catch (...) {
          msg = "fetch failed";
        }
      }
      if (msg.empty()) {
        try {
          pool.RethrowIfError();
        } catch (const std::exception& e) {
          msg = e.what();
        } catch (...) {
          msg = "encode failed";
        }
      }
      if (msg.empty()) msg = "encode pipeline failed";
      throw std::runtime_error(msg);
    }
    writer.Close();
    // v0.9 (SPEC4 item 1): join the input-hash threads BEFORE anything is
    // published, so a hash failure aborts while only the partials exist.
    hashes.Join();

    // ---- one publish sequence for the model AND its manifest ----
    // model partial -> close (writer.Close above) -> fsync; manifest (when
    // asked) to ITS partial -> close -> fsync; rename the model; rename the
    // manifest; fsync the containing directory (once per directory). Any
    // failure before the first rename leaves NEITHER final name; a failure
    // between the two renames leaves the model COMPLETE at <out> and only the
    // manifest absent (the branch catch below says exactly that).
    FsyncPath(partial, "model partial");
    outh.sha = manifest_file.empty() ? std::string() : out_hash_state.Hex();
    {
      struct stat stb;
      if (!::stat(partial.c_str(), &stb)) outh.bytes = (std::uint64_t)stb.st_size;
    }

    if (!manifest_file.empty()) {
      const std::string js = BuildManifestJson();
      {
        std::ofstream mfo(manifest_partial, std::ios::binary);
        if (!mfo)
          throw std::runtime_error("cannot write manifest partial " +
                                   manifest_partial + ": " + std::strerror(errno));
        mfo << js;
        mfo.flush();
        mfo.close();
        if (!mfo)
          throw std::runtime_error("cannot write manifest partial " +
                                   manifest_partial);
      }
      FsyncPath(manifest_partial, "manifest partial");
    }

    if (::rename(partial.c_str(), out.c_str()) != 0)
      throw std::runtime_error("cannot rename " + partial + " to " + out + ": " +
                               std::strerror(errno));
    model_published = true;
    if (!manifest_file.empty()) {
      if (::rename(manifest_partial.c_str(), manifest_file.c_str()) != 0)
        throw std::runtime_error("cannot rename " + manifest_partial + " to " +
                                 manifest_file + ": " + std::strerror(errno));
    }
    // Durability of the renames themselves (RV-LQP3 F3): fsync each containing
    // directory once.
    FsyncDirOnce(DirName(out));
    if (!manifest_file.empty() && DirName(manifest_file) != DirName(out))
      FsyncDirOnce(DirName(manifest_file));

    tlog("encode_wall",
         std::chrono::duration<double>(t_write1 - t_enc0).count());
    tlog("encode_cpu", (double)encode_cpu_ns.load() / 1e9);
    tlog("encode_max_tensor", max_tensor_s);
    tlog("output_write",
         std::chrono::duration<double>(t_write1 - t_write0).count());
    print_table();
    std::cerr << "wrote " << out << "\n";
    if (!manifest_file.empty()) std::cerr << "wrote " << manifest_file << "\n";
    } catch (const std::exception& e) {
      pipe_failed = true;
      pipe_msg = e.what();
    } catch (...) {
      pipe_failed = true;
      pipe_msg = "unknown failure";
    }
    // v0.9 (F6): the RAII joiner guarantees a join on every path; the explicit
    // join here reports a hash exception as the run's failure.
    if (!pipe_failed) {
      try {
        hashes.Join();
      } catch (const std::exception& e) {
        pipe_failed = true;
        pipe_msg = e.what();
      } catch (...) {
        pipe_failed = true;
        pipe_msg = "input hash failed";
      }
    } else {
      try {
        hashes.Join();
      } catch (...) {
      }
    }
    if (pipe_failed) {
      // v0.9 (SPEC4 item 1): remove our partials. If the model was already
      // renamed, it is COMPLETE at <out> and only the manifest is missing.
      if (model_published) {
        std::remove(manifest_partial.c_str());
        std::cerr << "error: model IS complete at " << out
                  << " but the manifest " << manifest_file
                  << " was not published: " << pipe_msg << "\n";
      } else {
        std::remove(partial.c_str());
        if (!manifest_partial.empty()) std::remove(manifest_partial.c_str());
        std::cerr << "error: " << pipe_msg << "\n";
      }
      return 2;
    }
  }
  return 0;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 2;
  } catch (...) {
    std::cerr << "error: unknown failure\n";
    return 2;
  }
}
