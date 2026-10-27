// superroc_quant_failure_test — LQP SPEC3 item 5 + SPEC4 item 3 + SPEC5 item 2:
// superroc-quant failure paths. A failed run must leave NOTHING shaped like a
// model behind, must never hang, and must exit 2 with one stderr line
// (RV-LQP2 F1/F2, RV-LQP3 F5). Each part below FAILS on the older binary and
// PASSES on v0.9 (and v0.9.1):
//
//   Part 0  in-process: a throwing pool task invokes the error callback, so a
//           waiter on the caller's own condition variable wakes (F2).
//   (a) RLIMIT_FSIZE below the output size -> exit 2, final path ABSENT,
//       partial absent, no .gq-lock left.
//   (b) a worker really fails in the MIDDLE of the encode (RV-LQP3 F5): the
//       test-only hooks binary throws from a chunk task; the run ends within
//       the bound, exit 2, no final file, no partial, no .gq-lock. On the v0.8
//       binary the hook env var is ignored, so this case FAILS there.
//   (c) a truncated input -> exit 2, no output, no .gq-lock.
//   (d) an output path that is a directory -> exit 2, no .gq-lock.
//   (e) a rejected float conversion gives the same message and exit code on
//       dry-run and real; no .gq-lock.
//   (f) a successful run leaves no partial and no .gq-lock beside the output.
//   (g) a `<out>.gq-lock` left by a SIGNAL-killed run is taken over and removed
//       (SPEC5 item 2): a child takes the real flock and is SIGKILLed, so the
//       file stays and the lock is free; the tool must then claim it and remove
//       it at exit. FAILS on v0.9 (the file is never removed).
//   (h) the loser of a race leaves the winner's lock file alone while the
//       winner runs (SPEC5 item 2): this process holds the flock, a full run is
//       refused with the pid, and the winner's file is untouched -- then a
//       later run takes the claim over and removes it.
//
// Usage: superroc_quant_failure_test <superroc-quant> <superroc-quant-hooks> [workdir]
// The hooks binary is the v0.9 tool built with -DSUPERROC_QUANT_TEST_HOOKS=1 (a
// CMake test target); the shipped binary never defines that macro. If the
// second argument is omitted, part (b) uses the first binary and is expected
// to FAIL (that is the v0.8 regression demonstration).

#include <fcntl.h>
#include <sys/file.h>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <mutex>
#include <string>
#include <vector>

#include "gguf.hpp"
#include "pool.hpp"
#include "vendored_quantize.hpp"  // GGML_FP32_TO_FP16 for the F16 fixture

namespace gqt = superroc::quant_tool;

namespace {

void Expect(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    std::exit(1);
  }
}

std::string ReadFile(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  Expect(!!f, "cannot open " + path);
  return std::string((std::istreambuf_iterator<char>(f)),
                     std::istreambuf_iterator<char>());
}

gqt::KvValue KvStr(const std::string& s) {
  gqt::KvValue v;
  v.type = gqt::ValType::kString;
  v.str = s;
  return v;
}
gqt::KvValue KvU32(std::uint32_t x) {
  gqt::KvValue v;
  v.type = gqt::ValType::kUint32;
  v.scalar_bytes.assign((const std::uint8_t*)&x, (const std::uint8_t*)&x + 4);
  return v;
}

// True when no partial of `out` survives: neither v0.8's fixed-name
// `<out>.partial` nor any per-run `<out>.partial.<pid>` (v0.9).
bool NoPartial(const std::string& out) {
  namespace fs = std::filesystem;
  if (fs::exists(out + ".partial"))
    return false;
  const fs::path p(out);
  const std::string prefix = p.filename().string() + ".partial.";
  fs::path dir = p.parent_path();
  std::error_code ec;
  for (auto& e :
       fs::directory_iterator(dir.empty() ? fs::path(".") : dir, ec)) {
    const std::string n = e.path().filename().string();
    if (n.rfind(prefix, 0) == 0)
      return false;
  }
  return true;
}

// True when the `<out>.gq-lock` claim file is gone. v0.9.1 removes it at every
// exit (success and every handled failure) -- a build of several variants must
// leave no stray lock beside the output.
bool NoLock(const std::string& out) {
  return !std::filesystem::exists(out + ".gq-lock");
}

// ---- child runner (fork/exec, no shell: paths may contain spaces) ----
// code: 0-255 = exit status, -1 = killed by a signal, 124 = timed out.
struct ProcResult {
  int code = -1;
};
ProcResult RunChild(
    const std::vector<std::string>& argv, const std::string& stdout_path,
    const std::string& stderr_path, int timeout_s, rlim_t fsize_limit = 0,
    const std::vector<std::pair<std::string, std::string>>& env = {}) {
  ProcResult r;
  pid_t pid = fork();
  Expect(pid >= 0, "fork failed");
  if (pid == 0) {
    // Child: redirects first, then the file-size limit, then env, then exec.
    if (!stdout_path.empty()) {
      int fd = ::open(stdout_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
      if (fd < 0)
        ::_exit(127);
      ::dup2(fd, 1);
      ::close(fd);
    }
    if (!stderr_path.empty()) {
      int fd = ::open(stderr_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
      if (fd < 0)
        ::_exit(127);
      ::dup2(fd, 2);
      ::close(fd);
    }
    if (fsize_limit > 0) {
      struct rlimit rl;
      rl.rlim_cur = fsize_limit;
      rl.rlim_max = fsize_limit;
      if (::setrlimit(RLIMIT_FSIZE, &rl) != 0)
        ::_exit(127);
    }
    for (auto& kv : env)
      if (::setenv(kv.first.c_str(), kv.second.c_str(), 1) != 0)
        ::_exit(127);
    std::vector<char*> args;
    for (auto& a : argv)
      args.push_back(const_cast<char*>(a.c_str()));
    args.push_back(nullptr);
    ::execvp(args[0], args.data());
    ::_exit(127);
  }
  // Parent: poll for the exit, kill after the timeout.
  const int step_ms = 20;
  int waited_ms = 0;
  for (;;) {
    int st = 0;
    pid_t w = ::waitpid(pid, &st, WNOHANG);
    if (w == pid) {
      if (WIFEXITED(st))
        r.code = WEXITSTATUS(st);
      else
        r.code = -1;
      return r;
    }
    if (waited_ms >= timeout_s * 1000) {
      ::kill(pid, SIGKILL);
      ::waitpid(pid, &st, 0);
      r.code = 124;  // like GNU timeout
      return r;
    }
    ::usleep((useconds_t)(step_ms * 1000));
    waited_ms += step_ms;
  }
}

// ---- fixtures ----
// One F16 tensor (the F2 repro shape): STRIX would keep/convert it, and an
// explicit float override exercises the pre-pass accept/reject.
std::string MakeF16One(const std::filesystem::path& dir,
                       const std::string& leaf) {
  gqt::GgufFile src;
  src.alignment = 32;
  src.kv.emplace_back("general.architecture", KvStr("qwen35"));
  src.kv.emplace_back("qwen35.block_count", KvU32(2));
  gqt::GgufTensor t;
  t.name = "bad.weight";
  t.dims = {256, 64};
  t.type = 1;  // F16 source
  std::uint64_t nelem = 256 * 64;
  for (std::uint64_t k = 0; k < nelem; ++k) {
    float v = (float)((k * 2654435761u) % 2000) / 1000.0f - 1.0f + 0.0005f;
    std::uint16_t h = ::GGML_FP32_TO_FP16(v);
    t.raw.push_back((std::uint8_t)(h & 0xFF));
    t.raw.push_back((std::uint8_t)(h >> 8));
  }
  t.size_bytes = t.raw.size();
  src.tensors.push_back(std::move(t));
  std::vector<gqt::Type> types = {1};
  std::vector<std::vector<std::uint8_t>> payloads = {src.tensors[0].raw};
  const std::string p = (dir / leaf).string();
  gqt::WriteGguf(p, src, types, payloads, /*ftype=*/0, {});
  return p;
}

// Small multi-tensor BF16 source (header ~hundreds of bytes, output ~100 KB,
// so RLIMIT_FSIZE=4096 lands mid-payload and truncation points are easy).
std::string MakeSmallBF16(const std::filesystem::path& dir,
                          const std::string& leaf) {
  gqt::GgufFile src;
  src.alignment = 32;
  src.kv.emplace_back("general.architecture", KvStr("qwen35"));
  src.kv.emplace_back("qwen35.block_count", KvU32(2));
  auto add = [&](const std::string& name, std::vector<std::uint64_t> dims) {
    gqt::GgufTensor t;
    t.name = name;
    t.dims = std::move(dims);
    t.type = 30;
    std::uint64_t nelem = 1;
    for (auto d : t.dims)
      nelem *= d;
    for (std::uint64_t k = 0; k < nelem; ++k) {
      float v = (float)((k * 2654435761u) % 2000) / 1000.0f - 1.0f + 0.0005f;
      std::uint32_t u;
      std::memcpy(&u, &v, 4);
      std::uint16_t h = (std::uint16_t)(u >> 16);
      t.raw.push_back((std::uint8_t)(h & 0xFF));
      t.raw.push_back((std::uint8_t)(h >> 8));
    }
    t.size_bytes = t.raw.size();
    src.tensors.push_back(std::move(t));
  };
  add("blk.0.attn_q.weight", {256, 64});
  add("blk.0.ffn_up.weight", {256, 128});
  add("token_embd.weight", {256, 32});
  add("output.weight", {256, 32});
  std::vector<gqt::Type> types(src.tensors.size(), 30);
  std::vector<std::vector<std::uint8_t>> payloads;
  for (auto& t : src.tensors)
    payloads.push_back(t.raw);
  const std::string p = (dir / leaf).string();
  gqt::WriteGguf(p, src, types, payloads, /*ftype=*/0, {});
  return p;
}

// ---- Part 0: the pool wakes the caller's waiter on a worker throw ----
void Part0() {
  std::mutex mu;
  std::condition_variable cv;
  bool woken = false;
  {
    gqt::ThreadPool pool(2);
    pool.SetErrorCallback([&]() {
      std::lock_guard<std::mutex> lk(mu);
      woken = true;
      cv.notify_all();
    });
    pool.Submit([]() { throw std::runtime_error("boom-from-worker"); });
    std::unique_lock<std::mutex> lk(mu);
    const bool ok =
        cv.wait_for(lk, std::chrono::seconds(5), [&]() { return woken; });
    Expect(ok, "(0): error callback woke the waiter within 5 s");
    Expect(pool.error(), "(0): pool records the error");
    pool.Shutdown();
    bool rethrew = false;
    try {
      pool.RethrowIfError();
    } catch (const std::runtime_error& e) {
      rethrew = (std::string(e.what()) == "boom-from-worker");
    }
    Expect(rethrew, "(0): first worker exception rethrown");
  }
  // A pool without a callback keeps the old behavior (done_cv_ only).
  {
    gqt::ThreadPool pool(2);
    pool.Submit([]() { throw std::runtime_error("x"); });
    Expect(!pool.WaitIdle(), "(0): WaitIdle still reports the error");
    pool.Shutdown();
  }
  std::cout << "Part 0 pool error callback PASS\n";
}

void PartA(const std::string& superroc_quant, const std::filesystem::path& dir) {
  const std::string src = MakeSmallBF16(dir, "a-src.gguf");
  const std::string out = (dir / "a-out.gguf").string();
  const std::string slog = (dir / "a-stdout.txt").string();
  const std::string elog = (dir / "a-stderr.txt").string();
  // 4096 bytes is below any successful output of this fixture (~100 KB): the
  // failure must land mid-payload, after the header was already written.
  ProcResult r = RunChild({superroc_quant, src, out, "STRIX", "2"}, slog, elog,
                          /*timeout_s=*/60, /*fsize_limit=*/4096);
  Expect(r.code == 2, "(a): RLIMIT_FSIZE rc 2, got " + std::to_string(r.code));
  Expect(!std::filesystem::exists(out), "(a): final path absent");
  Expect(NoPartial(out), "(a): partial absent");
  Expect(NoLock(out), "(a): no .gq-lock left");
  std::cout << "Part (a) file-size limit PASS\n";
}

void PartB(const std::string& superroc_quant_hooks,
           const std::filesystem::path& dir) {
  const std::string src = MakeSmallBF16(dir, "b-src.gguf");
  const std::string out = (dir / "b-out.gguf").string();
  const std::string slog = (dir / "b-stdout.txt").string();
  const std::string elog = (dir / "b-stderr.txt").string();
  // RV-LQP3 F5: a worker really fails in the MIDDLE of the encode. The
  // test-only hook throws from every chunk task, so no payload completes and
  // the writer must be woken (F2) and fail fast. On the v0.8 binary the env
  // var is meaningless -- the run SUCCEEDS -- so this case fails there.
  ProcResult r =
      RunChild({superroc_quant_hooks, src, out, "STRIX", "4"}, slog, elog,
               /*timeout_s=*/30,
               /*fsize_limit=*/0,
               /*env=*/{{"SUPERROC_QUANT_TEST_FAIL_CHUNK", "1"}});
  Expect(r.code == 2, "(b): mid-encode worker failure rc 2, got " +
                          std::to_string(r.code) + " (124 = hung)");
  Expect(!std::filesystem::exists(out), "(b): no final file");
  Expect(NoPartial(out), "(b): no partial left");
  Expect(NoLock(out), "(b): no .gq-lock left");
  const std::string e = ReadFile(elog);
  Expect(e.rfind("error:", 0) == 0, "(b): stderr is one `error:` line");
  std::cout << "Part (b) mid-encode worker failure PASS\n";
}

void PartC(const std::string& superroc_quant, const std::filesystem::path& dir) {
  const std::string src = MakeSmallBF16(dir, "c-src.gguf");
  // Truncate mid-body with the header intact: data_base + 64 bytes.
  std::uint64_t cut = 0;
  {
    std::ifstream in(src, std::ios::binary);
    Expect(!!in, "(c): cannot open fixture");
    gqt::HeaderResult hr = gqt::LoadHeaderStream(in);
    cut = hr.data_base + 64;
  }
  Expect(cut + 1024 < std::filesystem::file_size(src),
         "(c): truncation point inside the body");
  const std::string trunc = (dir / "c-trunc.gguf").string();
  {
    std::ifstream in(src, std::ios::binary);
    std::vector<char> b(cut);
    in.read(b.data(), (std::streamsize)cut);
    Expect((std::uint64_t)in.gcount() == cut, "(c): cannot cut fixture");
    std::ofstream o(trunc, std::ios::binary);
    o.write(b.data(), (std::streamsize)cut);
  }
  const std::string out = (dir / "c-out.gguf").string();
  ProcResult r = RunChild({superroc_quant, trunc, out, "STRIX", "2"},
                          (dir / "c-stdout.txt").string(),
                          (dir / "c-stderr.txt").string(), /*timeout_s=*/60);
  Expect(r.code == 2,
         "(c): truncated input rc 2, got " + std::to_string(r.code));
  Expect(!std::filesystem::exists(out), "(c): no output");
  Expect(NoPartial(out), "(c): no partial");
  Expect(NoLock(out), "(c): no .gq-lock left");
  std::cout << "Part (c) truncated input PASS\n";
}

void PartD(const std::string& superroc_quant, const std::filesystem::path& dir) {
  const std::string src = MakeSmallBF16(dir, "d-src.gguf");
  const std::string d = (dir / "d-outdir").string();
  std::error_code ec;
  std::filesystem::create_directories(d, ec);
  Expect(!ec, "(d): cannot create dir");
  ProcResult r = RunChild({superroc_quant, src, d, "STRIX", "2"},
                          (dir / "d-stdout.txt").string(),
                          (dir / "d-stderr.txt").string(), /*timeout_s=*/60);
  Expect(r.code == 2,
         "(d): directory output rc 2, got " + std::to_string(r.code));
  Expect(std::filesystem::is_directory(d), "(d): directory unchanged");
  Expect(NoPartial(d), "(d): no partial sibling");
  Expect(NoLock(d), "(d): no .gq-lock left");
  std::cout << "Part (d) directory output PASS\n";
}

void PartE(const std::string& superroc_quant, const std::filesystem::path& dir) {
  const std::string src = MakeF16One(dir, "e-src.gguf");
  const std::string out = (dir / "e-out.gguf").string();
  const std::string dry_err = (dir / "e-dry-stderr.txt").string();
  const std::string real_err = (dir / "e-real-stderr.txt").string();
  const std::string devnull = "/dev/null";
  ProcResult dry = RunChild({superroc_quant, "--tensor-type", "=bad.weight=BF16",
                             "--dry-run", src, out, "STRIX", "2"},
                            devnull, dry_err, /*timeout_s=*/60);
  ProcResult real = RunChild(
      {superroc_quant, "--tensor-type", "=bad.weight=BF16", src, out, "STRIX", "2"},
      devnull, real_err, /*timeout_s=*/60);
  Expect(dry.code == 2, "(e): dry-run rc 2, got " + std::to_string(dry.code));
  Expect(real.code == 2, "(e): real rc 2, got " + std::to_string(real.code));
  const std::string de = ReadFile(dry_err), re = ReadFile(real_err);
  Expect(de == re, "(e): same message on dry-run and real");
  Expect(de.find("unsupported float conversion") != std::string::npos,
         "(e): rejection names the conversion");
  Expect(!std::filesystem::exists(out), "(e): no output");
  Expect(NoLock(out), "(e): no .gq-lock left");
  std::cout << "Part (e) dry-run == real rejection PASS\n";
}

void PartF(const std::string& superroc_quant, const std::filesystem::path& dir) {
  const std::string src = MakeSmallBF16(dir, "f-src.gguf");
  const std::string out = (dir / "f-out.gguf").string();
  ProcResult r = RunChild({superroc_quant, src, out, "STRIX", "2"},
                          (dir / "f-stdout.txt").string(),
                          (dir / "f-stderr.txt").string(), /*timeout_s=*/60);
  Expect(r.code == 0, "(f): success rc 0, got " + std::to_string(r.code));
  Expect(std::filesystem::exists(out), "(f): output written");
  Expect(std::filesystem::file_size(out) > 0, "(f): output non-empty");
  Expect(NoPartial(out), "(f): no .partial left");
  Expect(NoLock(out), "(f): no .gq-lock left");
  std::cout << "Part (f) clean success PASS\n";
}

// (g) a `<out>.gq-lock` left by a SIGNAL-killed run is taken over cleanly and
// removed. We reproduce the exact on-disk state a killed writer leaves: a
// child takes the flock on the real lock file and is then SIGKILLed, so the
// kernel drops the lock but the file stays. The tool must claim that file
// (no live holder), run to success, and remove it at its own exit.
void PartG(const std::string& superroc_quant, const std::filesystem::path& dir) {
  const std::string src = MakeSmallBF16(dir, "g-src.gguf");
  const std::string out = (dir / "g-out.gguf").string();
  const std::string lock = out + ".gq-lock";
  int pfd[2];
  Expect(::pipe(pfd) == 0, "(g): pipe failed");
  pid_t holder = fork();
  Expect(holder >= 0, "(g): fork failed");
  if (holder == 0) {
    ::close(pfd[0]);
    int fd = ::open(lock.c_str(), O_CREAT | O_RDWR, 0644);
    if (fd < 0)
      ::_exit(3);
    if (::flock(fd, LOCK_EX) != 0)
      ::_exit(4);
    std::string p = std::to_string((long)::getpid());  // like the tool does
    ssize_t wr = ::write(fd, p.data(), p.size());
    (void)wr;
    char c = 'x';
    (void)!::write(pfd[1], &c, 1);  // tell the parent we hold the flock
    ::close(pfd[1]);
    for (;;)
      ::pause();  // hold until SIGKILL
  }
  ::close(pfd[1]);
  char c = 0;
  const ssize_t got = ::read(pfd[0], &c, 1);  // wait until the child holds it
  ::close(pfd[0]);
  Expect(got == 1, "(g): holder did not signal");
  ::kill(holder, SIGKILL);  // kernel drops the flock; the file remains
  int st = 0;
  ::waitpid(holder, &st, 0);
  Expect(std::filesystem::exists(lock),
         "(g): the killed run left the lock file behind");
  ProcResult r = RunChild({superroc_quant, src, out, "STRIX", "2"},
                          (dir / "g-stdout.txt").string(),
                          (dir / "g-stderr.txt").string(), /*timeout_s=*/60);
  Expect(r.code == 0, "(g): takeover run rc 0, got " + std::to_string(r.code));
  Expect(std::filesystem::exists(out), "(g): takeover run published");
  Expect(NoLock(out), "(g): the taken-over lock was removed");
  std::cout << "Part (g) stale lock taken over and removed PASS\n";
}

// (h) the loser of a race leaves the winner's lock alone while the winner
// runs, and is refused. This process plays the winner: it holds the flock on
// `<out>.gq-lock` (O_CLOEXEC, so the exec'd tool does not inherit it) and a
// full run to the same output must be refused with exit 2 and the pid in the
// message -- and must NOT remove the winner's file. After the winner releases,
// a real run takes the claim over and removes it.
void PartH(const std::string& superroc_quant, const std::filesystem::path& dir) {
  const std::string src = MakeSmallBF16(dir, "h-src.gguf");
  const std::string out = (dir / "h-out.gguf").string();
  const std::string lock = out + ".gq-lock";
  int wfd = ::open(lock.c_str(), O_CREAT | O_RDWR | O_CLOEXEC, 0644);
  Expect(wfd >= 0, "(h): cannot open claim file");
  Expect(::flock(wfd, LOCK_EX | LOCK_NB) == 0, "(h): winner cannot take claim");
  {
    std::string p = std::to_string((long)::getpid());
    ssize_t wr = ::write(wfd, p.data(), p.size());
    (void)wr;
  }

  ProcResult r = RunChild({superroc_quant, src, out, "STRIX", "2"},
                          (dir / "h-stdout.txt").string(),
                          (dir / "h-stderr.txt").string(), /*timeout_s=*/60);
  Expect(r.code == 2, "(h): loser refused rc 2, got " + std::to_string(r.code));
  Expect(!std::filesystem::exists(out), "(h): loser published nothing");
  Expect(NoPartial(out), "(h): loser left no partial");
  Expect(std::filesystem::exists(lock),
         "(h): loser removed the winner's lock file");
  const std::string e = ReadFile((dir / "h-stderr.txt").string());
  Expect(e.find("is being written by pid") != std::string::npos,
         "(h): refusal names the live writer");

  ::close(wfd);  // the winner exits, dropping the lock; the file remains
  Expect(std::filesystem::exists(lock),
         "(h): lock file still there after the winner released");
  ProcResult r2 = RunChild({superroc_quant, src, out, "STRIX", "2"},
                           (dir / "h2-stdout.txt").string(),
                           (dir / "h2-stderr.txt").string(), /*timeout_s=*/60);
  Expect(r2.code == 0,
         "(h): takeover run rc 0, got " + std::to_string(r2.code));
  Expect(std::filesystem::exists(out), "(h): takeover run published");
  Expect(NoLock(out), "(h): takeover run removed the lock");
  std::cout << "Part (h) loser leaves the winner's lock, then takeover PASS\n";
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "usage: superroc_quant_failure_test <superroc-quant> "
                 "<superroc-quant-hooks> [workdir]\n";
    return 2;
  }
  const std::string superroc_quant = argv[1];
  // Arg 2 is the test-hook build of the same tool (only it honours
  // SUPERROC_QUANT_TEST_FAIL_CHUNK). Omitting it makes part (b) use the shipped
  // binary, which lacks the hook -- the v0.8 regression demonstration.
  const std::string superroc_quant_hooks = (argc > 2) ? argv[2] : argv[1];
  std::filesystem::path dir;
  if (argc > 3) {
    dir = argv[3];
  } else {
    dir = std::filesystem::temp_directory_path() /
          ("superroc-quant-failure-test-" + std::to_string((long)getpid()));
  }
  std::error_code ec;
  std::filesystem::remove_all(dir, ec);
  std::filesystem::create_directories(dir, ec);
  Expect(!ec, "cannot create workdir " + dir.string());
  Expect(std::filesystem::exists(superroc_quant_hooks),
         "hook binary not found: " + superroc_quant_hooks);

  Part0();
  PartA(superroc_quant, dir);
  PartB(superroc_quant_hooks, dir);
  PartC(superroc_quant, dir);
  PartD(superroc_quant, dir);
  PartE(superroc_quant, dir);
  PartF(superroc_quant, dir);
  PartG(superroc_quant, dir);
  PartH(superroc_quant, dir);

  std::filesystem::remove_all(dir, ec);
  std::cout << "superroc_quant_failure_test PASS\n";
  return 0;
}
