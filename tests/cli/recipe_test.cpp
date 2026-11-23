// Unit tests for superroc-recipe/1 (--recipe): parse/validate
// refusals, env safety, env and command-line precedence, relative file paths.
// CPU only.
// Style follows tests/cli/lfm2_tool_parse_test.cpp: plain main, PASS/FAIL
// lines, nonzero exit on any failure.

#include "src/cli/serve/recipe.hpp"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#include "src/core/env.hpp"

namespace {

namespace fs = std::filesystem;
using superroc::cli::RecipeMerge;
using superroc::cli::RecipeSpec;

int failures = 0;

void Check(bool condition, const std::string& message) {
  if (condition) {
    std::cout << "PASS " << message << "\n";
  } else {
    std::cout << "FAIL " << message << "\n";
    ++failures;
  }
}

bool TestFlag(std::string_view name, bool* takes_value,
              std::string* canonical) {
  // Mirror the llm parser: short spellings resolve to their canonical long
  // name, which is what the recipe checks compare.
  std::string_view canon = name;
  if (name == "-m") {
    canon = "--model";
  } else if (name == "-p") {
    canon = "--port";
  } else if (name == "-i") {
    canon = "--host";
  } else if (name == "-j") {
    canon = "--sessions";
  } else if (name == "-c") {
    canon = "--context";
  } else if (name == "-t") {
    canon = "--temperature";
  } else if (name == "-d") {
    canon = "--draft-tokens";
  } else if (name == "-n") {
    canon = "--max-tokens";
  } else if (name == "-s") {
    canon = "--seed";
  }
  bool takes = false;
  bool known = true;
  if (canon == "--speculative" || canon == "--context" || canon == "--model" ||
      canon == "--draft-tokens" || canon == "--mtp-model" ||
      canon == "--mmproj" || canon == "--dflash-model" ||
      canon == "--dspark-model" || canon == "--cache-disk" ||
      canon == "--trace" || canon == "--draft-policy" || canon == "--think" ||
      canon == "--temperature" || canon == "--max-tokens" ||
      canon == "--reasoning-budget" || canon == "--seed" ||
      canon == "--recipe" || canon == "--port" || canon == "--host" ||
      canon == "--sessions" || canon == "--max-connections" ||
      canon == "--max-request-bytes" || canon == "--api-key" ||
      canon == "--api-key-file" || canon == "--served-model-name" ||
      canon == "--max-pending" || canon == "--request-timeout-ms" ||
      canon == "--cache-ram-bytes" || canon == "--log-level") {
    takes = true;
  } else if (canon == "--prompt-lookup" || canon == "--verbose") {
    takes = false;
  } else {
    known = false;
  }
  if (!known) {
    return false;
  }
  if (takes_value != nullptr) {
    *takes_value = takes;
  }
  if (canonical != nullptr) {
    *canonical = std::string(canon);
  }
  return true;
}

fs::path MakeDir(const std::string& tag) {
  fs::path dir = fs::temp_directory_path() / ("recipe_test_" + tag);
  fs::remove_all(dir);
  fs::create_directories(dir);
  return dir;
}

void Write(const fs::path& path, const std::string& text) {
  std::ofstream out(path);
  out << text;
  out.close();
  if (!out.good()) {
    Check(false, "write " + path.string());
  }
}

// Parses; on failure records FAIL and returns false.
bool MustParse(const fs::path& path, RecipeSpec* spec) {
  std::string err;
  const bool ok = superroc::cli::ParseRecipeFile(path.string(), spec, &err);
  Check(ok, "parse " + path.filename().string() + (ok ? "" : ": " + err));
  return ok;
}

void TestGoodFile() {
  const fs::path dir = MakeDir("good");
  const fs::path overlay = dir / "sidecar.gguf";
  Write(overlay, "x");
  Write(dir / "sidecar.gguf.mtp", "x");
  Write(
      dir / "r.json",
      R"({"schema": "superroc-recipe/1", "model": "main.gguf")"
      R"(, "env": {"SUPERROC_TAIL_CAPTURE": "64", "LFN_MTP_P_MIN": "0.5", "LING3_MLA_TILE": "1"})"
      R"(, "files": {"SUPERROC_TENSOR_OVERLAY": "sidecar.gguf"})"
      R"(, "file_args": {"--mtp-model": "sidecar.gguf.mtp"})"
      R"(, "args": ["--speculative", "mtp", "--prompt-lookup"])"
      R"(, "context": 131072, "notes": "free"})");
  RecipeSpec spec;
  if (!MustParse(dir / "r.json", &spec)) {
    fs::remove_all(dir);
    return;
  }
  Check(spec.model == "main.gguf", "good model");
  Check(spec.env.at("SUPERROC_TAIL_CAPTURE") == "64", "good env capture");
  Check(spec.env.at("LFN_MTP_P_MIN") == "0.5", "good env pmin");
  Check(spec.env.at("LING3_MLA_TILE") == "1", "good env ling3");
  Check(spec.files.at("SUPERROC_TENSOR_OVERLAY") == overlay.string(),
        "good file resolves against recipe dir");
  Check(spec.file_args.at("--mtp-model") == (dir / "sidecar.gguf.mtp").string(),
        "good file_args resolves against recipe dir");
  Check((spec.args ==
         std::vector<std::string>{"--speculative", "mtp", "--prompt-lookup"}),
        "good args");
  Check(spec.has_context && spec.context == 131072, "good context");
  std::string err;
  Check(superroc::cli::ValidateRecipeForModel(spec, "/m/main.gguf", TestFlag,
                                              &err),
        "good model basename validates");
  fs::remove_all(dir);
}

void MustRefuseFile(const std::string& tag, const std::string& body,
                    const std::string& want) {
  const fs::path dir = MakeDir(tag);
  Write(dir / "sidecar.gguf", "x");
  Write(dir / "r.json", body);
  RecipeSpec spec;
  std::string err;
  const bool ok =
      superroc::cli::ParseRecipeFile((dir / "r.json").string(), &spec, &err);
  Check(!ok, "refuse " + tag + ": parses");
  Check(err.find(want) != std::string::npos,
        "refuse " + tag + ": mentions '" + want + "' (got: " + err + ")");
  fs::remove_all(dir);
}

void TestRefusals() {
  MustRefuseFile("schema", R"({"schema": "nope", "model": "m.gguf"})",
                 "schema");
  MustRefuseFile("noschema", R"({"model": "m.gguf"})", "schema");
  MustRefuseFile("nomodel", R"({"schema": "superroc-recipe/1"})", "model");
  MustRefuseFile(
      "topkey",
      R"({"schema": "superroc-recipe/1", "model": "m.gguf", "bogus": 1})",
      "unknown key 'bogus'");
  MustRefuseFile(
      "envname",
      R"({"schema": "superroc-recipe/1", "model": "m.gguf", "env": {"PATH": "x"}})",
      "bad env key 'PATH'");
  MustRefuseFile(
      "envlower",
      R"({"schema": "superroc-recipe/1", "model": "m.gguf", "env": {"superroc_x": "1"}})",
      "bad env key");
  MustRefuseFile(
      "envempty",
      R"({"schema": "superroc-recipe/1", "model": "m.gguf", "env": {"SUPERROC_": "1"}})",
      "bad env key");
  MustRefuseFile(
      "envtype",
      R"({"schema": "superroc-recipe/1", "model": "m.gguf", "env": {"LFN_OK": 7}})",
      "must be a string");
  MustRefuseFile(
      "missingfile",
      R"({"schema": "superroc-recipe/1", "model": "m.gguf", "files": {"SUPERROC_TENSOR_OVERLAY": "nope.gguf"}})",
      "does not exist");
  MustRefuseFile(
      "argtype",
      R"({"schema": "superroc-recipe/1", "model": "m.gguf", "args": [7]})",
      "nonempty strings");
  MustRefuseFile(
      "contexttype",
      R"({"schema": "superroc-recipe/1", "model": "m.gguf", "context": "big"})",
      "context must be a number");
  MustRefuseFile(
      "contextzero",
      R"({"schema": "superroc-recipe/1", "model": "m.gguf", "context": 0})",
      "positive integer");
  MustRefuseFile("toplevel", "[1, 2]", "top level must be an object");
  MustRefuseFile("badjson", "{oops", "bad JSON");

  const fs::path dir = MakeDir("model");
  Write(
      dir / "r.json",
      R"({"schema": "superroc-recipe/1", "model": "main.gguf", "args": ["--speculative", "mtp"]})");
  RecipeSpec spec;
  if (!MustParse(dir / "r.json", &spec)) {
    fs::remove_all(dir);
    return;
  }
  std::string err;
  Check(!superroc::cli::ValidateRecipeForModel(spec, "/m/other.gguf", TestFlag,
                                               &err),
        "refuse model mismatch");
  Check(err.find("does not match basename") != std::string::npos,
        "mismatch message (got: " + err + ")");
  Check(!superroc::cli::ValidateRecipeForModel(spec, "", TestFlag, &err),
        "refuse missing model");
  RecipeSpec bad = spec;
  bad.args = {"--nope-flag", "1"};
  Check(!superroc::cli::ValidateRecipeForModel(bad, "/m/main.gguf", TestFlag,
                                               &err),
        "refuse unknown arg");
  Check(err.find("unknown llm flag '--nope-flag'") != std::string::npos,
        "unknown-arg message (got: " + err + ")");
  bad.args = {"lonely-value"};
  Check(!superroc::cli::ValidateRecipeForModel(bad, "/m/main.gguf", TestFlag,
                                               &err),
        "refuse value without flag");
  bad.args = {"--recipe", "off"};
  Check(!superroc::cli::ValidateRecipeForModel(bad, "/m/main.gguf", TestFlag,
                                               &err),
        "refuse --recipe inside args");
  bad.args = {"--speculative"};
  Check(!superroc::cli::ValidateRecipeForModel(bad, "/m/main.gguf", TestFlag,
                                               &err),
        "refuse flag without value");
  // Round 2: a recipe ships next to the model, so args must never carry a
  // host path. File-taking flags belong in file_args ...
  bad.args = {"--speculative", "mtp", "--mtp-model", "mtp.gguf"};
  Check(!superroc::cli::ValidateRecipeForModel(bad, "/m/main.gguf", TestFlag,
                                               &err),
        "refuse file flag in args");
  Check(err.find("belongs in file_args") != std::string::npos,
        "file-flag message (got: " + err + ")");
  bad.args = {"--mmproj", "mmproj.gguf"};
  Check(!superroc::cli::ValidateRecipeForModel(bad, "/m/main.gguf", TestFlag,
                                               &err),
        "refuse mmproj in args");
  // ... and no args value may be an absolute path, inline or separate.
  bad.args = {"--speculative", "mtp", "--draft-policy", "/etc/policy"};
  Check(!superroc::cli::ValidateRecipeForModel(bad, "/m/main.gguf", TestFlag,
                                               &err),
        "refuse absolute path value in args");
  Check(err.find("belongs in file_args") != std::string::npos,
        "abs-path message (got: " + err + ")");
  bad.args = {"--speculative=/abs/mtp"};
  Check(!superroc::cli::ValidateRecipeForModel(bad, "/m/main.gguf", TestFlag,
                                               &err),
        "refuse inline absolute path in args");
  // Relative values on non-file flags stay legal.
  bad.args = {"--speculative", "mtp", "--think", "off"};
  Check(superroc::cli::ValidateRecipeForModel(bad, "/m/main.gguf", TestFlag,
                                              &err),
        "relative values pass");
  fs::remove_all(dir);
}

void TestFileArgsRefusals() {
  // file_args keys must be llm model/file options; values must exist.
  MustRefuseFile(
      "fileargs-type",
      R"({"schema": "superroc-recipe/1", "model": "m.gguf", "file_args": ["--mtp-model"]})",
      "must be an object");
  MustRefuseFile(
      "fileargs-flag",
      R"({"schema": "superroc-recipe/1", "model": "m.gguf", "file_args": {"--speculative": "x.gguf"}})",
      "bad file_args flag '--speculative'");
  MustRefuseFile(
      "fileargs-missing",
      R"({"schema": "superroc-recipe/1", "model": "m.gguf", "file_args": {"--mtp-model": "nope.gguf"}})",
      "does not exist");
  MustRefuseFile(
      "fileargs-empty",
      R"({"schema": "superroc-recipe/1", "model": "m.gguf", "file_args": {"--mtp-model": ""}})",
      "nonempty path string");
}

void MustRefuseArgs(const std::string& tag,
                    const std::vector<std::string>& args,
                    const std::string& want) {
  RecipeSpec spec;
  spec.file = "test-" + tag;
  spec.model = "main.gguf";
  spec.args = args;
  std::string err;
  Check(!superroc::cli::ValidateRecipeForModel(spec, "/m/main.gguf", TestFlag,
                                               &err),
        "refuse " + tag);
  Check(err.find(want) != std::string::npos,
        "refuse " + tag + " mentions '" + want + "' (got: " + err + ")");
}

void TestLauncherOwnedRefusals() {
  // Launcher-owned flags stay with the command line, in either spelling.
  MustRefuseArgs("port-long", {"--port", "1"}, "--port");
  MustRefuseArgs("port-short", {"-p", "1"}, "--port");
  MustRefuseArgs("host", {"--host", "x"}, "--host");
  MustRefuseArgs("api-key", {"--api-key", "k"}, "--api-key");
  MustRefuseArgs("api-key-file", {"--api-key-file", "k"}, "--api-key-file");
  MustRefuseArgs("model", {"--model", "m"}, "--model");
  MustRefuseArgs("model-short", {"-m", "m"}, "--model");
  MustRefuseArgs("served-name", {"--served-model-name", "x"},
                 "--served-model-name");
  MustRefuseArgs("recipe", {"--recipe", "off"}, "--recipe");
  MustRefuseArgs("trace", {"--trace", "t"}, "--trace");
  MustRefuseArgs("cache-disk", {"--cache-disk", "d"}, "--cache-disk");
  MustRefuseArgs("sessions", {"--sessions", "1"}, "--sessions");
  MustRefuseArgs("max-connections", {"--max-connections", "1"},
                 "--max-connections");
  // Only the listed tuning options pass: limits, timeouts, cache budgets and
  // logging stay with the command line, in either spelling.
  const std::string owned = "only the command line may set";
  MustRefuseArgs("max-pending", {"--max-pending", "1"}, owned);
  MustRefuseArgs("request-timeout", {"--request-timeout-ms", "1"}, owned);
  MustRefuseArgs("cache-ram", {"--cache-ram-bytes", "1"}, owned);
  MustRefuseArgs("max-tokens", {"--max-tokens", "1"}, owned);
  MustRefuseArgs("max-tokens-short", {"-n", "1"}, "--max-tokens");
  MustRefuseArgs("max-tokens-inline", {"--max-tokens=1"}, owned);
  MustRefuseArgs("log-level", {"--log-level", "debug"}, owned);
  MustRefuseArgs("verbose", {"--verbose"}, owned);
  MustRefuseArgs("after-allowed", {"--prompt-lookup", "--max-pending", "1"},
                 "--max-pending");
  // --context has its own `context` field; the error says so.
  MustRefuseArgs("context", {"--context", "1"}, "--context");
  {
    RecipeSpec spec;
    spec.file = "test-context-msg";
    spec.model = "main.gguf";
    spec.args = {"--context", "1"};
    std::string err;
    Check(!superroc::cli::ValidateRecipeForModel(spec, "/m/main.gguf", TestFlag,
                                                 &err),
          "refuse --context message");
    Check(err.find("use the recipe's `context` field") != std::string::npos,
          "context error names the field (got: " + err + ")");
  }
  // file_args with a launcher-owned flag refuses, naming the flag.
  MustRefuseFile(
      "fileargs-trace",
      R"({"schema": "superroc-recipe/1", "model": "m.gguf", "file_args": {"--trace": "t"}})",
      "--trace");
  // Launcher-owned env is refused at parse time.
  MustRefuseFile(
      "env-host",
      R"({"schema": "superroc-recipe/1", "model": "m.gguf", "env": {"SUPERROC_HOST": "x"}})",
      "SUPERROC_HOST");
  // Host paths refuse: absolute paths, `..` components, directories.
  MustRefuseFile(
      "files-dotdot",
      R"({"schema": "superroc-recipe/1", "model": "m.gguf", "files": {"SUPERROC_TENSOR_OVERLAY": "../x"}})",
      "..");
  MustRefuseFile(
      "files-abs",
      R"({"schema": "superroc-recipe/1", "model": "m.gguf", "files": {"SUPERROC_TENSOR_OVERLAY": "/abs/x"}})",
      "absolute");
  MustRefuseFile(
      "fileargs-dotdot",
      R"({"schema": "superroc-recipe/1", "model": "m.gguf", "file_args": {"--mtp-model": "../x.gguf"}})",
      "..");
  {
    const fs::path dir = MakeDir("dirref");
    fs::create_directories(dir / "sub");
    Write(
        dir / "r.json",
        R"({"schema": "superroc-recipe/1", "model": "m.gguf", "files": {"SUPERROC_SOME_DIR": "sub"}})");
    RecipeSpec spec;
    std::string err;
    const bool ok =
        superroc::cli::ParseRecipeFile((dir / "r.json").string(), &spec, &err);
    Check(!ok, "refuse directory: parses");
    Check(err.find("regular file") != std::string::npos,
          "refuse directory mentions regular file (got: " + err + ")");
    fs::remove_all(dir);
  }
  // A flag with an inline value refuses.
  MustRefuseArgs("flag-value", {"--prompt-lookup=1"}, "takes no value");
}

void TestSymlinkAndNegative() {
  // A symlink to a regular file is accepted; negative numbers are values.
  const fs::path dir = MakeDir("link");
  Write(dir / "real.gguf", "x");
  std::error_code ec;
  fs::create_symlink(dir / "real.gguf", dir / "link.gguf", ec);
  Check(!ec, "symlink created");
  Write(
      dir / "r.json",
      R"({"schema": "superroc-recipe/1", "model": "m.gguf", "files": {"SUPERROC_TENSOR_OVERLAY": "link.gguf"}})");
  RecipeSpec spec;
  if (MustParse(dir / "r.json", &spec)) {
    Check(spec.files.at("SUPERROC_TENSOR_OVERLAY") ==
              (dir / "link.gguf").string(),
          "symlink to a regular file resolves");
  }
  fs::remove_all(dir);
  {
    // A link may not lead out of the recipe's folder ...
    const fs::path outer = MakeDir("linkout");
    const fs::path inner = outer / "model";
    fs::create_directories(inner);
    Write(outer / "secret.gguf", "x");
    std::error_code link_ec;
    fs::create_symlink(outer / "secret.gguf", inner / "leak.gguf", link_ec);
    Check(!link_ec, "outside symlink created");
    for (
        const char* body :
        {R"({"schema": "superroc-recipe/1", "model": "m.gguf", "files": {"SUPERROC_TENSOR_OVERLAY": "leak.gguf"}})",
         R"({"schema": "superroc-recipe/1", "model": "m.gguf", "file_args": {"--mtp-model": "leak.gguf"}})"}) {
      Write(inner / "r.json", body);
      RecipeSpec leak;
      std::string err;
      Check(!superroc::cli::ParseRecipeFile((inner / "r.json").string(), &leak,
                                            &err) &&
                err.find("resolves outside the recipe's folder") !=
                    std::string::npos,
            "a sidecar link out of the folder is refused (got: " + err + ")");
    }
    // ... except from a Hugging Face cache snapshot into its own repository's
    // blobs, which is how that cache stores every file.
    const fs::path repo = outer / "models--org--name";
    const fs::path snapshot = repo / "snapshots" / "abc123";
    fs::create_directories(snapshot);
    fs::create_directories(repo / "blobs");
    Write(repo / "blobs" / "0f0f", "x");
    fs::create_symlink(fs::path("../../blobs/0f0f"), snapshot / "overlay.gguf",
                       link_ec);
    Check(!link_ec, "cache symlink created");
    Write(
        snapshot / "r.json",
        R"({"schema": "superroc-recipe/1", "model": "m.gguf", "files": {"SUPERROC_TENSOR_OVERLAY": "overlay.gguf"}})");
    RecipeSpec cached;
    Check(MustParse(snapshot / "r.json", &cached) &&
              cached.files.count("SUPERROC_TENSOR_OVERLAY") == 1,
          "a cache snapshot link into its repository's blobs is accepted");
    fs::create_symlink(outer / "secret.gguf", snapshot / "leak.gguf", link_ec);
    Write(
        snapshot / "r2.json",
        R"({"schema": "superroc-recipe/1", "model": "m.gguf", "files": {"SUPERROC_TENSOR_OVERLAY": "leak.gguf"}})");
    RecipeSpec escaped;
    std::string err;
    Check(!superroc::cli::ParseRecipeFile((snapshot / "r2.json").string(),
                                          &escaped, &err) &&
              err.find("resolves outside the recipe's folder") !=
                  std::string::npos,
          "a cache snapshot link out of its repository is refused");
    fs::remove_all(outer);
  }
  for (const std::vector<std::string> args :
       {std::vector<std::string>{"--seed", "-1"},
        std::vector<std::string>{"--reasoning-budget", "-1"}}) {
    RecipeSpec neg;
    neg.file = "test-neg";
    neg.model = "main.gguf";
    neg.args = args;
    std::string err;
    Check(superroc::cli::ValidateRecipeForModel(neg, "/m/main.gguf", TestFlag,
                                                &err),
          "negative value validates for " + args[0] +
              (err.empty() ? "" : ": " + err));
  }
}

void TestCanonicalMerge() {
  // Canonical matching: recipe `-t 0` loses to command-line `--temperature`.
  RecipeSpec spec;
  spec.file = "test";
  spec.args = {"-t", "0"};
  const RecipeMerge got = superroc::cli::MergeRecipeArgs(
      spec, {"llm", "-m", "m.gguf", "--temperature", "1"}, TestFlag);
  Check(got.error.empty(), "canonical merge clean");
  Check(got.extra.empty(), "canonical merge appends nothing");
  Check(got.log.size() == 1 &&
            got.log[0] == "[recipe] kept command-line --temperature",
        "canonical merge keeps the command line");
  // A flag the recipe repeats is still one flag: the command line's short
  // spelling keeps every copy out, and without it the copies stay in order.
  RecipeSpec twice;
  twice.file = "test";
  twice.model = "main.gguf";
  twice.args = {"--draft-tokens", "5", "--draft-tokens", "9"};
  std::string err;
  Check(superroc::cli::ValidateRecipeForModel(twice, "/m/main.gguf", TestFlag,
                                              &err),
        "repeated flag validates" + (err.empty() ? "" : ": " + err));
  const RecipeMerge kept = superroc::cli::MergeRecipeArgs(
      twice, {"llm", "-m", "m.gguf", "-d", "3"}, TestFlag);
  Check(kept.error.empty() && kept.extra.empty(),
        "repeated flag loses to the command line");
  const RecipeMerge both =
      superroc::cli::MergeRecipeArgs(twice, {"llm", "-m", "m.gguf"}, TestFlag);
  Check(both.extra == std::vector<std::string>{"--draft-tokens", "5",
                                               "--draft-tokens", "9"},
        "repeated flag keeps its order");
}

void TestAppliedSubset() {
  // The /props record holds only what took effect: keys actually set, tokens
  // actually appended (not kept).
  ::setenv("SUPERROC_RECIPE_TEST_KEPT", "preset", 1);
  ::unsetenv("SUPERROC_RECIPE_TEST_NEW");
  RecipeSpec spec;
  spec.file = "test";
  spec.env = {{"SUPERROC_RECIPE_TEST_KEPT", "recipe"},
              {"SUPERROC_RECIPE_TEST_NEW", "recipe"}};
  std::map<std::string, std::string> applied_env, applied_files;
  superroc::cli::ApplyRecipeEnv(spec, &applied_env, &applied_files);
  Check(applied_env.size() == 1 &&
            applied_env.count("SUPERROC_RECIPE_TEST_NEW") == 1,
        "applied env holds only the set key");
  Check(applied_files.empty(), "no applied files");
  ::unsetenv("SUPERROC_RECIPE_TEST_KEPT");
  ::unsetenv("SUPERROC_RECIPE_TEST_NEW");
  RecipeSpec merge_spec;
  merge_spec.file = "test";
  merge_spec.args = {"--speculative", "mtp", "--prompt-lookup"};
  const RecipeMerge got = superroc::cli::MergeRecipeArgs(
      merge_spec, {"llm", "-m", "m.gguf", "--speculative", "off"}, TestFlag);
  Check(got.error.empty(), "applied merge clean");
  Check((got.extra == std::vector<std::string>{"--prompt-lookup"}),
        "applied args hold only appended tokens");
}

void TestFileArgsMerge() {
  // file_args append `<flag> <abs path>` unless the command line has the
  // flag; the command line wins, logged like the other args.
  const fs::path dir = MakeDir("famerge");
  Write(dir / "mtp.gguf", "x");
  Write(dir / "r.json", R"({"schema": "superroc-recipe/1", "model": "m.gguf")"
                        R"(, "file_args": {"--mtp-model": "mtp.gguf"}})");
  RecipeSpec spec;
  if (!MustParse(dir / "r.json", &spec)) {
    fs::remove_all(dir);
    return;
  }
  const std::string want = (dir / "mtp.gguf").string();
  const RecipeMerge appended =
      superroc::cli::MergeRecipeArgs(spec, {"llm", "-m", "m.gguf"}, TestFlag);
  Check(appended.error.empty(), "file_args merge clean");
  Check((appended.extra == std::vector<std::string>{"--mtp-model", want}),
        "file_args appends flag plus abs path");
  Check(appended.log.size() == 1 &&
            appended.log[0] == "[recipe] arg --mtp-model " + want,
        "file_args applied line");
  const RecipeMerge kept = superroc::cli::MergeRecipeArgs(
      spec, {"llm", "-m", "m.gguf", "--mtp-model", "/other/mtp.gguf"},
      TestFlag);
  Check(kept.error.empty(), "file_args kept clean");
  Check(kept.extra.empty(), "command-line file flag wins");
  Check(kept.log.size() == 1 &&
            kept.log[0] == "[recipe] kept command-line --mtp-model",
        "file_args kept line");
  fs::remove_all(dir);
}

void TestEnvPrecedence() {
  // A variable set beforehand wins: kept, not overwritten.
  ::setenv("SUPERROC_RECIPE_TEST_A", "preset", 1);
  ::unsetenv("SUPERROC_RECIPE_TEST_B");
  RecipeSpec spec;
  spec.file = "test";
  spec.env = {{"SUPERROC_RECIPE_TEST_A", "recipe"},
              {"SUPERROC_RECIPE_TEST_B", "recipe"}};
  const std::vector<std::string> log = superroc::cli::ApplyRecipeEnv(spec);
  Check(::superroc::core::GetEnv("SUPERROC_RECIPE_TEST_A") != nullptr &&
            std::string(::superroc::core::GetEnv("SUPERROC_RECIPE_TEST_A")) ==
                "preset",
        "preset env wins");
  Check(::superroc::core::GetEnv("SUPERROC_RECIPE_TEST_B") != nullptr &&
            std::string(::superroc::core::GetEnv("SUPERROC_RECIPE_TEST_B")) ==
                "recipe",
        "unset env applied");
  Check(log.size() == 2, "two env log lines");
  if (log.size() == 2) {
    Check(log[0] == "[recipe] kept env SUPERROC_RECIPE_TEST_A",
          "kept line (got: " + log[0] + ")");
    Check(log[1] == "[recipe] env SUPERROC_RECIPE_TEST_B=recipe",
          "applied line (got: " + log[1] + ")");
  }
  ::unsetenv("SUPERROC_RECIPE_TEST_A");
  ::unsetenv("SUPERROC_RECIPE_TEST_B");
}

void TestCmdlinePrecedence() {
  RecipeSpec spec;
  spec.file = "test";
  spec.args = {"--speculative", "mtp", "--prompt-lookup"};
  spec.has_context = true;
  spec.context = 131072;
  // Command line already sets --speculative and --context: both kept.
  const RecipeMerge got = superroc::cli::MergeRecipeArgs(
      spec, {"llm", "-m", "m.gguf", "--speculative", "off", "--context=8"},
      TestFlag);
  Check(got.error.empty(), "merge keeps clean");
  Check((got.extra == std::vector<std::string>{"--prompt-lookup"}),
        "merge appends only the absent switch");
  Check(got.log.size() == 3, "three merge lines");
  if (got.log.size() == 3) {
    Check(got.log[0] == "[recipe] kept command-line --speculative",
          "kept flag (got: " + got.log[0] + ")");
    Check(got.log[1] == "[recipe] arg --prompt-lookup",
          "applied switch (got: " + got.log[1] + ")");
    Check(got.log[2] == "[recipe] kept command-line --context",
          "kept context (got: " + got.log[2] + ")");
  }
  // Nothing on the command line: everything appended.
  const RecipeMerge all =
      superroc::cli::MergeRecipeArgs(spec, {"llm", "-m", "m.gguf"}, TestFlag);
  Check(all.error.empty(), "merge appends clean");
  Check((all.extra == std::vector<std::string>{"--speculative", "mtp",
                                               "--prompt-lookup", "--context",
                                               "131072"}),
        "merge appends flag, switch, context");
  if (all.log.size() == 3) {
    Check(all.log[0] == "[recipe] arg --speculative mtp",
          "arg line (got: " + all.log[0] + ")");
    Check(all.log[1] == "[recipe] arg --prompt-lookup",
          "switch line (got: " + all.log[1] + ")");
    Check(all.log[2] == "[recipe] arg --context 131072",
          "context line (got: " + all.log[2] + ")");
  } else {
    Check(false, "three applied lines");
  }
}

void TestRelativeFiles() {
  // Relative file values resolve against the recipe's directory.
  const fs::path dir = MakeDir("rel");
  const fs::path sub = dir / "sub";
  fs::create_directories(sub);
  Write(sub / "overlay.gguf", "x");
  Write(dir / "r.json",
        R"({"schema": "superroc-recipe/1", "model": "m.gguf")"
        R"(, "files": {"SUPERROC_TENSOR_OVERLAY": "sub/overlay.gguf"}})");
  RecipeSpec spec;
  if (!MustParse(dir / "r.json", &spec)) {
    fs::remove_all(dir);
    return;
  }
  const std::string want = (sub / "overlay.gguf").string();
  Check(spec.files.at("SUPERROC_TENSOR_OVERLAY") == want, "relative resolves");
  ::unsetenv("SUPERROC_TENSOR_OVERLAY");
  const std::vector<std::string> log = superroc::cli::ApplyRecipeEnv(spec);
  Check(log.size() == 1, "one file log line");
  if (!log.empty()) {
    Check(log[0] == "[recipe] file SUPERROC_TENSOR_OVERLAY=" + want,
          "file line (got: " + log[0] + ")");
  }
  Check(::superroc::core::GetEnv("SUPERROC_TENSOR_OVERLAY") != nullptr &&
            std::string(::superroc::core::GetEnv("SUPERROC_TENSOR_OVERLAY")) ==
                want,
        "file exported absolute");
  ::unsetenv("SUPERROC_TENSOR_OVERLAY");
  fs::remove_all(dir);
}

void TestRawScan() {
  using superroc::cli::ScanRawOption;
  const std::vector<std::string> args = {
      "serve",         "-i",        "127.0.0.1", "llm", "-m", "m.gguf",
      "--recipe=auto", "--context", "8"};
  Check(ScanRawOption(args, {"--recipe"}) == "auto", "scan --flag=value");
  Check(ScanRawOption(args, {"-m", "--model"}) == "m.gguf", "scan short flag");
  Check(!ScanRawOption(args, {"--missing"}).has_value(), "scan absent");
  // The parser keeps the last occurrence of an option; so does the scan, or a
  // later `--recipe off` would not switch an earlier `--recipe auto` off.
  const std::vector<std::string> twice = {
      "-m", "first.gguf", "--recipe",    "auto",
      "-m", "last.gguf",  "--recipe=off"};
  Check(ScanRawOption(twice, {"--recipe"}) == "off", "scan keeps the last one");
  Check(ScanRawOption(twice, {"-m", "--model"}) == "last.gguf",
        "scan keeps the last model");
  Check(superroc::cli::ScanRawPresent(args, {"-c", "--context"}),
        "scan present with value form");
  Check(!superroc::cli::ScanRawPresent(args, {"--prompt-lookup"}),
        "scan absent flag");
}

// A recipe is downloaded with a model: it may tune the engine, never name a
// program to run or a file to write, and never smuggle a path through `env`.
void TestEnvSafety() {
  const auto recipe = [](const std::string& body) {
    return R"({"schema": "superroc-recipe/1", "model": "m.gguf", )" + body +
           "}";
  };
  for (const char* key : {"SUPERROC_MTP_TF_DUMP", "SUPERROC_MTP_TF_LOGITS",
                          "SUPERROC_MTP_KV_DUMP", "LING3_KDA_CAPTURE",
                          "SUPERROC_HIPBLASLT_PLAN_CACHE", "SUPERROC_FFMPEG",
                          "SUPERROC_FFPROBE", "SUPERROC_EVAL_API_KEY"}) {
    MustRefuseFile(std::string("unsafe_") + key,
                   recipe(std::string(R"("env": {")") + key + R"(": "1"})"),
                   "names a program or an output file");
  }
  // Diagnostic switches change what the server prints, not how a model runs.
  for (const char* key :
       {"LFN_MTP_TRACE", "LFN_PROMPT_DUMP", "SUPERROC_VERIFY_DEBUG",
        "SUPERROC_KVC_TIMING", "LING3_BATCH_PROFILE",
        "SUPERROC_DSPARK_STEP_PROF", "LING3_MTP_VERBOSE", "LFN_MOE_R_CHECK",
        "LING3_KDA_CAPTURE_N"}) {
    MustRefuseFile(std::string("diag_") + key,
                   recipe(std::string(R"("env": {")") + key + R"(": "1"})"),
                   "a diagnostic switch");
  }
  MustRefuseFile("diagfile",
                 recipe(R"("files": {"LFN_MTP_TRACE": "sidecar.gguf"})"),
                 "a diagnostic switch");
  MustRefuseFile("envempty-value",
                 recipe(R"("env": {"SUPERROC_ROUTED_EXPERTS": ""})"),
                 "must be a plain setting");
  MustRefuseFile(
      "toolarge",
      recipe(R"("notes": ")" + std::string((1U << 20) + 1, 'x') + R"(")"),
      "larger than 1 MiB");
  {
    // A name holding a diagnostic word only as part of another word is a
    // setting: CHECKPOINT is not CHECK, CAPTURE alone is not a capture dump.
    const fs::path dir = MakeDir("notdiag");
    Write(
        dir / "r.json",
        recipe(
            R"("env": {"SUPERROC_INPASS_CHECKPOINT": "1", "SUPERROC_TAIL_CAPTURE": "64"})"));
    RecipeSpec spec;
    if (MustParse(dir / "r.json", &spec)) {
      Check(spec.env.count("SUPERROC_INPASS_CHECKPOINT") == 1 &&
                spec.env.count("SUPERROC_TAIL_CAPTURE") == 1,
            "settings that only resemble a diagnostic name are kept");
    }
    // The recipe path itself must be a regular file.
    RecipeSpec none;
    std::string err;
    Check(!superroc::cli::ParseRecipeFile(dir.string(), &none, &err) &&
              err.find("cannot read file") != std::string::npos,
          "a directory is not a recipe (got: " + err + ")");
    fs::remove_all(dir);
  }
  MustRefuseFile("envpath",
                 recipe(R"("env": {"SUPERROC_ROUTED_EXPERTS": "/tmp/x"})"),
                 "must be a plain setting");
  MustRefuseFile("envdot",
                 recipe(R"("env": {"SUPERROC_ROUTED_EXPERTS": ".bashrc"})"),
                 "must be a plain setting");
  MustRefuseFile("envspace",
                 recipe(R"("env": {"SUPERROC_ROUTED_EXPERTS": "7 8"})"),
                 "must be a plain setting");
  MustRefuseFile("envlong",
                 recipe(R"("env": {"SUPERROC_ROUTED_EXPERTS": ")" +
                        std::string(257, '7') + R"("})"),
                 "must be a plain setting");
  MustRefuseFile(
      "envfile",
      recipe(R"("env": {"SUPERROC_TENSOR_OVERLAY": "overlay.gguf"})"),
      "belongs in files");

  const fs::path dir = MakeDir("safety");
  Write(dir / "sidecar.gguf", "x");
  Write(dir / "unsafe.json",
        recipe(R"("files": {"SUPERROC_MTP_TF_DUMP": "sidecar.gguf"})"));
  Write(dir / "notinput.json",
        recipe(R"("files": {"SUPERROC_ROUTED_EXPERTS": "sidecar.gguf"})"));
  RecipeSpec spec;
  std::string err;
  Check(!superroc::cli::ParseRecipeFile((dir / "unsafe.json").string(), &spec,
                                        &err) &&
            err.find("names a program or an output file") != std::string::npos,
        "files refuses an output-file variable: " + err);
  err.clear();
  Check(!superroc::cli::ParseRecipeFile((dir / "notinput.json").string(), &spec,
                                        &err) &&
            err.find("not an input-file variable") != std::string::npos,
        "files refuses a variable that is not an input file: " + err);

  // Plain settings pass; a variable this build does not read is reported and
  // skipped, in env and in files, and is never exported.
  Write(
      dir / "ok.json",
      recipe(
          R"("env": {"SUPERROC_ROUTED_EXPERTS": "7", "LFN_MTP_P_MIN": "0.5", )"
          R"("SUPERROC_INT4_TENSORS": "ffn_gate,ffn_up", )"
          R"("SUPERROC_NOT_A_SETTING": "1"}, )"
          R"("files": {"SUPERROC_TENSOR_OVERLAY": "sidecar.gguf", )"
          R"("SUPERROC_FUTURE_INPUT": "sidecar.gguf"})"));
  ::unsetenv("SUPERROC_ROUTED_EXPERTS");
  ::unsetenv("LFN_MTP_P_MIN");
  ::unsetenv("SUPERROC_INT4_TENSORS");
  ::unsetenv("SUPERROC_TENSOR_OVERLAY");
  ::unsetenv("SUPERROC_NOT_A_SETTING");
  ::unsetenv("SUPERROC_FUTURE_INPUT");
  if (MustParse(dir / "ok.json", &spec)) {
    Check(spec.env.size() == 3 && spec.env.count("SUPERROC_NOT_A_SETTING") == 0,
          "unknown env variable is not kept as env");
    Check(spec.files.size() == 1 &&
              spec.files.count("SUPERROC_FUTURE_INPUT") == 0,
          "unknown files variable is not kept as a file");
    Check(spec.ignored.size() == 2, "both unknown variables are reported");
    const std::vector<std::string> log = superroc::cli::ApplyRecipeEnv(spec);
    bool reported = false;
    for (const std::string& line : log) {
      reported =
          reported || line ==
                          "[recipe] ignored SUPERROC_NOT_A_SETTING (this build "
                          "does not read it)";
    }
    Check(reported, "the startup log names the ignored variable");
    Check(::superroc::core::GetEnv("SUPERROC_NOT_A_SETTING") == nullptr &&
              ::superroc::core::GetEnv("SUPERROC_FUTURE_INPUT") == nullptr,
          "ignored variables are never exported");
    Check(::superroc::core::GetEnv("SUPERROC_ROUTED_EXPERTS") != nullptr,
          "known settings are exported");
  }
  ::unsetenv("SUPERROC_ROUTED_EXPERTS");
  ::unsetenv("LFN_MTP_P_MIN");
  ::unsetenv("SUPERROC_INT4_TENSORS");
  ::unsetenv("SUPERROC_TENSOR_OVERLAY");
  fs::remove_all(dir);
}

void TestPrescan() {
  // Off / missing: no-op. Mismatch: exit 2. Good file: env applied silently.
  ::unsetenv("SUPERROC_ROUTED_EXPERTS");
  const char* off_argv[] = {"superroc", "serve", "llm", "-m", "m.gguf"};
  Check(!superroc::cli::PrescanRecipeEnv({off_argv, 5}).has_value(),
        "prescan off is a no-op");
  const fs::path dir = MakeDir("pre");
  Write(
      dir / "r.json",
      R"({"schema": "superroc-recipe/1", "model": "m.gguf", "env": {"SUPERROC_ROUTED_EXPERTS": "1"}})");
  const std::string path = (dir / "r.json").string();
  const char* bad_argv[] = {"superroc",   "serve",    "llm",       "-m",
                            "other.gguf", "--recipe", path.c_str()};
  const std::optional<int> refused =
      superroc::cli::PrescanRecipeEnv({bad_argv, 7});
  Check(refused.has_value() && *refused == 2, "prescan refuses mismatch");
  Check(::superroc::core::GetEnv("SUPERROC_ROUTED_EXPERTS") == nullptr,
        "refused prescan applies nothing");
  const char* ok_argv[] = {"superroc", "serve",    "llm",       "-m",
                           "m.gguf",   "--recipe", path.c_str()};
  Check(!superroc::cli::PrescanRecipeEnv({ok_argv, 7}).has_value(),
        "prescan applies cleanly");
  Check(::superroc::core::GetEnv("SUPERROC_ROUTED_EXPERTS") != nullptr &&
            std::string(::superroc::core::GetEnv("SUPERROC_ROUTED_EXPERTS")) ==
                "1",
        "prescan env visible");
  const std::vector<std::string> stashed = superroc::cli::TakePrescanLog(path);
  Check(stashed.size() == 1 &&
            stashed[0] == "[recipe] env SUPERROC_ROUTED_EXPERTS=1",
        "prescan stashes the applied line");
  Check(superroc::cli::TakePrescanLog(path).empty(), "prescan log takes once");
  Check(superroc::cli::TakePrescanLog("other.json").empty(),
        "prescan log mismatched file");
  ::unsetenv("SUPERROC_ROUTED_EXPERTS");
  // Non-serve commands never touch the environment.
  Write(
      dir / "r2.json",
      R"({"schema": "superroc-recipe/1", "model": "m.gguf", "env": {"SUPERROC_ROUTED_EXPERTS": "2"}})");
  const std::string path2 = (dir / "r2.json").string();
  const char* prompt_argv[] = {"superroc", "prompt", "--recipe", path2.c_str(),
                               "hi"};
  Check(!superroc::cli::PrescanRecipeEnv({prompt_argv, 5}).has_value(),
        "prescan ignores non-serve");
  Check(::superroc::core::GetEnv("SUPERROC_ROUTED_EXPERTS") == nullptr,
        "non-serve prescan applies nothing");
  // `serve` must be the command itself, not a word later on the line.
  const char* eval_argv[] = {"superroc", "eval",   "serve",    "llm",
                             "-m",       "m.gguf", "--recipe", path2.c_str()};
  Check(!superroc::cli::PrescanRecipeEnv({eval_argv, 8}).has_value() &&
            ::superroc::core::GetEnv("SUPERROC_ROUTED_EXPERTS") == nullptr,
        "prescan ignores a command that only mentions serve");
  // A later `--recipe off` wins over an earlier recipe, as in the parser.
  const char* off_last_argv[] = {"superroc",    "serve",    "llm",
                                 "-m",          "m.gguf",   "--recipe",
                                 path2.c_str(), "--recipe", "off"};
  Check(!superroc::cli::PrescanRecipeEnv({off_last_argv, 9}).has_value() &&
            ::superroc::core::GetEnv("SUPERROC_ROUTED_EXPERTS") == nullptr,
        "prescan honours a later --recipe off");
  // Only the llm subcommand takes --recipe.
  const char* tts_argv[] = {"superroc", "serve",    "tts",        "-m",
                            "m.gguf",   "--recipe", path2.c_str()};
  Check(!superroc::cli::PrescanRecipeEnv({tts_argv, 7}).has_value() &&
            ::superroc::core::GetEnv("SUPERROC_ROUTED_EXPERTS") == nullptr,
        "prescan ignores serve subcommands other than llm");
  fs::remove_all(dir);
}

}  // namespace

int main() {
  TestGoodFile();
  TestRefusals();
  TestFileArgsRefusals();
  TestLauncherOwnedRefusals();
  TestSymlinkAndNegative();
  TestCanonicalMerge();
  TestAppliedSubset();
  TestFileArgsMerge();
  TestEnvPrecedence();
  TestCmdlinePrecedence();
  TestRelativeFiles();
  TestRawScan();
  TestEnvSafety();
  TestPrescan();
  if (failures == 0) {
    std::cout << "recipe_test PASS\n";
    return 0;
  }
  std::cout << "recipe_test FAIL (" << failures << " failures)\n";
  return 1;
}
