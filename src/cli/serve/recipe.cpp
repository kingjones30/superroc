#include "src/cli/serve/recipe.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <sstream>

#include "src/core/env.hpp"
#include "src/core/json.hpp"

namespace superroc::cli {
namespace {

constexpr std::string_view kSchema = "superroc-recipe/1";
constexpr const char* kAutoFileName = "superroc-recipe.json";
constexpr std::uintmax_t kMaxRecipeBytes = 1U << 20;

// Every `serve llm` option that takes a model/file path, from the llm parser
// (llm_parser in serve.cpp): --mmproj, --dflash-model, --dspark-model,
// --mtp-model. The PrintServeHelp parser repeats the same options.
// -m/--model is deliberately excluded: the recipe carries the model identity
// in its `model` field and the launcher always supplies -m itself.
// --cache-disk and --trace are NOT here: they take output paths owned by the
// launcher and are refused outright (see IsLauncherOwnedFlag).
constexpr std::string_view kFileArgFlags[] = {
    "--mmproj",
    "--dflash-model",
    "--dspark-model",
    "--mtp-model",
};

// Canonical names the recipe may never set, each refused with a message that
// names it: server identity, secrets and write paths from AddServerOptions
// (serve.cpp: -i/--host, -p/--port, -j/--sessions, --max-connections,
// --max-request-bytes, --api-key, --api-key-file) plus the log-option
// adder's only write path (--trace) and the llm options owned by the
// launcher (--model, --served-model-name, --recipe, --cache-disk, --context,
// which has its own `context` field). IsRecipeArgFlag below is the complete
// rule; this list only gives these names a clearer error.
bool IsLauncherOwnedFlag(std::string_view canonical) {
  constexpr std::string_view kOwned[] = {
      "--host",
      "--port",
      "--sessions",
      "--max-connections",
      "--max-request-bytes",
      "--api-key",
      "--api-key-file",
      "--model",
      "--served-model-name",
      "--recipe",
      "--trace",
      "--cache-disk",
      "--context",
  };
  for (const std::string_view owned : kOwned) {
    if (canonical == owned) {
      return true;
    }
  }
  return false;
}

// The llm options a recipe's `args` may set: speculative decoding, prompt
// lookup, prefill chunking, thinking and sampling defaults. Every other
// option (queue and output limits, timeouts, cache budgets, logging) stays
// with the command line, and so does any option added to the parser later
// until it is listed here.
bool IsRecipeArgFlag(std::string_view canonical) {
  constexpr std::string_view kAllowed[] = {
      "--chat-template-kwargs",
      "--draft-policy",
      "--draft-tokens",
      "--frequency-penalty",
      "--min-draft-tokens",
      "--min-keep",
      "--min-p",
      "--prefill-chunk",
      "--presence-penalty",
      "--preserve-thinking",
      "--prompt-lookup",
      "--reasoning-budget",
      "--reasoning-effort",
      "--repeat-last-n",
      "--repeat-penalty",
      "--seed",
      "--speculative",
      "--temperature",
      "--think",
      "--top-k",
      "--top-p",
  };
  for (const std::string_view allowed : kAllowed) {
    if (canonical == allowed) {
      return true;
    }
  }
  return false;
}

// SUPERROC_* keys the serve path reads as server/listen defaults before the
// llm options are parsed (serve.cpp: SUPERROC_HOST, SUPERROC_PORT). No other
// engine variable read in src/cli/serve sets a listen or auth default.
bool IsLauncherEnvKey(std::string_view key) {
  return key == "SUPERROC_HOST" || key == "SUPERROC_PORT";
}

// Variables whose value names a program to run or a file the engine writes
// (diagnostic dumps and captures), and the settings of the eval command. A
// recipe arrives with a downloaded model, so it may never set these.
bool IsRefusedEnvKey(std::string_view key) {
  constexpr std::string_view kRefused[] = {
      "LING3_KDA_CAPTURE",      "SUPERROC_FFMPEG",
      "SUPERROC_FFPROBE",       "SUPERROC_HIPBLASLT_PLAN_CACHE",
      "SUPERROC_MTP_KV_DUMP",   "SUPERROC_MTP_TF_DUMP",
      "SUPERROC_MTP_TF_LOGITS",
  };
  for (const std::string_view refused : kRefused) {
    if (key == refused) {
      return true;
    }
  }
  constexpr std::string_view kEvalPrefix = "SUPERROC_EVAL_";
  return key.substr(0, kEvalPrefix.size()) == kEvalPrefix;
}

// Diagnostic switches: traces, dumps, profiles, timing and self-check output.
// They change what the server prints, not how a model runs, so a recipe may
// not turn them on. Matched by whole words of the name (CHECKPOINT is not
// CHECK), plus the Ling-3 attention capture settings.
bool IsDiagnosticEnvKey(std::string_view key) {
  constexpr std::string_view kWords[] = {
      "CHECK", "DEBUG", "DUMP", "PROF", "PROFILE", "TIMING", "TRACE", "VERBOSE",
  };
  constexpr std::string_view kCapturePrefix = "LING3_KDA_CAPTURE";
  if (key.substr(0, kCapturePrefix.size()) == kCapturePrefix) {
    return true;
  }
  std::size_t start = 0;
  while (start <= key.size()) {
    std::size_t end = key.find('_', start);
    if (end == std::string_view::npos) {
      end = key.size();
    }
    const std::string_view word = key.substr(start, end - start);
    for (const std::string_view diagnostic : kWords) {
      if (word == diagnostic) {
        return true;
      }
    }
    start = end + 1;
  }
  return false;
}

// The variables whose value is a file the engine only reads; `files` may set
// these and nothing else.
bool IsInputFileEnvKey(std::string_view key) {
  return key == "SUPERROC_INT4_WEIGHTS" || key == "SUPERROC_TENSOR_OVERLAY";
}

// Every engine variable this build reads, generated from the source tree by
// tools/recipe/gen_env_keys.py (the recipe_env_keys_check test keeps the
// list current). Sorted, so the lookup is a binary search.
constexpr std::string_view kKnownEnvKeys[] = {
#include "src/cli/serve/recipe_env_keys.inc"
};

bool IsKnownEnvKey(std::string_view key) {
  return std::binary_search(std::begin(kKnownEnvKeys), std::end(kKnownEnvKeys),
                            key);
}

// An `env` value is a setting, never a path: 1 to 256 letters, digits and
// _ . , : + - only, not starting with a dot. Whatever variable a recipe
// names, its value then cannot point outside the working directory, and an
// empty value cannot switch on a variable that is only tested for presence.
bool IsPlainSetting(std::string_view value) {
  if (value.empty() || value.size() > 256 || value[0] == '.') {
    return false;
  }
  for (const char c : value) {
    const bool ok = (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') ||
                    (c >= '0' && c <= '9') || c == '_' || c == '.' ||
                    c == ',' || c == ':' || c == '+' || c == '-';
    if (!ok) {
      return false;
    }
  }
  return true;
}

bool IsFileArgFlag(std::string_view name) {
  for (const std::string_view flag : kFileArgFlags) {
    if (name == flag) {
      return true;
    }
  }
  return false;
}

// The directory a sidecar must resolve inside: the recipe's own, or the
// repository directory when the recipe sits in a Hugging Face cache snapshot
// (<repo>/snapshots/<revision>/, whose files are links into <repo>/blobs/).
// Empty when the recipe's directory cannot be resolved.
std::filesystem::path SidecarRoot(const std::string& recipe_dir) {
  std::error_code ec;
  const std::filesystem::path dir = std::filesystem::weakly_canonical(
      recipe_dir.empty() ? std::filesystem::path(".")
                         : std::filesystem::path(recipe_dir),
      ec);
  if (ec) {
    return {};
  }
  if (dir.parent_path().filename() == "snapshots") {
    return dir.parent_path().parent_path();
  }
  return dir;
}

// Whether `file`, with every symbolic link followed, is inside `root`. A
// sidecar may be a link (the Hugging Face cache stores files that way), but
// not one that leads out of the recipe's own folder or repository.
bool ResolvesInside(const std::filesystem::path& file,
                    const std::filesystem::path& root) {
  if (root.empty()) {
    return false;
  }
  std::error_code ec;
  const std::string real = std::filesystem::weakly_canonical(file, ec).string();
  if (ec) {
    return false;
  }
  std::string prefix = root.string();
  if (prefix.empty() || prefix.back() != '/') {
    prefix += '/';
  }
  return real.compare(0, prefix.size(), prefix) == 0;
}

bool LooksAbsolutePath(std::string_view value) {
  return !value.empty() && value[0] == '/';
}

bool HasDotDot(const std::filesystem::path& p) {
  for (const auto& comp : p) {
    if (comp == "..") {
      return true;
    }
  }
  return false;
}

// SUPERROC_*, LFN_* or LING3_* followed by [A-Z0-9_]+.
bool ValidEnvKey(std::string_view key) {
  constexpr std::string_view kEnginePrefix = "SUPERROC_";
  constexpr std::string_view kLfnPrefix = "LFN_";
  constexpr std::string_view kLing3Prefix = "LING3_";
  std::string_view rest;
  if (key.substr(0, kEnginePrefix.size()) == kEnginePrefix) {
    rest = key.substr(kEnginePrefix.size());
  } else if (key.substr(0, kLfnPrefix.size()) == kLfnPrefix) {
    rest = key.substr(kLfnPrefix.size());
  } else if (key.substr(0, kLing3Prefix.size()) == kLing3Prefix) {
    rest = key.substr(kLing3Prefix.size());
  } else {
    return false;
  }
  if (rest.empty()) {
    return false;
  }
  for (const char c : rest) {
    const bool ok =
        (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_';
    if (!ok) {
      return false;
    }
  }
  return true;
}

std::string BasenameOf(const std::string& path) {
  const std::string::size_type slash = path.find_last_of('/');
  return slash == std::string::npos ? path : path.substr(slash + 1);
}

// Long flag name without any =value suffix.
std::string_view FlagName(std::string_view token) {
  const std::string::size_type eq = token.find('=');
  return eq == std::string_view::npos ? token : token.substr(0, eq);
}

// A token matching ^-[0-9]+(\.[0-9]+)?$ is a value (e.g. `--max-tokens -1`),
// never a flag, everywhere TokenIsFlag is used.
bool IsNegativeNumber(std::string_view token) {
  if (token.size() < 2 || token[0] != '-') {
    return false;
  }
  std::size_t i = 1;
  bool digits = false;
  while (i < token.size() && token[i] >= '0' && token[i] <= '9') {
    ++i;
    digits = true;
  }
  if (!digits) {
    return false;
  }
  if (i == token.size()) {
    return true;
  }
  if (token[i] != '.') {
    return false;
  }
  ++i;
  bool frac = false;
  while (i < token.size() && token[i] >= '0' && token[i] <= '9') {
    ++i;
    frac = true;
  }
  return frac && i == token.size();
}

bool TokenIsFlag(std::string_view token) {
  return token.size() > 1 && token[0] == '-' && !IsNegativeNumber(token);
}

using FlagQuery = std::function<bool(std::string_view, bool*, std::string*)>;

// Canonical (long) name for a flag spelling; `name` itself when the query
// reports no canonical (unknown spellings are rejected by the callers).
std::string CanonicalFlag(const FlagQuery& known_flag, std::string_view name,
                          bool* takes_value) {
  bool takes = false;
  std::string canonical;
  if (!known_flag(name, &takes, &canonical) || canonical.empty()) {
    return std::string(name);
  }
  if (takes_value != nullptr) {
    *takes_value = takes;
  }
  return canonical;
}

bool CmdlineHas(const std::vector<std::string>& cmdline, std::string_view flag,
                const FlagQuery& known_flag) {
  const std::string want = CanonicalFlag(known_flag, flag, nullptr);
  for (const std::string& token : cmdline) {
    if (!TokenIsFlag(token)) {
      continue;
    }
    const std::string_view name = FlagName(token);
    if (name == flag || CanonicalFlag(known_flag, name, nullptr) == want) {
      return true;
    }
  }
  return false;
}

}  // namespace

std::optional<std::string> ScanRawOption(
    const std::vector<std::string>& args,
    std::initializer_list<std::string_view> names) {
  // The last occurrence wins, as in the argument parser, so the recipe and
  // model this scan reports are the ones the parser ends up with.
  std::optional<std::string> found;
  for (std::size_t i = 0; i < args.size(); ++i) {
    const std::string_view token = args[i];
    for (const std::string_view name : names) {
      if (token == name) {
        found = i + 1 < args.size() ? args[i + 1] : std::string();
        ++i;  // The next token is this option's value, not an option.
        break;
      }
      if (token.size() > name.size() && token.substr(0, name.size()) == name &&
          token[name.size()] == '=') {
        found = std::string(token.substr(name.size() + 1));
        break;
      }
    }
  }
  return found;
}

bool ScanRawPresent(const std::vector<std::string>& args,
                    std::initializer_list<std::string_view> names) {
  for (const std::string& token : args) {
    if (!TokenIsFlag(token)) {
      continue;
    }
    const std::string_view name = FlagName(token);
    for (const std::string_view want : names) {
      if (name == want) {
        return true;
      }
    }
  }
  return false;
}

bool ParseRecipeFile(const std::string& path, RecipeSpec* spec,
                     std::string* err) {
  if (spec != nullptr) {
    *spec = RecipeSpec();
    spec->file = path;
  }
  // A recipe is a small JSON file. Anything else under that name (a device,
  // a pipe, a huge file) is refused before a byte is read.
  std::error_code stat_ec;
  if (!std::filesystem::is_regular_file(path, stat_ec) || stat_ec) {
    if (err != nullptr) {
      *err = "[recipe] cannot read file '" + path + "'";
    }
    return false;
  }
  const std::uintmax_t bytes = std::filesystem::file_size(path, stat_ec);
  if (stat_ec || bytes > kMaxRecipeBytes) {
    if (err != nullptr) {
      *err = "[recipe] file is larger than 1 MiB: '" + path + "'";
    }
    return false;
  }
  std::ifstream in(path);
  if (!in) {
    if (err != nullptr) {
      *err = "[recipe] cannot read file '" + path + "'";
    }
    return false;
  }
  std::ostringstream text;
  text << in.rdbuf();
  json::Value root;
  try {
    root = json::parse(text.str());
  } catch (const std::exception& e) {
    if (err != nullptr) {
      *err = std::string("[recipe] bad JSON in '") + path + "': " + e.what();
    }
    return false;
  }
  if (!root.is_object()) {
    if (err != nullptr) {
      *err = "[recipe] top level must be an object in '" + path + "'";
    }
    return false;
  }
  const auto fail = [&](const std::string& msg) {
    if (err != nullptr) {
      *err = "[recipe] " + msg + " in '" + path + "'";
    }
    return false;
  };
  for (const auto& member : root.members()) {
    if (member.first != "schema" && member.first != "model" &&
        member.first != "env" && member.first != "files" &&
        member.first != "file_args" && member.first != "args" &&
        member.first != "context" && member.first != "notes") {
      return fail("unknown key '" + member.first + "'");
    }
  }
  const json::Value* schema = root.find("schema");
  if (schema == nullptr || !schema->is_string() || schema->str() != kSchema) {
    return fail("schema must be \"" + std::string(kSchema) + "\"");
  }
  const json::Value* model = root.find("model");
  if (model == nullptr || !model->is_string() || model->str().empty()) {
    return fail("model must be a nonempty basename string");
  }
  RecipeSpec out;
  out.file = path;
  out.model = model->str();
  if (const json::Value* env = root.find("env")) {
    if (!env->is_object()) {
      return fail("env must be an object");
    }
    for (const auto& kv : env->members()) {
      if (!ValidEnvKey(kv.first)) {
        return fail("bad env key '" + kv.first + "'");
      }
      if (IsLauncherEnvKey(kv.first)) {
        return fail("bad env key '" + kv.first +
                    "' (launcher-owned: serve listens on SUPERROC_HOST/"
                    "SUPERROC_PORT)");
      }
      if (IsRefusedEnvKey(kv.first)) {
        return fail("bad env key '" + kv.first +
                    "' (it names a program or an output file; a recipe "
                    "cannot set it)");
      }
      if (IsDiagnosticEnvKey(kv.first)) {
        return fail("bad env key '" + kv.first +
                    "' (a diagnostic switch; a recipe cannot set it)");
      }
      if (IsInputFileEnvKey(kv.first)) {
        return fail("env key '" + kv.first +
                    "' takes a file and belongs in files");
      }
      if (!kv.second.is_string()) {
        return fail("env value for '" + kv.first + "' must be a string");
      }
      if (!IsPlainSetting(kv.second.str())) {
        return fail("env value for '" + kv.first +
                    "' must be a plain setting (letters, digits and "
                    "_ . , : + -), not a path");
      }
      if (!IsKnownEnvKey(kv.first)) {
        out.ignored.push_back(kv.first);
        continue;
      }
      out.env[kv.first] = kv.second.str();
    }
  }
  const std::string dir = std::filesystem::path(path).parent_path().string();
  const std::filesystem::path sidecar_root = SidecarRoot(dir);
  if (const json::Value* files = root.find("files")) {
    if (!files->is_object()) {
      return fail("files must be an object");
    }
    for (const auto& kv : files->members()) {
      if (!ValidEnvKey(kv.first)) {
        return fail("bad files key '" + kv.first + "'");
      }
      if (IsLauncherEnvKey(kv.first)) {
        return fail("bad files key '" + kv.first +
                    "' (launcher-owned: serve listens on SUPERROC_HOST/"
                    "SUPERROC_PORT)");
      }
      if (IsRefusedEnvKey(kv.first)) {
        return fail("bad files key '" + kv.first +
                    "' (it names a program or an output file; a recipe "
                    "cannot set it)");
      }
      if (IsDiagnosticEnvKey(kv.first)) {
        return fail("bad files key '" + kv.first +
                    "' (a diagnostic switch; a recipe cannot set it)");
      }
      if (IsKnownEnvKey(kv.first) && !IsInputFileEnvKey(kv.first)) {
        return fail("bad files key '" + kv.first +
                    "' (not an input-file variable: SUPERROC_INT4_WEIGHTS or "
                    "SUPERROC_TENSOR_OVERLAY)");
      }
      if (!kv.second.is_string() || kv.second.str().empty()) {
        return fail("files value for '" + kv.first +
                    "' must be a nonempty path string");
      }
      const std::filesystem::path rel(kv.second.str());
      if (rel.is_absolute()) {
        return fail("files value for '" + kv.first +
                    "' must be a relative path, not an absolute path: '" +
                    rel.string() + "'");
      }
      if (HasDotDot(rel)) {
        return fail("files value for '" + kv.first +
                    "' must not contain '..': '" + rel.string() + "'");
      }
      const std::filesystem::path joined = std::filesystem::path(dir) / rel;
      std::error_code ec;
      if (!std::filesystem::is_regular_file(joined, ec) || ec) {
        return fail("files value for '" + kv.first +
                    "' does not exist or is not a regular file: '" +
                    joined.string() + "'");
      }
      if (!ResolvesInside(joined, sidecar_root)) {
        return fail("files value for '" + kv.first +
                    "' resolves outside the recipe's folder: '" +
                    joined.string() + "'");
      }
      // Exported as an absolute path even when the recipe and the caller's
      // working directory are both relative.
      if (!IsKnownEnvKey(kv.first)) {
        out.ignored.push_back(kv.first);
        continue;
      }
      std::error_code abs_ec;
      const std::string abs =
          std::filesystem::absolute(joined, abs_ec).string();
      out.files[kv.first] = abs_ec ? joined.string() : abs;
    }
  }
  if (const json::Value* file_args = root.find("file_args")) {
    if (!file_args->is_object()) {
      return fail("file_args must be an object");
    }
    for (const auto& kv : file_args->members()) {
      if (IsLauncherOwnedFlag(kv.first)) {
        return fail("bad file_args flag '" + kv.first +
                    "' (launcher-owned: the command line supplies it)");
      }
      if (!IsFileArgFlag(kv.first)) {
        return fail("bad file_args flag '" + kv.first +
                    "' (not an llm model/file option; see docs/recipes.md)");
      }
      if (!kv.second.is_string() || kv.second.str().empty()) {
        return fail("file_args value for '" + kv.first +
                    "' must be a nonempty path string");
      }
      const std::filesystem::path rel(kv.second.str());
      if (rel.is_absolute()) {
        return fail("file_args value for '" + kv.first +
                    "' must be a relative path, not an absolute path: '" +
                    rel.string() + "'");
      }
      if (HasDotDot(rel)) {
        return fail("file_args value for '" + kv.first +
                    "' must not contain '..': '" + rel.string() + "'");
      }
      const std::filesystem::path joined = std::filesystem::path(dir) / rel;
      std::error_code ec;
      if (!std::filesystem::is_regular_file(joined, ec) || ec) {
        return fail("file_args value for '" + kv.first +
                    "' does not exist or is not a regular file: '" +
                    joined.string() + "'");
      }
      if (!ResolvesInside(joined, sidecar_root)) {
        return fail("file_args value for '" + kv.first +
                    "' resolves outside the recipe's folder: '" +
                    joined.string() + "'");
      }
      std::error_code abs_ec;
      const std::string abs =
          std::filesystem::absolute(joined, abs_ec).string();
      out.file_args[kv.first] = abs_ec ? joined.string() : abs;
    }
  }
  if (const json::Value* args = root.find("args")) {
    if (!args->is_array()) {
      return fail("args must be an array of strings");
    }
    for (const auto& item : args->items()) {
      if (!item.is_string() || item.str().empty()) {
        return fail("args entries must be nonempty strings");
      }
      out.args.push_back(item.str());
    }
  }
  if (const json::Value* context = root.find("context")) {
    if (!context->is_number()) {
      return fail("context must be a number");
    }
    const std::uint32_t n = static_cast<std::uint32_t>(context->as_size(0));
    if (n == 0 || context->as_double(0.0) != static_cast<double>(n)) {
      return fail("context must be a positive integer");
    }
    out.has_context = true;
    out.context = n;
  }
  if (spec != nullptr) {
    *spec = std::move(out);
  }
  return true;
}

bool ValidateRecipeForModel(
    const RecipeSpec& spec, const std::string& model_path,
    const std::function<bool(std::string_view name, bool* takes_value,
                             std::string* canonical)>& known_flag,
    std::string* err) {
  const auto fail = [&](const std::string& msg) {
    if (err != nullptr) {
      *err = "[recipe] " + msg + " in '" + spec.file + "'";
    }
    return false;
  };
  if (model_path.empty()) {
    return fail("needs --model <PATH> (model is '" + spec.model + "')");
  }
  if (BasenameOf(model_path) != spec.model) {
    return fail("model '" + spec.model + "' does not match basename of '" +
                model_path + "'");
  }
  for (std::size_t i = 0; i < spec.args.size(); ++i) {
    const std::string_view token = spec.args[i];
    if (!TokenIsFlag(token)) {
      return fail("args entry '" + std::string(token) +
                  "' is a value without a flag");
    }
    const std::string_view name = FlagName(token);
    bool takes_value = false;
    std::string canonical;
    if (!known_flag(name, &takes_value, &canonical)) {
      return fail("unknown llm flag '" + std::string(name) + "'");
    }
    if (canonical.empty()) {
      canonical = std::string(name);
    }
    const bool inline_value = FlagName(token).size() != token.size();
    if (inline_value && !takes_value) {
      return fail("args flag '" + std::string(name) + "' takes no value");
    }
    // Compared by canonical name, so -p matches --port and -t matches
    // --temperature. Launcher-owned flags stay with the command line.
    if (IsLauncherOwnedFlag(canonical)) {
      if (canonical == "--context") {
        return fail("args flag '" + std::string(name) +
                    "' sets --context: use the recipe's `context` field");
      }
      return fail("args flag '" + std::string(name) +
                  "' sets launcher-owned '" + canonical + "'");
    }
    // A recipe ships next to the model, so it must never carry a host path:
    // file-taking flags belong in file_args, and no value may be absolute.
    if (IsFileArgFlag(canonical)) {
      return fail("args flag '" + std::string(name) +
                  "' takes a file path and belongs in file_args");
    }
    if (!IsRecipeArgFlag(canonical)) {
      return fail("args flag '" + std::string(name) + "' sets '" + canonical +
                  "', which only the command line may set (a recipe sets "
                  "speculative decoding, prompt lookup, prefill chunking, "
                  "thinking and sampling defaults)");
    }
    if (!takes_value) {
      continue;
    }
    if (inline_value) {
      if (LooksAbsolutePath(token.substr(name.size() + 1))) {
        return fail("args value for '" + std::string(name) +
                    "' is an absolute path and belongs in file_args");
      }
      continue;
    }
    if (i + 1 >= spec.args.size() || TokenIsFlag(spec.args[i + 1])) {
      return fail("llm flag '" + std::string(name) + "' needs a value");
    }
    if (LooksAbsolutePath(spec.args[i + 1])) {
      return fail("args value for '" + std::string(name) +
                  "' is an absolute path and belongs in file_args");
    }
    ++i;
  }
  // file_args flags are a subset of the llm model/file options, but confirm
  // each one against the live parser so a renamed option still refuses.
  for (const auto& kv : spec.file_args) {
    bool takes_value = false;
    std::string canonical;
    if (!known_flag(kv.first, &takes_value, &canonical) || !takes_value) {
      return fail("file_args flag '" + kv.first +
                  "' is not an llm file option");
    }
    if (canonical.empty()) {
      canonical = kv.first;
    }
    if (IsLauncherOwnedFlag(canonical)) {
      return fail("file_args must not set launcher-owned '" + canonical + "'");
    }
  }
  return true;
}

std::vector<std::string> ApplyRecipeEnv(
    const RecipeSpec& spec, std::map<std::string, std::string>* applied_env,
    std::map<std::string, std::string>* applied_files) {
  std::vector<std::string> log;
  for (const auto& kv : spec.env) {
    if (::superroc::core::GetEnv(kv.first.c_str()) != nullptr) {
      log.push_back("[recipe] kept env " + kv.first);
      continue;
    }
    ::setenv(kv.first.c_str(), kv.second.c_str(), 0);
    if (applied_env != nullptr) {
      (*applied_env)[kv.first] = kv.second;
    }
    log.push_back("[recipe] env " + kv.first + "=" + kv.second);
  }
  for (const auto& kv : spec.files) {
    if (::superroc::core::GetEnv(kv.first.c_str()) != nullptr) {
      log.push_back("[recipe] kept env " + kv.first);
      continue;
    }
    ::setenv(kv.first.c_str(), kv.second.c_str(), 0);
    if (applied_files != nullptr) {
      (*applied_files)[kv.first] = kv.second;
    }
    log.push_back("[recipe] file " + kv.first + "=" + kv.second);
  }
  for (const std::string& key : spec.ignored) {
    log.push_back("[recipe] ignored " + key + " (this build does not read it)");
  }
  return log;
}

RecipeMerge MergeRecipeArgs(
    const RecipeSpec& spec, const std::vector<std::string>& cmdline,
    const std::function<bool(std::string_view name, bool* takes_value,
                             std::string* canonical)>& known_flag) {
  RecipeMerge out;
  for (std::size_t i = 0; i < spec.args.size(); ++i) {
    const std::string& token = spec.args[i];
    const std::string name(FlagName(token));
    bool takes_value = false;
    std::string canonical;
    if (!known_flag(name, &takes_value, &canonical)) {
      out.error =
          "[recipe] unknown llm flag '" + name + "' in '" + spec.file + "'";
      return out;
    }
    if (canonical.empty()) {
      canonical = name;
    }
    std::string value;
    const bool inline_value = name.size() != token.size();
    if (inline_value && !takes_value) {
      out.error = "[recipe] args flag '" + name + "' takes no value in '" +
                  spec.file + "'";
      return out;
    }
    if (takes_value && !inline_value) {
      if (i + 1 >= spec.args.size()) {
        out.error = "[recipe] llm flag '" + name + "' needs a value in '" +
                    spec.file + "'";
        return out;
      }
      value = spec.args[i + 1];
      ++i;
    } else if (inline_value) {
      value = token.substr(name.size() + 1);
    }
    if (CmdlineHas(cmdline, name, known_flag)) {
      out.log.push_back("[recipe] kept command-line " + canonical);
      continue;
    }
    out.extra.push_back(name);
    std::string line = "[recipe] arg " + name;
    if (takes_value) {
      out.extra.push_back(value);
      line += " " + value;
    }
    out.log.push_back(std::move(line));
  }
  if (spec.has_context) {
    if (CmdlineHas(cmdline, "--context", known_flag)) {
      out.log.push_back("[recipe] kept command-line --context");
    } else {
      out.extra.push_back("--context");
      out.extra.push_back(std::to_string(spec.context));
      out.log.push_back("[recipe] arg --context " +
                        std::to_string(spec.context));
    }
  }
  // Portable file arguments: resolved absolute paths, appended as
  // `<flag> <abs path>` unless the flag is already on the command line.
  for (const auto& kv : spec.file_args) {
    bool takes_value = false;
    std::string canonical;
    if (!known_flag(kv.first, &takes_value, &canonical) || !takes_value) {
      out.error = "[recipe] file_args flag '" + kv.first +
                  "' is not an llm file option in '" + spec.file + "'";
      return out;
    }
    if (canonical.empty()) {
      canonical = kv.first;
    }
    if (CmdlineHas(cmdline, kv.first, known_flag)) {
      out.log.push_back("[recipe] kept command-line " + canonical);
      continue;
    }
    out.extra.push_back(kv.first);
    out.extra.push_back(kv.second);
    out.log.push_back("[recipe] arg " + kv.first + " " + kv.second);
  }
  return out;
}

namespace {
AppliedRecipe& StoredRecipe() {
  static AppliedRecipe stored;
  return stored;
}

struct PrescanSlot {
  bool ran = false;
  std::string file;
  std::vector<std::string> log;
  std::map<std::string, std::string> applied_env;
  std::map<std::string, std::string> applied_files;
};

PrescanSlot& Slot() {
  static PrescanSlot slot;
  return slot;
}
}  // namespace

void SetAppliedRecipe(AppliedRecipe recipe) {
  StoredRecipe() = std::move(recipe);
}

const AppliedRecipe& GetAppliedRecipe() {
  return StoredRecipe();
}

std::vector<std::string> TakePrescanLog(
    const std::string& file, std::map<std::string, std::string>* applied_env,
    std::map<std::string, std::string>* applied_files) {
  PrescanSlot& slot = Slot();
  if (!slot.ran || slot.file != file) {
    return {};
  }
  slot.ran = false;
  if (applied_env != nullptr) {
    *applied_env = std::move(slot.applied_env);
    slot.applied_env.clear();
  }
  if (applied_files != nullptr) {
    *applied_files = std::move(slot.applied_files);
    slot.applied_files.clear();
  }
  return std::move(slot.log);
}

std::optional<int> PrescanRecipeEnv(std::span<const char* const> args) {
  std::vector<std::string> tokens;
  tokens.reserve(args.size());
  for (const char* a : args) {
    tokens.emplace_back(a == nullptr ? "" : a);
  }
  // Only `serve ... llm` takes --recipe: `serve` must be the command (the
  // first argument, as main dispatches it) and `llm` its subcommand. Other
  // commands and serve subcommands never export a recipe's variables, and
  // only the llm arguments are scanned, as on the serve path.
  if (tokens.size() < 3 || tokens[1] != "serve") {
    return std::nullopt;
  }
  std::size_t llm_at = 2;
  while (llm_at < tokens.size() && tokens[llm_at] != "llm") {
    ++llm_at;
  }
  if (llm_at == tokens.size()) {
    return std::nullopt;
  }
  tokens.erase(tokens.begin(),
               tokens.begin() + static_cast<std::ptrdiff_t>(llm_at) + 1);
  const std::optional<std::string> recipe = ScanRawOption(tokens, {"--recipe"});
  if (!recipe.has_value() || *recipe == "off" || recipe->empty()) {
    return std::nullopt;
  }
  std::string path = *recipe;
  const std::optional<std::string> model =
      ScanRawOption(tokens, {"-m", "--model"});
  if (path == "auto") {
    if (!model.has_value() || model->empty()) {
      return std::nullopt;  // The serve path reports the missing --model.
    }
    path =
        (std::filesystem::path(*model).parent_path() / kAutoFileName).string();
  }
  RecipeSpec spec;
  std::string err;
  if (!ParseRecipeFile(path, &spec, &err)) {
    std::cerr << "Error: " << err << "\n";
    return 2;
  }
  if (model.has_value() && !model->empty() &&
      BasenameOf(*model) != spec.model) {
    std::cerr << "Error: [recipe] model '" << spec.model
              << "' does not match basename of '" << *model << "' in '" << path
              << "'\n";
    return 2;
  }
  PrescanSlot& slot = Slot();
  slot.ran = true;
  slot.file = path;
  slot.applied_env.clear();
  slot.applied_files.clear();
  slot.log = ApplyRecipeEnv(spec, &slot.applied_env, &slot.applied_files);
  return std::nullopt;
}

}  // namespace superroc::cli
