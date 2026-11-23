#pragma once

// Super Roc recipe: a model carries its own tuned setup (env, sidecar files,
// llm flags) in a superroc-recipe.json file next to it on Hugging Face, so
// any launcher gets the tuned setup. `--recipe off` (the default) leaves every
// existing command byte-identical.

#include <cstdint>
#include <functional>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace superroc::cli {

// A parsed superroc-recipe/1 file. `file` is the recipe path as given;
// `files` values are absolute paths resolved against the recipe's directory.
// `file_args` maps an llm model/file flag (e.g. --mtp-model) to the absolute
// path of the sidecar file resolved against the recipe's directory.
struct RecipeSpec {
  std::string file;
  std::string model;
  std::map<std::string, std::string> env;
  std::map<std::string, std::string> files;
  std::map<std::string, std::string> file_args;
  std::vector<std::string> args;
  // env/files keys this build does not read: reported and skipped, so a recipe
  // written for another engine version still loads.
  std::vector<std::string> ignored;
  bool has_context = false;
  std::uint32_t context = 0;
};

// Parse a recipe file and run the file-level checks (schema tag, unknown
// top-level keys, env key spelling, file existence). A recipe is downloaded
// with a model, so its env is limited to settings: keys that name a program
// or an output file are refused, values must be plain settings (no paths),
// and `files` may only set the engine's input-file variables. Pure: reads
// the file, touches no environment.
bool ParseRecipeFile(const std::string& path, RecipeSpec* spec,
                     std::string* err);

// Model + flag checks against the live command line: `model` must equal
// basename(model_path); every --flag in args must satisfy known_flag (which
// reports whether the flag takes a value and its canonical long name).
// Launcher-owned flags (server identity, secrets, write paths, --model,
// --recipe, --context) are refused here, as are launcher-owned env keys.
bool ValidateRecipeForModel(
    const RecipeSpec& spec, const std::string& model_path,
    const std::function<bool(std::string_view name, bool* takes_value,
                             std::string* canonical)>& known_flag,
    std::string* err);

// Export env + files with setenv(overwrite=0) so anything already set wins.
// Returns one `[recipe] ...` line per applied or kept item. When non-null,
// `applied_env`/`applied_files` receive only the keys this call actually set
// (kept keys are left out), for the GET /props record.
std::vector<std::string> ApplyRecipeEnv(
    const RecipeSpec& spec,
    std::map<std::string, std::string>* applied_env = nullptr,
    std::map<std::string, std::string>* applied_files = nullptr);

struct RecipeMerge {
  // Extra argv tokens to append (flags the command line did not supply).
  std::vector<std::string> extra;
  // One `[recipe] ...` line per applied or kept item.
  std::vector<std::string> log;
  std::string error;
};

// Decide which recipe args reach the parser: flags already on the command
// line (compared by canonical name, so -p matches --port) are skipped, the
// rest are appended. Context becomes --context only when the command line
// has none.
RecipeMerge MergeRecipeArgs(
    const RecipeSpec& spec, const std::vector<std::string>& cmdline,
    const std::function<bool(std::string_view name, bool* takes_value,
                             std::string* canonical)>& known_flag);

// Raw argv scan before any parser runs (supports `--flag value` and
// `--flag=value`). Used to find --recipe / -m early.
std::optional<std::string> ScanRawOption(
    const std::vector<std::string>& args,
    std::initializer_list<std::string_view> names);
bool ScanRawPresent(const std::vector<std::string>& args,
                    std::initializer_list<std::string_view> names);

// The recipe record exposed in GET /props. `active` is false when --recipe
// is off; then /props reports an empty object.
struct AppliedRecipe {
  bool active = false;
  std::string file;
  std::map<std::string, std::string> env;
  std::map<std::string, std::string> files;
  std::map<std::string, std::string> file_args;
  std::vector<std::string> args;
  // Keys the recipe named that this build does not read.
  std::vector<std::string> ignored;
};
void SetAppliedRecipe(AppliedRecipe recipe);
const AppliedRecipe& GetAppliedRecipe();

// The pre-scan applies env/files before the first SUPERROC_/LFN_ read and
// stashes the log lines it produced; the serve path prints them so the
// startup lines say what the recipe actually did (a fresh ApplyRecipeEnv in
// the serve path would report everything as kept). Returns the stashed lines
// when they belong to `file`, else an empty vector (then the caller applies
// the env itself). When non-null, `applied_env`/`applied_files` receive only
// the keys the pre-scan actually set (kept keys are left out).
std::vector<std::string> TakePrescanLog(
    const std::string& file,
    std::map<std::string, std::string>* applied_env = nullptr,
    std::map<std::string, std::string>* applied_files = nullptr);

// Early pass at the top of main(): when argv asks for `serve ... llm` with a
// non-off --recipe, load the file and export its env/files NOW, before the
// first SUPERROC_/LFN_ read (RunServe falls back to SUPERROC_HOST/PORT before
// the llm options are parsed, and several engine readers cache the value in a
// function-local static on first call). Silent: the serve path logs. Returns
// an exit code when the recipe refuses to start.
std::optional<int> PrescanRecipeEnv(std::span<const char* const> args);

}  // namespace superroc::cli
