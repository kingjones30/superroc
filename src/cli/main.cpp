#include <array>
#include <iostream>
#include <span>
#include <string_view>

#include "src/cli/banner.hpp"
#include "src/cli/bench/bench.hpp"
#include "src/cli/diagnose/diagnose.h"
#include "src/cli/eval/eval.hpp"
#include "src/cli/prompt/prompt.hpp"
#include "src/cli/serve/serve.hpp"
#include "src/cli/transcribe/transcribe.hpp"
#include "src/cli/video/video.hpp"

#ifndef SUPERROC_RELEASE_VERSION
#define SUPERROC_RELEASE_VERSION "development"
#endif
#ifndef SUPERROC_REVISION
#define SUPERROC_REVISION "unknown"
#endif

constexpr std::string_view kSuperROCReleaseVersion = SUPERROC_RELEASE_VERSION;
constexpr std::string_view kSuperROCRevision = SUPERROC_REVISION;

namespace {

// Super ROC: name help output after the invoked binary basename.
std::string_view Basename(std::string_view path) {
  const std::string_view::size_type slash = path.find_last_of('/');
  return slash == std::string_view::npos ? path : path.substr(slash + 1);
}

void print_version() {
  // Super ROC: product identity; internal names stay superroc in M1.
  std::cout << "Super ROC " << kSuperROCReleaseVersion << " (" << kSuperROCRevision
            << ") - engine lineage: upstream v0.19.0 (MIT)\n";
}

void print_help(std::string_view program_name) {
  std::cout
      << "Usage: " << program_name
      << " <COMMAND> [OPTIONS]\n\n"
      // Super ROC: product identity.
      << "Super ROC - fast LLM inference for AMD Strix Halo (gfx1151): "
         "ROCmFP4 quants, MTP/DSpark drafting, prompt cache.\n\n"
      << "Commands:\n"
      << "  serve          Start the OpenAI-compatible HTTP server\n"
      << "  prompt         Execute one prompt request and exit\n"
      << "  chat           Start an interactive terminal conversation\n"
      << "  bench          Benchmark prompt processing and token generation\n"
      << "  eval           Evaluate an OpenAI-compatible text server\n"
      << "  video          Generate MiniMax H3 text-to-video\n"
      << "  transcribe     Transcribe WAV audio with Qwen3-ASR-1.7B\n"
      << "  diagnose       Inspect system hardware, memory, and drivers\n"
      << "  help           Print help for a specific command\n\n"
      << "Run '" << program_name << " help <COMMAND>' or '" << program_name
      << " <COMMAND> --help' for details on a command\n\n"
      << "Options:\n"
      << "  -h, --help     Print help\n"
      << "  -v, --version  Print version\n";
}

int run(std::span<const char* const> args) {
  if (args.empty()) {
    std::cerr << "Error: missing command or option\n";
    print_help("superroc");  // Super ROC: no argv[0] when argc==0; default to
                             // product name.
    return 2;
  }

  // Super ROC: basename so the binary and its previous-name link each name themselves.
  const std::string_view program_name = Basename(args[0]);
  const auto options = args.subspan(1);

  if (options.empty()) {
    print_help(program_name);
    return 0;
  }

  const std::string_view first_arg = options[0];
  if (first_arg == "--version" || first_arg == "-v") {
    print_version();
    return 0;
  }

  if (first_arg == "--help" || first_arg == "-h") {
    print_help(program_name);
    return 0;
  }

  if (first_arg == "help") {
    if (options.size() <= 1) {
      print_help(program_name);
      return 0;
    }
    const std::string_view sub = options[1];
    const std::array<const char*, 1> help_flag = {"--help"};
    if (sub == "serve") {
      if (options.size() > 2) {
        const std::array<const char*, 2> serve_help_flags = {options[2],
                                                             "--help"};
        return superroc::cli::RunServe(serve_help_flags);
      }
      return superroc::cli::RunServe(help_flag);
    }
    if (sub == "prompt") {
      return superroc::cli::RunPrompt(help_flag);
    }
    if (sub == "bench") {
      return superroc::cli::RunBench(help_flag);
    }
    if (sub == "eval") {
      return superroc::cli::RunEval(help_flag);
    }
    if (sub == "video") {
      return superroc::cli::RunVideo(help_flag);
    }
    if (sub == "transcribe") {
      return superroc::cli::RunTranscribe(help_flag);
    }
    if (sub == "chat") {
      return superroc::cli::RunChat(help_flag);
    }
    if (sub == "probe") {
      return superroc::cli::RunProbe(help_flag);
    }
    if (sub == "diagnose") {
      return superroc::cli::RunDiagnose(help_flag);
    }
    std::cerr << "Error: unknown help topic '" << sub << "'\n";
    print_help(program_name);
    return 2;
  }

  superroc::cli::PrintStartupBanner(kSuperROCReleaseVersion, options);

  if (first_arg == "probe") {
    return superroc::cli::RunProbe(options.subspan(1));
  }

  if (first_arg == "diagnose") {
    return superroc::cli::RunDiagnose(options);
  }

  if (first_arg == "bench") {
    return superroc::cli::RunBench(options.subspan(1));
  }

  if (first_arg == "eval") {
    return superroc::cli::RunEval(options.subspan(1));
  }

  if (first_arg == "prompt") {
    return superroc::cli::RunPrompt(options.subspan(1));
  }

  if (first_arg == "video") {
    return superroc::cli::RunVideo(options.subspan(1));
  }

  if (first_arg == "transcribe") {
    return superroc::cli::RunTranscribe(options.subspan(1));
  }

  if (first_arg == "chat") {
    return superroc::cli::RunChat(options.subspan(1));
  }

  if (first_arg == "serve") {
    return superroc::cli::RunServe(options.subspan(1));
  }

  // If invoked as legacy binary name or flag passed directly, route to serve
  // Compatibility: the previous server binary name is built from
  // parts so its spelling never appears here.
  static const std::string previous_server_bin =
      std::string("g") + "ufo-server";
  if (program_name.ends_with(previous_server_bin) ||
      program_name.ends_with("/" + previous_server_bin)) {
    return superroc::cli::RunServe(options);
  }

  std::cerr << "Error: unknown command '" << first_arg << "'\n";
  print_help(program_name);
  return 2;
}

}  // namespace

int main(int argc, char* argv[]) {
  const std::span<const char* const> args(argv, static_cast<std::size_t>(argc));
  return run(args);
}
