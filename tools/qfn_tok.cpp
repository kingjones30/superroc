// FLASHNEXT: print superroc token ids for character cuts. Tokenizer only;
// does not upload weights. Usage: qfn_tok <gguf> <text> <cut>...
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include "src/core/gguf_reader.hpp"
#include "src/models/qwen/tokenizer.hpp"

int main(int argc, char** argv) {
  if (argc < 4) {
    std::cerr << "usage: qfn_tok <gguf> <text> <cut>...\n";
    return 2;
  }
  std::string error;
  auto reader = superroc::core::GgufReader::OpenFile(argv[1], &error);
  if (!reader) {
    std::cerr << "gguf: " << error << "\n";
    return 1;
  }
  auto tok = superroc::tokenization::QwenTokenizer::CreateFromGguf(*reader, &error);
  if (!tok) {
    std::cerr << "tokenizer: " << error << "\n";
    return 1;
  }
  std::ifstream in(argv[2]);
  if (!in) {
    std::cerr << "cannot open " << argv[2] << "\n";
    return 1;
  }
  std::ostringstream ss;
  ss << in.rdbuf();
  const std::string text = ss.str();
  std::cout << "{\n";
  for (int i = 3; i < argc; ++i) {
    const int cut = std::stoi(argv[i]);
    if (cut < 0 || static_cast<std::size_t>(cut) > text.size()) {
      std::cerr << "cut out of range " << cut << "\n";
      return 1;
    }
    const auto ids = tok->Encode(std::string_view(text).substr(0, cut));
    std::cout << " \"" << cut << "\": [";
    for (std::size_t j = 0; j < ids.size(); ++j) {
      if (j) std::cout << ",";
      std::cout << ids[j];
    }
    std::cout << "]" << (i + 1 == argc ? "\n" : ",\n");
  }
  std::cout << "}\n";
  return 0;
}
