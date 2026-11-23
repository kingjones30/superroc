# Security

## Reporting a vulnerability

Report it privately through this repository's **Security** tab
(**Report a vulnerability**). Please do not open a public issue for it.
Include the version (`superroc --version`), the command line, and the
smallest request or file that shows the problem.

## Supported versions

Fixes go into the next release. Only the latest release is supported.

## What the server assumes

- It listens on `127.0.0.1` unless `--host` says otherwise. Anything that
  can reach the port can use the model; set `--api-key` or
  `--api-key-file` before you listen on a network address, and put TLS in a
  reverse proxy.
- No CORS headers are sent unless `--cors-origin` is given.
- Model files, recipes and command-line options are trusted input from the
  operator. Requests are not: request bodies, JSON schemas, images, audio
  and tokenizer input are size- and work-limited, and request text is never
  read as a control token.
- Internal errors reach clients as fixed messages; details stay in the log.
- Clients of one server share its prompt cache (see
  [docs/SERVER.md](docs/SERVER.md#authentication-and-exposure)).
- The server never runs tools, shell commands, URLs or generated code.
