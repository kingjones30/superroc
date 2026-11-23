#!/usr/bin/env python3
"""growing_prompt.py - does a reply depend on where the prompt was cut?

The test sends a raw prompt that grows a little at a time. One server gets the
whole series, so every request after the first prefills only the few new
tokens behind the cache. For every step a fresh server gets the same full
prompt in one request.

If the answer does not depend on how the prompt was cut, the live server and
the fresh server give the same text at every step.

Two kinds of server can be tested:

  * a server this tool starts and stops itself (--line-file or --server-cmd);
  * any OpenAI-compatible server that the caller starts and stops
    (--other-server-start and --other-server-stop).

Request bodies are fixed: temperature 0, seed 1, stream off.

Exit code 0 when every step gives the same reply, 1 otherwise.
"""

import argparse
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
import urllib.request

ENV_ASSIGNMENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")

TOOL_NAME = "growing_prompt"

# The base text of the prompt. It is repeated so the prompt starts well above
# the first cache cut.
BASE_TEXT = (
    "The harbour office keeps a ledger of every vessel that ties up at the "
    "north quay. Each entry lists the name of the ship, its home port, the "
    "tonnage declared by the master, the cargo by class and weight, the berth "
    "assigned, and the hours of arrival and departure. The clerk on duty "
    "copies the figures from the pilot's slip, checks them against the "
    "manifest, and initials the line. When two figures disagree the clerk "
    "marks the entry and sends a runner to the customs shed. "
) * 3

# The words the prompt grows by. A leading space keeps every prompt a token
# prefix of the next one.
ADDITIONS = [
    "Today",
    "the first",
    "ship in was",
    "a collier from the east coast,",
    "and",
    "its master declared four hundred tons of coal,",
    "which",
    "the clerk entered",
    "without comment.",
    "The second",
    "arrived at noon carrying timber and a small quantity of salt,",
    "and here",
    "the",
    "manifest and the pilot's slip",
    "did not",
    "agree on the weight, so the entry was marked and",
    "a runner",
    "went",
    "to the shed.",
    "By evening the ledger showed",
]


def set_option(words, name, value):
    """Replace the value of a command line option, or append it when absent."""
    words = list(words)
    if name in words:
        words[words.index(name) + 1] = str(value)
        return words
    return words + [name, str(value)]


def parse_command(text):
    """Split a server command line into environment pairs and the command.

    The line may start with environment assignments, either as bare NAME=VALUE
    pairs or after an 'env' keyword.
    """
    words = shlex.split(text)
    env = {}
    if words and words[0] == "env":
        words.pop(0)
    while words and ENV_ASSIGNMENT.match(words[0]):
        name, value = words.pop(0).split("=", 1)
        env[name] = value
    if not words:
        raise SystemExit("no server command found")
    return env, words


def model_name_from(words):
    """The served model name a command line asks for, if it names one."""
    for flag in ("--served-model-name", "--served_model_name"):
        if flag in words:
            index = words.index(flag)
            if index + 1 < len(words):
                return words[index + 1]
    return None


class Server:
    """A server this tool owns: it starts the process and stops it again."""

    def __init__(self, args, words, env, tag, out):
        self.args = args
        self.words = list(words)
        self.env = env
        self.tag = tag
        self.out = out
        self.process = None
        self.log = None

    def start(self):
        self.log = open(self.out + ".srv-" + self.tag + ".log", "a")
        child_env = dict(os.environ)
        child_env.update(self.env)
        self.process = subprocess.Popen(
            self.words,
            env=child_env,
            stdout=self.log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        if not wait_ready(self.args.port, self.args.start_timeout, self.process):
            self.stop()
            raise SystemExit("server did not become ready on port %d" % self.args.port)

    def stop(self):
        if self.process is None:
            return
        if self.process.poll() is None:
            # TERM the process this tool started, then its own process group,
            # and wait for it to exit.
            try:
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
            except OSError:
                pass
            try:
                self.process.wait(self.args.stop_timeout)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                except OSError:
                    pass
                try:
                    self.process.wait(30)
                except subprocess.TimeoutExpired:
                    pass
        if self.log is not None:
            self.log.close()
            self.log = None
        self.process = None


class ExternalServer:
    """A server the caller owns: this tool only runs the start and stop
    commands it was given."""

    def __init__(self, args, tag, out):
        self.args = args
        self.tag = tag
        self.out = out
        self.log = open(self.out + ".srv-" + tag + ".log", "a")

    def start(self):
        subprocess.run(self.args.other_server_start, shell=True, stdout=self.log,
                       stderr=subprocess.STDOUT)
        if not wait_ready(self.args.port, self.args.start_timeout, None):
            raise SystemExit("server did not become ready on port %d" % self.args.port)

    def stop(self):
        subprocess.run(self.args.other_server_stop, shell=True, stdout=self.log,
                       stderr=subprocess.STDOUT)
        self.log.close()


def probe(port, path):
    try:
        urllib.request.urlopen("http://127.0.0.1:%d%s" % (port, path), timeout=5).read()
        return True
    except Exception:
        return False


def wait_ready(port, timeout, process):
    """Wait until the server answers, or give up after timeout seconds."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if process is not None and process.poll() is not None:
            return False
        if probe(port, "/v1/models") or probe(port, "/health"):
            return True
        time.sleep(0.5)
    return False


def complete(port, model, prompt, max_tokens):
    """One raw completion request, with the fixed sampling of the test."""
    body = json.dumps({
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
        "seed": 1,
        "stream": False,
    }).encode()
    request = urllib.request.Request(
        "http://127.0.0.1:%d/v1/completions" % port,
        data=body,
        headers={"Content-Type": "application/json"},
    )
    response = json.loads(urllib.request.urlopen(request, timeout=1800).read())
    usage = response.get("usage", {})
    return (
        response["choices"][0]["text"],
        usage.get("prompt_tokens"),
        (usage.get("prompt_tokens_details") or {}).get(
            "cached_tokens", usage.get("cached_tokens")),
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare the replies of a prompt that grew step by step behind a "
            "cache with the replies of the same prompt read from scratch by a "
            "fresh server. Exit code 0 when every reply matches, 1 otherwise."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--line-file",
        help=("file with one line: optional NAME=VALUE environment pairs "
              "followed by the server command"),
    )
    source.add_argument(
        "--server-cmd",
        help=("the server command as one string: optional NAME=VALUE "
              "environment pairs followed by the command"),
    )
    parser.add_argument(
        "--other-server-start",
        help=("shell command that starts an OpenAI-compatible server this tool "
              "does not own (use with --other-server-stop)"),
    )
    parser.add_argument(
        "--other-server-stop",
        help="shell command that stops the server started by --other-server-start",
    )
    parser.add_argument("--port", type=int, default=8080,
                        help="port the server listens on (default: 8080)")
    parser.add_argument("--model-name", default=None,
                        help="model name to send in the request body "
                             "(default: the served model name of the command)")
    parser.add_argument("--steps", type=int, default=len(ADDITIONS) + 1,
                        help="number of prompt steps to run (default: %d)"
                             % (len(ADDITIONS) + 1))
    parser.add_argument("--max-tokens", type=int, default=48,
                        help="reply length limit per request (default: 48)")
    parser.add_argument("--context", type=int, default=32768,
                        help="context size given to a server this tool starts "
                             "(default: 32768)")
    parser.add_argument("--out", default=TOOL_NAME + ".json",
                        help="file the JSON result is written to (default: %s.json)"
                             % TOOL_NAME)
    parser.add_argument("--start-timeout", type=float, default=900.0,
                        help="seconds to wait for a server to answer "
                             "(default: 900)")
    parser.add_argument("--stop-timeout", type=float, default=120.0,
                        help="seconds to wait for a server to exit after TERM "
                             "(default: 120)")
    args = parser.parse_args()

    if bool(args.other_server_start) != bool(args.other_server_stop):
        parser.error("--other-server-start and --other-server-stop go together")
    if args.other_server_start:
        if args.line_file or args.server_cmd:
            parser.error("use --other-server-start or --line-file/--server-cmd, "
                         "not both")
    elif not (args.line_file or args.server_cmd):
        parser.error("give --line-file, --server-cmd or --other-server-start")
    if args.steps < 1:
        parser.error("--steps must be at least 1")
    return args


def main():
    args = parse_args()
    out = args.out

    external = bool(args.other_server_start)
    env, words = ({}, [])
    if not external:
        if args.line_file:
            with open(args.line_file, "r", encoding="utf-8") as handle:
                text = handle.read().strip()
        else:
            text = args.server_cmd
        env, words = parse_command(text)
        words = set_option(words, "-p", args.port)
        words = set_option(words, "--context", args.context)

    model = args.model_name
    if not model:
        model = model_name_from(words)
    if not model:
        raise SystemExit("no model name: give --model-name")

    prompts = [BASE_TEXT.rstrip()]
    for addition in ADDITIONS[: max(args.steps - 1, 0)]:
        prompts.append(prompts[-1] + " " + addition)

    def make_server(tag):
        if external:
            return ExternalServer(args, tag, out)
        return Server(args, words, env, tag, out)

    server = make_server("live")
    server.start()
    live = [complete(args.port, model, prompt, args.max_tokens) for prompt in prompts]
    server.stop()

    rows = []
    same = 0
    result = {
        "tool": TOOL_NAME,
        "mode": "external" if external else "own-server",
        "port": args.port,
        "model": model,
        "steps": len(prompts),
        "max_tokens": args.max_tokens,
        "rows": rows,
        "live": [entry[0] for entry in live],
    }

    def save():
        """Write the JSON result, so a run that is cut short still leaves one."""
        reused = sum(1 for row in rows[1:] if (row["cached_live"] or 0) > 0)
        result["summary"] = {
            "same": same,
            "total": len(prompts),
            "steps_that_reused_the_cache": reused,
            "steps_with_a_cache": len(rows) - 1,
        }
        with open(out, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=1)

    for index, prompt in enumerate(prompts):
        server = make_server("cold")
        server.start()
        cold = complete(args.port, model, prompt, args.max_tokens)
        server.stop()
        equal = cold[0] == live[index][0]
        new_rows = (live[index][1] or 0) - (live[index][2] or 0)
        rows.append({
            "step": index,
            "ptok": live[index][1],
            "cached_live": live[index][2],
            "new_rows_live": new_rows,
            "same": equal,
            "live_sha": live[index][0],
            "cold_sha": cold[0],
        })
        same += equal
        print("step %2d ptok %5s cached %5s new %4d %s"
              % (index, live[index][1], live[index][2], new_rows,
                 "=" if equal else "X"), flush=True)
        save()

    reused = sum(1 for row in rows[1:] if (row["cached_live"] or 0) > 0)
    print("REPRO steps that reused the cache: %d of %d" % (reused, len(rows) - 1))
    verdict = "PASS" if same == len(prompts) else "FAIL"
    print("REPRO same reply live vs cold: %d of %d steps (%s)"
          % (same, len(prompts), verdict))
    return 0 if same == len(prompts) else 1


if __name__ == "__main__":
    sys.exit(main())
