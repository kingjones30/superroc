#!/usr/bin/env python3
"""live_vs_cold.py - does a reply depend on how the conversation reached the model?

The test runs the same short agent conversation two ways.

LIVE: one server, the conversation turn by turn. Every turn after the first
reuses the prompt cache of the turn before (the previous reply is read back as
prompt, tool results arrive as small prompt chunks).

COLD: for every turn a fresh server reads the SAME message history from
scratch, in one request, and answers it.

A server whose answers do not depend on the history gives the same reply to
the same history both ways. The two replies are compared character by
character.

Two kinds of server can be tested:

  * a server this tool starts and stops itself (--line-file or --server-cmd);
  * any OpenAI-compatible server that the caller starts and stops
    (--other-server-start and --other-server-stop).

Request bodies are fixed: temperature 0, seed 1, thinking off, stream off.

Exit code 0 when every turn gives the same reply, 1 otherwise.
"""

import argparse
import hashlib
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

TOOL_NAME = "live_vs_cold"

# The conversation the test runs. The system prompt asks for one JSON action
# per turn, so every turn has a short, comparable answer.
SYSTEM_PROMPT = (
    "You are an autonomous agent operating inside an isolated Linux container "
    "(workdir: /work). Solve the task by emitting ONE action per turn and "
    "reading the observation on the next turn.\n\n"
    "Reply with EXACTLY ONE JSON object and nothing else - no markdown fences, "
    "no prose:\n"
    '  run a shell command: {"tool":"shell","cmd":"<bash command>"}\n'
    '  write a file:        {"tool":"write","path":"<absolute path>",'
    '"content":"<text>"}\n'
    '  read a file:         {"tool":"read","path":"<absolute path>"}\n'
    '  finish:              {"tool":"final","summary":"<short summary>"}\n\n'
    "Rules:\n- Output ONLY the JSON object.\n"
    "- One tool call per turn; you see its result on the next turn.\n"
    "- This session has NO network access. Work only with what is already "
    "installed.\n- Work under /work.\n"
    "- You have at most 20 turns. Verify your own work before calling final.\n"
    "- Success is judged by hidden verify commands; ensure they will pass."
)


def build_tasks():
    """The task list: each entry is a user turn that starts one conversation."""
    def task(body, tests):
        return (
            "In /work create a file solution.py that solves this task:\n\n"
            + body
            + "\n\nYour function MUST satisfy these tests exactly (they reveal "
            "the required name and signature):\n```python\n"
            + tests
            + "\n```\n\nWrite the complete, correct solution (include any "
            "needed imports). Use the write tool, then finish."
        )

    return [
        task(
            "Given a string s, partition s such that every substring of the "
            "partition is a palindrome. Return the minimum number of cuts "
            "needed for a palindrome partitioning of s.",
            "assert min_cut('aab') == 1\nassert min_cut('a') == 0\n"
            "assert min_cut('ab') == 1",
        ),
        task(
            "Given a list of intervals [start, end], merge all overlapping "
            "intervals and return them sorted by start.",
            "assert merge([[1,3],[2,6],[8,10]]) == [[1,6],[8,10]]\n"
            "assert merge([[1,4],[4,5]]) == [[1,5]]",
        ),
        task(
            "Given two strings a and b, return the minimum number of "
            "single-character insertions, deletions or substitutions needed to "
            "turn a into b.",
            "assert edit_distance('kitten', 'sitting') == 3\n"
            "assert edit_distance('', 'abc') == 3",
        ),
        task(
            "Given the number of courses n and a list of prerequisite pairs "
            "[a, b] meaning b must be taken before a, return one valid order of "
            "all courses or an empty list if impossible.",
            "assert len(course_order(2, [[1,0]])) == 2\n"
            "assert course_order(2, [[1,0],[0,1]]) == []",
        ),
        task(
            "Implement a class LRUCache with capacity given at construction, "
            "get(key) returning -1 when missing, and put(key, value) evicting "
            "the least recently used entry when full. Both operations must run "
            "in O(1).",
            "c = LRUCache(2)\nc.put(1, 1)\nc.put(2, 2)\nassert c.get(1) == 1\n"
            "c.put(3, 3)\nassert c.get(2) == -1",
        ),
        task(
            "Given a list of non-negative integers heights representing an "
            "elevation map where each bar has width 1, compute how much water "
            "it can trap after raining.",
            "assert trap([0,1,0,2,1,0,1,3,2,1,2,1]) == 6\n"
            "assert trap([4,2,0,3,2,5]) == 9",
        ),
    ]


# The observation the test sends back after each action, so the conversation
# keeps going the same way in both arms.
OBSERVATIONS = [
    "wrote /work/solution.py (806 bytes)",
    "[exit 1]\n[stderr]\nTraceback (most recent call last):\n  File \"<string>\", "
    "line 3, in <module>\nAssertionError",
    "[exit 0]\n2 0 1",
    "wrote /work/solution.py (679 bytes)",
    "[exit 0]\nok",
]
TURNS_PER_TASK = len(OBSERVATIONS) + 1

# Reference notes put in front of the system prompt, so the conversation runs
# at depth (120 lines is about 3,300 more tokens, 1000 about 28,000).
def build_system_prompt(notes_lines):
    prompt = SYSTEM_PROMPT
    if not notes_lines:
        return prompt
    modules = ["scheduler", "ledger", "router", "indexer", "parser", "planner",
               "tracker", "resolver", "monitor", "exporter", "importer", "validator"]
    verbs = ["collects", "merges", "rejects", "defers", "replays", "rotates",
             "samples", "truncates", "promotes", "archives", "rebuilds", "audits"]
    objects = ["pending jobs", "stale leases", "duplicate keys", "partial batches",
               "orphan rows", "late events", "retry queues", "lock files",
               "shard maps", "audit trails", "cold segments", "open cursors"]
    state = 12345
    lines = []
    for index in range(notes_lines):
        state = (state * 1103515245 + 12345) % 2147483648
        module = state % 12
        state = (state * 1103515245 + 12345) % 2147483648
        verb = state % 12
        state = (state * 1103515245 + 12345) % 2147483648
        obj = state % 12
        lines.append(
            "Note %d: the %s module %s %s every %d seconds and reports %d "
            "counters to the %s module."
            % (index + 1, modules[module], verbs[verb], objects[obj],
               5 + (state % 55), 2 + (state % 7), modules[(module + verb + 1) % 12])
        )
    return ("Reference notes for this workspace (background only):\n"
            + "\n".join(lines) + "\n\n" + prompt)


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


def chat(port, model, messages, max_tokens):
    """One chat request, with the fixed sampling of the test."""
    body = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
        "seed": 1,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    request = urllib.request.Request(
        "http://127.0.0.1:%d/v1/chat/completions" % port,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    start = time.time()
    response = json.load(urllib.request.urlopen(request, timeout=1800))
    usage = response.get("usage", {})
    message = response["choices"][0]["message"]
    return {
        "text": message.get("content") or "",
        "ptok": usage.get("prompt_tokens"),
        "ctok": usage.get("completion_tokens"),
        "cached": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
        "sec": time.time() - start,
    }


def first_difference(left, right):
    """The character where two replies start to differ, or None when equal."""
    if left == right:
        return None
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return index
    return min(len(left), len(right))


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare the replies of a conversation that reached the model turn "
            "by turn (live) with the replies of the same conversation read from "
            "scratch by a fresh server (cold). Exit code 0 when every reply "
            "matches, 1 otherwise."
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
    parser.add_argument("--notes-lines", type=int, default=0,
                        help="reference notes in front of the system prompt, "
                             "about 26 tokens each (default: 0)")
    parser.add_argument("--tasks", type=int, default=len(build_tasks()),
                        help="number of tasks (conversations) to run "
                             "(default: %d)" % len(build_tasks()))
    parser.add_argument("--max-tokens", type=int, default=400,
                        help="reply length limit per request (default: 400)")
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
    if args.tasks < 1:
        parser.error("--tasks must be at least 1")
    if args.notes_lines < 0:
        parser.error("--notes-lines cannot be negative")
    return args


def main():
    args = parse_args()
    out = args.out
    tasks = build_tasks()[: args.tasks]
    system_prompt = build_system_prompt(args.notes_lines)

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
        words = set_option(words, "--think", "off")
        # Reasoning effort is not accepted together with thinking off.
        if "--reasoning-effort" in words:
            index = words.index("--reasoning-effort")
            del words[index:index + 2]

    model = args.model_name
    if not model:
        model = model_name_from(words)
    if not model:
        raise SystemExit("no model name: give --model-name")

    result = {
        "tool": TOOL_NAME,
        "mode": "external" if external else "own-server",
        "port": args.port,
        "model": model,
        "tasks": len(tasks),
        "notes_lines": args.notes_lines,
        "max_tokens": args.max_tokens,
        "rows": [],
    }
    same = 0
    total = 0
    live_stats = {"sec": 0.0, "turns": 0, "ctok": 0, "new": 0}

    def make_server(tag):
        if external:
            return ExternalServer(args, tag, out)
        return Server(args, words, env, tag, out)

    def save():
        """Write the JSON result, so a run that is cut short still leaves one."""
        result["summary"] = {
            "same": same,
            "total": total,
            "live_turns_after_first": live_stats["turns"],
            "live_wall_seconds": round(live_stats["sec"], 1),
            "live_seconds_per_turn": round(live_stats["sec"] / max(live_stats["turns"], 1), 2),
            "live_reply_tokens": live_stats["ctok"],
            "live_prompt_tokens_read": live_stats["new"],
        }
        with open(out, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=1)

    try:
        for index, user in enumerate(tasks):
            server = make_server("live")
            server.start()
            messages = [{"role": "system", "content": system_prompt},
                        {"role": "user", "content": user}]
            live = []
            history = []
            for turn in range(TURNS_PER_TASK):
                history.append([dict(m) for m in messages])
                reply = chat(args.port, model, messages, args.max_tokens)
                live.append(reply)
                messages.append({"role": "assistant", "content": reply["text"]})
                if turn < len(OBSERVATIONS):
                    messages.append({"role": "user",
                                     "content": "OBSERVATION:\n" + OBSERVATIONS[turn]})
            server.stop()

            live_stats["sec"] += sum(x["sec"] for x in live[1:])
            live_stats["turns"] += TURNS_PER_TASK - 1
            live_stats["ctok"] += sum(x["ctok"] or 0 for x in live[1:])
            live_stats["new"] += sum((x["ptok"] or 0) - (x["cached"] or 0)
                                     for x in live[1:])
            row = {"task": index, "turns": []}
            for turn in range(TURNS_PER_TASK):
                server = make_server("cold")
                server.start()
                reply = chat(args.port, model, history[turn], args.max_tokens)
                server.stop()
                a = live[turn]["text"]
                b = reply["text"]
                equal = a == b
                row["turns"].append({
                    "turn": turn,
                    "live": {k: live[turn][k] for k in ("ptok", "cached", "ctok")},
                    "cold": {k: reply[k] for k in ("ptok", "cached", "ctok")},
                    "same": equal,
                    "first_diff_char": first_difference(a, b),
                    "live_sha": hashlib.sha256(a.encode()).hexdigest()[:10],
                    "cold_sha": hashlib.sha256(b.encode()).hexdigest()[:10],
                })
                same += equal
                total += 1
            result["rows"].append(row)
            print(index, " ".join(
                "t%d:%s/%s:%s" % (x["turn"], x["live"]["cached"], x["live"]["ptok"],
                                  "=" if x["same"] else "X@%s" % x["first_diff_char"])
                for x in row["turns"]), flush=True)
            save()
    finally:
        save()

    print("LIVE turns after the first: %d turns, %.1f s wall (%.2f s per turn), "
          "%d reply tokens, %d prompt tokens read"
          % (live_stats["turns"], live_stats["sec"],
             live_stats["sec"] / max(live_stats["turns"], 1),
             live_stats["ctok"], live_stats["new"]))
    verdict = "PASS" if same == total else "FAIL"
    print("REPRO same reply live vs cold: %d of %d turns (%s)" % (same, total, verdict))
    return 0 if same == total else 1


if __name__ == "__main__":
    sys.exit(main())
