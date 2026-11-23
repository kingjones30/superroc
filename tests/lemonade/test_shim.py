"""DRYRUN tests for tools/lemonade/llama-server (the Super Roc shim).

Run from the repo root: python3 -m unittest discover tests/lemonade
"""

import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SHIM_SRC = ROOT / "tools" / "lemonade" / "llama-server"
SUPERROC_BIN = "/opt/superroc/bin/superroc"
STOCK_BIN = "/stock/llama-server"


def _gguf_string(text):
    data = text.encode()
    return struct.pack("<Q", len(data)) + data


def write_gguf(path, arch):
    """Minimal GGUF v3 header; arch sits behind other KV types on purpose."""
    kvs = b""
    kvs += _gguf_string("general.name") + struct.pack("<I", 8) + _gguf_string("tiny")
    kvs += _gguf_string("some.count") + struct.pack("<I", 4) + struct.pack("<I", 7)
    kvs += (_gguf_string("some.scores") + struct.pack("<I", 9)
            + struct.pack("<I", 6) + struct.pack("<Q", 2)
            + struct.pack("<ff", 0.5, 1.5))
    kvs += _gguf_string("general.architecture") + struct.pack("<I", 8) + _gguf_string(arch)
    with open(path, "wb") as fh:
        fh.write(b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0)
                 + struct.pack("<Q", 4) + kvs)


def write_raw(path, kv_count, kv_bytes):
    """GGUF v3 shell with caller-supplied kv count + bytes (cap tests)."""
    with open(path, "wb") as fh:
        fh.write(b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0)
                 + struct.pack("<Q", kv_count) + kv_bytes)


def _arch_kv(arch):
    return (_gguf_string("general.architecture") + struct.pack("<I", 8)
            + _gguf_string(arch))


class ShimTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="shim-test-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.shim = self.tmp / "llama-server"
        shutil.copy(SHIM_SRC, self.shim)
        (self.tmp / "superroc-models.json").write_text(json.dumps({
            "per-model.gguf": ["--think", "off"],
            "unsafe.gguf": ["--think", "off", "--host", "0.0.0.0",
                            "-p", "9999", "--trace", "--api-key", "secret",
                            "--cache-disk=/tmp/cache", "-m", "/evil.gguf"],
        }))
        self.model = self.tmp / "Model-Q4.gguf"
        write_gguf(self.model, "qwen35")

    def run_shim_full(self, args, extra_env=None):
        env = {k: v for k, v in os.environ.items() if not k.startswith("SUPERROC_")}
        env.update({
            "SUPERROC_SHIM_DRYRUN": "1",
            "SUPERROC_BIN": SUPERROC_BIN,
            "SUPERROC_FALLBACK_LLAMA_SERVER": STOCK_BIN,
        })
        if extra_env:
            env.update(extra_env)
        proc = subprocess.run(
            [sys.executable, str(self.shim)] + [str(a) for a in args],
            capture_output=True, text=True, env=env)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        return json.loads(proc.stdout), proc.stderr

    def run_shim(self, args, extra_env=None):
        return self.run_shim_full(args, extra_env)[0]

    def superroc_prefix(self, model, served):
        return [SUPERROC_BIN, "serve", "-i", "127.0.0.1", "-p", "8080",
                "-j", "1", "llm", "-m", str(model),
                "--served-model-name", served, "--accept-any-model-name"]

    def test_superroc_basic(self):
        out = self.run_shim(["-m", self.model])
        self.assertEqual(out["mode"], "superroc")
        self.assertEqual(out["argv"], self.superroc_prefix(self.model, "Model-Q4"))

    def test_fallback_gemma3(self):
        gemma = self.tmp / "gemma.gguf"
        write_gguf(gemma, "gemma3")
        out = self.run_shim(["-m", gemma, "--jinja"])
        self.assertEqual(out["mode"], "fallback")
        self.assertEqual(out["argv"], [STOCK_BIN, "-m", str(gemma), "--jinja"])

    def test_fallback_unreadable(self):
        bad = self.tmp / "bad.gguf"
        bad.write_bytes(b"not a gguf file at all")
        self.assertEqual(self.run_shim(["-m", bad])["mode"], "fallback")
        missing = self.tmp / "missing.gguf"
        self.assertEqual(self.run_shim(["-m", missing])["mode"], "fallback")

    def test_mtp_mapping(self):
        out = self.run_shim(["-m", self.model, "--spec-type", "draft-mtp",
                             "--spec-draft-n-max", "3"])
        self.assertEqual(out["argv"],
                         self.superroc_prefix(self.model, "Model-Q4")
                         + ["--speculative", "mtp", "-d", "3"])

    def test_dflash_mapping(self):
        out = self.run_shim(["-m", self.model, "--spec-type", "draft-dflash",
                             "--model-draft", "/draft.gguf"])
        self.assertEqual(out["argv"],
                         self.superroc_prefix(self.model, "Model-Q4")
                         + ["--speculative", "dflash2",
                            "--dflash-model", "/draft.gguf"])

    def test_ctx_host_port_mmproj(self):
        out = self.run_shim(["-m", self.model, "-c", "32768",
                             "--host", "0.0.0.0", "--port", "18080",
                             "--mmproj", "/mm.gguf"])
        self.assertEqual(out["argv"],
                         [SUPERROC_BIN, "serve", "-i", "0.0.0.0", "-p", "18080",
                          "-j", "1", "llm", "-m", str(self.model),
                          "--served-model-name", "Model-Q4",
                          "--accept-any-model-name",
                          "-c", "32768", "--mmproj", "/mm.gguf"])
        out0 = self.run_shim(["-m", self.model, "-c", "0"])
        self.assertNotIn("-c", out0["argv"])

    def test_unknown_flag_fallback(self):
        out = self.run_shim(["-m", self.model, "--some-unknown-flag"])
        self.assertEqual(out["mode"], "fallback")

    def test_embeddings_fallback(self):
        out = self.run_shim(["-m", self.model, "--embeddings"])
        self.assertEqual(out["mode"], "fallback")
        out2 = self.run_shim(["-m", self.model, "--pooling", "none"])
        self.assertEqual(out2["mode"], "fallback")

    def test_per_model_and_extra_args(self):
        pm = self.tmp / "per-model.gguf"
        write_gguf(pm, "qwen35")
        out = self.run_shim(["-m", pm],
                            {"SUPERROC_EXTRA_ARGS": "--foo bar"})
        self.assertEqual(out["argv"],
                         self.superroc_prefix(pm, "per-model")
                         + ["--think", "off", "--foo", "bar"])

    def test_header_caps_fallback(self):
        # kv_count over the 65536 cap, with a valid arch KV after it.
        p = self.tmp / "cap-kv.gguf"
        write_raw(p, 65537, _arch_kv("qwen35"))
        self.assertEqual(self.run_shim(["-m", p])["mode"], "fallback")
        # Key string length over the 1 MiB cap.
        p = self.tmp / "cap-str.gguf"
        write_raw(p, 1, struct.pack("<Q", (1 << 20) + 1) + b"x" * 8)
        self.assertEqual(self.run_shim(["-m", p])["mode"], "fallback")
        # Array element count over the 1<<20 cap, before the arch KV.
        arr = (_gguf_string("some.scores") + struct.pack("<I", 9)
               + struct.pack("<I", 4) + struct.pack("<Q", (1 << 20) + 1))
        p = self.tmp / "cap-arr.gguf"
        write_raw(p, 2, arr + _arch_kv("qwen35"))
        self.assertEqual(self.run_shim(["-m", p])["mode"], "fallback")

    def test_unsafe_args_dropped(self):
        um = self.tmp / "unsafe.gguf"
        write_gguf(um, "qwen35")
        out, err = self.run_shim_full(
            ["-m", um],
            {"SUPERROC_EXTRA_ARGS": "--port 9998 --host=0.0.0.0 --foo bar"})
        self.assertEqual(out["argv"],
                         self.superroc_prefix(um, "unsafe")
                         + ["--think", "off", "--foo", "bar"])
        self.assertIn("dropped unsafe arg --host from superroc-models.json",
                      err)
        self.assertIn("dropped unsafe arg --port from SUPERROC_EXTRA_ARGS",
                      err)

    def test_unsafe_args_all_forms_dropped(self):
        um = self.tmp / "unsafe2.gguf"
        write_gguf(um, "qwen35")
        out, err = self.run_shim_full(
            ["-m", um],
            {"SUPERROC_EXTRA_ARGS": "--api-key-file /tmp/k --api-key-file=/tmp/k "
                                    "--model /tmp/x.gguf -i 0.0.0.0 --foo bar"})
        self.assertEqual(out["argv"],
                         self.superroc_prefix(um, "unsafe2")
                         + ["--foo", "bar"])
        for flag in ("--api-key-file", "--model", "-i"):
            self.assertIn(f"dropped unsafe arg {flag} from SUPERROC_EXTRA_ARGS",
                          err)

    def test_lemonade_api_key_falls_back(self):
        out = self.run_shim(["-m", self.model, "--api-key", "secret"])
        self.assertEqual(out["mode"], "fallback")

    def test_version(self):
        proc = subprocess.run([sys.executable, str(self.shim), "--version"],
                              capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0)
        self.assertEqual(proc.stdout, "version: 1 (super-roc)\n")


if __name__ == "__main__":
    unittest.main()
