"""tools/lemonade/setup.sh against a stand-in for Lemonade's /internal API.

The stand-in keeps a config dict with the same merge rule as Lemonade's
/internal/set (objects merge key by key, other values replace). setup.sh
refuses to run anywhere but Linux x86-64, so these tests run only there.
"""

import http.server
import json
import os
import platform
import socket
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SETUP = ROOT / "tools" / "lemonade" / "setup.sh"
CATALOG = "https://raw.githubusercontent.com/kingjones30/superroc/main/docs/lemonade/catalog.json"
OTHER = {"id": "mine", "url": "https://example.org/catalog.json"}


class _Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _reply(self, obj):
        data = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path != "/internal/config":
            self.send_error(404)
            return
        self._reply(self.server.config)

    def do_POST(self):
        if self.path != "/internal/set":
            self.send_error(404)
            return
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.posts.append(body)
        for key, value in body.items():
            if isinstance(value, dict) and isinstance(self.server.config.get(key), dict):
                self.server.config[key].update(value)
            else:
                self.server.config[key] = value
        self._reply({"status": "success", "updated": body})


@unittest.skipUnless(sys.platform.startswith("linux") and platform.machine() == "x86_64",
                     "setup.sh runs on Linux x86-64 only")
class SetupScriptTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.prefix = self.dir / "prefix"
        self.binary = self.dir / "superroc"
        self.binary.write_text("#!/bin/sh\necho 'Super Roc test build'\n")
        self.binary.chmod(0o755)
        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self.server.config = {
            "llamacpp": {"backend": "auto", "rocm_bin": "builtin", "vulkan_bin": "builtin"},
            "model_catalogs": [dict(OTHER)],
        }
        self.server.posts = []
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = "http://127.0.0.1:%d" % self.server.server_address[1]

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.tmp.cleanup()

    def run_setup(self, *args, url=None):
        env = {k: v for k, v in os.environ.items() if not k.startswith("LEMONADE_")}
        cmd = ["bash", str(SETUP), "--force", "--prefix", str(self.prefix),
               "--lemonade-url", url or self.url, *args]
        return subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=120)

    def catalogs(self):
        return self.server.config["model_catalogs"]

    def test_install_keeps_other_catalogs_and_repeats_cleanly(self):
        r = self.run_setup("--superroc", str(self.binary))
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        launcher = str(self.prefix / "lemonade")
        self.assertEqual(self.server.config["llamacpp"]["rocm_bin"], launcher)
        self.assertEqual(self.server.config["llamacpp"]["vulkan_bin"], "builtin")
        self.assertEqual(self.catalogs(), [OTHER, {"id": "superroc", "url": CATALOG}])
        for name in ("bin/superroc", "lemonade/llama-server", "lemonade/superroc-models.json"):
            self.assertTrue((self.prefix / name).is_file(), name)

        # Again, from the installed binary: still one Super Roc entry.
        r = self.run_setup("--superroc", str(self.prefix / "bin" / "superroc"))
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual(self.catalogs(), [OTHER, {"id": "superroc", "url": CATALOG}])

    def test_undo_restores_builtin_and_keeps_other_catalogs(self):
        self.assertEqual(self.run_setup("--superroc", str(self.binary)).returncode, 0)
        r = self.run_setup("--undo")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual(self.server.config["llamacpp"]["rocm_bin"], "builtin")
        self.assertEqual(self.catalogs(), [OTHER])

    def test_dry_run_changes_nothing(self):
        before = json.loads(json.dumps(self.server.config))
        r = self.run_setup("--superroc", str(self.binary), "--dry-run")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual(self.server.config, before)
        self.assertEqual(self.server.posts, [])
        self.assertFalse(self.prefix.exists())

    def test_without_lemonade_installs_and_says_how_to_finish(self):
        probe = socket.socket()
        probe.bind(("127.0.0.1", 0))
        dead = "http://127.0.0.1:%d" % probe.getsockname()[1]
        probe.close()
        r = self.run_setup("--superroc", str(self.binary), url=dead)
        self.assertEqual(r.returncode, 3, r.stdout + r.stderr)
        self.assertTrue((self.prefix / "bin" / "superroc").is_file())
        self.assertIn("--superroc %s" % (self.prefix / "bin" / "superroc"), r.stderr)


if __name__ == "__main__":
    unittest.main()
