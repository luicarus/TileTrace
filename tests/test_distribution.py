"""Acceptance checks for the standalone Python and VS Code artifacts."""

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VSIX = ROOT / "dist/tiletrace-0.1.0.vsix"
WHEEL = ROOT / "dist/tiletrace-0.1.0-py3-none-any.whl"


@unittest.skipUnless(VSIX.exists(), "Build the VSIX with npm run package in extension/ first")
class ExtensionDistributionTests(unittest.TestCase):
    def test_installed_extension_contains_current_worker_and_entry_point(self):
        with zipfile.ZipFile(VSIX) as archive:
            names = set(archive.namelist())
            manifest = json.loads(archive.read("extension/package.json"))
            self.assertIn("extension/" + manifest["main"].removeprefix("./"), names)
            for source in (ROOT / "tiletrace").glob("*.py"):
                bundled = archive.read("extension/python/tiletrace/" + source.name)
                self.assertEqual(hashlib.sha256(bundled).hexdigest(),
                                 hashlib.sha256(source.read_bytes()).hexdigest(), source.name)
            self.assertFalse(any("node_modules/" in name or "/.venv/" in name or "__pycache__" in name for name in names))

    def test_packaged_worker_ignores_workspace_modules_and_startup_environment(self):
        with tempfile.TemporaryDirectory() as directory, zipfile.ZipFile(VSIX) as archive:
            target = Path(directory).resolve()
            for name in archive.namelist():
                if not (name.startswith("extension/python/tiletrace/") or name == "extension/out/launch.js") or name.endswith("/"):
                    continue
                destination = (target / name).resolve()
                self.assertTrue(destination.is_relative_to(target))
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(archive.read(name))
            workspace = target / "unrelated-workspace"
            workspace.mkdir()
            sentinels = []
            for module in ("tiletrace", "json", "sitecustomize"):
                sentinel = workspace / (module + ".executed")
                sentinels.append(sentinel)
                (workspace / (module + ".py")).write_text(
                    "from pathlib import Path\nPath(" + repr(str(sentinel)) + ").write_text('EXECUTED')\n",
                    encoding="utf-8")
            source = ("import triton\nimport triton.language as tl\n@triton.jit\n"
                      "def demo():\n    x = tl.arange(0, 8)\n    y = x.reshape((2,4))\n    z = tl.trans(y)\n")
            request = {"id": "distribution", "method": "analyze", "params": {
                "source": source, "document_id": "standalone.py", "version": 99}}
            # Exercise the exact production launch helper extracted from the VSIX.
            helper_script = (
                "const {workerLaunchOptions}=require(process.argv[1]);"
                "const o=workerLaunchOptions(process.argv[2],process.argv[3],process.argv[4],()=>{});"
                "process.stdout.write(JSON.stringify({command:o.command,args:o.args,cwd:o.cwd}));")
            helper = subprocess.run(["node", "-e", helper_script, str(target / "extension/out/launch.js"),
                                     sys.executable, str(target / "extension/python"), str(workspace / "sessions")],
                                    cwd=target, text=True, capture_output=True, timeout=10)
            self.assertEqual(helper.returncode, 0, helper.stderr)
            launch = json.loads(helper.stdout)
            self.assertEqual(Path(launch["cwd"]), target / "extension/python")
            self.assertIn("-I", launch["args"])
            self.assertIn("-S", launch["args"])
            environment = dict(os.environ, PYTHONPATH=str(workspace), PYTHONHOME=str(workspace))
            result = subprocess.run([launch["command"], *launch["args"]],
                                    cwd=launch["cwd"], env=environment, input=json.dumps(request) + "\n",
                                    text=True, capture_output=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            for sentinel in sentinels:
                self.assertFalse(sentinel.exists(), sentinel.name)
            response = json.loads(result.stdout)
            self.assertEqual(response["id"], "distribution")
            self.assertEqual(response["result"]["version"], 99)
            transposed = next(n for n in response["result"]["nodes"] if n["name"] == "z")
            self.assertEqual(transposed["shape"], [4, 2])


@unittest.skipUnless(WHEEL.exists(), "Build the Python wheel with pip wheel --no-deps --wheel-dir dist . first")
class PythonDistributionTests(unittest.TestCase):
    def test_python_wheel_contains_current_modules_and_optional_mcp_extra(self):
        with zipfile.ZipFile(WHEEL) as archive:
            for source in (ROOT / "tiletrace").glob("*.py"):
                self.assertEqual(archive.read("tiletrace/" + source.name), source.read_bytes(), source.name)
            metadata_name = next(n for n in archive.namelist() if n.endswith(".dist-info/METADATA"))
            metadata = archive.read(metadata_name).decode("utf-8")
            self.assertIn("Provides-Extra: mcp", metadata)


if __name__ == "__main__":
    unittest.main()
