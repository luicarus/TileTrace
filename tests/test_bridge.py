import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tiletrace.bridge import Bridge


SOURCE = """import triton
import triton.language as tl
@triton.jit
def kernel(BLOCK: tl.constexpr):
    x = tl.arange(0, BLOCK)
    y = tl.reshape(x, (2, 4))
    z = tl.trans(y)
"""


class BridgeTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.bridge = Bridge(self.directory.name)

    def analysis(self):
        return self.bridge.dispatch("analyze", {"source": SOURCE, "parameters": {"BLOCK": 8},
                                               "document_id": "kernel.py", "version": 3})

    def test_analyze_and_inspect_use_same_graph(self):
        analysis = self.analysis()
        node = next(n for n in analysis["nodes"] if n["name"] == "z")
        result = self.bridge.dispatch("inspect", {"analysis": analysis, "node_id": node["id"], "index": [1, 0]})
        self.assertEqual(result["status"], "exact")
        self.assertEqual(result["origins"][0]["indices"], [[0, 1]])

    def test_publishes_context_only_when_explicitly_requested(self):
        analysis = self.analysis()
        self.assertEqual(self.bridge.dispatch("get_context", {} )["sessions"], [])
        node = next(n for n in analysis["nodes"] if n["name"] == "z")
        context = {"document_id": "kernel.py", "version": 3, "analysis": analysis,
                   "selected_node_id": node["id"]}
        self.bridge.dispatch("sync_context", {"session_id": "editor-1", "context": context})
        self.assertEqual(self.bridge.dispatch("get_context", {"session_id": "editor-1"})["selected_node_id"], node["id"])
        inspected = self.bridge.dispatch("inspect", {"session_id": "editor-1", "node_id": node["id"], "index": [1, 0]})
        self.assertEqual(inspected["status"], "exact")

    def test_rejects_stale_session_inspection(self):
        analysis = self.analysis()
        self.bridge.dispatch("sync_context", {"session_id": "editor-1", "context": {
            "document_id": "kernel.py", "version": 4, "analysis": analysis, "stale": True}})
        with self.assertRaisesRegex(ValueError, "stale"):
            self.bridge.dispatch("inspect", {"session_id": "editor-1", "node_id": "irrelevant"})

    def test_cannot_publish_analysis_for_wrong_document(self):
        with self.assertRaises(ValueError):
            self.bridge.dispatch("sync_context", {"session_id": "editor-1", "context": {
                "document_id": "different.py", "version": 3, "analysis": self.analysis()}})

    def test_unknown_method_returns_clear_error(self):
        with self.assertRaisesRegex(ValueError, "Unknown method"):
            self.bridge.dispatch("execute_python", {})

    def test_worker_recovers_after_invalid_json(self):
        requests = "not json\n" + json.dumps({"id": "good", "method": "analyze", "params": {
            "source": SOURCE, "parameters": {"BLOCK": 8}}}) + "\n"
        result = subprocess.run([sys.executable, "-m", "tiletrace", "worker", "--session-dir", self.directory.name],
                                input=requests, text=True, capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        lines = [json.loads(line) for line in result.stdout.splitlines()]
        self.assertIn("error", lines[0])
        self.assertEqual(lines[1]["id"], "good")
        self.assertEqual(lines[1]["result"]["kernel"], "kernel")

    def test_worker_recovers_after_invalid_ids_and_nonfinite_numbers(self):
        invalid = ['NaN', 'Infinity', '-Infinity', '1e999', 'true', '[]', '{}']
        requests = []
        for request_id in invalid:
            for method in ('get_context', 'unknown'):
                requests.append('{"id":' + request_id + ',"method":"' + method + '"}')
        requests.append('{"id":"bad-parameter","method":"analyze","params":{"source":"",'
                        '"parameters":{"BLOCK":1e999}}}')
        requests.append(json.dumps({"id": "good-中文", "method": "get_context"}, ensure_ascii=False))
        result = subprocess.run([sys.executable, "-m", "tiletrace", "worker", "--session-dir", self.directory.name],
                                input='\n'.join(requests) + '\n', encoding='utf-8', capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        lines = [json.loads(line) for line in result.stdout.splitlines()]
        self.assertEqual(len(lines), len(requests))
        for response in lines[:-1]:
            self.assertIn("error", response)
        for response in lines[:-2]:
            self.assertIsNone(response["id"])
        self.assertEqual(lines[-1]["id"], "good-中文")
        self.assertEqual(lines[-1]["result"]["sessions"], [])

    def test_cli_rejects_oversized_source_before_parsing(self):
        path = Path(self.directory.name) / "oversized.py"
        # Write bounded chunks; invalid syntax ensures parsing would produce a
        # different error if the common source validation were bypassed.
        with path.open('w', encoding='utf-8') as stream:
            stream.write('!')
            for _ in range(100):
                stream.write(' ' * 10_000)
        result = subprocess.run([sys.executable, "-m", "tiletrace", "analyze", str(path)],
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 2, result.stderr)
        error = json.loads(result.stdout)["error"]
        self.assertEqual(error["type"], "ValueError")
        self.assertIn("one-million-character", error["message"])

    def test_cli_reads_example_without_importing_triton(self):
        path = Path(self.directory.name) / "example.py"
        path.write_text(SOURCE, encoding="utf-8")
        result = subprocess.run([sys.executable, "-m", "tiletrace", "analyze", str(path),
                                 "--parameters", '{"BLOCK":8}'], capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        analysis = json.loads(result.stdout)
        self.assertTrue(any(n["name"] == "z" and n["shape"] == [4, 2] for n in analysis["nodes"]))


@unittest.skipUnless(os.name == 'nt', 'Windows setup script')
class SetupScriptTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.project = Path(self.directory.name) / "O'Brien 中文"
        (self.project / 'scripts').mkdir(parents=True)
        shutil.copyfile(Path(__file__).resolve().parents[1] / 'scripts' / 'setup.ps1',
                        self.project / 'scripts' / 'setup.ps1')
        interpreter = self.project / '.venv' / 'Scripts' / 'python.exe'
        interpreter.parent.mkdir(parents=True)
        interpreter.touch()
        self.configuration = self.project / '.codex' / 'config.toml'

    def run_setup(self):
        # Intercept only package installation; run the actual configuration
        # logic in its copied checkout with real filesystem/path operations.
        command = ('function global:.venv/Scripts/python.exe { $global:LASTEXITCODE = 0 }; '
                   '& $env:TILETRACE_TEST_SETUP -SkipExtension')
        result = subprocess.run(['powershell.exe', '-NoProfile', '-NonInteractive', '-Command', command],
                                env={**os.environ, 'TILETRACE_TEST_SETUP': str(self.project / 'scripts' / 'setup.ps1')},
                                capture_output=True, encoding='utf-8', errors='replace', timeout=15,
                                creationflags=subprocess.CREATE_NO_WINDOW)
        self.assertEqual(result.returncode, 0, result.stderr)
        return result

    def test_setup_preserves_existing_generated_and_user_configuration(self):
        for prefix in ('# Project-local server. Generated by scripts/setup.ps1\n', '# My configuration\n'):
            with self.subTest(prefix=prefix):
                original = (prefix + '[unrelated]\nsetting = "keep"\n').encode('utf-8')
                self.configuration.parent.mkdir(exist_ok=True)
                self.configuration.write_bytes(original)
                result = self.run_setup()
                self.assertEqual(self.configuration.read_bytes(), original)
                self.assertIn('Existing Codex configuration preserved', result.stdout)

    def test_setup_generated_paths_are_valid_toml_without_bom(self):
        try:
            import tomllib
        except ImportError:
            self.skipTest('TOML quoting verification requires Python 3.11+')
        self.run_setup()
        content = self.configuration.read_bytes()
        self.assertFalse(content.startswith(b'\xef\xbb\xbf'))
        server = tomllib.loads(content.decode('utf-8'))['mcp_servers']['tiletrace']
        self.assertEqual(Path(server['cwd']).resolve(), self.project.resolve())
        self.assertEqual(Path(server['command']).resolve(), (self.project / '.venv/Scripts/python.exe').resolve())
        self.assertEqual(server['args'], ['-m', 'tiletrace', 'mcp', '--session-dir', '.tiletrace'])


if __name__ == "__main__":
    unittest.main()
