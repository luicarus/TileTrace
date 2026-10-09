// Real analyzer output keeps presentation tests tied to a private fixture.
const fs = require('node:fs');
const path = require('node:path');
const {spawnSync} = require('node:child_process');
let cached;
module.exports = function broadcastAnalysis() {
  if (cached) return structuredClone(cached);
  const root = path.resolve(__dirname, '../..');
  const venv = path.join(root, '.venv', process.platform === 'win32' ? 'Scripts/python.exe' : 'bin/python');
  const python = fs.existsSync(venv) ? venv : 'python';
  const code = 'import sys,json; from pathlib import Path; sys.path.insert(0,sys.argv[1]); from tiletrace import analyze; print(json.dumps(analyze(Path(sys.argv[2]).read_text(encoding="utf-8"),kernel="broadcast_demo",document_id="demo.py",version=1)))';
  const result = spawnSync(python, ['-I', '-S', '-c', code, root, path.join(__dirname, 'fixtures/broadcast.py')], {encoding:'utf8', cwd:root, windowsHide:true, timeout:10000});
  if (result.status !== 0) throw new Error(result.stderr || result.error?.message || 'Cannot analyze demo');
  cached = JSON.parse(result.stdout);
  return structuredClone(cached);
};
