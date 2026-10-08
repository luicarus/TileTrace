const test=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const {spawnSync}=require('node:child_process');
const {operationNodes,operationId,contextInputs}=require('../media/operations');
const {selectAt}=require('../out/state');
function analyze(source,parameters={}) {
 const root=path.resolve(__dirname,'../..');const python=path.join(root,'.venv',process.platform==='win32'?'Scripts/python.exe':'bin/python');
 const code='import json,sys; sys.path.insert(0,sys.argv[1]); from tiletrace import analyze; print(json.dumps(analyze(sys.argv[2], parameters=json.loads(sys.argv[3]))))';
 const result=spawnSync(fs.existsSync(python)?python:'python',['-I','-S','-c',code,root,source,JSON.stringify(parameters)],{cwd:root,encoding:'utf8',windowsHide:true,timeout:10000});
 assert.equal(result.status,0,result.stderr);return JSON.parse(result.stdout);
}
test('unsupported scalar arithmetic remains a step and cannot fold into unrelated parameters',()=>{
 const result=analyze('import triton\nimport triton.language as tl\n@triton.jit\ndef k():\n    bad = 1 / 0\n    x = tl.arange(0, 8)\n');
 const bad=result.nodes.find(n=>n.name==='bad');assert.equal(bad.status,'unsupported');
 assert.ok(operationNodes(result.nodes).some(n=>n.id===bad.id));assert.equal(operationId(result.nodes,bad.id),bad.id);
 assert.equal(selectAt(result.nodes,bad.source.start_line,bad.source.start_col,undefined),bad.id);
 const fake={...bad,id:'use',op:'add',inputs:[bad.id],shape:[8]};assert.equal(contextInputs(fake,[...result.nodes,fake]).length,0);
});
for(const source of [
 'import triton\nimport triton.language as tl\n@triton.jit\ndef k(A: tl.constexpr=4, B: tl.constexpr=4*2):\n    x=tl.arange(0,A)\n    y=tl.arange(0,B)\n',
 'import triton\nimport triton.language as tl\n@triton.jit\ndef k(\n    A: tl.constexpr=4,\n    B: tl.constexpr=\n        8\n):\n    x=tl.arange(0,A)\n    y=tl.arange(0,B)\n'
]) test('default source ownership follows the exact expression, including compound and multiline defaults',()=>{
 const result=analyze(source);const b=result.nodes.find(n=>n.name==='B');const y=result.nodes.find(n=>n.name==='y');
 assert.ok(b.attrs.default_source);
 const span=b.attrs.default_source;
 assert.equal(selectAt(result.nodes,span.start_line,span.start_col,undefined,source),y.id);
 assert.equal(selectAt(result.nodes,span.end_line,span.end_col-1,undefined,source),y.id);
 const overridden=analyze(source,{B:16});assert.equal(selectAt(overridden.nodes,span.start_line,span.start_col,undefined,source),overridden.nodes.find(n=>n.name==='y').id);
});
