const test = require('node:test');
const assert = require('node:assert/strict');
const {SelectionState, selectAt, validateParameters} = require('../out/state');
const node = (id, start, end, name=id) => ({id,name,op:'arange',inputs:[],shape:[8],status:'resolved',source:{start_line:1,start_col:start,end_line:1,end_col:end},attrs:{},dtype:null});
const analysis = (document_id='file:///a.py',version=1,nodes=[node('outer',0,20),node('inner',5,10)]) => ({document_id,version,kernel:'k',kernels:['k'],nodes,diagnostics:[],missing_parameters:[],parameters:{}});
test('selection picks smallest containing expression and blank lines preserve chosen node', () => {
 assert.equal(selectAt(analysis().nodes,1,7,'outer'),'inner');
 assert.equal(selectAt(analysis().nodes,2,0,'inner'),'inner');
 assert.equal(selectAt(analysis().nodes,1,10,'outer'),'outer');
});
test('assignment target selection picks its output while RHS still picks the most specific expression',()=>{
 const source='    matrix = tl.reshape(x, (2, 4))\n    matrix_other = matrix\n';
 const outer=node('reshape',13,34,'matrix');const inner=node('x',24,25,'x');
 assert.equal(selectAt([inner,outer],1,5,'x',source),'reshape');
 assert.equal(selectAt([inner,outer],1,24,'reshape',source),'x');
 assert.equal(selectAt([inner,outer],1,11,'x',source),'x');
 assert.equal(selectAt([inner,outer],2,5,'x',source),'x');
});
test('old version and generation cannot replace accepted document', () => {
 const s=new SelectionState(); const old=s.begin('file:///a.py',1); const newer=s.begin('file:///a.py',2);
 assert.equal(s.accept(old,analysis()),false); assert.equal(s.accept(newer,analysis('file:///a.py',2)),true);
 const inspect=s.inspectionToken(); s.select('inner'); assert.equal(s.matchesInspection(inspect),false);
 const pending=s.begin('file:///b.py',1); assert.equal(s.stale,true); assert.equal(s.analysis,undefined);
 assert.equal(s.accept(newer,analysis('file:///a.py',2)),false); assert.equal(s.accept(pending,analysis('file:///b.py',1)),true);
});
test('inspection invalidated by edits and node remaps by name after analysis', () => {
 const s=new SelectionState(); s.accept(s.begin('file:///a.py',1),analysis()); s.select('inner'); const token=s.inspectionToken();
 const next=s.begin('file:///a.py',2); assert.equal(s.matchesInspection(token),false);
 s.accept(next,analysis('file:///a.py',2,[node('renamed',5,10,'inner')])); assert.equal(s.selected,'renamed');
});
test('JSON options reject arrays for dictionaries and noninteger program IDs', () => {
 assert.equal(validateParameters({parameters:{BLOCK:8},input_shapes:{x:[2,4]},program_ids:[0,1]}).parameters.BLOCK,8);
 assert.throws(()=>validateParameters({parameters:[],program_ids:[]}));
 assert.throws(()=>validateParameters({parameters:{},program_ids:[0.5]}));
 assert.throws(()=>validateParameters({parameters:{},input_shapes:{x:[0]}}));
});

test('default selection promotes literals and parameter defaults to tensor operations, detailed view preserves literals',()=>{
 const demo=require('./operations-fixture.cjs')();
 const rows=demo.nodes.find(n=>n.name==='rows');const cols=demo.nodes.find(n=>n.name==='cols');
 const zero=demo.nodes.find(n=>n.op==='constant'&&n.source.start_line===rows.source.start_line);
 assert.equal(selectAt(demo.nodes,zero.source.start_line,zero.source.start_col,undefined),'n5');
 assert.equal(selectAt(demo.nodes,zero.source.start_line,zero.source.start_col,undefined,undefined,true),zero.id);
 const defaultFour=demo.nodes.find(n=>n.op==='constant'&&n.attrs.value===4);
 assert.equal(selectAt(demo.nodes,defaultFour.source.start_line,defaultFour.source.start_col,undefined),cols.id);
 const axis=demo.nodes.find(n=>n.op==='constant'&&n.attrs.value===1);
 assert.equal(selectAt(demo.nodes,axis.source.start_line,axis.source.start_col,undefined),demo.nodes.find(n=>n.name==='row_sum').id);
 const s=new SelectionState();s.accept(s.begin('demo.py',1),demo);assert.equal(s.selected,rows.id);
 const detailed=new SelectionState();detailed.accept(detailed.begin('demo.py',1),demo,true);assert.equal(detailed.selected,demo.nodes[0].id);
});
