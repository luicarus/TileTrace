const test=require('node:test');const assert=require('node:assert/strict');const fs=require('node:fs');const path=require('node:path');const {JSDOM}=require('jsdom');
const node=(id,shape,inputs=[])=>({id,name:id,op:'reshape',shape,inputs,status:'resolved',source:{start_line:1,start_col:0,end_line:1,end_col:10},attrs:{},dtype:null});
function setup(nodes) {const messages=[];const dom=new JSDOM('<main id="app"></main>',{runScripts:'outside-only'});dom.window.acquireVsCodeApi=()=>({postMessage:m=>messages.push(m),getState:()=>null,setState:()=>{}});dom.window.eval(fs.readFileSync(path.join(__dirname,'../media/operations.js'),'utf8'));dom.window.eval(fs.readFileSync(path.join(__dirname,'../media/viewer.js'),'utf8'));const send=data=>dom.window.dispatchEvent(new dom.window.MessageEvent('message',{data}));send({type:'state',state:{analysis:{nodes,kernels:['k'],kernel:'k',diagnostics:[],missing_parameters:[]},selected:nodes[nodes.length-1].id,stale:false,file:'a.py',version:1,parameters:{},input_shapes:{},program_ids:[]}});return {dom,messages,send};}
const inspectionResponse=(request,output,inspection)=>({type:'inspection',request_id:request.request_id,generation:request.generation,node_id:request.node_id,inspection:{node:output,...inspection}});
test('output click requests exact index and highlights immediate input origin',()=>{const {dom,messages,send}=setup([node('in',[8]),node('out',[2,4],['in'])]);dom.window.document.querySelector('[data-output="true"][data-index="[0,1]"]').click();assert.equal(messages.at(-1).type,'inspectIndex');assert.deepEqual(JSON.parse(JSON.stringify(messages.at(-1).index)),[0,1]);send(inspectionResponse(messages.at(-1),node('out',[2,4],['in']),{status:'exact',output_index:[0,1],origins:[{node_id:'in',indices:[[1]],total:1,truncated:false}]}));assert.ok(dom.window.document.querySelector('[data-node="in"][data-index="[1]"]').classList.contains('origin'));send({type:'state',state:{stale:true,file:'a.py',version:2}});assert.equal(dom.window.document.querySelectorAll('.origin').length,0);assert.equal(dom.window.document.querySelectorAll('[data-output="true"]').length,0);});
test('huge tensors stay capped, rank three supports prefix slices, symbolic tensors have no grid',()=>{const {dom,messages}=setup([node('x',[4,1000000,1000000])]);assert.ok(dom.window.document.querySelectorAll('[data-index]').length<=128);const slice=dom.window.document.querySelector('input[data-axis="0"]');assert.ok(slice);slice.value='3';slice.dispatchEvent(new dom.window.Event('change'));const cell=dom.window.document.querySelector('[data-output="true"]');cell.click();assert.equal(messages.at(-1).index[0],3);assert.match(dom.window.document.body.textContent,/显示.*总计/);const symbolic=setup([node('s',['BLOCK',4])]);assert.equal(symbolic.dom.window.document.querySelectorAll('[data-index]').length,0);assert.match(symbolic.dom.window.document.body.textContent,/符号/);});
test('malicious source text is rendered literally and unavailable mappings clear origins',()=>{const x=node('x',[1]);x.name='<img src=x onerror=alert(1)>';const {dom,messages,send}=setup([x]);assert.equal(dom.window.document.querySelectorAll('img').length,0);dom.window.document.querySelector('[data-output="true"]').click();send(inspectionResponse(messages.at(-1),x,{status:'unavailable',output_index:[0],origins:[],message:'symbolic mapping'}));assert.match(dom.window.document.body.textContent,/symbolic mapping/);});
test('off-slice origins are explained and navigation reveals the mapped input cell',()=>{
 const {dom,messages,send}=setup([node('in',[4,2,4]),node('out',[2,4],['in'])]);
 dom.window.document.querySelector('[data-output="true"]').click();
 send(inspectionResponse(messages.at(-1),node('out',[2,4],['in']),{status:'exact',output_index:[0,0],origins:[{node_id:'in',indices:[[3,1,2]],total:1,truncated:false}]}));
 assert.match(dom.window.document.body.textContent,/高亮 0 \/ 1/);assert.match(dom.window.document.body.textContent,/范围之外/);
 const jump=Array.from(dom.window.document.querySelectorAll('button')).find(b=>b.textContent==='定位首个来源坐标');jump.click();
 const cell=dom.window.document.querySelector('[data-node="in"][data-index="[3,1,2]"]');assert.ok(cell.classList.contains('origin'));assert.match(dom.window.document.body.textContent,/高亮 1 \/ 1/);
});
test('scalar output requests the empty coordinate and invalid JSON sends no apply request',()=>{
 const {dom,messages}=setup([node('scalar',[])]);dom.window.document.querySelector('[data-output="true"]').click();assert.deepEqual(JSON.parse(JSON.stringify(messages.at(-1).index)),[]);
 const input=dom.window.document.querySelector('textarea');input.value='{invalid';input.dispatchEvent(new dom.window.Event('input'));
 Array.from(dom.window.document.querySelectorAll('button')).find(b=>b.textContent==='应用参数并重新分析').click();assert.match(dom.window.document.body.textContent,/JSON 格式无效/);assert.equal(messages.some(m=>m.type==='applyParameters'),false);
});
test('new analysis resets reused node ID slices after shape shrink and rank change',()=>{
 const {dom,messages,send}=setup([node('x',[4,8,8])]);
 const prefix=dom.window.document.querySelector('input[data-axis="0"]');prefix.value='3';prefix.dispatchEvent(new dom.window.Event('change'));
 const state=(shape,generation,file)=>({type:'state',state:{analysis:{nodes:[node('x',shape)],kernels:['k'],kernel:'k',diagnostics:[],missing_parameters:[]},selected:'x',stale:false,file,version:2,generation,parameters:{},input_shapes:{},program_ids:[]}});
 send(state([2,8,8],2,'a.py'));dom.window.document.querySelector('[data-output="true"]').click();
 assert.deepEqual(JSON.parse(JSON.stringify(messages.at(-1).index)),[0,0,0]);
 send(state([8,8],3,'b.py'));dom.window.document.querySelector('[data-output="true"]').click();
 assert.deepEqual(JSON.parse(JSON.stringify(messages.at(-1).index)),[0,0]);
});
test('two-click and new-generation inspection responses require current browser token',()=>{
 const {dom,messages,send}=setup([node('x',[2])]);
 dom.window.document.querySelector('[data-output="true"][data-index="[0]"]').click();const first=messages.at(-1);
 dom.window.document.querySelector('[data-output="true"][data-index="[1]"]').click();const second=messages.at(-1);
 const response=(request,index)=>({type:'inspection',request_id:request.request_id,generation:request.generation,node_id:'x',inspection:{node:node('x',[2]),status:'exact',output_index:index,origins:[]}});
 send(response(first,[0]));assert.doesNotMatch(dom.window.document.querySelector('.mapping').textContent,/输出 \[0\]/);
 send({...response(first,[0]),inspection:{node:node('x',[2]),status:'unavailable',output_index:[0],origins:[],message:'old-error'}});assert.doesNotMatch(dom.window.document.querySelector('.mapping').textContent,/old-error/);
 send(response(second,[1]));assert.match(dom.window.document.querySelector('.mapping').textContent,/输出 \[1\]/);
 send({type:'state',state:{analysis:{nodes:[node('x',[2])],kernels:['k'],kernel:'k',diagnostics:[],missing_parameters:[]},selected:'x',stale:false,file:'b.py',version:2,generation:2,parameters:{},input_shapes:{},program_ids:[]}});
 dom.window.document.querySelector('[data-output="true"][data-index="[0]"]').click();const third=messages.at(-1);
 send(response(second,[1]));assert.doesNotMatch(dom.window.document.querySelector('.mapping').textContent,/输出 \[1\]/);
 send({...response(third,[0]),generation:1});assert.doesNotMatch(dom.window.document.querySelector('.mapping').textContent,/输出 \[0\] →/);
 send(response(third,[0]));assert.match(dom.window.document.querySelector('.mapping').textContent,/输出 \[0\] →/);
});
test('output slice changes clear current mappings and invalidate pending inspection',()=>{
 const {dom,messages,send}=setup([node('x',[4,2,4])]);dom.window.document.querySelector('[data-output="true"]').click();const first=messages.at(-1);
 const reply=inspectionResponse(first,node('x',[4,2,4]),{status:'exact',output_index:[0,0,0],origins:[]});send(reply);
 const slice=dom.window.document.querySelector('input[data-axis="0"]');slice.value='3';slice.dispatchEvent(new dom.window.Event('change'));
 assert.doesNotMatch(dom.window.document.querySelector('.mapping').textContent,/输出 \[0, 0, 0\]/);assert.equal(dom.window.document.querySelectorAll('.selected-cell').length,0);
 dom.window.document.querySelector('[data-output="true"]').click();const second=messages.at(-1);
 const nextSlice=dom.window.document.querySelector('input[data-axis="0"]');nextSlice.value='2';nextSlice.dispatchEvent(new dom.window.Event('change'));
 send(inspectionResponse(second,node('x',[4,2,4]),{status:'exact',output_index:[3,0,0],origins:[]}));assert.doesNotMatch(dom.window.document.querySelector('.mapping').textContent,/输出 \[3, 0, 0\]/);
});
test('input slice navigation retains current pending output query and its valid origins',()=>{
 const {dom,messages,send}=setup([node('in',[4,2,4]),node('out',[2,4],['in'])]);dom.window.document.querySelector('[data-output="true"]').click();const request=messages.at(-1);
 const slice=dom.window.document.querySelector('[data-card-node="in"] input[data-axis="0"]');slice.value='3';slice.dispatchEvent(new dom.window.Event('change'));
 send(inspectionResponse(request,node('out',[2,4],['in']),{status:'exact',output_index:[0,0],origins:[{node_id:'in',indices:[[3,1,2]],total:1,truncated:false}]}));
 assert.ok(dom.window.document.querySelector('[data-node="in"][data-index="[3,1,2]"]').classList.contains('origin'));assert.match(dom.window.document.querySelector('.mapping').textContent,/高亮 1 \/ 1/);
});

test('demo defaults to nine tensor steps and detailed toggle restores sixteen with matching navigation',()=>{
 const demo=require('./operations-fixture.cjs')();const {dom,messages,send}=setup(demo.nodes);
 assert.equal(dom.window.document.querySelectorAll('.operation').length,9);
 assert.deepEqual(Array.from(dom.window.document.querySelectorAll('.operation'),b=>b.dataset.node),demo.nodes.filter(n=>!['constant','parameter'].includes(n.op)).map(n=>n.id));
 const state={analysis:demo,selected:demo.nodes.find(n=>n.name==='rows').id,stale:false,file:'demo.py',version:1,generation:1,parameters:{},input_shapes:{},program_ids:[],show_all:false};
 send({type:'state',state});
 const next=Array.from(dom.window.document.querySelectorAll('button')).find(b=>b.textContent==='下一步 →');next.click();
 assert.equal(messages.at(-1).node_id,demo.nodes.find(n=>n.name==='cols').id);
 const toggle=dom.window.document.querySelector('input[data-role="show-all-nodes"]');assert.ok(toggle);toggle.checked=true;toggle.dispatchEvent(new dom.window.Event('change'));
 assert.equal(messages.at(-1).type,'setNodeVisibility');assert.equal(messages.at(-1).show_all,true);
 send({type:'state',state:{...state,show_all:true}});assert.equal(dom.window.document.querySelectorAll('.operation').length,16);
 const detailedNext=Array.from(dom.window.document.querySelectorAll('button')).find(b=>b.textContent==='下一步 →');detailedNext.click();
 assert.equal(messages.at(-1).node_id,demo.nodes[6].id);
});

test('arange folds its scalar bounds into readable attributes without singleton input grids',()=>{
 const demo=require('./operations-fixture.cjs')();const rows=demo.nodes.find(n=>n.name==='rows');const {dom,send}=setup(demo.nodes);
 send({type:'state',state:{analysis:demo,selected:rows.id,stale:false,file:'demo.py',version:1,generation:1,parameters:{},input_shapes:{},program_ids:[],show_all:false}});
 assert.equal(dom.window.document.querySelectorAll('.flow [data-card-node]').length,1);
 assert.match(dom.window.document.querySelector('.operation-details').textContent,/起点.*0/);
 assert.match(dom.window.document.querySelector('.operation-details').textContent,/终点.*BLOCK_M.*2/);
 assert.match(dom.window.document.querySelector('.operation-details').textContent,/长度.*2/);
 send({type:'state',state:{analysis:demo,selected:rows.id,stale:false,file:'demo.py',version:1,generation:1,parameters:{},input_shapes:{},program_ids:[],show_all:true}});
 assert.equal(dom.window.document.querySelectorAll('.flow [data-card-node]').length,3);
});

test('scalar reduction and its arithmetic remain visible while scalar index preparation folds',()=>{
 const make=(id,op,shape,inputs=[])=>({...node(id,shape,inputs),op});
 const nodes=[make('pid','program_id',[]),make('block','parameter',[]),make('start','mul',[],['pid','block']),make('x','arange',[8]),make('sum','sum',[],['x']),make('eps','constant',[]),make('denominator','add',[],['sum','eps']),make('result','div',[8],['x','denominator'])];
 const {dom,messages,send}=setup(nodes);
 assert.deepEqual(Array.from(dom.window.document.querySelectorAll('.operation'),b=>b.dataset.node),['x','sum','denominator','result']);
 send({type:'state',state:{analysis:{nodes,kernels:['k'],kernel:'k',diagnostics:[],missing_parameters:[]},selected:'denominator',stale:false,file:'a.py',version:1,generation:1,parameters:{},input_shapes:{},program_ids:[]}});
 assert.ok(dom.window.document.querySelector('[data-card-node="sum"]'));assert.equal(dom.window.document.querySelector('[data-card-node="eps"]'),null);
 assert.ok(dom.window.document.querySelector('[data-output="true"][data-index="[]"]'));
 dom.window.document.querySelector('[data-output="true"]').click();
 send(inspectionResponse(messages.at(-1),make('denominator','add',[],['sum','eps']),{status:'exact',output_index:[],origins:[{node_id:'sum',indices:[[]],total:1,truncated:false},{node_id:'eps',indices:[[]],total:1,truncated:false}]}));
 const folded=dom.window.document.querySelector('[data-origin-node="eps"]');assert.ok(folded);assert.match(folded.textContent,/参数与属性/);assert.doesNotMatch(folded.textContent,/网格高亮 1/);assert.equal(folded.querySelectorAll('button').length,0);
});
