const test=require('node:test');const assert=require('node:assert/strict');const fs=require('node:fs');const path=require('node:path');const {JSDOM}=require('jsdom');
const node=(id,shape,inputs=[])=>({id,name:id,op:'reshape',shape,inputs,status:'resolved',source:{start_line:1,start_col:0,end_line:1,end_col:10},attrs:{},dtype:null});
function setup(nodes) {const messages=[];const dom=new JSDOM('<main id="app"></main>',{runScripts:'outside-only'});dom.window.acquireVsCodeApi=()=>({postMessage:m=>messages.push(m),getState:()=>null,setState:()=>{}});dom.window.eval(fs.readFileSync(path.join(__dirname,'../media/operations.js'),'utf8'));dom.window.eval(fs.readFileSync(path.join(__dirname,'../media/viewer.js'),'utf8'));const send=data=>dom.window.dispatchEvent(new dom.window.MessageEvent('message',{data}));send({type:'state',state:{analysis:{nodes,kernels:['k'],kernel:'k',diagnostics:[],missing_parameters:[]},selected:nodes[nodes.length-1].id,stale:false,file:'a.py',version:1,parameters:{},input_shapes:{},program_ids:[]}});return {dom,messages,send};}
const inspectionResponse=(request,output,inspection)=>({type:'inspection',request_id:request.request_id,generation:request.generation,node_id:request.node_id,inspection:{node:output,...inspection}});

test('flow keeps its operator and accessible meaning while CSS controls arrow direction',()=>{
 const {dom}=setup([node('in',[8]),node('out',[2,4],['in'])]);const arrow=dom.window.document.querySelector('.flow-arrow');
 assert.ok(arrow.querySelector('.flow-direction'));
 assert.equal(arrow.querySelector('.flow-operator').textContent,'reshape');
 assert.equal(arrow.getAttribute('role'),'img');
 assert.match(arrow.getAttribute('aria-label'),/输入.*reshape.*输出/);
 assert.equal(arrow.querySelector('.flow-direction').getAttribute('aria-hidden'),'true');
});

test('tensor cards keep one metadata line and move directions and counts to tooltips',()=>{
 const rows=node('rows',[16]);const expanded=node('expanded',[16,1],['rows']);expanded.op='expand_dims';expanded.attrs={mapping:'expand_dims',axes:[1]};
 const {dom,messages}=setup([rows,expanded]);const document=dom.window.document;
 for(const card of document.querySelectorAll('.tensor-card')) {
  assert.equal(card.querySelectorAll(':scope > p').length,1);
  assert.doesNotMatch(card.textContent,/一维索引|长度超过|每卡最多|显示.*总计|当前切片包含/);
  assert.match(card.querySelector('.pill').title,/16/);assert.match(card.querySelector('.grid-scroll').title,/显示 6.*总计 16/);
  assert.match(card.querySelector('table').getAttribute('aria-label'),/显示 6.*总计 16/);
  assert.ok(card.querySelector('[data-role="toggle-grid-window"]'));
 }
 assert.match(document.querySelector('[data-card-role="input"] .pill').title,/行轴/);
 document.querySelector('[data-output="true"][data-index="[15,0]"]').click();assert.deepEqual(JSON.parse(JSON.stringify(messages.at(-1).index)),[15,0]);
});

test('long vectors show head and tail with an accessible ellipsis and exact tail coordinates',()=>{
 for(const size of [8,9,16,1000000]) {
  const v=node('v',[size]);const {dom,messages}=setup([v]);const card=dom.window.document.querySelector('[data-card-role="output"]');
  const indices=Array.from(card.querySelectorAll('[data-index]'),c=>JSON.parse(c.dataset.index)[0]);
  assert.deepEqual(indices,size<=8?Array.from({length:size},(_,i)=>i):[0,1,2,size-3,size-2,size-1]);
  assert.equal(card.querySelectorAll('.axis-ellipsis').length,size<=8?0:1);
  assert.equal(card.querySelector('.pill').textContent,`[${size}]`);
  if(size>8){const gap=card.querySelector('.axis-ellipsis');assert.match(gap.getAttribute('aria-label'),/3.*省略/);card.querySelector(`[data-index="[${size-1}]"]`).click();assert.deepEqual(JSON.parse(JSON.stringify(messages.at(-1).index)),[size-1]);}
 }
});

test('matrix overview folds both axes and expanding a gap reveals actual middle coordinates',()=>{
 const {dom,messages}=setup([node('x',[16,32])]);const document=dom.window.document;
 let card=document.querySelector('[data-card-role="output"]');
 assert.equal(card.querySelectorAll('[data-index]').length,36);assert.ok(card.querySelector('[data-index="[15,31]"]'));assert.equal(card.querySelector('[data-index="[7,7]"]'),null);
 const gap=card.querySelector('.axis-ellipsis[data-axis="0"]');gap.focus();gap.click();
 assert.equal(document.activeElement.dataset.role,'toggle-grid-window');
 card=document.querySelector('[data-card-role="output"]');assert.ok(card.querySelector('[data-index="[3,0]"]'));assert.equal(card.querySelectorAll('.axis-ellipsis').length,0);assert.ok(card.querySelectorAll('[data-index]').length<=128);
 const col=card.querySelector('input[data-axis="1"]');col.value='5';col.dispatchEvent(new dom.window.Event('change'));
 document.querySelector('[data-output="true"][data-index="[3,5]"]').click();assert.deepEqual(JSON.parse(JSON.stringify(messages.at(-1).index)),[3,5]);
 document.querySelector('[data-card-role="output"] [data-role="toggle-grid-window"]').click();
 assert.ok(document.querySelector('[data-output="true"][data-index="[15,31]"]'));assert.equal(document.querySelectorAll('[data-card-role="output"] .axis-ellipsis').length,8);
});

test('collapsed input origins count only visible head/tail cells and can reveal omitted sources',()=>{
 const {dom,messages,send}=setup([node('in',[16]),node('out',[1],['in'])]);const document=dom.window.document;
 document.querySelector('[data-output="true"]').click();
 send(inspectionResponse(messages.at(-1),node('out',[1],['in']),{status:'exact',output_index:[0],origins:[{node_id:'in',indices:[[7]],total:1,truncated:false}]}));
 assert.equal(document.querySelectorAll('.origin').length,0);assert.match(document.querySelector('.mapping').textContent,/高亮 0 \/ 1/);
 document.querySelector('[data-origin-target="in"]').click();assert.ok(document.querySelector('[data-node="in"][data-index="[7]"].origin'));assert.match(document.querySelector('.mapping').textContent,/高亮 1 \/ 1/);
});

test('origin text abbreviates returned coordinates without inventing an unenumerated tail',()=>{
 const {dom,messages,send}=setup([node('in',[1000000]),node('out',[1],['in'])]);const document=dom.window.document;
 document.querySelector('[data-output="true"]').click();
 send(inspectionResponse(messages.at(-1),node('out',[1],['in']),{status:'exact',output_index:[0],origins:[{node_id:'in',indices:Array.from({length:16},(_,i)=>[i]),total:1000000,truncated:true}]}));
 const text=document.querySelector('.coordinates').textContent;assert.equal(text,'[0] · [1] · [2] · … · [13] · [14] · [15]');assert.doesNotMatch(text,/999999/);
 assert.match(document.querySelector('.mapping').textContent,/来源枚举已截断/);assert.match(document.querySelector('.mapping').textContent,/返回坐标的首尾/);
});

test('expanding and collapsing keeps page position and toggle focus',()=>{
 const {dom}=setup([node('out',[16])]);const document=dom.window.document;const scroll=scrollModel(dom);let toggle=document.querySelector('[data-role="toggle-grid-window"]');
 toggle.focus();scroll.set(0,420);toggle.click();
 assert.deepEqual(scroll.position(),[0,420]);assert.equal(document.activeElement.dataset.role,'toggle-grid-window');assert.equal(document.querySelectorAll('[data-output="true"]').length,16);
 toggle=document.activeElement;toggle.click();assert.deepEqual(scroll.position(),[0,420]);assert.equal(document.activeElement.dataset.role,'toggle-grid-window');assert.equal(document.querySelectorAll('[data-output="true"]').length,6);
});

test('one-dimensional row indices follow new-axis use without changing coordinate rank',()=>{
 const rows=node('rows',[16]);const expanded=node('expanded',[16,1],['rows']);expanded.op='expand_dims';expanded.attrs={mapping:'expand_dims',axes:[1]};
 const {dom,messages,send}=setup([rows,expanded]);const document=dom.window.document;
 assert.equal(document.querySelector('[data-card-role="output"] tbody').rows.length,7);
 send({type:'state',state:{analysis:{nodes:[rows,expanded],kernels:['k'],kernel:'k',diagnostics:[],missing_parameters:[]},selected:'rows',stale:false,file:'a.py',version:1,parameters:{},input_shapes:{},program_ids:[]}});
 const card=document.querySelector('[data-card-role="output"]');
 assert.equal(card.querySelectorAll('tbody tr').length,7);
 assert.equal(card.querySelectorAll('tbody tr:first-child td').length,1);
 assert.equal(card.querySelector('.pill').textContent,'[16]');
 assert.match(card.querySelector('.pill').title,/行轴/);
 card.querySelector('[data-index="[15]"]').click();assert.deepEqual(JSON.parse(JSON.stringify(messages.at(-1).index)),[15]);
});

test('vector direction follows uses and selected transform, never variable names',()=>{
 const vector=node('query_rows',[16]);const rowUse=node('r',[16,1],['query_rows']);rowUse.op='expand_dims';rowUse.attrs={mapping:'expand_dims',axes:[1]};
 const colUse=node('c',[1,16],['query_rows']);colUse.op='expand_dims';colUse.attrs={mapping:'expand_dims',axes:[0]};
 const {dom,send}=setup([vector,rowUse,colUse]);const document=dom.window.document;
 const select=id=>send({type:'state',state:{analysis:{nodes:[vector,rowUse,colUse],kernels:['k'],kernel:'k',diagnostics:[],missing_parameters:[]},selected:id,stale:false,file:'a.py',version:1,parameters:{},input_shapes:{},program_ids:[]}});
 select('query_rows');assert.match(document.querySelector('[data-card-role="output"] .pill').title,/未指定行列方向/);
 select('r');assert.equal(document.querySelector('[data-card-role="input"] tbody').rows.length,7);
 select('c');assert.equal(document.querySelector('[data-card-role="input"] tbody').rows.length,1);
 const unused=setup([node('query_rows',[16])]);assert.match(unused.dom.window.document.querySelector('[data-card-role="output"] .pill').title,/未指定行列方向/);
});

test('vertical vector origins and windows preserve the same logical index across directions',()=>{
 const vector=node('v',[64]);const r=node('r',[64,1],['v']);r.op='expand_dims';r.attrs={mapping:'expand_dims',axes:[1]};
 const c=node('c',[1,64],['v']);c.op='expand_dims';c.attrs={mapping:'expand_dims',axes:[0]};
 const {dom,send,messages}=setup([vector,c,r]);const document=dom.window.document;
 document.querySelector('[data-output="true"][data-index="[0,0]"]').click();
 send(inspectionResponse(messages.at(-1),r,{status:'exact',output_index:[0,0],origins:[{node_id:'v',indices:[[47]],total:1,truncated:false}]}));
 document.querySelector('[data-origin-target="v"]').click();
 let card=document.querySelector('[data-card-role="input"]');assert.ok(card.querySelector('[data-index="[47]"].origin'));assert.equal(card.querySelector('input[data-axis="0"]').value,'47');
 const start=card.querySelector('input[data-axis="0"]');start.value='32';start.dispatchEvent(new dom.window.Event('change'));
 card=document.querySelector('[data-card-role="input"]');assert.equal(card.querySelector('tbody tr').firstChild.textContent,'32');assert.ok(card.querySelector('[data-index="[47]"].origin'));
 assert.match(document.querySelector('.mapping').textContent,/当前网格高亮 1 \/ 1/);
 send({type:'state',state:{analysis:{nodes:[vector,c,r],kernels:['k'],kernel:'k',diagnostics:[],missing_parameters:[]},selected:'c',stale:false,file:'a.py',version:1,parameters:{},input_shapes:{},program_ids:[]}});
 card=document.querySelector('[data-card-role="input"]');assert.equal(card.querySelector('tbody').rows.length,1);assert.equal(card.querySelector('input[data-axis="0"]').value,'32');assert.ok(card.querySelector('[data-index="[32]"]'));
});
function scrollModel(dom,clampLoading=false) {
 let x=0,y=0;
 Object.defineProperties(dom.window,{scrollX:{configurable:true,get:()=>x},scrollY:{configurable:true,get:()=>y}});
 dom.window.scrollTo=(left,top)=>{x=left;y=clampLoading&&!dom.window.document.querySelector('[data-output="true"]')?0:top;};
 const root=dom.window.document.getElementById('app');const replace=root.replaceChildren.bind(root);
 root.replaceChildren=(...children)=>{x=0;y=0;replace(...children);};
 return {set:(left,top)=>{x=left;y=top;},position:()=>[x,y]};
}

test('FlashAttention shows both KV iterations and dot origins as a row and column',()=>{
 const {spawnSync}=require('node:child_process');const root=path.resolve(__dirname,'../..');
 const venv=path.join(root,'.venv',process.platform==='win32'?'Scripts/python.exe':'bin/python');
 const code='import sys,json; from pathlib import Path; sys.path.insert(0,sys.argv[1]); from tiletrace import analyze,inspect_transform; a=analyze(Path(sys.argv[2]).read_text(encoding="utf-8")); n=[n for n in a["nodes"] if n["op"]=="dot"][2]; print(json.dumps({"analysis":a,"node":n,"inspection":inspect_transform(a,n["id"],[0,1])}))';
 const result=spawnSync(fs.existsSync(venv)?venv:'python',['-I','-S','-c',code,root,path.join(root,'examples/flash_attention.py')],{encoding:'utf8',cwd:root,windowsHide:true,timeout:10000});
 assert.equal(result.status,0,result.stderr);const data=JSON.parse(result.stdout);
 const {dom,send,messages}=setup(data.analysis.nodes);
 const select=id=>send({type:'state',state:{analysis:data.analysis,selected:id,stale:false,file:'attention.py',version:1,parameters:{},input_shapes:{},program_ids:[]}});
 select(data.analysis.nodes.find(n=>n.name==='query_rows').id);
 assert.equal(dom.window.document.querySelector('[data-card-role="output"] tbody').rows.length,7);
 select(data.analysis.nodes.find(n=>n.name==='head_cols').id);
 assert.equal(dom.window.document.querySelector('[data-card-role="output"] tbody').rows.length,1);
 select(data.analysis.nodes.find(n=>n.name==='q').id);
 assert.match(dom.window.document.querySelector('[data-card-role="output"] .pill').title,/16 行 × 32 列/);
 select(data.node.id);
 const document=dom.window.document;const buttons=Array.from(document.querySelectorAll('.operation'));
 assert.ok(buttons.some(b=>/dot.*start_n=0/.test(b.textContent)));assert.ok(buttons.some(b=>/dot.*start_n=32/.test(b.textContent)));
 assert.match(document.body.textContent,/矩阵乘法归约长度/);
 document.querySelector('[data-output="true"][data-index="[0,1]"]').click();
 send(inspectionResponse(messages.at(-1),data.node,data.inspection));
 // The overview shows six head/tail coordinates per axis; origins retain 32.
 assert.equal(document.querySelectorAll('[data-node="'+data.node.inputs[0]+'"].origin').length,6);
 assert.equal(document.querySelectorAll('[data-node="'+data.node.inputs[1]+'"].origin').length,6);
 assert.equal(data.inspection.origins[0].total,32);
 assert.equal(data.inspection.origins[1].total,32);
});

test('output clicks and matching replies keep page/list/grid scroll, expanded parameters and focus',()=>{
 const inputs=[node('in',[8]),node('out',[2,4],['in'])];const {dom,messages,send}=setup(inputs);const document=dom.window.document;const scroll=scrollModel(dom);
 document.querySelector('.parameters').open=true;document.querySelector('.operation-list').scrollTop=80;document.querySelector('.expression').scrollTop=12;
 document.querySelector('[data-card-node="in"] .grid-scroll').scrollLeft=65;document.querySelector('[data-card-node="out"] .grid-scroll').scrollLeft=35;
 const cell=document.querySelector('[data-output="true"][data-index="[1,2]"]');cell.focus();scroll.set(7,620);cell.click();
 const check=()=>{assert.deepEqual(scroll.position(),[7,620]);assert.equal(document.querySelector('.parameters').open,true);assert.equal(document.querySelector('.operation-list').scrollTop,80);assert.equal(document.querySelector('.expression').scrollTop,12);assert.equal(document.querySelector('[data-card-node="in"] .grid-scroll').scrollLeft,65);assert.equal(document.querySelector('[data-card-node="out"] .grid-scroll').scrollLeft,35);assert.equal(document.activeElement.dataset.index,'[1,2]');};
 check();send(inspectionResponse(messages.at(-1),inputs[1],{status:'exact',output_index:[1,2],origins:[{node_id:'in',indices:[[6]],total:1,truncated:false}]}));check();
});

test('parameter caret/size/scroll and manual collapsed state survive background updates',()=>{
 const nodes=[node('out',[2,4])];const {dom,send}=setup(nodes);const document=dom.window.document;const scroll=scrollModel(dom);
 document.querySelector('.parameters').open=true;const textarea=document.querySelector('textarea');textarea.value='{"BLOCK":8}';textarea.dispatchEvent(new dom.window.Event('input'));textarea.style.height='96px';textarea.scrollTop=14;textarea.focus();textarea.setSelectionRange(4,8);scroll.set(0,200);
 const state={analysis:{nodes,kernels:['k'],kernel:'k',diagnostics:[],missing_parameters:[]},selected:'out',stale:false,file:'a.py',version:1,parameters:{},input_shapes:{},program_ids:[]};
 send({type:'state',state});assert.deepEqual(scroll.position(),[0,200]);const restored=document.querySelector('textarea');assert.equal(document.activeElement,restored);assert.equal(restored.selectionStart,4);assert.equal(restored.selectionEnd,8);assert.equal(restored.scrollTop,14);assert.equal(restored.style.height,'96px');
 document.querySelector('.parameters').open=false;send({type:'state',state});assert.equal(document.querySelector('.parameters').open,false);
});

test('temporary stale layout cannot overwrite saved viewport and a different file starts at top',()=>{
 const nodes=[node('out',[2,4])];const {dom,send}=setup(nodes);const document=dom.window.document;const scroll=scrollModel(dom,true);scroll.set(0,620);document.querySelector('.operation-list').scrollTop=80;document.querySelector('.parameters').open=true;
 const analysis={nodes,kernels:['k'],kernel:'k',diagnostics:[],missing_parameters:[]};
 send({type:'state',state:{stale:true,file:'a.py',version:2,generation:2,parameters:{},input_shapes:{},program_ids:[]}});assert.equal(scroll.position()[1],0);
 send({type:'state',state:{analysis,selected:'out',stale:false,file:'a.py',version:2,generation:2,parameters:{},input_shapes:{},program_ids:[]}});assert.equal(scroll.position()[1],620);assert.equal(document.querySelector('.operation-list').scrollTop,80);assert.equal(document.querySelector('.parameters').open,true);
 send({type:'state',state:{analysis,selected:'out',stale:false,file:'b.py',version:1,generation:3,parameters:{},input_shapes:{},program_ids:[]}});assert.equal(scroll.position()[1],0);assert.equal(document.querySelector('.operation-list').scrollTop,0);assert.equal(document.querySelector('.parameters').open,false);
});

test('automatically restored surviving focus must not turn loading clamp into the saved viewport',()=>{
 for(const control of ['textarea','apply']) {
  const nodes=[node('out',[2,4])];const {dom,send}=setup(nodes);const document=dom.window.document;const scroll=scrollModel(dom,true);
  document.querySelector('.parameters').open=true;
  const focused=control==='textarea'?document.querySelector('textarea'):Array.from(document.querySelectorAll('button')).find(b=>b.textContent==='应用参数并重新分析');focused.focus();scroll.set(0,200);
  const fields={file:'a.py',version:2,generation:2,parameters:{},input_shapes:{},program_ids:[]};
  send({type:'state',state:{...fields,stale:true}});assert.equal(scroll.position()[1],0);assert.notEqual(document.activeElement,document.body);
  send({type:'state',state:{...fields,stale:false,analysis:{nodes,kernels:['k'],kernel:'k',diagnostics:[],missing_parameters:[]},selected:'out'}});
  assert.equal(scroll.position()[1],200,control);
 }
});

test('intentional parameter input during loading keeps the new position and caret',()=>{
 const nodes=[node('out',[2,4])];const {dom,send}=setup(nodes);const document=dom.window.document;const scroll=scrollModel(dom,true);scroll.set(0,620);
 const fields={file:'a.py',version:2,generation:2,parameters:{},input_shapes:{},program_ids:[]};
 send({type:'state',state:{...fields,stale:true}});
 const textarea=document.querySelector('textarea');textarea.focus();textarea.value='{"BLOCK":16}';textarea.setSelectionRange(3,7);textarea.dispatchEvent(new dom.window.Event('input',{bubbles:true}));
 send({type:'state',state:{...fields,stale:false,analysis:{nodes,kernels:['k'],kernel:'k',diagnostics:[],missing_parameters:[]},selected:'out'}});
 assert.equal(scroll.position()[1],0);assert.equal(document.activeElement,document.querySelector('textarea'));assert.equal(document.activeElement.selectionStart,3);assert.equal(document.activeElement.selectionEnd,7);
});
test('output click requests exact index and highlights immediate input origin',()=>{const {dom,messages,send}=setup([node('in',[8]),node('out',[2,4],['in'])]);dom.window.document.querySelector('[data-output="true"][data-index="[0,1]"]').click();assert.equal(messages.at(-1).type,'inspectIndex');assert.deepEqual(JSON.parse(JSON.stringify(messages.at(-1).index)),[0,1]);send(inspectionResponse(messages.at(-1),node('out',[2,4],['in']),{status:'exact',output_index:[0,1],origins:[{node_id:'in',indices:[[1]],total:1,truncated:false}]}));assert.ok(dom.window.document.querySelector('[data-node="in"][data-index="[1]"]').classList.contains('origin'));send({type:'state',state:{stale:true,file:'a.py',version:2}});assert.equal(dom.window.document.querySelectorAll('.origin').length,0);assert.equal(dom.window.document.querySelectorAll('[data-output="true"]').length,0);});
test('huge tensors stay capped, rank three supports prefix slices, symbolic tensors have no grid',()=>{const {dom,messages}=setup([node('x',[4,1000000,1000000])]);assert.ok(dom.window.document.querySelectorAll('[data-index]').length<=128);const slice=dom.window.document.querySelector('input[data-axis="0"]');assert.ok(slice);slice.value='3';slice.dispatchEvent(new dom.window.Event('change'));const cell=dom.window.document.querySelector('[data-output="true"]');cell.click();assert.equal(messages.at(-1).index[0],3);assert.match(dom.window.document.querySelector(".grid-scroll").title,/显示.*总计/);const symbolic=setup([node('s',['BLOCK',4])]);assert.equal(symbolic.dom.window.document.querySelectorAll('[data-index]').length,0);assert.match(symbolic.dom.window.document.body.textContent,/符号/);});
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
