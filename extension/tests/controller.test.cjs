const test=require('node:test');const assert=require('node:assert/strict');const path=require('node:path');const fs=require('node:fs');const os=require('node:os');const {spawnSync}=require('node:child_process');
const {createController,until}=require('./controller-harness.cjs');
test('production Controller startup ignores conflicting workspace modules and PYTHONPATH',async()=>{
 const root=fs.mkdtempSync(path.join(os.tmpdir(),'triton-isolation-'));const python=path.resolve(__dirname,'../../.venv',process.platform==='win32'?'Scripts/python.exe':'bin/python');
 const harness=createController({root,python});
 try{
  for(const name of ['tiletrace','json','sitecustomize'])fs.writeFileSync(path.join(root,name+'.py'),`from pathlib import Path\nPath(${JSON.stringify(path.join(root,name+'.sentinel'))}).write_text('EXECUTED')\nprint('WORKSPACE_EXECUTED')\n`);
  harness.controller.activateEditor(harness.editor);await until(()=>harness.launches.length===1);
  const options=harness.launches[0];
  const request={id:'isolated',method:'analyze',params:{source:harness.editor.document.getText(),document_id:'isolated.py',version:7}};
  const result=spawnSync(options.command,options.args,{cwd:options.cwd,env:{...options.env,PYTHONPATH:root,PYTHONHOME:root},input:JSON.stringify(request)+'\n',encoding:'utf8',timeout:10000});
  assert.equal(result.status,0,result.stderr);assert.equal(fs.existsSync(path.join(root,'tiletrace.sentinel')),false);assert.equal(fs.existsSync(path.join(root,'json.sentinel')),false);assert.equal(fs.existsSync(path.join(root,'sitecustomize.sentinel')),false);
  assert.equal(JSON.parse(result.stdout).result.version,7);assert.notEqual(options.cwd,root);assert.ok(options.args.includes('-I'));assert.ok(options.args.includes('-S'));
 }finally{harness.releaseOld();harness.controller.dispose();fs.rmSync(root,{recursive:true,force:true});}
});
test('Controller restart confines delayed old publication and cleanup to its old session',async()=>{
 const root=fs.mkdtempSync(path.join(os.tmpdir(),'triton-session-'));const h=createController({root,holdOld:true});
 try{
  h.controller.activateEditor(h.editor);await until(()=>h.calls.some(c=>c.id===1&&c.method==='sync_context'));
  const oldSession=h.messages.at(-1).state.session;
  await h.controller.restart();await until(()=>h.calls.some(c=>c.id===2&&c.method==='sync_context'&&!c.p.context.stale));
  const newSession=h.messages.at(-1).state.session;assert.ok(h.registry.has(newSession));
  assert.ok(h.calls.some(c=>c.id===2&&c.method==='sync_context'&&c.p.context.stale));
  h.releaseOld();await until(()=>h.calls.some(c=>c.id===1&&c.method==='clear_context'));await new Promise(resolve=>setTimeout(resolve,10));
  assert.ok(h.registry.has(newSession),'old cleanup deleted current session');assert.notEqual(oldSession,newSession);assert.equal(h.registry.has(oldSession),false);
  assert.ok(h.calls.filter(c=>c.id===1&&c.kind==='context').every(c=>c.p.session_id===oldSession));
  await h.controller.copyPrompt();assert.ok(h.clipboard().includes(newSession));assert.equal(h.clipboard().includes(oldSession),false);
 }finally{h.releaseOld();h.controller.dispose();fs.rmSync(root,{recursive:true,force:true});}
});
test('untrusted workspace cannot start an automatic or manually opened worker',async()=>{
 const h=createController({root:os.tmpdir(),trusted:false});try{h.controller.activateEditor(h.editor);h.controller.open();await new Promise(resolve=>setTimeout(resolve,20));assert.equal(h.launches.length,0);}finally{h.controller.dispose();}
});
for(const inspectError of [false,true])test(`Controller echoes browser correlation for inspection ${inspectError?'error':'success'} and rejects invalid requests`,async()=>{
 const h=createController({root:os.tmpdir(),inspectError});
 try{
  h.controller.activateEditor(h.editor);await until(()=>h.messages.some(m=>m.type==='state'&&!m.state.stale));
  const generation=h.messages.filter(m=>m.type==='state').at(-1).state.generation;
  const request={type:'inspectIndex',request_id:'browser-7',generation,node_id:'n1',index:[1]};
  await h.controller.message({...request,request_id:undefined});await h.controller.message({...request,generation:generation+1});
  assert.equal(h.calls.some(c=>c.method==='inspect'),false);
  await h.controller.message(request);const response=h.messages.at(-1);
  assert.equal(response.type,'inspection');assert.equal(response.request_id,'browser-7');assert.equal(response.generation,generation);assert.equal(response.node_id,'n1');assert.equal(response.inspection.output_index[0],1);assert.equal(response.inspection.status,inspectError?'unavailable':'exact');
 }finally{h.controller.dispose();}
});
test('Controller ignores old inspection errors after another click or closed document',async()=>{
 const h=createController({root:os.tmpdir(),holdInspections:true});
 try{
  h.controller.activateEditor(h.editor);await until(()=>h.messages.some(m=>m.type==='state'&&!m.state.stale));
  const generation=h.messages.filter(m=>m.type==='state').at(-1).state.generation;
  const request={type:'inspectIndex',generation,node_id:'n1',index:[0]};
  const first=h.controller.message({...request,request_id:'first'});const second=h.controller.message({...request,index:[1],request_id:'second'});
  h.pendingInspections[0].reject(new Error('old-query-error'));await first;assert.equal(h.messages.some(m=>m.type==='inspection'),false);assert.equal(h.messages.some(m=>m.state?.error==='old-query-error'),false);
  h.pendingInspections[1].resolve();await second;assert.equal(h.messages.at(-1).request_id,'second');
  const third=h.controller.message({...request,request_id:'third'});h.controller.closed(h.editor.document);h.pendingInspections[2].reject(new Error('closed-query-error'));await third;
  assert.equal(h.messages.some(m=>m.state?.error==='closed-query-error'),false);assert.equal(h.messages.some(m=>m.request_id==='third'),false);
 }finally{h.controller.dispose();}
});

test('Controller keeps default and detailed source selection aligned with the published full graph',async()=>{
 const demo=require('./operations-fixture.cjs')();const sourceText=fs.readFileSync(path.resolve(__dirname,'../../examples/transforms.py'),'utf8');
 const h=createController({root:os.tmpdir(),analysisNodes:demo.nodes,sourceText});
 try {
  h.controller.activateEditor(h.editor);await until(()=>h.messages.some(m=>m.type==='state'&&!m.state.stale));
  const latest=()=>h.messages.filter(m=>m.type==='state').at(-1).state;
  const rows=demo.nodes.find(n=>n.name==='rows');const zero=demo.nodes.find(n=>n.op==='constant'&&n.source.start_line===rows.source.start_line);
  assert.equal(latest().selected,rows.id);assert.equal(latest().show_all,false);
  h.editor.selection.active={line:zero.source.start_line-1,character:zero.source.start_col};h.controller.selection(h.editor);assert.equal(latest().selected,rows.id);
  await h.controller.message({type:'setNodeVisibility',generation:latest().generation+1,show_all:true});assert.equal(latest().show_all,false);
  await h.controller.message({type:'setNodeVisibility',generation:latest().generation,show_all:true});h.controller.selection(h.editor);assert.equal(latest().selected,zero.id);
  await h.controller.message({type:'setNodeVisibility',generation:latest().generation,show_all:false});assert.equal(latest().selected,rows.id);
  await until(()=>{const context=h.calls.filter(c=>c.method==='sync_context').at(-1)?.p.context;return context?.selected_node_id===rows.id&&context.view_mode==='tensor_steps';});
  const snapshot=h.calls.filter(c=>c.method==='sync_context').at(-1).p.context;assert.equal(snapshot.analysis.nodes.length,16);assert.equal(snapshot.view_mode,'tensor_steps');assert.equal(snapshot.visible_node_ids.length,9);
 } finally {h.controller.dispose();}
});
