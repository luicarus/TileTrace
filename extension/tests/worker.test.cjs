const test=require('node:test'); const assert=require('node:assert/strict'); const path=require('node:path');
const {WorkerClient,WorkerPair}=require('../out/worker');
const make=(timeout=1000,log=()=>{})=>new WorkerClient({command:process.execPath,args:[path.join(__dirname,'worker-fixture.cjs')],cwd:__dirname,env:process.env,timeout,log});
test('worker correlates out of order responses and separates stderr',async()=>{
 const logs=[]; const w=make(1000,x=>logs.push(x));
 try {const [a,b]=await Promise.all([w.request('echo',{delay:60,value:'a'}),w.request('stderr',{value:'b'})]); assert.equal(a.value,'a');assert.equal(b.value,'b');assert.ok(logs.join('').includes('diagnostic only'));await assert.rejects(w.request('error',{}),/invalid shape/);} finally{w.dispose();}
});
test('worker exit rejects every pending request',async()=>{const w=make();try {await assert.rejects(Promise.all([w.request('wait',{}),w.request('exit',{})]),/退出|closed|exit/i);}finally{w.dispose();}});
test('timeout closes process and allows next request to restart',async()=>{const w=make(1000);try{await assert.rejects(w.request('wait',{}),/超时|timeout/i); assert.deepEqual(await w.request('echo',{value:8}),{value:8});}finally{w.dispose();}});
test('stale context is published while analysis worker is still busy',async()=>{
 const pair=new WorkerPair({command:process.execPath,args:[path.join(__dirname,'worker-fixture.cjs')],cwd:__dirname,env:process.env,timeout:3000,log:()=>{}});
 let finished=false;const pending=pair.analysis.request('analyze',{value:'analysis'}).then(()=>{finished=true;});
 try{const result=await pair.context.request('sync_context',{session_id:'test',context:{document_id:'a.py',version:2,stale:true}});assert.equal(result.context.stale,true);assert.equal(finished,false);await pending;}finally{pair.dispose();}
});
test('spawn error rejects instead of hanging',async()=>{const w=new WorkerClient({command:'__nonexistent_triton_python__',args:[],cwd:__dirname,env:process.env,timeout:500,log:()=>{}});try{await assert.rejects(w.request('echo',{}),/Python|ENOENT/);}finally{w.dispose();}});
