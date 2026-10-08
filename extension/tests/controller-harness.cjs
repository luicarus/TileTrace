const fs=require('node:fs');const path=require('node:path');const vm=require('node:vm');
function createController({root,python,holdOld=false,trusted=true,inspectError=false,holdInspections=false,analysisNodes,sourceText}={}) {
 const messages=[],calls=[],launches=[],pendingInspections=[],registry=new Map();let releaseOld;
 const gate=new Promise(resolve=>{releaseOld=resolve;});let ordinal=0;
 const node={id:'n1',name:'x',op:'arange',inputs:[],shape:[2],status:'resolved',source:{start_line:4,start_col:8,end_line:4,end_col:22},attrs:{},dtype:null};
 const uri={scheme:'file',fsPath:path.join(root,'kernel.py'),toString:()=>`file:///${root}/kernel.py`};
 const folder={uri:{fsPath:root,toString:()=>`file:///${root}`}};
 const document={uri,languageId:'python',fileName:uri.fsPath,version:1,getText:()=> sourceText??'import triton\nimport triton.language as tl\n@triton.jit\ndef k():\n    x = tl.arange(0,2)\n'};
 const editor={document,selection:{active:{line:4,character:4}},viewColumn:1};
 let clipboard='';
 const disposable={dispose(){}};
 const panel={webview:{cspSource:'vscode-resource:',asWebviewUri:x=>String(x),postMessage:m=>{messages.push(m);return Promise.resolve(true);},onDidReceiveMessage:()=>disposable},onDidDispose:()=>disposable,reveal(){},dispose(){}};
 const vscode={workspace:{isTrusted:trusted,getWorkspaceFolder:()=>folder,getConfiguration:()=>({get:(key,fallback)=>key==='pythonPath'&&python?python:fallback})},window:{activeTextEditor:editor,createOutputChannel:()=>({append(){},appendLine(){},dispose(){}}),createWebviewPanel:()=>panel,showInformationMessage:()=>Promise.resolve()},env:{clipboard:{writeText:async text=>{clipboard=text;}}},Uri:{joinPath:(uri,...pieces)=>String(uri)+'/'+pieces.join('/')},ViewColumn:{Beside:2},Range:class{constructor(line,col,endLine,endCol){this.start={line,character:col};this.end={line:endLine,character:endCol};}}};
 class WorkerPair {
  constructor(options){const id=++ordinal;launches.push(options);this.analysis={dispose(){},request:async(method,p)=>{calls.push({id,kind:'analysis',method,p});if(method==='inspect'){const result={node,inputs:[],output_index:p.index,status:'exact',origins:[]};if(holdInspections)return new Promise((resolve,reject)=>pendingInspections.push({resolve:()=>resolve(result),reject}));if(inspectError)throw new Error('inspection fixture unavailable');return result;}return {document_id:p.document_id,version:p.version,kernel:'k',kernels:['k'],nodes:analysisNodes??[node],diagnostics:[],missing_parameters:[],parameters:{}};}};
   this.context={dispose(){},request:async(method,p)=>{calls.push({id,kind:'context',method,p});if(id===1&&holdOld&&method==='sync_context')await gate;if(method==='sync_context')registry.set(p.session_id,p.context);if(method==='clear_context')registry.delete(p.session_id);return {};}};
  }
 }
 const filename=path.resolve(__dirname,'../out/extension.js');const source=fs.readFileSync(filename,'utf8');const exported={};
 const sandbox={exports:exported,require:name=>name==='vscode'?vscode:name==='./worker'?{WorkerPair}:require(name.startsWith('.')?path.resolve(path.dirname(filename),name):name),setTimeout,clearTimeout,process,Buffer};
 vm.runInNewContext(source+'\nexports.TestController = Controller;',sandbox,{filename});
 const controller=new exported.TestController({extensionPath:path.resolve(__dirname,'..'),extensionUri:'file:///extension',subscriptions:[]});
 return {controller,editor,messages,calls,launches,pendingInspections,registry,releaseOld,clipboard:()=>clipboard,panel};
}
const until=async(predicate)=>{const end=Date.now()+3000;while(!predicate()){if(Date.now()>end)throw new Error('Controller condition timed out');await new Promise(resolve=>setTimeout(resolve,10));}};
module.exports={createController,until};
