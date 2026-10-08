import * as vscode from 'vscode';
import * as path from 'node:path';
import * as fs from 'node:fs';
import {randomUUID,randomBytes} from 'node:crypto';
import {WorkerClient,WorkerPair} from './worker';
import {workerLaunchOptions} from './launch';
import {SelectionState,selectAt,validateParameters} from './state';
import {Analysis,Inspection,Options,Node} from './protocol';
import {operationId,operationNodes} from '../media/operations';

function relevant(document:vscode.TextDocument):boolean {
  if(document.uri.scheme!=='file'||document.languageId!=='python'||!document.fileName.endsWith('.py'))return false;
  const source=document.getText();
  const modules=['triton',...Array.from(source.matchAll(/^\s*import\s+triton\s+as\s+(\w+)/gm),m=>m[1])];
  const jitNames=Array.from(source.matchAll(/^\s*from\s+triton\s+import\s+jit(?:\s+as\s+(\w+))?/gm),m=>m[1]??'jit');
  return modules.some(name=>new RegExp('@\\s*'+name+'\\s*\\.\\s*jit\\b').test(source))||jitNames.some(name=>new RegExp('@\\s*'+name+'\\b').test(source));
}
export function activate(context:vscode.ExtensionContext):void {
  const controller=new Controller(context);
  context.subscriptions.push(controller,
    vscode.commands.registerCommand('tiletrace.open',()=>controller.open()),
    vscode.commands.registerCommand('tiletrace.copyAgentPrompt',()=>controller.copyPrompt()),
    vscode.commands.registerCommand('tiletrace.restartWorker',()=>controller.restart()),
    vscode.window.onDidChangeActiveTextEditor(editor=>controller.activateEditor(editor)),
    vscode.workspace.onDidChangeTextDocument(event=>controller.changed(event.document)),
    vscode.workspace.onDidCloseTextDocument(document=>controller.closed(document)),
    vscode.window.onDidChangeTextEditorSelection(event=>controller.selection(event.textEditor)),
    vscode.workspace.onDidChangeConfiguration(event=>{if(event.affectsConfiguration('tiletrace'))void controller.restart();})
  );
  controller.activateEditor(vscode.window.activeTextEditor);
}
class Controller implements vscode.Disposable {
  private panel?:vscode.WebviewPanel;private editor?:vscode.TextEditor;private worker?:WorkerClient;private contextWorker?:WorkerClient;private folder?:vscode.WorkspaceFolder;
  private session=randomUUID();private state=new SelectionState();private output=vscode.window.createOutputChannel('TileTrace');
  private timer?:NodeJS.Timeout;private queue:Promise<void>=Promise.resolve();private inspectionSerial=0;private error='';private disposed=false;
  private options=new Map<string,Options>();
  private showAllNodes=false;
  constructor(private context:vscode.ExtensionContext){}
  private get config():vscode.WorkspaceConfiguration {return vscode.workspace.getConfiguration('tiletrace',this.editor?.document.uri);}
  private currentOptions():Options {return this.options.get(this.state.document)??{parameters:{},input_shapes:{},program_ids:[]};}
  activateEditor(editor:vscode.TextEditor|undefined):void {
    // VS Code emits undefined when the webview takes focus; preserve its source editor.
    if(!editor||this.disposed)return;
    if(!vscode.workspace.isTrusted){if(this.editor)this.detach('工作区未受信任，无法启动 Python 分析进程。');return;}
    if(!relevant(editor.document)||!vscode.workspace.getWorkspaceFolder(editor.document.uri)){
      if(this.editor&&editor.document.uri.toString()!==this.editor.document.uri.toString())this.detach('选择工作区中带 @triton.jit 的已保存 Python 文件。');
      return;
    }
    if(this.editor?.document===editor.document){this.editor=editor;return;}
    this.editor=editor;
    if(this.config.get<boolean>('autoOpen',true))this.createPanel();
    if(this.panel)this.schedule(0);
  }
  open():void {
    if(!vscode.workspace.isTrusted){void vscode.window.showInformationMessage('请先信任此工作区，再启动 Python 分析。工作区中的 Python 可执行文件也必须可信。');return;}
    const candidate=vscode.window.activeTextEditor??this.editor;
    if(!candidate||!relevant(candidate.document)||!vscode.workspace.getWorkspaceFolder(candidate.document.uri)){
      void vscode.window.showInformationMessage('请先打开工作区中包含 @triton.jit 的已保存 .py 文件。未保存文件和工作区外文件不建立会话。');return;
    }
    this.editor=candidate;this.createPanel();this.panel?.reveal(vscode.ViewColumn.Beside,true);this.schedule(0);
  }
  private createPanel():void {
    if(this.panel)return;
    this.session=randomUUID();
    this.panel=vscode.window.createWebviewPanel('tiletrace','TileTrace · 变换',vscode.ViewColumn.Beside,{enableScripts:true,retainContextWhenHidden:true,localResourceRoots:[vscode.Uri.joinPath(this.context.extensionUri,'media')]});
    const nonce=randomBytes(18).toString('base64');const webview=this.panel.webview;
    const css=webview.asWebviewUri(vscode.Uri.joinPath(this.context.extensionUri,'media','viewer.css'));
    const js=webview.asWebviewUri(vscode.Uri.joinPath(this.context.extensionUri,'media','viewer.js'));
    const operations=webview.asWebviewUri(vscode.Uri.joinPath(this.context.extensionUri,'media','operations.js'));
    webview.html=`<!doctype html><html lang="zh-CN"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src ${webview.cspSource}; script-src 'nonce-${nonce}';"><link rel="stylesheet" href="${css}"><title>TileTrace 变换</title></head><body><main id="app"></main><script nonce="${nonce}" src="${operations}"></script><script nonce="${nonce}" src="${js}"></script></body></html>`;
    this.panel.onDidDispose(()=>{this.panel=undefined;this.detach('面板已关闭。');},null,this.context.subscriptions);
    this.panel.webview.onDidReceiveMessage(message=>{void this.message(message).catch(error=>this.fail(error));},null,this.context.subscriptions);
  }
  private ensureWorker():void {
    if(!this.editor)return;
    const folder=vscode.workspace.getWorkspaceFolder(this.editor.document.uri);if(!folder)return;
    if(this.worker&&this.folder?.uri.toString()===folder.uri.toString())return;
    this.releaseWorker();this.folder=folder;
    const root=folder.uri.fsPath;let command=this.config.get<string>('pythonPath','python');
    if(command==='python'){const venv=path.join(root,'.venv',process.platform==='win32'?'Scripts':'bin',process.platform==='win32'?'python.exe':'python');if(fs.existsSync(venv))command=venv;}
    const configured=this.config.get<string>('sessionDirectory','.tiletrace');const directory=path.resolve(root,configured);
    const bundle=path.join(this.context.extensionPath,'python');const pythonRoot=fs.existsSync(path.join(bundle,'tiletrace','__main__.py'))?bundle:path.resolve(this.context.extensionPath,'..');
    const pair=new WorkerPair(workerLaunchOptions(command,pythonRoot,directory,text=>this.output.append(text)));
    this.session=randomUUID();
    this.worker=pair.analysis;this.contextWorker=pair.context;
    this.queue=Promise.resolve();
  }
  private releaseWorker():void {
    const worker=this.contextWorker;const session=this.session;
    this.worker?.dispose();
    if(worker){const queue=this.queue;void queue.catch(()=>{}).then(()=>worker.request('clear_context',{session_id:session})).catch(error=>this.output.appendLine(String(error))).finally(()=>worker.dispose());}
    this.worker=undefined;this.contextWorker=undefined;this.folder=undefined;
  }
  private publish():void {
    if(!this.contextWorker||!this.state.document)return;
    const worker=this.contextWorker;const options=this.currentOptions();
    const payload={session_id:this.session,context:{document_id:this.state.document,version:this.state.version,stale:this.state.stale,...options,...(!this.state.stale?{analysis:this.state.analysis,selected_node_id:this.state.selected,view_mode:this.showAllNodes?'all':'tensor_steps',visible_node_ids:operationNodes(this.state.analysis?.nodes??[],this.showAllNodes).map(node=>node.id)}: {})}};
    this.queue=this.queue.catch(()=>{}).then(async()=>{await worker.request('sync_context',payload);}).catch(error=>{this.output.appendLine(String(error));if(worker===this.contextWorker)this.fail(error);});
  }
  private send():void {
    const expressions:Record<string,string>={};
    if(this.editor&&!this.state.stale)for(const node of this.state.analysis?.nodes??[])expressions[node.id]=this.editor.document.getText(this.range(node));
    void this.panel?.webview.postMessage({type:'state',state:{analysis:this.state.analysis,selected:this.state.selected,show_all:this.showAllNodes,stale:this.state.stale,file:this.editor?.document.fileName??'',version:this.state.version,generation:this.state.generation,error:this.error,session:this.session,expressions,...this.currentOptions()}});
  }
  private schedule(delay=250):void {
    if(!this.editor||!this.panel)return;
    if(!vscode.workspace.isTrusted){this.detach('工作区未受信任，无法启动 Python 分析进程。');return;}
    clearTimeout(this.timer);this.inspectionSerial++;this.error='';
    this.ensureWorker();const token=this.state.begin(this.editor.document.uri.toString(),this.editor.document.version);this.send();this.publish();
    this.timer=setTimeout(()=>{void this.analyze(token);},delay);
  }
  private async analyze(token:ReturnType<SelectionState['begin']>):Promise<void> {
    if(!this.editor||!this.worker||!this.state.current(token))return;
    const worker=this.worker;const document=this.editor.document;const options=this.currentOptions();const source=document.getText();
    try{
      const analysis=await worker.request<Analysis>('analyze',{source,...options,document_id:token.document,version:token.version});
      if(worker!==this.worker||document!==this.editor?.document||document.version!==token.version||!this.state.accept(token,analysis,this.showAllNodes))return;
      const cursor=this.editor.selection.active;const chosen=selectAt(analysis.nodes,cursor.line+1,cursor.character,this.state.selected,source,this.showAllNodes);if(chosen)this.state.select(chosen);
      this.send();this.publish();
    }catch(error){if(this.state.current(token)&&worker===this.worker){this.fail(error);this.publish();}}
  }
  changed(document:vscode.TextDocument):void {if(document===this.editor?.document)this.schedule();}
  closed(document:vscode.TextDocument):void {if(document===this.editor?.document)this.detach('源文件已关闭。');}
  private detach(reason:string):void {
    clearTimeout(this.timer);this.inspectionSerial++;this.editor=undefined;this.state.begin('',0);this.error=reason;this.send();this.releaseWorker();
  }
  selection(editor:vscode.TextEditor):void {
    if(editor.document!==this.editor?.document||this.state.stale||!this.state.analysis)return;
    this.editor=editor;const cursor=editor.selection.active;const id=selectAt(this.state.analysis.nodes,cursor.line+1,cursor.character,this.state.selected,editor.document.getText(),this.showAllNodes);
    if(id&&id!==this.state.selected){this.state.select(id);this.inspectionSerial++;this.send();this.publish();}
  }
  private range(node:Node):vscode.Range {return new vscode.Range(node.source.start_line-1,node.source.start_col,node.source.end_line-1,node.source.end_col);}
  private async message(message:unknown):Promise<void> {
    if(!message||typeof message!=='object')return;
    const m=message as Record<string,unknown>;
    if(m.type==='ready'){this.send();return;}
    if(m.type==='copyPrompt'){await this.copyPrompt();return;}
    if(m.type==='setNodeVisibility'){
      if(m.generation!==this.state.generation||typeof m.show_all!=='boolean')return;
      this.showAllNodes=m.show_all;this.inspectionSerial++;
      const nodes=this.state.analysis?.nodes??[];
      const id=this.showAllNodes?(this.state.selected??nodes[0]?.id):operationId(nodes,this.state.selected);
      if(id)this.state.select(id);else this.state.selected=undefined;
      this.send();this.publish();return;
    }
    if(m.type==='applyParameters'){
      if(!this.editor||m.generation!==this.state.generation)return;
      const options=validateParameters(m.options);if(options.kernel&&!this.state.analysis?.kernels.includes(options.kernel))throw new Error('请选择分析发现的 kernel。');
      this.options.set(this.state.document,options);this.schedule(0);return;
    }
    if(this.state.stale||!this.state.analysis||m.generation!==this.state.generation)return;
    if(typeof m.node_id!=='string')return;
    const node=this.state.analysis.nodes.find(n=>n.id===m.node_id);if(!node)return;
    if(m.type==='selectNode'){const id=this.showAllNodes?node.id:operationId(this.state.analysis.nodes,node.id,this.state.selected);if(id)this.state.select(id);this.inspectionSerial++;this.send();this.publish();return;}
    if(m.type==='revealSource'&&this.editor){const document=await vscode.workspace.openTextDocument(this.editor.document.uri);if(document.uri.toString()!==this.state.document||document.version!==this.state.version)return;const editor=await vscode.window.showTextDocument(document,{viewColumn:this.editor.viewColumn,preserveFocus:false});editor.selection=new vscode.Selection(this.range(node).start,this.range(node).end);editor.revealRange(this.range(node),vscode.TextEditorRevealType.InCenterIfOutsideViewport);return;}
    if(m.type==='inspectIndex'&&node.id===this.state.selected&&this.worker){
      if(typeof m.request_id!=='string'||m.request_id.length<1||m.request_id.length>100)return;
      if(!Array.isArray(m.index)||m.index.length!==node.shape.length||!m.index.every((x,i)=>Number.isSafeInteger(x)&&x>=0&&typeof node.shape[i]==='number'&&x<(node.shape[i] as number)))return;
      const token=this.state.inspectionToken();const serial=++this.inspectionSerial;const worker=this.worker;
      const correlation={request_id:m.request_id,generation:token.generation,node_id:node.id};
      try {
        const result=await worker.request<Inspection>('inspect',{analysis:this.state.analysis,node_id:node.id,index:m.index,limit:128});
        if(serial===this.inspectionSerial&&worker===this.worker&&this.state.matchesInspection(token))void this.panel?.webview.postMessage({type:'inspection',...correlation,inspection:result});
      } catch(error) {
        if(serial===this.inspectionSerial&&worker===this.worker&&this.state.matchesInspection(token)){
          this.fail(error);
          void this.panel?.webview.postMessage({type:'inspection',...correlation,inspection:{node,status:'unavailable',origins:[],output_index:m.index,message:this.error}});
        }
      }
    }
  }
  private fail(error:unknown):void {this.error=error instanceof Error?error.message:String(error);this.output.appendLine(this.error);this.send();}
  async copyPrompt():Promise<void> {
    if(!this.panel){this.open();if(!this.panel)return;}
    await vscode.env.clipboard.writeText(`请使用 TileTrace MCP：先调用 get_visualization_context(session_id="${this.session}") 获取此 VS Code 面板的当前上下文。检查 stale 和文档版本；使用 inspect_transform(session_id="${this.session}", node_id=<当前 selected_node_id>, index=<完整逻辑坐标>) 查看直接输入来源。仅解释静态形状与逻辑坐标，不宣称运行 kernel 或获得实际数值。若上下文过期，请等待编辑器完成分析。`);
    void vscode.window.showInformationMessage(`已复制 Agent 提示词，会话 ${this.session}。`);
  }
  async restart():Promise<void> {this.releaseWorker();if(this.editor&&this.panel)this.schedule(0);}
  dispose():void {this.disposed=true;clearTimeout(this.timer);this.releaseWorker();this.panel?.dispose();this.output.dispose();}
}
