import {spawn,ChildProcessWithoutNullStreams} from 'node:child_process';
import {createInterface,Interface} from 'node:readline';
export interface WorkerOptions {command:string;args:string[];cwd:string;env:NodeJS.ProcessEnv;timeout?:number;log:(text:string)=>void;}
interface Pending {resolve:(value:unknown)=>void;reject:(reason:Error)=>void;timer:NodeJS.Timeout;}
export class WorkerClient {
  private process?:ChildProcessWithoutNullStreams; private reader?:Interface;private pending=new Map<string,Pending>();private serial=0;private disposed=false;
  constructor(private options:WorkerOptions){}
  private start():ChildProcessWithoutNullStreams {
    if(this.disposed)throw new Error('分析进程已关闭。');
    if(this.process)return this.process;
    const child=spawn(this.options.command,this.options.args,{cwd:this.options.cwd,env:this.options.env,shell:false,windowsHide:true,stdio:'pipe'});
    this.process=child;this.reader=createInterface({input:child.stdout});
    this.reader.on('line',line=>{
      if(this.process!==child)return;
      try {const response=JSON.parse(line);const entry=this.pending.get(String(response.id));if(!entry)return;this.pending.delete(String(response.id));clearTimeout(entry.timer);
        if(response.error)entry.reject(new Error(`${response.error.type??'WorkerError'}: ${response.error.message??'未知错误'}`));else entry.resolve(response.result);
      } catch {this.options.log('忽略非 JSON worker 输出。\n');}
    });
    child.stderr.on('data',chunk=>this.options.log(String(chunk)));
    child.on('error',error=>{if(this.process===child)this.stop(new Error(`无法启动 Python：${error.message}。请设置 tritonTransform.pythonPath。`));});
    child.on('exit',(code,signal)=>{if(this.process===child)this.stop(new Error(`分析进程退出 (${code??signal})，可使用 Triton: 重启分析进程。`));});
    child.stdin.on('error',error=>{if(this.process===child)this.stop(error);});
    return child;
  }
  request<T=unknown>(method:string,params:unknown):Promise<T> {
    return new Promise<T>((resolve,reject)=>{
      let child:ChildProcessWithoutNullStreams;try{child=this.start();}catch(error){reject(error);return;}
      const id=String(++this.serial);
      const timer=setTimeout(()=>this.stop(new Error('分析请求超时，请减少源码规模或重启分析进程。')),this.options.timeout??15000);
      this.pending.set(id,{resolve:value=>resolve(value as T),reject,timer});
      try{child.stdin.write(JSON.stringify({id,method,params})+'\n');}catch(error){this.stop(error instanceof Error?error:new Error(String(error)));}
    });
  }
  private stop(error:Error):void {
    const child=this.process;this.process=undefined;this.reader?.close();this.reader=undefined;
    for(const p of this.pending.values()){clearTimeout(p.timer);p.reject(error);}this.pending.clear();
    if(child){child.stdin.end();if(!child.killed)child.kill();}
  }
  dispose():void {this.disposed=true;this.stop(new Error('分析进程已关闭。'));}
}
/** Context writes must never wait behind a potentially expensive analysis. */
export class WorkerPair {
  readonly analysis:WorkerClient;
  readonly context:WorkerClient;
  constructor(options:WorkerOptions){this.analysis=new WorkerClient(options);this.context=new WorkerClient(options);}
  dispose():void {this.analysis.dispose();this.context.dispose();}
}
