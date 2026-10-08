import * as path from 'node:path';
import {WorkerOptions} from './worker';
// Owned code only: paths travel as argv data, never as interpolated Python source.
export const WORKER_BOOTSTRAP='import sys,runpy; sys.path.insert(0,sys.argv.pop(1)); runpy.run_module("tiletrace",run_name="__main__",alter_sys=True)';
export function workerLaunchOptions(command:string,pythonRoot:string,sessionDirectory:string,log:(text:string)=>void):WorkerOptions {
  const trustedRoot=path.resolve(pythonRoot);
  return {command,args:['-I','-S','-u','-c',WORKER_BOOTSTRAP,trustedRoot,'worker','--session-dir',path.resolve(sessionDirectory)],cwd:trustedRoot,env:{...process.env},log};
}
