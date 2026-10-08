export interface SourceRange { start_line:number; start_col:number; end_line:number; end_col:number; }
export interface Node { id:string; op:string; name:string; inputs:string[]; shape:(number|string)[]; status:'resolved'|'symbolic'|'unsupported'; source:SourceRange; attrs:Record<string,unknown>; dtype:string|null; }
export interface Analysis { document_id:string; version:number; kernel:string|null; kernels:string[]; nodes:Node[]; diagnostics:{severity:string;message:string;source?:SourceRange}[]; missing_parameters:string[]; parameters:Record<string,unknown>; }
export interface Inspection { node:Node; inputs:Node[]; output_index:number[]|null; status:'exact'|'unavailable'; origins:{node_id:string;indices:number[][];total:number;truncated:boolean}[];message?:string; }
export interface Options { kernel?:string; parameters:Record<string,unknown>; input_shapes:Record<string,(number|string)[]>; program_ids:number[]; }
