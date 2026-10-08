import {Analysis,Node,Options} from './protocol';
export function selectAt(nodes:Node[],line:number,col:number,previous?:string,source?:string):string|undefined {
  const contains=(n:Node)=> (line>n.source.start_line || line===n.source.start_line&&col>=n.source.start_col) && (line<n.source.end_line||line===n.source.end_line&&col<n.source.end_col);
  const span=(n:Node)=>(n.source.end_line-n.source.start_line)*1000000+n.source.end_col-n.source.start_col;
  const expression=nodes.filter(contains).sort((a,b)=>span(a)-span(b))[0];
  if(expression)return expression.id;
  if(source){
    const text=source.split(/\r?\n/)[line-1]??'';
    for(const node of nodes){
      if(node.source.start_line!==line)continue;
      const prefix=text.slice(0,node.source.start_col);
      const assignment=/^(\s*)([\p{L}_][\p{L}\p{N}_]*)(?:\s*:\s*[^=]+)?\s*=\s*$/u.exec(prefix);
      if(assignment&&assignment[2]===node.name&&col>=assignment[1].length&&col<assignment[1].length+assignment[2].length)return node.id;
    }
  }
  return nodes.some(n=>n.id===previous)?previous:nodes[0]?.id;
}
interface Token {document:string;version:number;generation:number; selected?:string;}
export class SelectionState {
  document=''; version=0; generation=0; stale=true; analysis?:Analysis; selected?:string;
  private previous?:Node;
  begin(document:string,version:number):Token {
    this.previous=document===this.document?(this.analysis?.nodes.find(n=>n.id===this.selected)??this.previous):undefined;
    this.document=document;this.version=version;this.generation++;this.stale=true;this.analysis=undefined;this.selected=undefined;
    return {document,version,generation:this.generation};
  }
  current(token:Token):boolean {return token.document===this.document&&token.version===this.version&&token.generation===this.generation;}
  accept(token:Token,analysis:Analysis):boolean {
    if(!this.current(token)||analysis.document_id!==this.document||analysis.version!==this.version) return false;
    this.analysis=analysis;this.stale=false;
    this.selected=analysis.nodes.find(n=>n.name===this.previous?.name&&n.source.start_line===this.previous?.source.start_line)?.id??analysis.nodes[0]?.id;
    return true;
  }
  select(id:string):boolean {if(this.stale||!this.analysis?.nodes.some(n=>n.id===id))return false;this.selected=id;return true;}
  inspectionToken():Token {return {document:this.document,version:this.version,generation:this.generation,selected:this.selected};}
  matchesInspection(token:Token):boolean {return !this.stale&&this.current(token)&&token.selected===this.selected;}
}
function dictionary(value:unknown):value is Record<string,unknown> {return !!value&&typeof value==='object'&&!Array.isArray(value);}
export function validateParameters(value:unknown):Options {
  if(!dictionary(value)||!dictionary(value.parameters)||!dictionary(value.input_shapes??{})||!Array.isArray(value.program_ids??[]))throw new Error('constexpr 与 input_shapes 必须为 JSON 对象，program_ids 必须为数组。');
  const shapes=value.input_shapes as Options['input_shapes']??{};
  for(const shape of Object.values(shapes))if(!Array.isArray(shape)||!shape.every(x=>typeof x==='string'&&x.length>0||typeof x==='number'&&Number.isSafeInteger(x)&&x>0))throw new Error('input_shapes 的每个形状须为正整数或符号维度数组。');
  const ids=value.program_ids as number[]??[];
  if(!ids.every(x=>Number.isSafeInteger(x)&&x>=0))throw new Error('program_ids 必须为非负整数数组。');
  if(value.kernel!==undefined&&typeof value.kernel!=='string')throw new Error('kernel 必须为字符串。');
  return {parameters:value.parameters,input_shapes:shapes,program_ids:ids,kernel:value.kernel as string|undefined};
}
