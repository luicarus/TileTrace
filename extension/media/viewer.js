/* Original logical-coordinate viewer. No kernel values are evaluated here. */
(() => {
  'use strict';
  const vscode = acquireVsCodeApi();
  const {operationNodes, operationId, contextInputs} = TileTraceOperations;
  const root = document.getElementById('app');
  let state = {stale:true,version:0,generation:0};
  let inspection = null;
  let outputIndex = null;
  let inspectionSequence = 0;
  let pendingInspection = null;
  let localError = '';
  const slices = new Map();
  const drafts = new Map();
  const remembered = vscode.getState() || {};
  const readingPositions = new Map();
  const consumerGraphs = new WeakMap();
  let flowLayoutFrame=null;
  const flowObserver=typeof ResizeObserver==='function'?new ResizeObserver(()=>queueFlowLayout()):null;
  let renderedFile = null;
  let renderedStale = false;
  let interactedWhileStale = false;
  const markStaleInteraction=()=>{if(renderedStale)interactedWhileStale=true;};
  // Restoring focus is not user activity. Only explicit interaction can replace
  // the last full-view viewport while the loading layout is temporarily short.
  for(const event of ['pointerdown','keydown','wheel','input','change'])root.addEventListener(event,markStaleInteraction,{capture:true,passive:event==='wheel'});
  const post = (type, data = {}) => vscode.postMessage({type,generation:state.generation,...data});
  const el = (tag, text, className) => {
    const element = document.createElement(tag);
    if(text !== undefined) element.textContent = String(text);
    if(className) element.className = className;
    return element;
  };
  const button = (text, action, className = '') => {
    const b = el('button',text,className); b.type='button'; b.addEventListener('click',action); return b;
  };
  const shapeText = node => node.shape.length ? '['+node.shape.join(', ')+']' : '[] · 标量';
  const coordText = index => '['+index.join(', ')+']';
  const loopText = node => (node.attrs?.loops||[]).map(loop=>`${loop.variable}=${loop.value}`).join(' · ');
  function section(title) {const box=el('section',undefined,'section');box.append(el('h2',title));return box;}
  function selected() {
    const nodes=state.analysis?.nodes??[];
    const id=state.show_all?state.selected:operationId(nodes,state.selected);
    return nodes.find(node=>node.id===id);
  }

  function operationDetails(node, nodes) {
    const fields=[];
    const display=value=>value===undefined||value===null?'未知':typeof value==='string'?value:JSON.stringify(value);
    const scalar=input=>{
      if(!input)return '未知';
      if(Object.hasOwn(input.attrs,'value'))return input.op==='constant'?display(input.attrs.value):`${input.name} = ${display(input.attrs.value)}`;
      return `${input.attrs.symbol||input.name}（符号）`;
    };
    const context=contextInputs(node,nodes);
    if(loopText(node))fields.push(['循环分块',loopText(node)]);
    if(node.op==='arange') {
      const inputs=node.inputs.map(id=>nodes.find(n=>n.id===id));
      fields.push(['起点',inputs[0]?scalar(inputs[0]):display(node.attrs.start)],['终点',inputs[1]?scalar(inputs[1]):display(node.attrs.end)],['长度',display(node.shape[0])]);
    } else {
      for(const input of context)fields.push([input.op==='constant'?'常量':input.name,scalar(input)]);
      for(const [key,label] of [['axes','轴'],['permutation','轴排列'],['keep_dims','保留归约轴'],['can_reorder','允许重排'],['fill_value','静态填充值'],['contraction','矩阵乘法归约长度'],['target_dtype','目标类型']]) {
        if(Object.hasOwn(node.attrs,key))fields.push([label,display(node.attrs[key])]);
      }
    }
    if(!fields.length)return null;
    const box=el('div',undefined,'operation-details');box.append(el('h2','参数与属性'));
    const list=el('dl');
    for(const [label,value] of fields)list.append(el('dt',label),el('dd',value));
    box.append(list);return box;
  }
  function draft() {
    const key=state.file||'';
    if(!drafts.has(key)) drafts.set(key,remembered[key] || {
      parameters:JSON.stringify(state.parameters||{},null,2),
      input_shapes:JSON.stringify(state.input_shapes||{},null,2),
      program_ids:JSON.stringify(state.program_ids||[])
    });
    return drafts.get(key);
  }
  function parameters() {
    const details=el('details',undefined,'parameters');details.open=!!state.analysis?.missing_parameters.length||!state.analysis;
    details.append(el('summary','分析参数 · JSON'));
    const fields=draft();
    for(const [key,label] of [['parameters','constexpr / 标量参数'],['input_shapes','输入形状（可选）'],['program_ids','program ID（可选）']]){
      const wrapper=el('label',undefined,'field');wrapper.append(el('span',label));
      const input=el('textarea');input.value=fields[key];input.rows=key==='program_ids'?1:3;input.spellcheck=false;input.setAttribute('aria-label',label);
      input.addEventListener('input',()=>{fields[key]=input.value;remembered[state.file||'']={...fields};vscode.setState(remembered);});wrapper.append(input);details.append(wrapper);
    }
    details.append(button('应用参数并重新分析',()=>{
      try {
        const options={kernel:state.analysis?.kernel||undefined,parameters:JSON.parse(fields.parameters),input_shapes:JSON.parse(fields.input_shapes),program_ids:JSON.parse(fields.program_ids)};
        localError='';inspection=null;outputIndex=null;pendingInspection=null;post('applyParameters',{options});
      } catch(error) {localError='JSON 格式无效：'+error.message;render();}
    },'primary'));
    return details;
  }
  function sliceFor(node) {
    if(!slices.has(node.id)) slices.set(node.id,{prefix:node.shape.slice(0,-2).map(()=>0),row:0,col:0,expanded:false});
    return slices.get(node.id);
  }
  function numericControl(label,axis,value,max,change,output) {
    const wrapper=el('label',undefined,'slice-control');wrapper.append(el('span',label));
    const input=el('input');input.type='number';input.min='0';input.max=String(max);input.step='1';input.value=String(value);input.dataset.axis=String(axis);input.setAttribute('aria-label',label);
    input.addEventListener('change',()=>{const n=Number(input.value);if(!Number.isSafeInteger(n)||n<0||n>max){input.value=String(value);return;}if(output){pendingInspection=null;outputIndex=null;inspection=null;}change(n);render();});
    wrapper.append(input);return wrapper;
  }
  function vectorAxis(node) {
    // Rank-one tensors have no inherent row/column direction. Use actual
    // new-axis consumers to choose a presentation, while preserving shape.
    const nodes=state.analysis?.nodes||[];
    if(!consumerGraphs.has(nodes)) {
      const graph=new Map();
      for(const use of nodes)for(const id of use.inputs){if(!graph.has(id))graph.set(id,[]);graph.get(id).push(use);}
      consumerGraphs.set(nodes,graph);
    }
    const consumers=consumerGraphs.get(nodes);
    const expansionAxis=(use,id)=>{
      if(use.status==='unsupported'||use.attrs?.mapping!=='expand_dims'||!use.inputs.includes(id)||use.shape.length!==2)return null;
      const axes=use.attrs.axes;
      return axes?.length===1&&axes[0]===1?0:axes?.length===1&&axes[0]===0?1:null;
    };
    const current=selected();const contextual=current&&expansionAxis(current,node.id);
    if(contextual===0||contextual===1)return contextual;
    const axes=new Set(),seen=new Set(),queue=[node];
    for(let i=0;i<queue.length;i++) {
      const tensor=queue[i];if(seen.has(tensor.id))continue;seen.add(tensor.id);
      for(const use of consumers.get(tensor.id)||[]) {
        if(use.status==='unsupported')continue;
        const axis=expansionAxis(use,tensor.id);if(axis!==null)axes.add(axis);
        if(use.shape.length===1&&use.shape[0]===tensor.shape[0]&&['identity','broadcast'].includes(use.attrs?.mapping)&&!seen.has(use.id))queue.push(use);
      }
    }
    return axes.size===1?[...axes][0]:null;
  }
  function axisCoordinates(length,start,limit,overview) {
    if(overview&&length>8)return [0,1,2,null,length-3,length-2,length-1];
    return Array.from({length:Math.min(length-start,limit)},(_,i)=>start+i);
  }
  function gridLayout(node) {
    const rank=node.shape.length,axis=rank===1?vectorAxis(node):null,vertical=axis===0;
    const rows=rank>=2?node.shape[rank-2]:vertical?node.shape[0]:1;
    const cols=rank>=2?node.shape[rank-1]:rank===1&&!vertical?node.shape[0]:1;
    const slice=sliceFor(node);
    const maxRows=vertical||cols===1?16:8,maxCols=16;
    const startRow=slice.expanded?Math.min(rank===1?(vertical?slice.col:0):slice.row,rows-1):0;
    const startCol=slice.expanded?Math.min(vertical?0:slice.col,cols-1):0;
    return {axis,vertical,rows,cols,maxRows,maxCols,startRow,startCol,
      rowIndices:axisCoordinates(rows,startRow,maxRows,!slice.expanded),
      colIndices:axisCoordinates(cols,startCol,maxCols,!slice.expanded)};
  }
  function gridMode(node,output,expanded,axis=null,start=0) {
    const slice=sliceFor(node);slice.expanded=expanded;
    if(axis!==null){if(axis===node.shape.length-1)slice.col=start;else slice.row=start;}
    if(output){pendingInspection=null;outputIndex=null;inspection=null;}
    render();
  }
  function ellipsis(node,output,axis,length,key) {
    const gap=button('…',event=>{
      const owner=event.currentTarget.closest('[data-card-node]');const occurrence=owner.dataset.cardOccurrence;
      gridMode(node,output,true,axis,3);
      const card=Array.from(root.querySelectorAll('[data-card-node]')).find(candidate=>candidate.dataset.cardNode===node.id&&candidate.dataset.cardRole===(output?'output':'input')&&candidate.dataset.cardOccurrence===occurrence);
      card?.querySelector('[data-role="toggle-grid-window"]')?.focus({preventScroll:true});
    },'axis-ellipsis');
    gap.dataset.axis=String(axis);gap.dataset.gapKey=key;
    const label=`轴 ${axis} 的 3–${length-4} 坐标已省略，点击查看中间窗口`;
    gap.title=label;gap.setAttribute('aria-label',label);return gap;
  }
  function tensorCard(node,output,occurrence=0) {
    const card=el('article',undefined,'tensor-card');card.dataset.cardNode=node.id;
    card.dataset.cardRole=output?'output':'input';card.dataset.cardOccurrence=String(occurrence);
    const header=el('div',undefined,'card-title');const shapeBadge=el('span',shapeText(node),'pill');header.append(el('strong',node.name||node.op),shapeBadge);card.append(header);
    card.append(el('p',`rank ${node.shape.length} · ${node.op} · ${node.status}`,'muted'));
    if(node.status==='unsupported'){card.append(el('p','此操作未获支持，无法证明精确坐标映射。','notice'));return card;}
    if(!node.shape.every(dim=>Number.isSafeInteger(dim)&&dim>=0)) {card.append(el('p','符号形状：补充参数后可显示逻辑坐标。','notice'));return card;}
    const shape=node.shape;const rank=shape.length;const slice=sliceFor(node);
    let total=1n;for(const size of shape)total*=BigInt(size);
    if(total===0n){card.append(el('p','空张量，没有可选坐标。','notice'));return card;}
    const layout=gridLayout(node);const {rows,cols,maxRows,maxCols,vertical}=layout;
    shapeBadge.title=rank===0?'标量，shape []':rank===1?(layout.axis===0?'一维索引，对应矩阵行轴，纵向展示；shape '+shapeText(node):layout.axis===1?'一维索引，对应矩阵列轴，横向展示；shape '+shapeText(node):'一维向量，未指定行列方向；shape '+shapeText(node)):`当前切片包含 ${rows} 行 × ${cols} 列；展开窗口最多展示 ${maxRows} 行 × ${maxCols} 列。`;
    const controls=el('div',undefined,'slice-controls');
    if(rows>8||cols>8){const toggle=button(slice.expanded?'收起为首尾概览':'展开显示窗口',()=>gridMode(node,output,!slice.expanded));toggle.dataset.role='toggle-grid-window';toggle.setAttribute('aria-expanded',String(slice.expanded));toggle.title='长度超过 8 时概览保留首尾各 3 项；展开查看中间坐标。';controls.append(toggle);}
    if(rank>2){controls.title='前缀轴选择一个切片；网格对应最后两轴。';for(let axis=0;axis<rank-2;axis++)controls.append(numericControl(`轴 ${axis}`,axis,slice.prefix[axis],shape[axis]-1,n=>{slice.prefix[axis]=n;},output));}
    if(slice.expanded&&rows>maxRows)controls.append(numericControl(`轴 ${vertical?0:rank-2} 起点`,vertical?0:rank-2,vertical?slice.col:slice.row,rows-1,n=>{if(vertical)slice.col=n;else slice.row=n;},output));
    if(slice.expanded&&cols>maxCols)controls.append(numericControl(`轴 ${rank-1} 起点`,rank-1,slice.col,cols-1,n=>{slice.col=n;},output));
    if(controls.childNodes.length)card.append(controls);
    const table=el('table',undefined,'index-grid');table.setAttribute('aria-label',`${node.name} 逻辑坐标`);
    const head=el('thead');const heading=el('tr');heading.append(el('th',rank>=2?`轴 ${rank-2} / ${rank-1}`:'坐标'));
    const {rowIndices,colIndices}=layout;
    for(const col of colIndices){const th=el('th');if(col===null&&rank>=2)th.append(ellipsis(node,output,rank-1,cols,'header-col'));else th.textContent=col===null?'…':vertical?'索引':rank?String(col):'标量';heading.append(th);}head.append(heading);table.append(head);
    const body=el('tbody');let displayed=0;
    for(const row of rowIndices){
      const tr=el('tr');tr.append(el('th',row===null?'…':rank>=2||vertical?row:'—'));
      if(row===null){const td=el('td');td.colSpan=colIndices.length;td.append(ellipsis(node,output,vertical?0:rank-2,rows,'row'));tr.append(td);body.append(tr);continue;}
      for(const col of colIndices){
        if(col===null){const td=el('td');td.append(ellipsis(node,output,rank-1,cols,`col:${row}`));tr.append(td);continue;}
        const index=rank===0?[]:rank===1?[vertical?row:col]:[...slice.prefix,row,col];
        const td=el('td');const cell=button(rank===0?'·':rank===1?String(index[0]):`${index[rank-2]},${index[rank-1]}`,()=>{
          if(!output)return;outputIndex=index;inspection=null;
          const request_id=String(++inspectionSequence);pendingInspection={request_id,generation:state.generation,node_id:node.id,index};
          post('inspectIndex',{node_id:node.id,index,request_id});render();
        },'cell');
        cell.dataset.node=node.id;cell.dataset.index=JSON.stringify(index);if(output)cell.dataset.output='true';
        cell.title=`${node.name} ${coordText(index)} · 逻辑坐标，非数值`;cell.setAttribute('aria-label',cell.title);cell.disabled=!output;
        if(output&&JSON.stringify(outputIndex)===JSON.stringify(index))cell.classList.add('selected-cell');
        const origin=inspection?.status==='exact'?inspection.origins.find(origin=>origin.node_id===node.id):undefined;
        if(!output&&origin?.indices.some(coordinate=>JSON.stringify(coordinate)===JSON.stringify(index)))cell.classList.add('origin');
        td.append(cell);tr.append(td);displayed++;
      }
      body.append(tr);
    }
    table.append(body);const scroll=el('div',undefined,'grid-scroll');scroll.title=`显示 ${displayed} 个逻辑坐标 / 总计 ${total.toString()} 个${rank>2?'（当前切片）':''}；每卡最多 128 个。`;
    table.setAttribute('aria-label',`${node.name} 逻辑坐标；${shapeBadge.title} ${scroll.title}${rank>2?' '+controls.title:''}`);
    scroll.append(table);card.append(scroll);
    return card;
  }
  function mapping() {
    const box=section('元素来源');box.classList.add('mapping');
    if(!inspection){box.append(el('p',outputIndex?`正在查询输出 ${coordText(outputIndex)}…`:'点击输出网格坐标，查看直接输入中的对应位置。','muted'));return box;}
    if(inspection.status!=='exact'){box.append(el('p',`坐标映射不可用：${inspection.message||'符号参数、未知重排或未支持操作无法证明映射。'}`,'notice'));return box;}
    box.append(el('p',`输出 ${coordText(inspection.output_index||[])} → 直接输入坐标`));
    const current=selected();const folded=new Set(state.show_all||!current?[]:contextInputs(current,state.analysis.nodes).map(input=>input.id));
    for(const origin of inspection.origins){
      const n=state.analysis.nodes.find(n=>n.id===origin.node_id);const line=el('div',undefined,'origin-description');line.dataset.originNode=origin.node_id;line.append(el('strong',n?.name||origin.node_id));
      const coordinates=origin.indices.map(coordText);const compact=coordinates.length>8?[...coordinates.slice(0,3),'…',...coordinates.slice(-3)]:coordinates;
      line.append(el('p',compact.join(' · ')||'无输入坐标','coordinates'));
      line.append(el('p',`返回 ${origin.indices.length} / 总计 ${origin.total}${origin.truncated?' · 来源枚举已截断':''}${origin.indices.length>8?' · 文本只显示返回坐标的首尾各 3 项':''}`,'muted'));
      if(folded.has(origin.node_id)){line.append(el('p','该标量输入已显示在参数与属性中。','muted'));box.append(line);continue;}
      if(n){const slice=sliceFor(n);const rank=n.shape.length;const layout=gridLayout(n);const visible=origin.indices.filter(index=>{
        if(rank===0)return true;
        if(rank>2&&!slice.prefix.every((v,i)=>v===index[i]))return false;
        const row=rank>=2?index[rank-2]:layout.vertical?index[0]:0;const col=layout.vertical?0:index[rank-1];
        return layout.rowIndices.includes(row)&&layout.colIndices.includes(col);
      }).length;line.append(el('p',`当前网格高亮 ${visible} / ${origin.indices.length} 个返回来源${visible<origin.indices.length?'；其余位于当前切片、省略区域或显示范围之外。':''}`,'muted'));}
      if(n&&origin.indices.length){const jump=button('定位首个来源坐标',()=>{const index=origin.indices[0];const slice=sliceFor(n);slice.expanded=true;slice.prefix=index.slice(0,-2);slice.row=index.length>=2?index[index.length-2]:0;slice.col=index.length?index[index.length-1]:0;render();});jump.dataset.originTarget=origin.node_id;line.append(jump);}
      box.append(line);
    }
    if(!inspection.origins.length)box.append(el('p','此节点没有直接输入来源。','muted'));
    return box;
  }
  function viewKey(element) {
    if(!element)return null;
    const card=element.closest('[data-card-node]');
    const scope=card?`${card.dataset.cardRole}:${card.dataset.cardNode}:${card.dataset.cardOccurrence}`:'';
    if(element.matches('.operation-list'))return 'operations';
    if(element.matches('.expression'))return `expression:${element.dataset.ownerNode}`;
    if(element.matches('.grid-scroll'))return `grid:${scope}`;
    if(element.matches('.cell'))return `cell:${scope}:${element.dataset.index}`;
    if(element.matches('.axis-ellipsis'))return `gap:${scope}:${element.dataset.gapKey}`;
    if(element.matches('[data-role="toggle-grid-window"]'))return `grid-mode:${scope}`;
    if(element.matches('.operation'))return `operation:${element.dataset.node}`;
    if(element.matches('textarea'))return `parameter:${element.getAttribute('aria-label')}`;
    if(element.matches('input[type="number"]'))return `slice:${scope}:${element.dataset.axis}:${element.getAttribute('aria-label')}`;
    if(element.matches('[data-role="show-all-nodes"]'))return 'visibility';
    if(element.matches('select[aria-label="Kernel"]'))return 'kernel';
    if(element.matches('.parameters summary'))return 'parameters-summary';
    if(element.dataset.originTarget)return `origin:${element.dataset.originTarget}`;
    if(element.matches('button'))return `button:${element.textContent}`;
    return null;
  }
  function captureReadingPosition() {
    const scrolls=new Map();
    for(const element of root.querySelectorAll('.operation-list,.expression,.grid-scroll,textarea')) {
      const key=viewKey(element);
      if(key)scrolls.set(key,{top:element.scrollTop,left:element.scrollLeft,height:element.style.height,width:element.style.width});
    }
    const active=document.activeElement;
    const key=root.contains(active)&&document.hasFocus()?viewKey(active):null;
    const focus=key?{key}:null;
    if(focus&&active.tagName==='TEXTAREA')Object.assign(focus,{start:active.selectionStart,end:active.selectionEnd,direction:active.selectionDirection});
    return {x:window.scrollX,y:window.scrollY,scrolls,open:root.querySelector('.parameters')?.open,focus};
  }
  function restoreReadingPosition(position) {
    const elements=Array.from(root.querySelectorAll('button,input,textarea,select,summary,.operation-list,.expression,.grid-scroll'));
    const details=root.querySelector('.parameters');
    if(details&&position.open!==undefined)details.open=position.open;
    for(const element of elements) {
      const saved=position.scrolls?.get(viewKey(element));
      if(saved&&element.tagName==='TEXTAREA'){element.style.height=saved.height;element.style.width=saved.width;}
    }
    const focus=position.focus&&elements.find(element=>viewKey(element)===position.focus.key);
    if(focus){focus.focus({preventScroll:true});if(focus.tagName==='TEXTAREA'&&position.focus.start!==undefined)focus.setSelectionRange(position.focus.start,position.focus.end,position.focus.direction);}
    for(const element of elements) {
      const saved=position.scrolls?.get(viewKey(element));
      if(saved){element.scrollTop=saved.top;element.scrollLeft=saved.left;}
    }
    if(window.scrollX!==position.x||window.scrollY!==position.y)window.scrollTo(position.x,position.y);
  }
  function queueFlowLayout() {
    if(flowLayoutFrame!==null)return;
    if(typeof requestAnimationFrame!=='function'){fitFlowLayout();return;}
    flowLayoutFrame=requestAnimationFrame(()=>{flowLayoutFrame=null;fitFlowLayout();});
  }
  function fitFlowLayout() {
    const flow=root.querySelector('.flow');if(!flow)return;
    const input=flow.querySelector('.flow-inputs'),output=flow.querySelector('.flow-output'),arrow=flow.querySelector('.flow-arrow');
    if(!input||!output||!arrow)return;
    const style=getComputedStyle(root);const available=root.getBoundingClientRect().width-(parseFloat(style.paddingLeft)||0)-(parseFloat(style.paddingRight)||0);
    if(available<=0)return;
    const gap=parseFloat(getComputedStyle(flow).gap)||12;
    const needed=input.getBoundingClientRect().width+output.getBoundingClientRect().width+arrow.getBoundingClientRect().width+2*gap;
    const layout=Math.ceil(needed)<=available?'horizontal':'vertical';
    if(flow.dataset.layout!==layout)flow.dataset.layout=layout;
  }
  function observeFlowLayout() {
    flowObserver?.disconnect();fitFlowLayout();
    if(flowObserver){flowObserver.observe(root);for(const part of root.querySelectorAll('.flow-inputs,.flow-output'))flowObserver.observe(part);}
  }
  window.addEventListener('resize',fitFlowLayout);
  function render() {
    if(renderedFile!==null) {
      const current=captureReadingPosition();const saved=readingPositions.get(renderedFile);
      if(!renderedStale||!saved)readingPositions.set(renderedFile,current);
      else {
        // A loading placeholder clamps scroll to zero. Keep the last full-view
        // position unless the user is actively working in a surviving control.
        saved.open=current.open;
        for(const [key,value] of current.scrolls)saved.scrolls.set(key,value);
        if(interactedWhileStale){saved.x=current.x;saved.y=current.y;saved.focus=current.focus;}
      }
    }
    const file=state.file||'';
    const position=readingPositions.get(file)??{x:0,y:0,scrolls:new Map()};
    const fragment=document.createDocumentFragment();
    renderContent(fragment);
    // Build the complete replacement off-page, then restore reading state.
    root.replaceChildren(fragment);
    observeFlowLayout();
    restoreReadingPosition(position);
    renderedFile=file;renderedStale=!!state.stale;interactedWhileStale=false;
  }
  function renderContent(target) {
    const header=el('header');const title=el('div',undefined,'title-row');title.append(el('h1','TileTrace · 变换'),el('span',state.stale?'等待分析':'静态分析','status'));header.append(title);
    header.append(el('p','形状与逻辑坐标 · 不执行 kernel','muted'));
    const file=el('p',`${state.file||'尚未选择源文件'} · v${state.version||0}`,'file');file.title=state.file||'';header.append(file);
    const actions=el('div',undefined,'toolbar');actions.append(button('复制 Agent 提示词',()=>post('copyPrompt')));
    if(state.analysis?.kernels.length){const label=el('label',undefined,'kernel-select');label.append(el('span','Kernel'));const select=el('select');select.setAttribute('aria-label','Kernel');for(const name of state.analysis.kernels){const option=el('option',name);option.value=name;option.selected=name===state.analysis.kernel;select.append(option);}select.addEventListener('change',()=>{try{const fields=draft();post('applyParameters',{options:{kernel:select.value,parameters:JSON.parse(fields.parameters),input_shapes:JSON.parse(fields.input_shapes),program_ids:JSON.parse(fields.program_ids)}});}catch(error){localError='JSON 格式无效：'+error.message;render();}});label.append(select);actions.append(label);}
    header.append(actions);target.append(header,parameters());
    if(localError||state.error)target.append(el('p',localError||state.error,'error'));
    if(state.stale){target.append(el('p','源码或参数已变化，坐标映射已清除。等待当前版本分析完成。','notice'));return;}
    const analysis=state.analysis;if(!analysis)return;
    if(analysis.missing_parameters.length)target.append(el('p','待补参数：'+analysis.missing_parameters.join('、'),'notice'));
    if(analysis.diagnostics.length){const box=section('诊断');for(const diagnostic of analysis.diagnostics)box.append(el('p',`${diagnostic.severity} · ${diagnostic.message}`,'diagnostic'));target.append(box);}
    if(!analysis.nodes.length){target.append(el('p','没有可显示的操作。请检查 kernel、语法和诊断，补齐所需参数。','notice'));return;}
    const steps=operationNodes(analysis.nodes,!!state.show_all);const node=selected();
    const browse=section(`${state.show_all?'全部节点':'张量步骤'} · ${steps.length}`);
    const visibility=el('label',undefined,'visibility-control');const toggle=el('input');toggle.type='checkbox';toggle.checked=!!state.show_all;toggle.dataset.role='show-all-nodes';
    toggle.addEventListener('change',()=>{pendingInspection=null;inspection=null;outputIndex=null;post('setNodeVisibility',{show_all:toggle.checked});});visibility.append(toggle,el('span','显示全部节点'));browse.append(visibility);
    if(!state.show_all)browse.append(el('p',`已折叠 ${analysis.nodes.length-steps.length} 个参数、常量或准备节点。`,'muted'));
    const position=steps.findIndex(n=>n.id===node?.id);const toolbar=el('div',undefined,'toolbar');
    const change=delta=>{const nextNode=steps[position+delta];if(nextNode)post('selectNode',{node_id:nextNode.id});};
    const prev=button('← 上一步',()=>change(-1));prev.disabled=position<=0;const next=button('下一步 →',()=>change(1));next.disabled=position<0||position>=steps.length-1;toolbar.append(prev,el('span',`${position+1} / ${steps.length}`,'muted'),next);browse.append(toolbar);
    const list=el('div',undefined,'operation-list');for(const item of steps){const b=button(`${item.name||item.op} · ${item.op} ${shapeText(item)}${loopText(item)?' · '+loopText(item):''}`,()=>post('selectNode',{node_id:item.id}),'operation');b.dataset.node=item.id;b.classList.toggle('active',item.id===node?.id);b.setAttribute('aria-current',String(item.id===node?.id));list.append(b);}browse.append(list);target.append(browse);
    if(!steps.length){target.append(el('p','当前没有张量步骤；可开启“显示全部节点”查看参数和其他分析节点。','notice'));return;}
    if(!node)return;
    const operation=section(`${node.name||node.op} · ${node.op}`);const expression=el('pre',state.expressions?.[node.id]||node.name||node.op,'expression');expression.dataset.ownerNode=node.id;operation.append(expression,button(`查看源码 · 第 ${node.source.start_line} 行`,()=>post('revealSource',{node_id:node.id})));
    const details=operationDetails(node,analysis.nodes);if(details)operation.append(details);target.append(operation);
    const hidden=new Set(state.show_all?[]:contextInputs(node,analysis.nodes).map(input=>input.id));
    const dataInputs=node.inputs.map(id=>analysis.nodes.find(n=>n.id===id)).filter(input=>input&&!hidden.has(input.id));
    const flow=el('div',undefined,'flow');const inputs=section('输入');inputs.classList.add('flow-inputs');dataInputs.forEach((input,index)=>inputs.append(tensorCard(input,false,index)));
    if(!dataInputs.length)inputs.append(el('p',hidden.size?'标量输入已列入上方参数与属性。':'此操作创建逻辑形状，没有直接输入。','muted'));
    const arrow=el('div',undefined,'flow-arrow');arrow.setAttribute('role','img');arrow.setAttribute('aria-label',`输入经过 ${node.op} 得到输出`);
    const direction=el('span',undefined,'flow-direction');direction.setAttribute('aria-hidden','true');const operator=el('span',node.op,'flow-operator');operator.setAttribute('aria-hidden','true');arrow.append(direction,operator);
    flow.append(inputs,arrow);const output=section('输出');output.classList.add('flow-output');output.append(tensorCard(node,true));flow.append(output);target.append(flow,mapping());
  }
  window.addEventListener('message',event=>{
    const message=event.data;if(!message||typeof message!=='object')return;
    if(message.type==='state'){
      const next=message.state;
      if(next.file!==state.file||next.generation!==state.generation||next.version!==state.version)slices.clear();
      if(next.stale||next.show_all!==state.show_all||next.file!==state.file||next.selected!==state.selected||next.generation!==state.generation||next.version!==state.version){inspection=null;outputIndex=null;pendingInspection=null;}
      state=next;render();
    } else if(message.type==='inspection'&&!state.stale){
      if(!pendingInspection||message.request_id!==pendingInspection.request_id||message.generation!==state.generation||message.generation!==pendingInspection.generation||message.node_id!==state.selected||message.node_id!==pendingInspection.node_id)return;
      if(message.inspection.node?.id!==state.selected||JSON.stringify(message.inspection.output_index)!==JSON.stringify(pendingInspection.index))return;
      pendingInspection=null;
      inspection=message.inspection;render();
    }
  });
  render();post('ready');
})();
