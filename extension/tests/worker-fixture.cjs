const readline=require('node:readline');
readline.createInterface({input:process.stdin}).on('line',line=>{
 const r=JSON.parse(line);
 if(r.method==='exit') process.exit(3);
 if(r.method==='wait') return;
 if(r.method==='analyze') Atomics.wait(new Int32Array(new SharedArrayBuffer(4)),0,0,1800);
 if(r.method==='stderr') {process.stderr.write('diagnostic only\n');}
 if(r.method==='error') process.stdout.write(JSON.stringify({id:r.id,error:{type:'ValueError',message:'invalid shape'}})+'\n');
 else setTimeout(()=>process.stdout.write(JSON.stringify({id:r.id,result:r.params})+'\n'),r.params.delay||0);
});
