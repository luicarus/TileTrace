const fs=require('node:fs');const path=require('node:path');
const root=path.resolve(__dirname,'../..');const target=path.resolve(__dirname,'../python/triton_transform');
fs.mkdirSync(target,{recursive:true});fs.mkdirSync(path.join(root,'dist'),{recursive:true});
for(const name of fs.readdirSync(path.join(root,'triton_transform')))if(name.endsWith('.py'))fs.copyFileSync(path.join(root,'triton_transform',name),path.join(target,name));
console.log('Bundled stdlib Python worker in extension/python/triton_transform');
