const fs=require('node:fs');const path=require('node:path');
const root=path.resolve(__dirname,'../..');const bundle=path.resolve(__dirname,'../python');const target=path.join(bundle,'tiletrace');
fs.mkdirSync(path.join(root,'dist'),{recursive:true});
// Rebuild from scratch so renamed or deleted modules cannot survive as stale copies in the VSIX.
fs.rmSync(bundle,{recursive:true,force:true});
fs.mkdirSync(target,{recursive:true});
for(const name of fs.readdirSync(path.join(root,'tiletrace')))if(name.endsWith('.py'))fs.copyFileSync(path.join(root,'tiletrace',name),path.join(target,name));
// vsce requires the license inside the extension folder; keep one source of truth at the repo root.
fs.copyFileSync(path.join(root,'LICENSE'),path.resolve(__dirname,'../LICENSE'));
console.log('Bundled stdlib Python worker in extension/python/tiletrace and LICENSE');
