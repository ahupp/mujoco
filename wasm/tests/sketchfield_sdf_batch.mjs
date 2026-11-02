// Compare the scalar and bulk-query engines using the same pure SDFs. Run:
// node wasm/tests/sketchfield_sdf_batch.mjs OLD/mujoco.js NEW/mujoco.js
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {dirname,join,resolve} from 'node:path';
import {pathToFileURL} from 'node:url';
const paths=process.argv.slice(2);assert.equal(paths.length,2);
const engines=await Promise.all(paths.map(async file=>{
  const path=resolve(file),{default:load}=await import(pathToFileURL(path));
  return load({wasmBinary:await readFile(join(dirname(path),'mujoco.wasm'))});
}));
const sphere=(x,y,z)=>Math.hypot(x,y,z)-1;
const box=(x,y,z)=>{const q=[Math.abs(x)-.8,Math.abs(y)-.6,Math.abs(z)-.7];return Math.hypot(...q.map(v=>Math.max(0,v)))+Math.min(0,Math.max(...q));};
const torus=(x,y,z)=>Math.hypot(Math.hypot(x,y)-.8,z)-.25;
let fields;const counts=engines.map(()=>({scalar:0,batches:0,points:0,max:0}));
for(const [i,engine]of engines.entries()){
  const distance=(id,x,y,z)=>fields[id](x,y,z);
  const gradient=(id,...p)=>p.map((v,axis)=>{const lo=[...p],hi=[...p];lo[axis]=v-1e-5;hi[axis]=v+1e-5;return (distance(id,...hi)-distance(id,...lo))/(hi[axis]-lo[axis]);});
  engine.sketchfieldDistance=(...args)=>{counts[i].scalar++;return distance(...args);};
  engine.sketchfieldGradient=(...args)=>{counts[i].scalar++;return gradient(...args);};
  engine.sketchfieldBatch=(id,heap,points,n,dist,grad)=>{
    counts[i].batches++;counts[i].points+=n;counts[i].max=Math.max(counts[i].max,n);
    for(let j=0;j<n;j++){const p=Array.from(heap.subarray(points+3*j,points+3*j+3));heap[dist+j]=distance(id,...p);if(grad)heap.set(gradient(id,...p),grad+3*j);}
  };
}
function xml(starts,iterations,position){
  const attributes=i=>['field','cx','cy','cz','hx','hy','hz'].map((name,j)=>`<config key="${name}" value="${[i,0,0,0,1.2,1.2,1.2][j]}"/>`).join('');
  return `<mujoco><option sdf_initpoints="${starts}" sdf_iterations="${iterations}"/><extension><plugin plugin="sketchfield.sdf"><instance name="a">${attributes(0)}</instance><instance name="b">${attributes(1)}</instance></plugin></extension><asset><mesh name="artifact" vertex="-.001 -.001 -.001 .001 -.001 -.001 -.001 .001 -.001 .001 .001 -.001 -.001 -.001 .001 .001 -.001 .001 -.001 .001 .001 .001 .001 .001"/></asset><worldbody><geom name="a" type="sdf" mesh="artifact"><plugin instance="a"/></geom><body pos="${position}" euler="13 7 23"><freejoint/><inertial mass="1" pos="0 0 0" diaginertia="1 1 1"/><geom name="b" type="sdf" mesh="artifact" density="0"><plugin instance="b"/></geom></body></worldbody><contact><pair geom1="a" geom2="b"/></contact></mujoco>`;
}
function contacts(engine,source){
  const model=engine.MjModel.from_xml_string(source),data=new engine.MjData(model);
  try{engine.mj_forward(model,data);const list=data.contact,result=[];
    try{for(let j=0;j<data.ncon;j++){const c=list.get(j);try{result.push([c.dist,...c.pos,...c.frame]);}finally{c.delete();}}}finally{list.delete();}
    return result;
  }finally{data.delete();model.delete();}
}
let cases=0,contactCases=0,maxDifference=0;
for(const shape of [sphere,box,torus])for(const starts of [1,31,32,33,65])for(const iterations of [0,1,30])for(const position of ['0 0 0','1.4 .1 .2','3 0 0']){
  fields=[sphere,shape];const source=xml(starts,iterations,position),a=contacts(engines[0],source),b=contacts(engines[1],source);
  assert.equal(a.length,b.length,`contact count: starts=${starts}, iterations=${iterations}, pos=${position}`);
  for(let i=0;i<a.length;i++)for(let j=0;j<a[i].length;j++){
    const delta=Math.abs(a[i][j]-b[i][j]);maxDifference=Math.max(maxDifference,delta);assert(delta<1e-11,`contact order/value: ${delta}`);
  }
  cases++;if(a.length)contactCases++;
}
assert(contactCases>0);assert.equal(counts[0].batches,0);assert(counts[1].batches>0);assert.equal(counts[1].max,32);
console.log(JSON.stringify({cases,contactCases,maxDifference,counts},null,2));
