import{C as w,T as d}from"./training.worker-B7LZuQQb.js";class h{name="webgpu";fallback=new w;devicePromise;static isSupported(){return typeof navigator<"u"&&"gpu"in navigator}binary(e,a,r){return this.fallback.binary(e,a,r)}unary(e,a,r){return this.fallback.unary(e,a,r)}pow(e,a,r){return this.fallback.pow(e,a,r)}sum(e){return this.fallback.sum(e)}max(e){return this.fallback.max(e)}min(e){return this.fallback.min(e)}broadcastTo(e,a){return this.fallback.broadcastTo(e,a)}reshape(e,a){return this.fallback.reshape(e,a)}transpose(e,a,r){return this.fallback.transpose(e,a,r)}matmul(e,a,r){return this.fallback.matmul(e,a,r)}softmax(e,a,r){return this.fallback.softmax(e,a,r)}dispose(e){this.fallback.dispose(e)}async addAsync(e,a){if(!E(e.shape,a.shape))return e.add(a);const r=await this.runElementwise(_,[e.data,a.data],e.size);return new d(r,e.shape)}async reluAsync(e){const a=await this.runElementwise(k,[e.data],e.size);return new d(a,e.shape)}async matmulAsync(e,a){if(e.shape.length!==2||a.shape.length!==2)return e.matmul(a);const[r,n]=e.shape,[i,u]=a.shape;if(n!==i)return e.matmul(a);const o=await this.device(),s=r*u,f=[p(o,e.data),p(o,a.data),l(o,s),m(o,new Uint32Array([r,n,u,0]))];return await g(o,v,f,Math.ceil(s/64)),new d(await b(o,f[2],s),[r,u])}async runElementwise(e,a,r){const n=await this.device(),i=[...a.map(u=>p(n,u)),l(n,r),m(n,new Uint32Array([r,0,0,0]))];return await g(n,e,i,Math.ceil(r/64)),b(n,i[a.length],r)}async device(){if(!h.isSupported())throw new Error("WebGPU is not available in this browser");return this.devicePromise??=(async()=>{const a=await navigator.gpu.requestAdapter();if(!a)throw new Error("No WebGPU adapter found");return a.requestDevice()})(),this.devicePromise}}const _=`
struct Meta {
  size: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
}

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> meta: Meta;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i >= meta.size) {
    return;
  }
  out[i] = a[i] + b[i];
}
`,k=`
struct Meta {
  size: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
}

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> meta: Meta;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i >= meta.size) {
    return;
  }
  out[i] = max(x[i], 0.0);
}
`,v=`
struct Meta {
  m: u32,
  k: u32,
  n: u32,
  _pad0: u32,
}

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> meta: Meta;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x;
  let total = meta.m * meta.n;
  if (index >= total) {
    return;
  }
  let row = index / meta.n;
  let col = index % meta.n;
  var sum = 0.0;
  for (var inner = 0u; inner < meta.k; inner = inner + 1u) {
    sum = sum + a[row * meta.k + inner] * b[inner * meta.n + col];
  }
  out[index] = sum;
}
`;function E(t,e){return t.length===e.length&&t.every((a,r)=>a===e[r])}function p(t,e){const a=t.createBuffer({size:e.byteLength,usage:c().STORAGE|c().COPY_DST});return t.queue.writeBuffer(a,0,e),a}function l(t,e){return t.createBuffer({size:e*Float32Array.BYTES_PER_ELEMENT,usage:c().STORAGE|c().COPY_SRC})}function m(t,e){const a=t.createBuffer({size:e.byteLength,usage:c().UNIFORM|c().COPY_DST});return t.queue.writeBuffer(a,0,e),a}async function g(t,e,a,r){const n=t.createShaderModule({code:e}),i=t.createComputePipeline({layout:"auto",compute:{module:n,entryPoint:"main"}}),u=t.createBindGroup({layout:i.getBindGroupLayout(0),entries:a.map((f,y)=>({binding:y,resource:{buffer:f}}))}),o=t.createCommandEncoder(),s=o.beginComputePass();s.setPipeline(i),s.setBindGroup(0,u),s.dispatchWorkgroups(r),s.end(),t.queue.submit([o.finish()]),await t.queue.onSubmittedWorkDone()}async function b(t,e,a){const r=a*Float32Array.BYTES_PER_ELEMENT,n=t.createBuffer({size:r,usage:c().COPY_DST|c().MAP_READ}),i=t.createCommandEncoder();i.copyBufferToBuffer(e,0,n,0,r),t.queue.submit([i.finish()]),await n.mapAsync(S().READ);const u=new Float32Array(n.getMappedRange().slice(0));return n.unmap(),u}function c(){return globalThis.GPUBufferUsage}function S(){return globalThis.GPUMapMode}export{h as WebGPUBackend,_ as addShaderWGSL,v as matmulShaderWGSL,k as reluShaderWGSL};
