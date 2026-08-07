const lanesEl = document.querySelector('#lanes');
const clockEl = document.querySelector('#clock');
const playBtn = document.querySelector('#play');
const speedEl = document.querySelector('#speed');
const speedValue = document.querySelector('#speedValue');
const eventEl = document.querySelector('#event');
const detailToggle = document.querySelector('#detailToggle');
const modes = { none: '无双缓冲、无多 Stream', cpu: '只有 CPU 双缓冲', both: 'CPU双缓冲+2个 CUDA stream' };
let mode = 'both', time = 0, playing = false, detailed = false, last = performance.now(), selected = null;
const total = 11;
const base = {
  none: [{name:'CPU / 单缓冲',jobs:[['F0 预处理',0,1,'prep'],['F0 后处理',2.9,3.4,'post'],['F1 预处理',3.4,4.4,'prep'],['F1 后处理',6.3,6.8,'post'],['F2 预处理',6.8,7.8,'prep'],['F2 后处理',9.7,10.2,'post']]},{name:'GPU / 默认 Stream',jobs:[['F0 H2D',1,1.4,'h2d'],['F0 计算',1.4,2.6,'compute'],['F0 D2H',2.6,2.9,'result'],['F1 H2D',4.4,4.8,'h2d'],['F1 计算',4.8,6,'compute'],['F1 D2H',6,6.3,'result'],['F2 H2D',7.8,8.2,'h2d'],['F2 计算',8.2,9.4,'compute'],['F2 D2H',9.4,9.7,'result']]}],
  cpu: [{name:'CPU / slot A',jobs:[['F0 预处理',0,1.5,'prep'],['F2 预处理',3.6,5.1,'prep']]},{name:'CPU / slot B',jobs:[['F1 预处理',1.5,3,'prep'],['F3 预处理',5.5,7,'prep']]},{name:'GPU / 推理',jobs:[['F0 H2D',1.5,1.9,'h2d'],['F0 计算',1.9,3.2,'compute'],['F0 D2H',3.2,3.5,'result'],['F1 H2D',3.5,3.9,'h2d'],['F1 计算',3.9,5.2,'compute'],['F1 D2H',5.2,5.5,'result'],['F2 H2D',5.5,5.9,'h2d'],['F2 计算',5.9,7.2,'compute'],['F2 D2H',7.2,7.5,'result'],['F3 H2D',7.5,7.9,'h2d'],['F3 计算',7.9,9.2,'compute'],['F3 D2H',9.2,9.5,'result']]},{name:'CPU / 后处理 worker',jobs:[['F0 后处理',3.5,4,'post'],['F1 后处理',5.5,6,'post'],['F2 后处理',7.1,7.6,'post'],['F3 后处理',8.6,9.1,'post']]}],
  both: [{name:'CPU / slot A',jobs:[['F0 预处理',0,1.5,'prep'],['F2 预处理',3.8,5.3,'prep'],['F4 预处理',7.6,8.5,'prep']]},{name:'CPU / slot B',jobs:[['F1 预处理',1.5,3,'prep'],['F3 预处理',5.3,6.8,'prep']]},{name:'GPU / Stream 0',jobs:[['F0 H2D',1.5,2.2,'h2d'],['F0 计算',2.2,3.6,'compute'],['F0 D2H',3.6,3.8,'result'],['F2 H2D',5.3,6,'h2d'],['F2 计算',6,7.4,'compute'],['F2 D2H',7.4,7.6,'result'],['F4 H2D',8.5,9.2,'h2d'],['F4 计算',9.2,10.6,'compute'],['F4 D2H',10.6,10.8,'result']]},{name:'GPU / Stream 1',jobs:[['F1 H2D',3,3.7,'h2d'],['F1 计算',3.7,5.1,'compute'],['F1 D2H',5.1,5.3,'result'],['F3 H2D',6.8,7.5,'h2d'],['F3 计算',7.5,8.9,'compute'],['F3 D2H',8.9,9.1,'result']]},{name:'CPU / 后处理 worker',jobs:[['F0 后处理',3.8,4.2,'post'],['F1 后处理',5.3,5.7,'post'],['F2 后处理',7.6,8,'post'],['F3 后处理',9.1,9.5,'post'],['F4 后处理',10.8,11,'post']]}]
};
const resourceDetail = [
  {name:'slot A / Host input', jobs:[['F0 H2D 后可复用',1.5,2.2,'prep'],['F2 H2D 后可复用',5.3,6,'prep'],['F4 H2D 后可复用',8.5,9.2,'prep']]},
  {name:'slot A / Device + context', jobs:[['F0 stream 0',1.5,3.6,'compute'],['F2 stream 0',5.3,7.4,'compute'],['F4 stream 0',8.5,10.6,'compute']]},
  {name:'slot A / Output', jobs:[['F0 D2H 后可复用',3.6,3.8,'result'],['F2 D2H 后可复用',7.4,7.6,'result'],['F4 D2H 后可复用',10.6,10.8,'result']]},
  {name:'slot B / Host input', jobs:[['F1 H2D 后可复用',1.5,3.7,'prep'],['F3 H2D 后可复用',5.3,7.5,'prep']]},
  {name:'slot B / Device + context', jobs:[['F1 stream 1',3,5.1,'compute'],['F3 stream 1',6.8,8.9,'compute']]},
  {name:'slot B / Output', jobs:[['F1 D2H 后可复用',5.1,5.3,'result'],['F3 D2H 后可复用',8.9,9.1,'result']]}
];
function render(){ lanesEl.innerHTML=''; const active=[]; base[mode].forEach(lane=>{ const row=document.createElement('div'); row.className='lane'; row.innerHTML=`<div class="lane-label">${lane.name}</div><div class="track"></div>`; const track=row.lastElementChild; lane.jobs.forEach(j=>{const [label,start,end,kind]=j; const el=document.createElement('div'); el.className=`job ${kind}${time>=start&&time<end?' running':''}`; el.textContent=label; el.style.left=`${start/total*100}%`; el.style.width=`${(end-start)/total*100}%`; el.dataset.info=`${label}：${start.toFixed(1)}–${end.toFixed(1)} s`; if(time>=start&&time<end) active.push(`${lane.name} 正在 ${label}`); el.onclick=()=>{document.querySelectorAll('.job').forEach(x=>x.classList.remove('selected'));el.classList.add('selected');eventEl.textContent=el.dataset.info;}; track.appendChild(el);}); lanesEl.appendChild(row); }); if (detailed && mode === 'both') {
  const heading = document.createElement('div'); heading.className = 'detail-heading'; heading.textContent = '细粒度资源生命周期（虚线之外表示资源已可复用）'; lanesEl.appendChild(heading);
  resourceDetail.forEach(lane => { const row=document.createElement('div'); row.className='lane detail-lane'; row.innerHTML=`<div class="lane-label">${lane.name}</div><div class="track"></div>`; const track=row.lastElementChild; lane.jobs.forEach(j=>{const [label,start,end,kind]=j; const el=document.createElement('div'); el.className=`job ${kind}${time>=start&&time<end?' running':''}`; el.textContent=label; el.style.left=`${start/total*100}%`; el.style.width=`${(end-start)/total*100}%`; track.appendChild(el);}); lanesEl.appendChild(row); });
}
const playhead=document.createElement('div'); playhead.className='playhead'; playhead.style.left=`calc(var(--label-width) + (100% - var(--label-width)) * ${time / total})`; lanesEl.appendChild(playhead); clockEl.textContent=`${time.toFixed(1)} s · ${modes[mode]}`; eventEl.textContent=active.length ? `此刻并行发生：${active.join('；')}` : '此刻没有任务执行：等待可用 slot 或下一帧。'; }
function tick(now){ const dt=(now-last)/1000; last=now; if(playing){time+=dt*Number(speedEl.value); if(time>total){time=0;} render();} requestAnimationFrame(tick); }
document.querySelectorAll('[data-mode]').forEach(b=>b.onclick=()=>{mode=b.dataset.mode;document.querySelectorAll('[data-mode]').forEach(x=>x.classList.toggle('active',x===b));time=0; if(mode!=='both'){detailed=false; detailToggle.classList.remove('active'); detailToggle.setAttribute('aria-pressed','false'); detailToggle.textContent='▸ 资源生命周期';} detailToggle.classList.toggle('visible',mode==='both'); render();});
detailToggle.onclick=()=>{ detailed=!detailed; detailToggle.classList.toggle('active',detailed); detailToggle.setAttribute('aria-pressed',detailed); detailToggle.textContent=detailed?'▾ 隐藏资源生命周期':'▸ 资源生命周期'; render(); };
document.querySelectorAll('[data-mode]').forEach(b=>b.addEventListener('click',()=>{ detailToggle.classList.toggle('visible',b.dataset.mode==='both'); }));
playBtn.onclick=()=>{playing=!playing;playBtn.textContent=playing?'❚❚ 暂停':'▶ 播放';}; document.querySelector('#reset').onclick=()=>{time=0;playing=false;playBtn.textContent='▶ 播放';render();}; speedEl.oninput=()=>speedValue.textContent=`${speedEl.value}×`;
render(); requestAnimationFrame(tick);
