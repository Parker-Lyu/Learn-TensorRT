const lanesEl = document.querySelector('#lanes');
const clockEl = document.querySelector('#clock');
const playBtn = document.querySelector('#play');
const speedEl = document.querySelector('#speed');
const speedValue = document.querySelector('#speedValue');
const eventEl = document.querySelector('#event');
const modes = { none: '无双缓冲、无多 Stream', cpu: '只有 CPU 双缓冲', gpu: '只有多 CUDA Stream', both: '组合（生产环境）' };
let mode = 'both', time = 0, playing = false, last = performance.now(), selected = null;
const total = 10;
const base = {
  none: [{name:'CPU / 单缓冲',jobs:[['F0 准备',0,1.4,'prep'],['F1 准备',3.4,4.8,'prep'],['F2 准备',6.8,8.2,'prep']]},{name:'GPU / 默认 Stream',jobs:[['F0 H2D',1.4,2,'h2d'],['F0 计算',2,3.4,'compute'],['F1 H2D',4.8,5.4,'h2d'],['F1 计算',5.4,6.8,'compute'],['F2 H2D',8.2,8.8,'h2d'],['F2 计算',8.8,10,'compute']]}],
  cpu: [{name:'CPU / 双缓冲', jobs:[['F0 准备',0,1.5,'prep'],['F1 准备',1.5,3,'prep'],['F2 准备',3,4.5,'prep'],['F3 准备',4.5,6,'prep'],['F4 准备',6,7.5,'prep']]} , {name:'GPU / 推理',jobs:[['F0 推理',1.7,3.5,'compute'],['F1 推理',3.7,5.5,'compute'],['F2 推理',5.7,7.5,'compute'],['F3 推理',7.7,9.5,'compute']]}],
  gpu: [{name:'CPU / 准备（串行）',jobs:[['F0 准备',0,2,'prep'],['F1 准备',2,4,'prep'],['F2 准备',4,6,'prep'],['F3 准备',6,8,'prep']]} , {name:'GPU / Stream 0',jobs:[['F0 H2D',2,3,'h2d'],['F0 计算',3,5,'compute'],['F2 H2D',6,7,'h2d'],['F2 计算',7,9,'compute']]}, {name:'GPU / Stream 1',jobs:[['F1 H2D',4,5,'h2d'],['F1 计算',5,7,'compute'],['F3 H2D',8,9,'h2d'],['F3 计算',9,10,'compute']]}],
  both: [{name:'CPU / 准备（slot A）',jobs:[['F0 准备',0,1.5,'prep'],['F2 准备',3.6,5.1,'prep'],['F4 准备',7.2,8.1,'prep']]},{name:'CPU / 准备（slot B）',jobs:[['F1 准备',1.5,3,'prep'],['F3 准备',5.1,6.6,'prep']]},{name:'GPU / Stream 0',jobs:[['F0 H2D',1.5,2.2,'h2d'],['F0 计算',2.2,3.6,'compute'],['F2 H2D',5.1,5.8,'h2d'],['F2 计算',5.8,7.2,'compute'],['F4 H2D',8.1,8.8,'h2d'],['F4 计算',8.8,10,'compute']]},{name:'GPU / Stream 1',jobs:[['F1 H2D',3,3.7,'h2d'],['F1 计算',3.7,5.1,'compute'],['F3 H2D',6.6,7.3,'h2d'],['F3 计算',7.3,8.7,'compute']]}]
};
function render(){ lanesEl.innerHTML=''; const active=[]; base[mode].forEach(lane=>{ const row=document.createElement('div'); row.className='lane'; row.innerHTML=`<div class="lane-label">${lane.name}</div><div class="track"></div>`; const track=row.lastElementChild; lane.jobs.forEach(j=>{const [label,start,end,kind]=j; const el=document.createElement('div'); el.className=`job ${kind}${time>=start&&time<end?' running':''}`; el.textContent=label; el.style.left=`${start/total*100}%`; el.style.width=`${(end-start)/total*100}%`; el.dataset.info=`${label}：${start.toFixed(1)}–${end.toFixed(1)} s`; if(time>=start&&time<end) active.push(`${lane.name} 正在 ${label}`); el.onclick=()=>{document.querySelectorAll('.job').forEach(x=>x.classList.remove('selected'));el.classList.add('selected');eventEl.textContent=el.dataset.info;}; track.appendChild(el);}); lanesEl.appendChild(row); }); clockEl.textContent=`${time.toFixed(1)} s · ${modes[mode]}`; eventEl.textContent=active.length ? `此刻并行发生：${active.join('；')}` : '此刻没有任务执行：等待可用 slot 或下一帧。'; }
function tick(now){ const dt=(now-last)/1000; last=now; if(playing){time+=dt*Number(speedEl.value); if(time>total){time=0;} render();} requestAnimationFrame(tick); }
document.querySelectorAll('[data-mode]').forEach(b=>b.onclick=()=>{mode=b.dataset.mode;document.querySelectorAll('[data-mode]').forEach(x=>x.classList.toggle('active',x===b));time=0;render();});
playBtn.onclick=()=>{playing=!playing;playBtn.textContent=playing?'❚❚ 暂停':'▶ 播放';}; document.querySelector('#reset').onclick=()=>{time=0;playing=false;playBtn.textContent='▶ 播放';render();}; speedEl.oninput=()=>speedValue.textContent=`${speedEl.value}×`;
render(); requestAnimationFrame(tick);
