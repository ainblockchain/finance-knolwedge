(function(){const p=location.pathname.split('/').pop()||'index.html';
const links=[['index.html','채팅 (전 / 후)'],['bench.html','벤치마크'],['trace.html','추론 추적'],['method.html','트레이닝 기법'],['privy.html','에이전트 결제 (Privy)']];
document.body.insertAdjacentHTML('afterbegin',`<nav><span class="brand">Qwen3.8-Flash-Next · 51B n-gram 지식 주입 데모</span>${links.map(([h,t])=>`<a href="${h}" class="${p===h?'on':''}">${t}</a>`).join('')}<span class="st">서버 <span class="dot" id="d-before"></span>학습 전 <span class="dot" id="d-after"></span><span id="d-after-label">학습 후</span></span></nav><div id="patch-warn" class="banner" hidden></div>`);
/* "학습 후" 점은 서버가 떠 있는지가 아니라 패치가 테이블에 올라가 있는지를 나타낸다.
   서버만 뜬 상태를 초록으로 칠했던 탓에, 2026-09-12 에는 원본 모델이 22시간 동안
   "학습 후" 로 표시됐다. 확인하지 못한 경우(applied: null)도 초록이 아니다. */
fetch('/api/status').then(r=>r.json()).then(s=>{
  document.getElementById('d-before').classList.toggle('on',!!s.before);
  const pt=s.patch||{},ok=s.after&&pt.applied===true;
  document.getElementById('d-after').classList.toggle('on',ok);
  const lbl=document.getElementById('d-after-label');
  if(s.after&&pt.applied===false)lbl.textContent='학습 후 (패치 빠짐)';
  else if(s.after&&pt.applied==null)lbl.textContent='학습 후 (확인 불가)';
  const w=document.getElementById('patch-warn');
  if(!s.after){w.textContent='학습 후 서버가 응답하지 않습니다. 오른쪽 패널은 답하지 못합니다.';w.hidden=false;}
  else if(pt.applied===false){w.innerHTML=`<b>학습 내용이 모델에 올라가 있지 않습니다.</b> 오른쪽 패널은 지금 <b>학습 전과 같은 원본 모델</b>입니다 — 두 답이 같게 나오는 것이 정상입니다. `+
    `<span class="note">${pt.npz||''}${pt.rows?' · '+pt.rows.toLocaleString()+'행':''} · ${pt.detail||''} · 재적용: scripts/patch_watchdog.py 확인</span>`;w.hidden=false;}
  else if(pt.applied==null){w.innerHTML=`<b>학습 내용이 올라가 있는지 확인하지 못했습니다.</b> <span class="note">${pt.detail||''}</span>`;w.hidden=false;}
}).catch(()=>{});})();
