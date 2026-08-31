(function(){const p=location.pathname.split('/').pop()||'index.html';
const links=[['index.html','채팅 (전 / 후)'],['bench.html','벤치마크'],['trace.html','추론 추적'],['method.html','트레이닝 기법']];
document.body.insertAdjacentHTML('afterbegin',`<nav><span class="brand">Qwen3.8-Flash-Next · 51B n-gram 지식 주입 데모</span>${links.map(([h,t])=>`<a href="${h}" class="${p===h?'on':''}">${t}</a>`).join('')}<span class="st">서버 <span class="dot" id="d-before"></span>학습 전 <span class="dot" id="d-after"></span>학습 후</span></nav>`);
fetch('/api/status').then(r=>r.json()).then(s=>{for(const k of['before','after'])document.getElementById('d-'+k).classList.toggle('on',!!s[k]);}).catch(()=>{});})();
