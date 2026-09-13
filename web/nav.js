/* 헤더: 탭, 서버/패치 상태, 언어 선택.
 *
 * 탭은 데이터다. 한 줄이 한 탭이고, 페이지가 아직 없는 탭(`optional`)은 실제로 있는지
 * 확인해서 없으면 아예 그리지 않는다 — 브랜치마다 있는 페이지가 다른데 헤더를 손으로
 * 맞추면, 머지할 때마다 같은 배열에서 충돌이 나고 어느 브랜치에선 404 링크가 남는다.
 *
 * 언어는 window.I18N 하나로 노출한다. 각 페이지가 자기 본문을 번역할 때는 I18N.lang 을
 * 읽고 'langchange' 이벤트를 들으면 된다. 헤더가 먼저 두 언어를 갖추고, 본문은 페이지별로
 * 따라오면 된다 — 헤더가 언어의 단일 출처다.
 */
(function () {
  const TABS = [
    { href: 'index.html', ko: '채팅 (전 / 후)', en: 'Chat (before / after)' },
    { href: 'bench.html', ko: '벤치마크', en: 'Benchmark' },
    { href: 'trace.html', ko: '추론 추적', en: 'Inference trace' },
    { href: 'method.html', ko: '트레이닝 기법', en: 'Method' },
    // 다른 브랜치에서 들어오는 페이지. 있으면 보이고, 없으면 조용히 빠진다.
    { href: 'privy.html', ko: 'Privy', en: 'Privy', optional: true },
  ];

  const S = {
    ko: {
      brand: 'Qwen3.8-Flash-Next · 51B n-gram 지식 주입 데모',
      server: '서버', before: '학습 전', after: '학습 후',
      afterStale: '학습 후 (패치 빠짐)', afterUnknown: '학습 후 (확인 불가)',
      down: '학습 후 서버가 응답하지 않습니다. 오른쪽 패널은 답하지 못합니다.',
      staleTitle: '학습 내용이 모델에 올라가 있지 않습니다.',
      staleBody: '오른쪽 패널은 지금 <b>학습 전과 같은 원본 모델</b>입니다 — 두 답이 같게 나오는 것이 정상입니다.',
      unknownTitle: '학습 내용이 올라가 있는지 확인하지 못했습니다.',
      rows: (n) => n.toLocaleString('ko') + '행',
      reapply: '재적용: scripts/patch_watchdog.py 확인',
      langLabel: '언어',
    },
    en: {
      brand: 'Qwen3.8-Flash-Next · 51B n-gram knowledge injection demo',
      server: 'Servers', before: 'before', after: 'after',
      afterStale: 'after (patch not loaded)', afterUnknown: 'after (unverified)',
      down: 'The "after" server is not responding. The right-hand panel cannot answer.',
      staleTitle: 'The lesson is not loaded into the model.',
      staleBody: 'The right-hand panel is currently <b>the same original model as the left</b> — the two answers matching is expected.',
      unknownTitle: 'Could not verify whether the lesson is loaded.',
      rows: (n) => n.toLocaleString('en') + ' rows',
      reapply: 'to re-apply: check scripts/patch_watchdog.py',
      langLabel: 'Language',
    },
  };

  // 두 언어뿐이라 드롭다운이 아니라 세그먼트 토글이다: 지금 어느 쪽인지가 펼치지 않아도
  // 보이고, 바꾸는 데 클릭이 한 번이면 된다. [코드, 버튼에 쓸 약어, 툴팁에 쓸 이름]
  const LANGS = [['ko', 'KO', '한국어'], ['en', 'EN', 'English']];
  const KEY = 'ainize.lang';

  function initialLang() {
    try {
      const saved = localStorage.getItem(KEY);
      if (saved && S[saved]) return saved;
    } catch (e) { /* 사생활 보호 모드 등 — 저장소를 못 써도 동작해야 한다 */ }
    return (navigator.language || '').toLowerCase().startsWith('ko') ? 'ko' : 'en';
  }

  let lang = initialLang();
  const t = () => S[lang];

  const page = location.pathname.split('/').pop() || 'index.html';
  const tabHtml = (x) => `<a href="${x.href}" data-tab="${x.href}"${x.optional ? ' hidden' : ''}` +
    ` class="${page === x.href ? 'on' : ''}">${x[lang]}</a>`;

  document.body.insertAdjacentHTML('afterbegin',
    `<nav><span class="brand" id="nav-brand">${t().brand}</span>` +
    TABS.map(tabHtml).join('') +
    '<span class="st">' +
      `<div class="langtog" id="nav-lang" role="group" aria-labelledby="nav-lang-label" data-lang="${lang}">` +
        `<span class="vh" id="nav-lang-label">${t().langLabel}</span>` +
        '<span class="langtog-ind" aria-hidden="true"></span>' +
        LANGS.map(([v, short, full]) => `<button type="button" class="langtog-b" data-set-lang="${v}" title="${full}" aria-pressed="${v === lang}">${short}</button>`).join('') +
      '</div>' +
      `<span id="nav-server">${t().server}</span> <span class="dot" id="d-before"></span><span id="d-before-label">${t().before}</span>` +
      ` <span class="dot" id="d-after"></span><span id="d-after-label">${t().after}</span>` +
    '</span></nav><div id="patch-warn" class="banner" hidden></div>');

  /* 있는 페이지만 남긴다. HEAD 한 번이면 되고, 없는 탭은 처음부터 hidden 이라 404 링크가
     잠깐이라도 보이는 일이 없다. */
  TABS.filter((x) => x.optional).forEach((x) => {
    fetch(x.href, { method: 'HEAD' })
      .then((r) => { if (r.ok) document.querySelector(`[data-tab="${x.href}"]`).hidden = false; })
      .catch(() => {});
  });

  /* 마지막 /api/status 응답을 들고 있다가 언어가 바뀌면 같은 응답으로 다시 그린다 —
     상태 문구가 안 따라오면 헤더 절반만 언어가 바뀐다.

     "학습 후" 점은 서버가 떠 있는지가 아니라 패치가 테이블에 올라가 있는지를 나타낸다.
     서버만 뜬 상태를 초록으로 칠했던 탓에, 2026-09-12 에는 원본 모델이 22시간 동안
     "학습 후" 로 표시됐다. 확인하지 못한 경우(applied: null)도 초록이 아니다. */
  let lastStatus = null;

  function renderStatus(s) {
    lastStatus = s;
    const pt = (s && s.patch) || {};
    document.getElementById('d-before').classList.toggle('on', !!(s && s.before));
    document.getElementById('d-after').classList.toggle('on', !!(s && s.after) && pt.applied === true);
    document.getElementById('d-after-label').textContent =
      !s || !s.after ? t().after
        : pt.applied === false ? t().afterStale
          : pt.applied == null ? t().afterUnknown
            : t().after;

    const w = document.getElementById('patch-warn');
    if (!s || !s.after) {
      w.textContent = t().down; w.hidden = false;
    } else if (pt.applied === false) {
      const meta = [pt.npz, pt.rows ? t().rows(pt.rows) : '', pt.detail, t().reapply].filter(Boolean).join(' · ');
      w.innerHTML = `<b>${t().staleTitle}</b> ${t().staleBody} <span class="note">${meta}</span>`;
      w.hidden = false;
    } else if (pt.applied == null) {
      w.innerHTML = `<b>${t().unknownTitle}</b> <span class="note">${pt.detail || ''}</span>`;
      w.hidden = false;
    } else {
      w.hidden = true;
    }
  }

  const tog = document.getElementById('nav-lang');

  function applyLang() {
    document.documentElement.lang = lang;
    document.getElementById('nav-brand').textContent = t().brand;
    document.getElementById('nav-lang-label').textContent = t().langLabel;
    document.getElementById('nav-server').textContent = t().server;
    document.getElementById('d-before-label').textContent = t().before;
    TABS.forEach((x) => {
      const a = document.querySelector(`[data-tab="${x.href}"]`);
      if (a) a.textContent = x[lang];
    });
    // 표시기는 CSS 가 이 속성 하나로 옮긴다 — 위치 계산을 JS 가 들고 있으면 창 크기나
    // 글꼴이 바뀔 때마다 어긋난다.
    tog.dataset.lang = lang;
    tog.querySelectorAll('.langtog-b').forEach((b) => {
      b.setAttribute('aria-pressed', String(b.dataset.setLang === lang));
    });
    renderStatus(lastStatus);
    window.dispatchEvent(new CustomEvent('langchange', { detail: { lang } }));
  }

  function setLang(next) {
    if (!S[next] || next === lang) return;
    lang = next;
    try { localStorage.setItem(KEY, lang); } catch (err) { /* 저장 못 해도 이번 방문에는 적용된다 */ }
    applyLang();
  }

  tog.addEventListener('click', (e) => {
    const b = e.target.closest('.langtog-b');
    if (b) setLang(b.dataset.setLang);
  });
  // ←/→ 로도 넘어간다. 세그먼트 토글은 라디오처럼 읽히므로 그렇게 움직이는 게 맞다.
  tog.addEventListener('keydown', (e) => {
    if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return;
    const codes = LANGS.map(([v]) => v);
    const next = codes[(codes.indexOf(lang) + (e.key === 'ArrowRight' ? 1 : codes.length - 1)) % codes.length];
    e.preventDefault();
    setLang(next);
    tog.querySelector(`[data-set-lang="${next}"]`).focus();
  });

  // 각 페이지가 자기 본문을 번역할 때 쓰는 것.
  window.I18N = {
    get lang() { return lang; },
    t(dict, key) { return (dict[lang] || dict.en || {})[key]; },
  };

  document.documentElement.lang = lang;
  fetch('/api/status').then((r) => r.json()).then(renderStatus).catch(() => renderStatus(null));
})();
