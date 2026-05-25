/* ════════════════════════════════════════════════════
   Windows 12 — Core JS
   ════════════════════════════════════════════════════ */

let zTop = 100;
let openWindows = {};
let wallpaperIndex = 0;

const WALLPAPERS = [
  { type: 'gradient', stops: ['#0d0d2b','#0a3d62','#1a1a2e'], angle: 135 },
  { type: 'aurora',   colors: ['#0f2027','#203a43','#2c5364'] },
  { type: 'bloom',    colors: ['#1a0533','#0d1b2a','#0f0c29'] },
  { type: 'flow',     colors: ['#000428','#004e92'] },
];

/* ─── BOOT SEQUENCE ───────────────────────────── */
window.addEventListener('load', () => {
  drawWallpaper();
  updateClock();
  setInterval(updateClock, 1000);

  setTimeout(() => {
    document.getElementById('boot-screen').style.display = 'none';
    const ls = document.getElementById('lock-screen');
    ls.style.display = 'flex';
    updateLockClock();
    setInterval(updateLockClock, 1000);

    ls.addEventListener('click', unlockScreen);
    ls.addEventListener('keydown', unlockScreen);
  }, 2200);
});

function unlockScreen() {
  const ls = document.getElementById('lock-screen');
  ls.style.transition = 'opacity 0.5s';
  ls.style.opacity = '0';
  setTimeout(() => {
    ls.style.display = 'none';
    document.getElementById('desktop').style.display = 'flex';
    showToast('Windows 12', 'Welcome back, User!');
    buildCalendar();
  }, 500);
}

/* ─── CLOCK ───────────────────────────────────── */
function updateClock() {
  const now = new Date();
  const t = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  const d = now.toLocaleDateString([], { month: 'numeric', day: 'numeric', year: 'numeric' });
  document.getElementById('clock-time').textContent = t;
  document.getElementById('clock-date').textContent = d;
  const bt = document.getElementById('cal-big-time');
  if (bt) bt.textContent = now.toLocaleTimeString();
}

function updateLockClock() {
  const now = new Date();
  document.getElementById('lock-time').textContent =
    now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  document.getElementById('lock-date').textContent =
    now.toLocaleDateString([], { weekday: 'long', month: 'long', day: 'numeric' });
}

/* ─── WALLPAPER ───────────────────────────────── */
function drawWallpaper() {
  const canvas = document.getElementById('wallpaper-canvas');
  const W = canvas.width = window.innerWidth;
  const H = canvas.height = window.innerHeight;
  const ctx = canvas.getContext('2d');
  const wp = WALLPAPERS[wallpaperIndex % WALLPAPERS.length];

  if (wp.type === 'gradient') {
    const r = Math.PI * wp.angle / 180;
    const gx = W / 2 + Math.cos(r) * W;
    const gy = H / 2 + Math.sin(r) * H;
    const g = ctx.createLinearGradient(W / 2 - Math.cos(r) * W, H / 2 - Math.sin(r) * H, gx, gy);
    wp.stops.forEach((c, i) => g.addColorStop(i / (wp.stops.length - 1), c));
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, W, H);
  } else {
    const g = ctx.createLinearGradient(0, 0, W, H);
    wp.colors.forEach((c, i) => g.addColorStop(i / (wp.colors.length - 1), c));
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, W, H);
  }

  /* Subtle noise overlay */
  for (let i = 0; i < 3000; i++) {
    ctx.fillStyle = `rgba(255,255,255,${Math.random() * 0.04})`;
    ctx.fillRect(Math.random() * W, Math.random() * H, 1, 1);
  }

  /* Glowing orbs */
  [[W * 0.2, H * 0.3, '#0078d4'], [W * 0.8, H * 0.7, '#8a2be2'], [W * 0.5, H * 0.15, '#00bcd4']].forEach(([x, y, c]) => {
    const rg = ctx.createRadialGradient(x, y, 0, x, y, W * 0.4);
    rg.addColorStop(0, c + '33');
    rg.addColorStop(1, 'transparent');
    ctx.fillStyle = rg;
    ctx.fillRect(0, 0, W, H);
  });

  /* Windows logo watermark */
  ctx.save();
  ctx.globalAlpha = 0.06;
  ctx.translate(W - 100, H - 100);
  ctx.scale(2, 2);
  drawWinLogo(ctx);
  ctx.restore();
}

function drawWinLogo(ctx) {
  const paths = [
    [0, -1.5, 4.5, -2, 4.5, 2.5, 0, 2.5],
    [0, 3, 4.5, 3, 4.5, 7.5, 0, 7],
    [5, -2, 11, -2.5, 11, 2.5, 5, 2.5],
    [5, 3, 11, 3, 11, 7.5, 5, 7],
  ];
  ctx.fillStyle = '#fff';
  paths.forEach(p => {
    ctx.beginPath();
    ctx.moveTo(p[0] * 4, p[1] * 4);
    for (let i = 2; i < p.length; i += 2) ctx.lineTo(p[i] * 4, p[i + 1] * 4);
    ctx.closePath();
    ctx.fill();
  });
}

function changeWallpaper() {
  wallpaperIndex++;
  drawWallpaper();
  hideContextMenu();
}

window.addEventListener('resize', drawWallpaper);

/* ─── CONTEXT MENU ────────────────────────────── */
document.getElementById('desktop').addEventListener('contextmenu', e => {
  e.preventDefault();
  const menu = document.getElementById('context-menu');
  const x = Math.min(e.clientX, window.innerWidth - 210);
  const y = Math.min(e.clientY, window.innerHeight - 200);
  menu.style.left = x + 'px';
  menu.style.top = y + 'px';
  menu.style.display = 'block';
});

document.addEventListener('click', e => {
  const menu = document.getElementById('context-menu');
  if (!menu.contains(e.target)) hideContextMenu();
  if (!document.getElementById('start-menu').contains(e.target) &&
      !document.getElementById('start-btn').contains(e.target)) {
    closeStart();
  }
  if (!document.getElementById('calendar-panel').contains(e.target) &&
      !document.getElementById('taskbar-clock').contains(e.target)) {
    document.getElementById('calendar-panel').style.display = 'none';
  }
  if (!document.getElementById('action-center').contains(e.target) &&
      !document.getElementById('action-center-btn').contains(e.target)) {
    document.getElementById('action-center').style.display = 'none';
  }
  if (!document.getElementById('power-menu').contains(e.target)) {
    document.getElementById('power-menu').style.display = 'none';
  }
});

function hideContextMenu() {
  document.getElementById('context-menu').style.display = 'none';
}

function sortIcons() { hideContextMenu(); showToast('Desktop', 'Icons sorted by name'); }
function refreshDesktop() { hideContextMenu(); drawWallpaper(); showToast('Desktop', 'Refreshed'); }

/* ─── START MENU ──────────────────────────────── */
function toggleStart() {
  const m = document.getElementById('start-menu');
  const btn = document.getElementById('start-btn');
  if (m.style.display === 'none' || !m.style.display) {
    m.style.display = 'block';
    btn.classList.add('active');
    m.querySelector('.start-search').focus();
  } else {
    closeStart();
  }
}

function closeStart() {
  document.getElementById('start-menu').style.display = 'none';
  document.getElementById('start-btn').classList.remove('active');
}

function filterApps(q) {
  q = q.toLowerCase();
  document.querySelectorAll('.start-app').forEach(a => {
    a.style.display = a.querySelector('span').textContent.toLowerCase().includes(q) ? '' : 'none';
  });
}

/* ─── POWER ───────────────────────────────────── */
function showPowerMenu() {
  closeStart();
  const pm = document.getElementById('power-menu');
  pm.style.display = pm.style.display === 'none' ? 'block' : 'none';
}

function shutDown() {
  document.getElementById('desktop').style.transition = 'opacity 1s';
  document.getElementById('desktop').style.opacity = '0';
  setTimeout(() => {
    document.getElementById('desktop').style.display = 'none';
    document.getElementById('boot-screen').style.display = 'flex';
    document.getElementById('boot-screen').style.opacity = '1';
    setTimeout(() => location.reload(), 3000);
  }, 1000);
}

/* ─── CALENDAR ────────────────────────────────── */
function toggleCalendar() {
  const p = document.getElementById('calendar-panel');
  const ac = document.getElementById('action-center');
  ac.style.display = 'none';
  p.style.display = p.style.display === 'none' ? 'block' : 'none';
}

function buildCalendar() {
  const now = new Date();
  document.getElementById('cal-big-date').textContent =
    now.toLocaleDateString([], { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' });
  const grid = document.getElementById('cal-grid');
  grid.innerHTML = '';
  ['Su','Mo','Tu','We','Th','Fr','Sa'].forEach(d => {
    const el = document.createElement('div');
    el.className = 'day-label'; el.textContent = d;
    grid.appendChild(el);
  });
  const first = new Date(now.getFullYear(), now.getMonth(), 1);
  const lastDay = new Date(now.getFullYear(), now.getMonth() + 1, 0).getDate();
  for (let i = 0; i < first.getDay(); i++) {
    const el = document.createElement('div'); el.className = 'day other-month';
    el.textContent = new Date(now.getFullYear(), now.getMonth(), -first.getDay() + i + 1).getDate();
    grid.appendChild(el);
  }
  for (let d = 1; d <= lastDay; d++) {
    const el = document.createElement('div');
    el.className = 'day' + (d === now.getDate() ? ' today' : '');
    el.textContent = d;
    grid.appendChild(el);
  }
}

/* ─── ACTION CENTER ───────────────────────────── */
function toggleActionCenter() {
  const ac = document.getElementById('action-center');
  const cal = document.getElementById('calendar-panel');
  cal.style.display = 'none';
  ac.style.display = ac.style.display === 'none' ? 'block' : 'none';
}

/* ─── SEARCH ──────────────────────────────────── */
function openSearch() {
  document.getElementById('search-overlay').style.display = 'flex';
  setTimeout(() => document.getElementById('search-input').focus(), 50);
}

function closeSearch(e) {
  if (!e || e.target === document.getElementById('search-overlay')) {
    document.getElementById('search-overlay').style.display = 'none';
    document.getElementById('search-input').value = '';
    doSearch('');
  }
}

const ALL_APPS = [
  { name: 'Microsoft Edge', icon: '🌐', id: 'browser' },
  { name: 'File Explorer', icon: '📁', id: 'explorer' },
  { name: 'Settings', icon: '⚙️', id: 'settings' },
  { name: 'Notepad', icon: '📝', id: 'notepad' },
  { name: 'Calculator', icon: '🧮', id: 'calculator' },
  { name: 'Terminal', icon: '💻', id: 'terminal' },
  { name: 'Paint', icon: '🎨', id: 'paint' },
  { name: 'Mail', icon: '📧', id: 'mail' },
  { name: 'Photos', icon: '🖼️', id: 'photos' },
  { name: 'Microsoft Store', icon: '🏪', id: 'store' },
  { name: 'Task Manager', icon: '📊', id: 'taskmanager' },
  { name: 'Copilot', icon: '✨', id: 'copilot' },
  { name: 'This PC', icon: '💻', id: 'mypc' },
];

function doSearch(q) {
  const res = document.getElementById('search-results');
  if (!q) {
    res.innerHTML = `<div class="search-category">Top apps</div>` +
      ALL_APPS.slice(0, 5).map(a =>
        `<div class="search-item" onclick="openApp('${a.id}');closeSearch()"><span>${a.icon}</span> ${a.name}</div>`
      ).join('');
    return;
  }
  q = q.toLowerCase();
  const matches = ALL_APPS.filter(a => a.name.toLowerCase().includes(q));
  res.innerHTML = matches.length
    ? `<div class="search-category">Apps</div>` +
        matches.map(a =>
          `<div class="search-item" onclick="openApp('${a.id}');closeSearch()"><span>${a.icon}</span> ${a.name}</div>`
        ).join('')
    : `<div class="search-category" style="text-align:center;padding:20px">No results for "${q}"</div>`;
}

document.addEventListener('keydown', e => {
  if (e.key === 'Escape') {
    closeSearch();
    closeStart();
    document.getElementById('calendar-panel').style.display = 'none';
    document.getElementById('action-center').style.display = 'none';
    document.getElementById('power-menu').style.display = 'none';
  }
  if ((e.metaKey || e.ctrlKey) && e.key === 'f') { e.preventDefault(); openSearch(); }
});

/* ─── TOAST ───────────────────────────────────── */
function showToast(title, body, duration = 4000) {
  const c = document.getElementById('toast-container');
  const t = document.createElement('div');
  t.className = 'toast';
  t.innerHTML = `<div class="toast-title">${title}</div><div class="toast-body">${body}</div>`;
  c.appendChild(t);
  setTimeout(() => {
    t.style.transition = 'opacity 0.4s, transform 0.4s';
    t.style.opacity = '0';
    t.style.transform = 'translateX(30px)';
    setTimeout(() => t.remove(), 400);
  }, duration);
}

/* ════════════════════════════════════════════════
   WINDOW MANAGER
   ════════════════════════════════════════════════ */
function openApp(id) {
  if (openWindows[id]) {
    const w = openWindows[id];
    w.style.display = 'flex';
    w.classList.remove('minimized');
    bringToFront(w);
    return;
  }
  const cfg = APP_CONFIG[id];
  if (!cfg) return;
  createWindow(id, cfg);
  closeStart();
}

const APP_CONFIG = {
  notepad:     { title: 'Notepad', icon: '📝', w: 640, h: 480, build: buildNotepad },
  calculator:  { title: 'Calculator', icon: '🧮', w: 320, h: 480, build: buildCalculator },
  explorer:    { title: 'File Explorer', icon: '📁', w: 800, h: 520, build: buildExplorer },
  settings:    { title: 'Settings', icon: '⚙️', w: 720, h: 540, build: buildSettings },
  browser:     { title: 'Microsoft Edge', icon: '🌐', w: 900, h: 600, build: buildBrowser },
  terminal:    { title: 'Windows Terminal', icon: '💻', w: 680, h: 440, build: buildTerminal },
  paint:       { title: 'Paint', icon: '🎨', w: 760, h: 540, build: buildPaint },
  copilot:     { title: 'Copilot', icon: '✨', w: 420, h: 560, build: buildCopilot },
  mail:        { title: 'Mail', icon: '📧', w: 800, h: 540, build: buildMail },
  photos:      { title: 'Photos', icon: '🖼️', w: 700, h: 520, build: buildPhotos },
  store:       { title: 'Microsoft Store', icon: '🏪', w: 780, h: 560, build: buildStore },
  taskmanager: { title: 'Task Manager', icon: '📊', w: 700, h: 480, build: buildTaskManager },
  recyclebin:  { title: 'Recycle Bin', icon: '🗑️', w: 700, h: 480, build: buildRecycleBin },
  mypc:        { title: 'This PC', icon: '💻', w: 780, h: 520, build: buildThisPC },
};

function createWindow(id, cfg) {
  const container = document.getElementById('windows-container');
  const cw = window.innerWidth, ch = window.innerHeight - 48;
  const w = Math.min(cfg.w, cw - 40);
  const h = Math.min(cfg.h, ch - 40);
  const x = Math.floor((cw - w) / 2) + (Object.keys(openWindows).length % 5) * 20;
  const y = Math.floor((ch - h) / 2) + (Object.keys(openWindows).length % 5) * 20;

  const win = document.createElement('div');
  win.className = 'win';
  win.id = 'win-' + id;
  win.style.cssText = `width:${w}px;height:${h}px;left:${x}px;top:${y}px;z-index:${++zTop}`;

  win.innerHTML = `
    <div class="win-titlebar" id="tb-${id}">
      <div class="win-icon">${cfg.icon}</div>
      <div class="win-title">${cfg.title}</div>
      <div class="win-controls">
        <div class="win-btn" onclick="minimizeWin('${id}')" title="Minimize">
          <svg viewBox="0 0 12 12" width="12" height="12"><rect y="5.5" width="12" height="1" fill="currentColor"/></svg>
        </div>
        <div class="win-btn" onclick="toggleMaximize('${id}')" title="Maximize">
          <svg viewBox="0 0 12 12" width="12" height="12"><rect x="1" y="1" width="10" height="10" rx="1" fill="none" stroke="currentColor" stroke-width="1.2"/></svg>
        </div>
        <div class="win-btn close" onclick="closeWin('${id}')" title="Close">
          <svg viewBox="0 0 12 12" width="12" height="12"><path d="M1 1l10 10M11 1L1 11" stroke="currentColor" stroke-width="1.4" stroke-linecap="round"/></svg>
        </div>
      </div>
    </div>
    <div class="win-body" id="body-${id}"></div>
    <div class="win-resize" id="rs-${id}"></div>
  `;

  container.appendChild(win);
  openWindows[id] = win;

  cfg.build(document.getElementById('body-' + id));

  makeDraggable(win, document.getElementById('tb-' + id));
  makeResizable(win, document.getElementById('rs-' + id));

  win.addEventListener('mousedown', () => bringToFront(win));

  addTaskbarEntry(id, cfg);
}

function bringToFront(win) {
  win.style.zIndex = ++zTop;
}

function minimizeWin(id) {
  openWindows[id].classList.add('minimized');
  const entry = document.querySelector(`.taskbar-running-btn[data-id="${id}"]`);
  if (entry) entry.classList.remove('active-win');
}

function toggleMaximize(id) {
  openWindows[id].classList.toggle('maximized');
}

function closeWin(id) {
  const win = openWindows[id];
  win.style.transition = 'opacity 0.15s, transform 0.15s';
  win.style.opacity = '0';
  win.style.transform = 'scale(0.95)';
  setTimeout(() => {
    win.remove();
    delete openWindows[id];
    removeTaskbarEntry(id);
  }, 150);
}

function addTaskbarEntry(id, cfg) {
  const tb = document.getElementById('taskbar-running');
  const btn = document.createElement('div');
  btn.className = 'taskbar-pin running';
  btn.dataset.id = id;
  btn.title = cfg.title;
  btn.textContent = cfg.icon;
  btn.style.cssText = 'font-size:18px;border-radius:8px;width:42px;height:42px;display:flex;align-items:center;justify-content:center;cursor:pointer;transition:background 0.15s;position:relative;';
  btn.addEventListener('click', () => {
    const w = openWindows[id];
    if (w.classList.contains('minimized')) {
      w.classList.remove('minimized');
      bringToFront(w);
    } else {
      minimizeWin(id);
    }
  });
  btn.addEventListener('mouseover', () => btn.style.background = 'rgba(255,255,255,0.1)');
  btn.addEventListener('mouseout', () => btn.style.background = '');
  tb.appendChild(btn);
}

function removeTaskbarEntry(id) {
  const btn = document.querySelector(`.taskbar-pin[data-id="${id}"]`);
  if (btn) btn.remove();
}

/* ─── DRAG ────────────────────────────────────── */
function makeDraggable(win, handle) {
  let ox, oy, dragging = false;
  handle.addEventListener('mousedown', e => {
    if (e.target.closest('.win-btn')) return;
    if (win.classList.contains('maximized')) return;
    dragging = true;
    ox = e.clientX - win.offsetLeft;
    oy = e.clientY - win.offsetTop;
    document.body.style.cursor = 'move';
    bringToFront(win);
  });
  document.addEventListener('mousemove', e => {
    if (!dragging) return;
    const maxX = window.innerWidth - win.offsetWidth;
    const maxY = window.innerHeight - 48 - win.offsetHeight;
    win.style.left = Math.max(0, Math.min(e.clientX - ox, maxX)) + 'px';
    win.style.top = Math.max(0, Math.min(e.clientY - oy, maxY)) + 'px';
  });
  document.addEventListener('mouseup', () => {
    dragging = false;
    document.body.style.cursor = '';
  });

  /* Double-click titlebar → toggle maximize */
  handle.addEventListener('dblclick', e => {
    if (e.target.closest('.win-btn')) return;
    toggleMaximize(win.id.replace('win-', ''));
  });
}

/* ─── RESIZE ──────────────────────────────────── */
function makeResizable(win, handle) {
  let resizing = false, startX, startY, startW, startH;
  handle.addEventListener('mousedown', e => {
    resizing = true;
    startX = e.clientX; startY = e.clientY;
    startW = win.offsetWidth; startH = win.offsetHeight;
    e.stopPropagation();
  });
  document.addEventListener('mousemove', e => {
    if (!resizing) return;
    win.style.width = Math.max(320, startW + e.clientX - startX) + 'px';
    win.style.height = Math.max(200, startH + e.clientY - startY) + 'px';
  });
  document.addEventListener('mouseup', () => { resizing = false; });
}

/* ════════════════════════════════════════════════
   APP BUILDERS
   ════════════════════════════════════════════════ */

/* ─── NOTEPAD ─────────────────────────────────── */
function buildNotepad(body) {
  body.innerHTML = `
    <div class="notepad-toolbar">
      <div class="np-menu" onclick="npAction('file')">File</div>
      <div class="np-menu" onclick="npAction('edit')">Edit</div>
      <div class="np-menu" onclick="npAction('format')">Format</div>
      <div class="np-menu" onclick="npAction('view')">View</div>
    </div>
    <textarea class="notepad-area" placeholder="Start typing..." spellcheck="true"></textarea>
  `;
  setTimeout(() => body.querySelector('.notepad-area').focus(), 50);
}

function npAction(m) {
  const msgs = { file: 'File menu', edit: 'Edit menu', format: 'Format menu', view: 'View menu' };
  showToast('Notepad', msgs[m]);
}

/* ─── CALCULATOR ──────────────────────────────── */
function buildCalculator(body) {
  let expr = '', result = '0';
  const buttons = [
    ['%','CE','C','⌫'],
    ['1/x','x²','√x','÷'],
    ['7','8','9','×'],
    ['4','5','6','−'],
    ['1','2','3','+'],
    ['+/−','0','.','='],
  ];
  body.innerHTML = `
    <div class="calc-wrap">
      <div class="calc-display">
        <div class="calc-expression" id="calc-expr"></div>
        <div class="calc-value" id="calc-val">0</div>
      </div>
      <div class="calc-grid" id="calc-grid"></div>
    </div>
  `;
  const grid = body.querySelector('#calc-grid');
  const valEl = body.querySelector('#calc-val');
  const exprEl = body.querySelector('#calc-expr');
  buttons.forEach(row => {
    row.forEach(btn => {
      const el = document.createElement('div');
      el.className = 'calc-btn' +
        (['÷','×','−','+','='].includes(btn) ? (' eq') : '') +
        (['%','CE','C','⌫','1/x','x²','√x','+/−'].includes(btn) ? ' func' : '');
      if (btn === '=') el.style.background = 'var(--accent)';
      el.textContent = btn;
      el.onclick = () => calcClick(btn);
      grid.appendChild(el);
    });
  });

  function calcClick(b) {
    if (b === 'C') { expr = ''; valEl.textContent = '0'; exprEl.textContent = ''; return; }
    if (b === 'CE') { valEl.textContent = '0'; return; }
    if (b === '⌫') {
      expr = expr.slice(0, -1);
      valEl.textContent = expr || '0';
      return;
    }
    if (b === '=') {
      try {
        let e = expr.replace(/×/g,'*').replace(/÷/g,'/').replace(/−/g,'-');
        exprEl.textContent = expr + ' =';
        let r = eval(e);
        valEl.textContent = parseFloat(r.toFixed(10)).toString();
        expr = r.toString();
      } catch { valEl.textContent = 'Error'; expr = ''; }
      return;
    }
    if (b === '%') { try { let r = eval(expr) / 100; valEl.textContent = r; expr = r.toString(); } catch {} return; }
    if (b === '1/x') { try { let r = 1 / eval(expr); valEl.textContent = r; expr = r.toString(); } catch {} return; }
    if (b === 'x²') { try { let r = Math.pow(eval(expr), 2); valEl.textContent = r; expr = r.toString(); } catch {} return; }
    if (b === '√x') { try { let r = Math.sqrt(eval(expr)); valEl.textContent = r; expr = r.toString(); } catch {} return; }
    if (b === '+/−') { try { let r = -eval(expr); valEl.textContent = r; expr = r.toString(); } catch {} return; }
    expr += b;
    valEl.textContent = expr;
  }
}

/* ─── FILE EXPLORER ───────────────────────────── */
function buildExplorer(body) {
  const folders = [
    { name: 'Desktop', icon: '🖥️' }, { name: 'Documents', icon: '📁' },
    { name: 'Downloads', icon: '⬇️' }, { name: 'Music', icon: '🎵' },
    { name: 'Pictures', icon: '🖼️' }, { name: 'Videos', icon: '🎬' },
  ];
  const files = [
    { name: 'Documents', icon: '📁' }, { name: 'Downloads', icon: '📁' },
    { name: 'Desktop', icon: '📁' }, { name: 'Music', icon: '📁' },
    { name: 'Pictures', icon: '📁' }, { name: 'Videos', icon: '📁' },
    { name: 'README.txt', icon: '📄' }, { name: 'report.docx', icon: '📝' },
    { name: 'budget.xlsx', icon: '📊' }, { name: 'photo.png', icon: '🖼️' },
    { name: 'song.mp3', icon: '🎵' }, { name: 'movie.mp4', icon: '🎬' },
    { name: 'notes.txt', icon: '📄' }, { name: 'setup.exe', icon: '⚙️' },
  ];

  body.innerHTML = `
    <div class="explorer-wrap">
      <div class="explorer-toolbar">
        <div class="exp-nav-btn">←</div>
        <div class="exp-nav-btn">→</div>
        <div class="exp-nav-btn">↑</div>
        <input class="exp-addr" value="This PC > " readonly>
        <div class="exp-nav-btn">🔍</div>
      </div>
      <div class="explorer-body">
        <div class="explorer-sidebar">
          ${folders.map((f, i) => `<div class="sidebar-item${i===0?' active':''}" onclick="this.parentElement.querySelectorAll('.sidebar-item').forEach(x=>x.classList.remove('active'));this.classList.add('active')">${f.icon} ${f.name}</div>`).join('')}
        </div>
        <div class="explorer-files">
          ${files.map(f => `<div class="file-item" ondblclick="showToast('Explorer','Opening ${f.name}')"><div style="font-size:32px">${f.icon}</div><span>${f.name}</span></div>`).join('')}
        </div>
      </div>
    </div>
  `;
}

/* ─── SETTINGS ────────────────────────────────── */
function buildSettings(body) {
  const sections = [
    { icon: '🖥️', name: 'System' }, { icon: '📶', name: 'Bluetooth & devices' },
    { icon: '🌐', name: 'Network' }, { icon: '🎨', name: 'Personalization' },
    { icon: '📱', name: 'Apps' }, { icon: '👤', name: 'Accounts' },
    { icon: '⏰', name: 'Time & language' }, { icon: '🎮', name: 'Gaming' },
    { icon: '♿', name: 'Accessibility' }, { icon: '🔒', name: 'Privacy & security' },
    { icon: '🔄', name: 'Windows Update' },
  ];

  body.innerHTML = `
    <div class="settings-wrap">
      <div class="settings-nav">
        ${sections.map((s, i) => `<div class="settings-nav-item${i===0?' active':''}" onclick="loadSettingsSection('${s.name}', this)">${s.icon} ${s.name}</div>`).join('')}
      </div>
      <div class="settings-content" id="settings-content">
        <div class="settings-section-title">System</div>
        <div class="settings-card settings-row">
          <div><div class="settings-card-title">Dark mode</div><div class="settings-card-desc">Use dark theme across apps</div></div>
          <div class="settings-toggle on" onclick="this.classList.toggle('on')"></div>
        </div>
        <div class="settings-card settings-row">
          <div><div class="settings-card-title">Notifications</div><div class="settings-card-desc">Show app notifications</div></div>
          <div class="settings-toggle on" onclick="this.classList.toggle('on')"></div>
        </div>
        <div class="settings-card settings-row">
          <div><div class="settings-card-title">Storage Sense</div><div class="settings-card-desc">Automatically free up space</div></div>
          <div class="settings-toggle" onclick="this.classList.toggle('on')"></div>
        </div>
        <div class="settings-card settings-row">
          <div><div class="settings-card-title">Night light</div><div class="settings-card-desc">Reduce blue light at night</div></div>
          <div class="settings-toggle" onclick="this.classList.toggle('on')"></div>
        </div>
        <div class="settings-card settings-row">
          <div><div class="settings-card-title">Transparency effects</div><div class="settings-card-desc">Mica and Acrylic blur effects</div></div>
          <div class="settings-toggle on" onclick="this.classList.toggle('on')"></div>
        </div>
      </div>
    </div>
  `;
}

window.loadSettingsSection = function(name, el) {
  el.closest('.settings-nav').querySelectorAll('.settings-nav-item').forEach(x => x.classList.remove('active'));
  el.classList.add('active');
  const content = document.getElementById('settings-content');
  const defaults = `
    <div class="settings-card settings-row">
      <div><div class="settings-card-title">Feature A</div><div class="settings-card-desc">Configure ${name}</div></div>
      <div class="settings-toggle on" onclick="this.classList.toggle('on')"></div>
    </div>
    <div class="settings-card settings-row">
      <div><div class="settings-card-title">Feature B</div><div class="settings-card-desc">More options for ${name}</div></div>
      <div class="settings-toggle" onclick="this.classList.toggle('on')"></div>
    </div>
  `;
  content.innerHTML = `<div class="settings-section-title">${name}</div>${defaults}`;
};

/* ─── BROWSER ─────────────────────────────────── */
function buildBrowser(body) {
  body.innerHTML = `
    <div class="browser-wrap">
      <div class="browser-toolbar">
        <div class="browser-nav-btn" onclick="">←</div>
        <div class="browser-nav-btn" onclick="">→</div>
        <div class="browser-nav-btn" onclick="">↺</div>
        <input class="browser-url" id="browser-url-input" value="edge://newtab"
          onkeydown="if(event.key==='Enter') browserNav(this.value)">
        <div class="browser-nav-btn">☆</div>
        <div class="browser-nav-btn">⋯</div>
      </div>
      <div class="browser-body">
        <div class="browser-new-tab" id="browser-newtab">
          <div>
            <svg viewBox="0 0 88 88" width="48" height="48"><path d="M0 12.402l35.687-4.86.016 34.423-35.67.203zm35.67 33.529l.028 34.453L.028 75.48.026 45.7zm4.326-38.931L87.314 0v41.527l-47.318.376zm47.329 43.308l-.011 41.34-47.318-6.686-.066-34.739z" fill="rgba(255,255,255,0.6)"/></svg>
          </div>
          <h2>Good afternoon</h2>
          <div style="display:flex;gap:8px;margin-bottom:8px">
            <input style="flex:1;background:rgba(255,255,255,0.1);border:1px solid rgba(255,255,255,0.2);border-radius:20px;padding:10px 18px;color:#fff;font-size:14px;outline:none" placeholder="Search or enter URL"
              onkeydown="if(event.key==='Enter')browserNav(this.value)">
          </div>
          <div class="browser-shortcuts">
            <div class="browser-shortcut" onclick="browserNav('https://google.com')">🔍<span>Google</span></div>
            <div class="browser-shortcut" onclick="browserNav('https://youtube.com')">▶️<span>YouTube</span></div>
            <div class="browser-shortcut" onclick="browserNav('https://github.com')">🐙<span>GitHub</span></div>
            <div class="browser-shortcut" onclick="browserNav('https://bing.com')">🌐<span>Bing</span></div>
          </div>
        </div>
      </div>
    </div>
  `;
}

window.browserNav = function(url) {
  if (!url.startsWith('http')) url = 'https://' + url;
  document.getElementById('browser-url-input').value = url;
  const body = document.querySelector('.browser-body');
  body.innerHTML = `<iframe src="${url}" sandbox="allow-scripts allow-same-origin allow-forms"></iframe>`;
};

/* ─── TERMINAL ────────────────────────────────── */
function buildTerminal(body) {
  body.innerHTML = `
    <div class="terminal-wrap">
      <div class="terminal-output" id="term-out">
        <div>Windows Terminal [Version 4.0.0]</div>
        <div style="color:#888">(c) Microsoft Corporation 2026. All rights reserved.</div>
        <div>&nbsp;</div>
      </div>
      <div class="terminal-input-row">
        <span class="terminal-prompt">C:\\Users\\User&gt;&nbsp;</span>
        <input class="terminal-input" id="term-input" placeholder="" spellcheck="false"
          onkeydown="termKey(event)">
      </div>
    </div>
  `;
  setTimeout(() => body.querySelector('#term-input').focus(), 50);
}

window.termKey = function(e) {
  if (e.key !== 'Enter') return;
  const input = e.target;
  const cmd = input.value.trim();
  const out = document.getElementById('term-out');
  const line = document.createElement('div');
  line.innerHTML = `<span style="color:#4caf50">C:\\Users\\User&gt;</span> ${escHtml(cmd)}`;
  out.appendChild(line);
  input.value = '';
  termExec(cmd, out);
  out.scrollTop = out.scrollHeight;
};

function escHtml(s) { return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }

const TERM_FS = { 'Documents': {}, 'Downloads': {}, 'Desktop': {}, 'readme.txt': 'Hello from Windows 12!' };

function termExec(cmd, out) {
  const add = (text, cls='') => {
    const d = document.createElement('div');
    if (cls) d.className = cls;
    d.textContent = text;
    out.appendChild(d);
  };
  const c = cmd.toLowerCase().split(' ');
  if (!cmd) return;
  switch (c[0]) {
    case 'help': add('Available commands: help, cls, echo, dir, whoami, date, ver, ping, ipconfig, shutdown, exit'); break;
    case 'cls': out.innerHTML = ''; break;
    case 'echo': add(cmd.slice(5) || ''); break;
    case 'dir':
      add('  Directory of C:\\Users\\User');
      add('');
      Object.keys(TERM_FS).forEach(f => add(`  ${typeof TERM_FS[f] === 'object' ? '<DIR>' : '     '}  ${f}`));
      break;
    case 'whoami': add('desktop\\User'); break;
    case 'date': add(new Date().toDateString()); break;
    case 'ver': add('Microsoft Windows 12 [Version 12.0.00000]'); break;
    case 'ping': add(`Pinging ${c[1] || 'localhost'}...`); add('Reply from 127.0.0.1: bytes=32 time<1ms TTL=128'); break;
    case 'ipconfig':
      add('Windows IP Configuration');
      add('');
      add('Ethernet adapter:');
      add('   IPv4 Address: 192.168.1.100');
      add('   Subnet Mask: 255.255.255.0');
      add('   Default Gateway: 192.168.1.1');
      break;
    case 'shutdown': shutDown(); break;
    case 'exit': closeWin('terminal'); break;
    default: add(`'${c[0]}' is not recognized as an internal or external command.`, 'error');
  }
  out.scrollTop = out.scrollHeight;
}

/* ─── PAINT ───────────────────────────────────── */
function buildPaint(body) {
  body.innerHTML = `
    <div class="paint-wrap">
      <div class="paint-toolbar">
        <div class="paint-tool active" id="ptool-pencil" onclick="setPaintTool('pencil',this)">✏️ Pencil</div>
        <div class="paint-tool" id="ptool-brush" onclick="setPaintTool('brush',this)">🖌️ Brush</div>
        <div class="paint-tool" id="ptool-eraser" onclick="setPaintTool('eraser',this)">🧹 Eraser</div>
        <div class="paint-tool" id="ptool-line" onclick="setPaintTool('line',this)">📏 Line</div>
        <div class="paint-tool" id="ptool-rect" onclick="setPaintTool('rect',this)">▭ Rectangle</div>
        <div class="paint-tool" id="ptool-circle" onclick="setPaintTool('circle',this)">⬭ Circle</div>
        <div class="paint-tool" id="ptool-fill" onclick="setPaintTool('fill',this)">🪣 Fill</div>
        <input type="color" class="paint-color-picker" id="paint-color" value="#ff0000" title="Color">
        <input type="range" class="paint-size" id="paint-size" min="1" max="40" value="4" title="Size">
        <div class="paint-tool" onclick="clearPaintCanvas()">🗑️ Clear</div>
        <div class="paint-tool" onclick="savePainting()">💾 Save</div>
      </div>
      <div class="paint-canvas-wrap" id="paint-canvas-wrap">
        <canvas id="paint-canvas" class="paint-canvas"></canvas>
      </div>
    </div>
  `;
  initPaint(body);
}

function initPaint(body) {
  const wrap = body.querySelector('#paint-canvas-wrap');
  const canvas = body.querySelector('#paint-canvas');
  let ctx;

  function resize() {
    const oldData = canvas.toDataURL();
    canvas.width = wrap.clientWidth;
    canvas.height = wrap.clientHeight;
    ctx = canvas.getContext('2d');
    ctx.fillStyle = '#fff';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    const img = new Image();
    img.onload = () => ctx.drawImage(img, 0, 0);
    img.src = oldData;
  }
  setTimeout(resize, 50);

  let drawing = false, startX, startY, snapshot;
  let tool = 'pencil';

  body.querySelector('#paint-canvas').addEventListener('mousedown', e => {
    drawing = true;
    const rect = canvas.getBoundingClientRect();
    startX = e.clientX - rect.left;
    startY = e.clientY - rect.top;
    snapshot = ctx.getImageData(0, 0, canvas.width, canvas.height);
    if (tool === 'pencil' || tool === 'brush' || tool === 'eraser') {
      ctx.beginPath(); ctx.moveTo(startX, startY);
    }
  });

  body.querySelector('#paint-canvas').addEventListener('mousemove', e => {
    if (!drawing) return;
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    const color = body.querySelector('#paint-color').value;
    const size = +body.querySelector('#paint-size').value;
    ctx.strokeStyle = tool === 'eraser' ? '#fff' : color;
    ctx.fillStyle = color;
    ctx.lineWidth = tool === 'brush' ? size * 3 : size;
    ctx.lineCap = 'round';

    if (tool === 'pencil' || tool === 'brush' || tool === 'eraser') {
      ctx.lineTo(mx, my); ctx.stroke();
    } else {
      ctx.putImageData(snapshot, 0, 0);
      ctx.beginPath();
      if (tool === 'line') { ctx.moveTo(startX, startY); ctx.lineTo(mx, my); ctx.stroke(); }
      if (tool === 'rect') { ctx.strokeRect(startX, startY, mx - startX, my - startY); }
      if (tool === 'circle') { ctx.ellipse(startX + (mx-startX)/2, startY + (my-startY)/2, Math.abs(mx-startX)/2, Math.abs(my-startY)/2, 0, 0, Math.PI*2); ctx.stroke(); }
    }
  });

  body.querySelector('#paint-canvas').addEventListener('mouseup', e => {
    if (tool === 'fill' && drawing) {
      const rect = canvas.getBoundingClientRect();
      ctx.fillStyle = body.querySelector('#paint-color').value;
      ctx.fillRect(0, 0, canvas.width, canvas.height);
    }
    drawing = false;
  });

  body.querySelector('#paint-canvas').addEventListener('mouseleave', () => { drawing = false; });
  window._paintBody = body;
  window._paintCanvas = canvas;
}

window.setPaintTool = function(t, el) {
  window._paintTool = t;
  document.querySelectorAll('.paint-tool').forEach(x => x.classList.remove('active'));
  el.classList.add('active');
  window._activePaintTool = t;
  if (window._paintBody) {
    window._paintBody.querySelector('#paint-canvas')._tool = t;
  }
};

window.clearPaintCanvas = function() {
  const canvas = window._paintCanvas;
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = '#fff'; ctx.fillRect(0, 0, canvas.width, canvas.height);
};

window.savePainting = function() {
  const canvas = window._paintCanvas;
  if (!canvas) return;
  const a = document.createElement('a');
  a.download = 'painting.png';
  a.href = canvas.toDataURL();
  a.click();
};

/* ─── COPILOT ─────────────────────────────────── */
function buildCopilot(body) {
  body.innerHTML = `
    <div class="copilot-wrap">
      <div class="copilot-header">
        <h3>✨ Copilot</h3>
        <p>Your AI assistant for Windows 12</p>
      </div>
      <div class="copilot-messages" id="copilot-msgs">
        <div class="copilot-msg bot">
          <div class="copilot-bubble">Hello! I'm Copilot, your AI assistant. How can I help you today?</div>
        </div>
      </div>
      <div class="copilot-input-row">
        <input class="copilot-input" id="copilot-input" placeholder="Ask me anything..."
          onkeydown="if(event.key==='Enter')copilotSend()">
        <div class="copilot-send" onclick="copilotSend()">
          <svg viewBox="0 0 24 24" width="18" height="18" fill="white"><path d="M2 21l21-9L2 3v7l15 2-15 2v7z"/></svg>
        </div>
      </div>
    </div>
  `;
}

const COPILOT_RESPONSES = [
  "I can help you with that! Let me look into it.",
  "Great question! Here's what I know about Windows 12...",
  "Sure! I can assist you with Windows 12 features and settings.",
  "I understand. Let me provide some helpful information.",
  "Windows 12 introduces many exciting new features including improved AI integration.",
  "You can find that setting in the Settings app under System > Display.",
  "That's a great idea! Windows 12 supports that natively.",
  "I'm here to make your Windows 12 experience better. What else can I help with?",
];

window.copilotSend = function() {
  const input = document.getElementById('copilot-input');
  const msgs = document.getElementById('copilot-msgs');
  if (!input || !input.value.trim()) return;
  const text = input.value.trim();
  input.value = '';
  const userMsg = document.createElement('div');
  userMsg.className = 'copilot-msg user';
  userMsg.innerHTML = `<div class="copilot-bubble">${escHtml(text)}</div>`;
  msgs.appendChild(userMsg);
  msgs.scrollTop = msgs.scrollHeight;
  setTimeout(() => {
    const botMsg = document.createElement('div');
    botMsg.className = 'copilot-msg bot';
    const r = COPILOT_RESPONSES[Math.floor(Math.random() * COPILOT_RESPONSES.length)];
    botMsg.innerHTML = `<div class="copilot-bubble">${r}</div>`;
    msgs.appendChild(botMsg);
    msgs.scrollTop = msgs.scrollHeight;
  }, 800 + Math.random() * 600);
};

/* ─── MAIL ────────────────────────────────────── */
function buildMail(body) {
  const emails = [
    { from: 'Alex Johnson', subject: 'Project Update', time: '9:41 AM', body: 'Hi, just wanted to give you an update on the project. Everything is on track for the deadline next week.' },
    { from: 'Microsoft', subject: 'Windows Update Available', time: 'Yesterday', body: 'A new update for Windows 12 is available. This update includes security fixes and performance improvements.' },
    { from: 'Sarah Chen', subject: 'Meeting Tomorrow', time: 'Mon', body: 'Don\'t forget our team meeting tomorrow at 10 AM in conference room B.' },
    { from: 'GitHub', subject: 'New PR Review', time: 'Sun', body: 'Someone requested your review on pull request #42: "Add Windows 12 UI components"' },
    { from: 'Newsletter', subject: 'Tech News This Week', time: 'Sat', body: 'This week in tech: AI breakthroughs, new hardware releases, and more...' },
  ];

  body.innerHTML = `
    <div class="mail-wrap">
      <div class="mail-sidebar">
        <div class="mail-folder active">📥 Inbox <span style="margin-left:auto;background:var(--accent);border-radius:10px;padding:1px 6px;font-size:11px">5</span></div>
        <div class="mail-folder">⭐ Starred</div>
        <div class="mail-folder">📤 Sent</div>
        <div class="mail-folder">📋 Drafts</div>
        <div class="mail-folder">🗑️ Trash</div>
      </div>
      <div class="mail-list">
        ${emails.map((m, i) => `
          <div class="mail-item${i===0?' active':''}" onclick="mailOpen(${i},this)">
            <div class="mail-from">${m.from}</div>
            <div class="mail-subject">${m.subject}</div>
            <div class="mail-time">${m.time}</div>
          </div>`).join('')}
      </div>
      <div class="mail-reading" id="mail-reading">
        <div class="mail-reading-from">From: ${emails[0].from}</div>
        <div class="mail-reading-subj">${emails[0].subject}</div>
        <div class="mail-reading-body">${emails[0].body}</div>
      </div>
    </div>
  `;
  window._mailData = emails;
}

window.mailOpen = function(i, el) {
  document.querySelectorAll('.mail-item').forEach(x => x.classList.remove('active'));
  el.classList.add('active');
  const m = window._mailData[i];
  document.getElementById('mail-reading').innerHTML = `
    <div class="mail-reading-from">From: ${m.from}</div>
    <div class="mail-reading-subj">${m.subject}</div>
    <div class="mail-reading-body">${m.body}</div>
  `;
};

/* ─── TASK MANAGER ────────────────────────────── */
function buildTaskManager(body) {
  const processes = [
    { name: 'System', cpu: 2, mem: 8, disk: 0 },
    { name: 'explorer.exe', cpu: 0.5, mem: 15, disk: 0.1 },
    { name: 'msedge.exe', cpu: 4, mem: 320, disk: 0.5 },
    { name: 'SearchHost.exe', cpu: 0.1, mem: 12, disk: 0 },
    { name: 'dwm.exe', cpu: 1, mem: 40, disk: 0 },
    { name: 'RuntimeBroker', cpu: 0, mem: 18, disk: 0 },
    { name: 'SecurityHealth', cpu: 0.2, mem: 22, disk: 0 },
    { name: 'Discord.exe', cpu: 3, mem: 180, disk: 0.2 },
    { name: 'Code.exe', cpu: 5, mem: 280, disk: 0.8 },
    { name: 'Spotify.exe', cpu: 1, mem: 130, disk: 0 },
  ];

  body.innerHTML = `
    <div class="tm-wrap">
      <div class="tm-tabs">
        <div class="tm-tab active">Processes</div>
        <div class="tm-tab" onclick="showToast('Task Manager','Performance tab')">Performance</div>
        <div class="tm-tab" onclick="showToast('Task Manager','Startup tab')">Startup</div>
        <div class="tm-tab" onclick="showToast('Task Manager','Users tab')">Users</div>
      </div>
      <div class="tm-content">
        <table class="tm-table">
          <thead><tr><th>Name</th><th>CPU %</th><th>Memory</th><th>Disk</th></tr></thead>
          <tbody>
            ${processes.map(p => `
              <tr>
                <td>${p.name}</td>
                <td><div class="tm-bar"><div class="tm-bar-fill${p.cpu>5?' crit':p.cpu>2?' warn':''}" style="width:${Math.min(p.cpu*10,100)}%"></div></div> ${p.cpu}%</td>
                <td>${p.mem} MB</td>
                <td>${p.disk} MB/s</td>
              </tr>`).join('')}
          </tbody>
        </table>
      </div>
    </div>
  `;
}

/* ─── STORE ───────────────────────────────────── */
function buildStore(body) {
  const featured = [
    { icon: '🎮', name: 'Minecraft', cat: 'Games', rating: '⭐⭐⭐⭐⭐' },
    { icon: '🎵', name: 'Spotify', cat: 'Music', rating: '⭐⭐⭐⭐⭐' },
    { icon: '📺', name: 'Netflix', cat: 'Entertainment', rating: '⭐⭐⭐⭐' },
    { icon: '💬', name: 'Discord', cat: 'Social', rating: '⭐⭐⭐⭐⭐' },
    { icon: '✏️', name: 'Canva', cat: 'Design', rating: '⭐⭐⭐⭐' },
  ];
  const productivity = [
    { icon: '📝', name: 'Notion', cat: 'Productivity', rating: '⭐⭐⭐⭐⭐' },
    { icon: '📊', name: 'Trello', cat: 'Productivity', rating: '⭐⭐⭐⭐' },
    { icon: '🔑', name: 'Bitwarden', cat: 'Security', rating: '⭐⭐⭐⭐⭐' },
    { icon: '📷', name: 'Lightroom', cat: 'Photos', rating: '⭐⭐⭐⭐' },
  ];

  const appCard = a => `<div class="store-app-card" onclick="showToast('Store','Installing ${a.name}...')"><div class="store-app-icon">${a.icon}</div><div class="store-app-name">${a.name}</div><div class="store-app-cat">${a.cat}</div><div class="store-app-rating">${a.rating}</div></div>`;

  body.innerHTML = `
    <div class="store-wrap">
      <div class="store-nav">
        <span style="font-size:16px">🏪</span>
        <input class="store-search" placeholder="Search apps, games, and more">
      </div>
      <div class="store-body">
        <div class="store-section">
          <h3>🔥 Featured</h3>
          <div class="store-apps">${featured.map(appCard).join('')}</div>
        </div>
        <div class="store-section">
          <h3>💼 Productivity</h3>
          <div class="store-apps">${productivity.map(appCard).join('')}</div>
        </div>
      </div>
    </div>
  `;
}

/* ─── PHOTOS ──────────────────────────────────── */
function buildPhotos(body) {
  const colors = ['#ff6b6b','#ffd93d','#6bcb77','#4d96ff','#c77dff','#ff9a3c','#48cae4','#e8a598','#95d5b2','#f9c74f','#90e0ef','#b5e48c'];
  const emojis = ['🌅','🏔️','🌊','🌸','🦋','🌿','🏙️','🌌','🦁','🌺','🎨','🌈'];
  body.innerHTML = `
    <div class="photos-wrap">
      <div class="photos-toolbar">
        <span style="font-size:13px;font-weight:500">All Photos</span>
        <div style="flex:1"></div>
        <div class="paint-tool" onclick="showToast('Photos','Import photos')">📥 Import</div>
      </div>
      <div class="photos-body">
        <div style="font-size:11px;color:rgba(255,255,255,0.4);margin-bottom:12px">Today · ${colors.length} items</div>
        <div class="photos-grid">
          ${colors.map((c, i) => `<div class="photo-thumb" style="background:${c}22;border:1px solid ${c}44" onclick="showToast('Photos','${emojis[i]} Photo ${i+1}')">${emojis[i]}</div>`).join('')}
        </div>
      </div>
    </div>
  `;
}

/* ─── RECYCLE BIN ─────────────────────────────── */
function buildRecycleBin(body) {
  body.innerHTML = `
    <div class="explorer-wrap">
      <div class="explorer-toolbar">
        <div style="font-size:13px;color:var(--text-dim)">Recycle Bin is empty</div>
        <div style="flex:1"></div>
      </div>
      <div style="flex:1;display:flex;align-items:center;justify-content:center;flex-direction:column;gap:12px;color:var(--text-dim)">
        <div style="font-size:64px">🗑️</div>
        <div style="font-size:16px">The Recycle Bin is empty</div>
        <div style="font-size:12px">Items deleted will appear here</div>
      </div>
    </div>
  `;
}

/* ─── THIS PC ─────────────────────────────────── */
function buildThisPC(body) {
  const drives = [
    { name: 'Local Disk (C:)', icon: '💾', total: 512, used: 280 },
    { name: 'Local Disk (D:)', icon: '💽', total: 1024, used: 420 },
    { name: 'USB Drive (E:)', icon: '🔌', total: 64, used: 12 },
  ];
  body.innerHTML = `
    <div class="explorer-wrap">
      <div class="explorer-toolbar">
        <div class="exp-nav-btn">←</div>
        <div class="exp-nav-btn">→</div>
        <input class="exp-addr" value="This PC" readonly>
      </div>
      <div style="flex:1;padding:20px;overflow-y:auto">
        <div style="font-size:12px;color:var(--text-dim);margin-bottom:12px;font-weight:600;text-transform:uppercase;letter-spacing:0.5px">Devices and Drives</div>
        ${drives.map(d => `
          <div style="background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08);border-radius:10px;padding:16px;margin-bottom:10px;cursor:pointer;transition:background 0.15s" onmouseover="this.style.background='rgba(255,255,255,0.08)'" onmouseout="this.style.background='rgba(255,255,255,0.04)'" ondblclick="showToast('This PC','Opening ${d.name}')">
            <div style="display:flex;align-items:center;gap:12px;margin-bottom:10px">
              <span style="font-size:28px">${d.icon}</span>
              <div>
                <div style="font-size:13px;font-weight:500">${d.name}</div>
                <div style="font-size:11px;color:var(--text-dim)">${d.used} GB used of ${d.total} GB</div>
              </div>
            </div>
            <div style="height:6px;background:rgba(255,255,255,0.1);border-radius:3px;overflow:hidden">
              <div style="height:100%;width:${(d.used/d.total*100).toFixed(0)}%;background:${d.used/d.total>0.8?'#f44336':'var(--accent)'};border-radius:3px"></div>
            </div>
          </div>`).join('')}
      </div>
    </div>
  `;
}

/* ─── INIT ────────────────────────────────────── */
// Paint tool references via canvas data attribute
document.addEventListener('click', e => {
  if (window._paintCanvas && e.target === window._paintCanvas) return;
});
