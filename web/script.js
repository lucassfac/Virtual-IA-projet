/**
 * script.js — Neural Forge Eel Bridge
 * Gère : chat, streaming, attachements, tabs, toggle Turbo, settings
 */

'use strict';

// ── État global ──────────────────────────────────────────────────────

const State = {
  generating:      false,
  turboEnabled:    false,
  turboEligible:   false,
  attachedFile:    null,   // { path, type:'image'|'doc', name, meta }
  streamBuf:       '',
  streamElemId:    null,
  conversationId:  0,
};

// ── Utilitaires DOM ──────────────────────────────────────────────────

const $  = id => document.getElementById(id);
const el = (tag, cls, html) => {
  const e = document.createElement(tag);
  if (cls)  e.className = cls;
  if (html) e.innerHTML = html;
  return e;
};

// ── Markdown basique ─────────────────────────────────────────────────

function mdToHtml(text) {
  return text
    .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
    .replace(/\*\*(.+?)\*\*/g, '<b>$1</b>')
    .replace(/\*(.+?)\*/g,     '<i>$1</i>')
    .replace(/`(.+?)`/g,       '<code>$1</code>')
    .replace(/^### (.+)$/gm,   '<b>$1</b>')
    .replace(/^## (.+)$/gm,    '<b>$1</b>')
    .replace(/^# (.+)$/gm,     '<b>$1</b>')
    .replace(/^[-*] (.+)$/gm,  '<li>$1</li>')
    .replace(/\n/g,            '<br/>');
}

// ── Scroll ───────────────────────────────────────────────────────────

function scrollBottom(smooth = true) {
  const s = $('chat-scroll');
  s.scrollTo({ top: s.scrollHeight, behavior: smooth ? 'smooth' : 'instant' });
}

// ── Ajout de bulle ───────────────────────────────────────────────────

function addBubble(type, html, id = null) {
  const wrap = el('div', `bubble-wrap ${type}`);
  const bub  = el('div', `bubble ${type}`, html);
  if (id) bub.id = id;
  wrap.appendChild(bub);
  $('messages').appendChild(wrap);
  scrollBottom();
  return bub;
}

function addSys(text) {
  addBubble('sys', text);
}

function addThinking() {
  const id = 'thinking-' + Date.now();
  addBubble('ai',
    '<div class="thinking">'
    + '<span></span><span></span><span></span>'
    + '</div>',
    id
  );
  return id;
}

function removeById(id) {
  const el = document.getElementById(id);
  if (el) el.closest('.bubble-wrap')?.remove();
}

// ── Tabs ─────────────────────────────────────────────────────────────

function switchTab(name, btn) {
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
  $('tab-' + name).classList.add('active');
  if (btn) btn.classList.add('active');
}

// ── Turbo toggle ─────────────────────────────────────────────────────

function toggleTurbo() {
  const tog = $('turbo-toggle');
  if (tog.classList.contains('locked')) return;
  State.turboEnabled = !State.turboEnabled;
  tog.classList.toggle('on', State.turboEnabled);
  $('turbo-sub').textContent = State.turboEnabled ? 'Turbo ⚡' : 'Standard';
}

function setTurboEligible(eligible, reason = '') {
  State.turboEligible = eligible;
  const tog = $('turbo-toggle');
  tog.classList.toggle('locked', !eligible);
  tog.title = eligible
    ? 'Cliquez pour activer le Speculative Decoding'
    : reason;
}

// ── Attachements ─────────────────────────────────────────────────────

function showAttachMenu(event) {
  const menu = $('attach-menu');
  const rect = event.currentTarget.getBoundingClientRect();
  menu.style.left   = rect.left + 'px';
  menu.style.bottom = (window.innerHeight - rect.top + 8) + 'px';
  menu.classList.toggle('hidden');
  menu.style.display = menu.classList.contains('hidden') ? 'none' : 'flex';
  menu.style.flexDirection = 'column';
  setTimeout(() => document.addEventListener('click', closeAttachMenu, { once: true }), 0);
}

function closeAttachMenu() {
  const m = $('attach-menu');
  m.classList.add('hidden');
  m.style.display = 'none';
}

async function attachImage() {
  closeAttachMenu();
  const path = await eel.browse_file('image')();
  if (!path) return;
  const name = path.split(/[\\/]/).pop();
  State.attachedFile = { path, type: 'image', name, meta: 'Image' };
  showAttachPreview('🖼', name, 'Image');
  $('attach-btn').classList.add('active');
}

async function attachDocument() {
  closeAttachMenu();
  const path = await eel.browse_file('document')();
  if (!path) return;
  const name = path.split(/[\\/]/).pop();
  const ext  = name.split('.').pop().toUpperCase();
  State.attachedFile = { path, type: 'doc', name, meta: ext };
  showAttachPreview('📄', name, ext);
  $('attach-btn').classList.add('active');
}

function showAttachPreview(icon, name, meta) {
  $('attach-thumb').textContent = icon;
  $('attach-name').textContent  = name;
  $('attach-meta').textContent  = meta;
  $('attach-preview').classList.remove('hidden');
}

function removeAttachment() {
  State.attachedFile = null;
  $('attach-preview').classList.add('hidden');
  $('attach-btn').classList.remove('active');
}

// ── Input auto-resize ─────────────────────────────────────────────────

function autoResize(el) {
  el.style.height = 'auto';
  el.style.height = Math.min(el.scrollHeight, 120) + 'px';
}

function handleKey(e) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
}

// ── Envoi message ─────────────────────────────────────────────────────

async function sendMessage() {
  if (State.generating) return;
  const input = $('user-input');
  const text  = input.value.trim();
  if (!text && !State.attachedFile) return;

  input.value = '';
  input.style.height = 'auto';

  // Bulle utilisateur
  if (State.attachedFile) {
    const f = State.attachedFile;
    addBubble('user',
      `<div style="display:flex;align-items:center;gap:8px;margin-bottom:${text?'8px':'0'};">
         <span style="font-size:20px;">${f.type==='image'?'🖼':'📄'}</span>
         <div>
           <div style="font-size:12px;font-weight:500;">${f.name}</div>
           <div style="font-size:11px;opacity:0.6;">${f.meta}</div>
         </div>
       </div>${text ? mdToHtml(text) : ''}`
    );
  } else {
    addBubble('user', mdToHtml(text));
  }

  const attached = State.attachedFile;
  removeAttachment();

  // Thinking
  const thinkId = addThinking();
  setGenerating(true);

  // Appel Python
  try {
    await eel.send_message(
      text,
      State.turboEnabled,
      attached ? attached.path : null,
      attached ? attached.type : null
    )();
  } catch (e) {
    removeById(thinkId);
    addSys('Erreur : ' + e);
    setGenerating(false);
  }
}

// ── Streaming (appelé token par token depuis Python) ──────────────────

eel.expose(display_token);
function display_token(token) {
  // Retire le thinking au premier token
  if (!State.streamBuf) {
    document.querySelectorAll('.thinking').forEach(el => {
      el.closest('.bubble-wrap')?.remove();
    });
    // Crée une bulle vide
    const id = 'stream-' + Date.now();
    State.streamElemId = id;
    addBubble('ai', '', id);
    document.getElementById(id).classList.add('cursor-blink');
  }
  State.streamBuf += token;
  const bub = document.getElementById(State.streamElemId);
  if (bub) bub.innerHTML = mdToHtml(State.streamBuf);
  scrollBottom(false);
}

eel.expose(stream_done);
function stream_done(error = null) {
  const bub = document.getElementById(State.streamElemId);
  if (bub) bub.classList.remove('cursor-blink');
  if (error) {
    document.querySelectorAll('.thinking').forEach(el => el.closest('.bubble-wrap')?.remove());
    addSys('Erreur : ' + error);
  }
  State.streamBuf    = '';
  State.streamElemId = null;
  setGenerating(false);
}

// ── Stop ─────────────────────────────────────────────────────────────

async function stopGeneration() {
  await eel.stop_generation()();
  const bub = document.getElementById(State.streamElemId);
  if (bub) {
    bub.classList.remove('cursor-blink');
    bub.innerHTML += ' <span style="color:#ef4444;font-size:11px;">[arrêté]</span>';
  }
  State.streamBuf    = '';
  State.streamElemId = null;
  setGenerating(false);
}

function setGenerating(on) {
  State.generating = on;
  $('send-btn').classList.toggle('loading', on);
  $('stop-btn').classList.toggle('visible', on);
  $('user-input').disabled = on;
}

// ── Historique ────────────────────────────────────────────────────────

async function saveConversation() {
  const msgs  = $('messages').innerHTML;
  const title = $('messages').querySelector('.bubble.user')?.textContent.slice(0,50) || 'Conversation';
  await eel.save_conversation(title, msgs)();
  addSys('Conversation sauvegardée.');
}

async function showHistory() {
  const convs = await eel.list_conversations()();
  if (!convs || !convs.length) { addSys('Aucune conversation sauvegardée.'); return; }

  let html = convs.slice(0,15).map(c =>
    `<div class="attach-item" onclick="loadConversation('${c.id}')">${c.date} — ${c.title}</div>`
  ).join('');
  // Simple inline popup
  const existing = document.getElementById('history-popup');
  if (existing) existing.remove();
  const pop = el('div', '', html);
  pop.id = 'history-popup';
  pop.style.cssText = (
    'position:fixed;top:60px;left:50%;transform:translateX(-50%);'
    + 'background:#1c1c20;border:1px solid rgba(255,255,255,0.10);'
    + 'border-radius:14px;padding:6px;min-width:320px;max-height:300px;'
    + 'overflow-y:auto;z-index:100;box-shadow:0 16px 40px rgba(0,0,0,0.6);'
  );
  document.body.appendChild(pop);
  setTimeout(() => document.addEventListener('click', () => pop.remove(), { once: true }), 0);
}

async function loadConversation(id) {
  const html = await eel.get_conversation(id)();
  if (html) {
    $('messages').innerHTML = html;
    addSys('Conversation restaurée.');
    scrollBottom(false);
  }
}

function clearChat() {
  $('messages').innerHTML = '';
  addSys('Conversation effacée.');
}

// ── Settings ──────────────────────────────────────────────────────────

async function loadModel() {
  const modelPath = $('model-path-input').value.trim();
  const loraPath  = $('lora-path-input').value.trim();
  const draftPath = $('draft-path-input').value.trim();
  if (!modelPath) { $('settings-status').textContent = '⚠ Chemin du modèle requis.'; return; }

  $('load-btn').textContent = 'Chargement…';
  $('load-btn').style.background = '#1e293b';
  $('settings-status').textContent = '';

  const result = await eel.load_model(modelPath, loraPath, draftPath, State.turboEnabled)();
  if (result.ok) {
    $('load-btn').textContent = 'Chargé ✓';
    $('load-btn').style.background = '#16a34a';
    $('settings-status').textContent = result.message;
    updateModelStatus('llm', true, result.name);
    // Notifier le chat
    $('chat-model-label').textContent = result.name;
    $('chat-dot').className = 'dot dot-green';
    addSys(`Modèle prêt : ${result.name}`);
    switchTab('chat', document.querySelector('.nav-btn'));
  } else {
    $('load-btn').textContent = 'Charger le modèle';
    $('load-btn').style.background = '#2563eb';
    $('settings-status').textContent = '✕ ' + result.message;
    $('settings-status').style.color = '#ef4444';
  }
}

async function browseModel()  { const p = await eel.browse_file('model')();    if (p) $('model-path-input').value  = p; }
async function browseLora()   { const p = await eel.browse_file('lora')();     if (p) $('lora-path-input').value   = p; }
async function browseDraft()  { const p = await eel.browse_file('draft')();    if (p) $('draft-path-input').value  = p; }

async function saveHFToken() {
  const t = $('hf-token-input').value.trim();
  if (!t) return;
  await eel.save_hf_token(t)();
  $('hf-token-input').value = '';
  addSys('Token HuggingFace sauvegardé.');
}

// ── Status helpers ────────────────────────────────────────────────────

function updateModelStatus(type, ready, name = '') {
  const dot   = $(type + '-dot');
  const label = $(type + '-label');
  dot.className = 'dot ' + (ready ? 'dot-green' : 'dot-idle');
  label.textContent = ready ? name.slice(0, 10) : '—';
}

// ── Python → JS callbacks ─────────────────────────────────────────────

eel.expose(set_hw_info);
function set_hw_info(badge, color, turbo_eligible, turbo_reason) {
  const b = $('perf-badge');
  b.textContent = badge;
  b.style.color  = color;
  b.style.borderColor = color + '44';
  setTurboEligible(turbo_eligible, turbo_reason);
}

eel.expose(set_model_status);
function set_model_status(type, ready, name) {
  updateModelStatus(type, ready, name);
  if (type === 'llm' && ready) {
    $('chat-dot').className   = 'dot dot-green';
    $('chat-model-label').textContent = name;
  }
}

// ── Init ─────────────────────────────────────────────────────────────

window.addEventListener('DOMContentLoaded', async () => {
  addSys('Neural Forge — Edge AI Studio');

  // Charger infos matérielles depuis Python
  try {
    const hw = await eel.get_hw_info()();
    set_hw_info(hw.badge, hw.color, hw.turbo_eligible, hw.turbo_reason);
  } catch(e) {
    console.warn('hw info unavailable', e);
  }

  // Charger le dernier modèle si disponible
  try {
    const last = await eel.get_last_model()();
    if (last && last.model_path) {
      $('model-path-input').value = last.model_path;
      if (last.lora_path) $('lora-path-input').value = last.lora_path;
      addSys('Chargement automatique du dernier modèle…');
      const result = await eel.load_model(last.model_path, last.lora_path, '', false)();
      if (result.ok) {
        set_model_status('llm', true, result.name);
        addSys('Modèle prêt : ' + result.name);
      }
    }
  } catch(e) {
    console.warn('auto-load failed', e);
  }
});

// Fermer les menus en cliquant ailleurs
document.addEventListener('click', e => {
  const menu = $('attach-menu');
  if (!menu.contains(e.target) && e.target.id !== 'attach-btn') {
    menu.classList.add('hidden');
    menu.style.display = 'none';
  }
});
