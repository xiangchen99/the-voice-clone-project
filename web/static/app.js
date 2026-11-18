/**
 * Digital Self Studio - Client Controller
 * Pure Vanilla JavaScript with Web Audio API (raw PCM microphone capture).
 */

// Global State
let currentTab = 'tab-chat';
let interviewQuestions = [];
let activeQuestionIndex = 0;
let voicePrompts = [];
let isChatRecording = false;
let chatRecorder = null;
let chatRecordingTimerInterval = null;
let chatRecordingStartTime = 0;

// Prompt Recorder State
let promptRecorder = null;
let activeRecordingPromptId = null;

document.addEventListener('DOMContentLoaded', () => {
  initTabs();
  initChat();
  initInterview();
  initVoiceStudio();
  initMemories();
  initPersona();
  refreshSystemStatus();
  setInterval(() => {
    if (!document.hidden) refreshSystemStatus();
  }, 15000);
});

// ========================================================
// RAW PCM MICROPHONE RECORDER -> 16-bit RIFF WAV
// Captures samples directly via the Web Audio API, so no browser codec
// (WebM/Opus/Ogg) or decodeAudioData() support is required.
// ========================================================
function encodeWav(samples, inputRate, targetRate = 24000) {
  let data = samples;
  let rate = inputRate;
  if (inputRate !== targetRate && samples.length > 1) {
    const outLen = Math.max(1, Math.round(samples.length * targetRate / inputRate));
    data = new Float32Array(outLen);
    const ratio = (samples.length - 1) / Math.max(1, outLen - 1);
    for (let i = 0; i < outLen; i++) {
      const pos = i * ratio;
      const idx = Math.floor(pos);
      const frac = pos - idx;
      const next = idx + 1 < samples.length ? samples[idx + 1] : samples[idx];
      data[i] = samples[idx] * (1 - frac) + next * frac;
    }
    rate = targetRate;
  }

  const buffer = new ArrayBuffer(44 + data.length * 2);
  const view = new DataView(buffer);
  const writeStr = (off, s) => { for (let i = 0; i < s.length; i++) view.setUint8(off + i, s.charCodeAt(i)); };

  writeStr(0, 'RIFF');
  view.setUint32(4, 36 + data.length * 2, true);
  writeStr(8, 'WAVE');
  writeStr(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);          // PCM
  view.setUint16(22, 1, true);          // mono
  view.setUint32(24, rate, true);
  view.setUint32(28, rate * 2, true);   // byte rate
  view.setUint16(32, 2, true);          // block align
  view.setUint16(34, 16, true);         // bits per sample
  writeStr(36, 'data');
  view.setUint32(40, data.length * 2, true);

  let off = 44;
  for (let i = 0; i < data.length; i++) {
    const s = Math.max(-1, Math.min(1, data[i]));
    view.setInt16(off, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
    off += 2;
  }
  return new Blob([buffer], { type: 'audio/wav' });
}

class PcmRecorder {
  async start() {
    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
    if (!AudioContextClass || !navigator.mediaDevices?.getUserMedia) {
      throw new Error('This browser does not support microphone capture via the Web Audio API.');
    }
    this.stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    this.ctx = new AudioContextClass();
    if (this.ctx.state === 'suspended') await this.ctx.resume();
    this.chunks = [];
    this.source = this.ctx.createMediaStreamSource(this.stream);
    this.processor = this.ctx.createScriptProcessor(4096, 1, 1);
    this.processor.onaudioprocess = (e) => {
      this.chunks.push(new Float32Array(e.inputBuffer.getChannelData(0)));
    };
    // The processor only fires while connected to the destination; route it
    // through a muted gain node so the microphone is not played back.
    this.mute = this.ctx.createGain();
    this.mute.gain.value = 0;
    this.source.connect(this.processor);
    this.processor.connect(this.mute);
    this.mute.connect(this.ctx.destination);
  }

  /** Stops capture and returns a WAV Blob, or null if nothing was captured. */
  async stop(targetRate = 24000) {
    const inputRate = this.ctx ? this.ctx.sampleRate : targetRate;
    try {
      if (this.processor) this.processor.onaudioprocess = null;
      this.source?.disconnect();
      this.processor?.disconnect();
      this.mute?.disconnect();
    } catch (_) {}
    this.stream?.getTracks().forEach(t => t.stop());
    try { await this.ctx?.close(); } catch (_) {}

    const total = (this.chunks || []).reduce((n, c) => n + c.length, 0);
    if (total === 0) return null;
    const merged = new Float32Array(total);
    let off = 0;
    for (const c of this.chunks) { merged.set(c, off); off += c.length; }
    this.chunks = [];
    return encodeWav(merged, inputRate, targetRate);
  }
}

// ========================================================
// 1. TABS NAVIGATION
// ========================================================
function initTabs() {
  const tabBtns = document.querySelectorAll('.tab-btn');
  tabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      const targetTab = btn.getAttribute('data-tab');
      tabBtns.forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));

      btn.classList.add('active');
      const pane = document.getElementById(targetTab);
      if (pane) pane.classList.add('active');
      currentTab = targetTab;

      if (targetTab === 'tab-interview') loadQuestions();
      if (targetTab === 'tab-voice') loadPrompts();
      if (targetTab === 'tab-memories') loadAllMemories();
      if (targetTab === 'tab-persona') loadPersona();
    });
  });
}

// ========================================================
// 2. SYSTEM STATUS REFRESH
// ========================================================
async function refreshSystemStatus() {
  try {
    const res = await fetch('/api/status');
    if (!res.ok) throw new Error('Status request failed');
    const data = await res.json();
    document.getElementById('stat-memory-count').textContent = data.memories_count || 0;
    document.getElementById('stat-voice-count').textContent = data.voice_samples_count || 0;
    renderEngineStatus(data.llm_status, data.llm_model);
    const ttsEl = document.getElementById('stat-tts-engine');
    const ttsDot = document.getElementById('stat-tts-dot');
    if (ttsEl && ttsDot) {
      if (data.tts_is_mock) {
        ttsEl.textContent = 'Demo';
        ttsDot.className = 'status-dot';
        ttsDot.style.background = '#f59e0b';
      } else {
        ttsEl.textContent = data.tts_ready ? 'XTTS-v2' : 'XTTS-v2 · standby';
        ttsDot.classList.toggle('active', data.tts_ready);
        ttsDot.style.background = data.tts_ready ? '' : 'var(--text-muted)';
      }
    }
  } catch (err) {
    renderEngineStatus('dashboard_unavailable');
    console.error('Failed to fetch status:', err);
  }
}

function renderEngineStatus(state, model = '') {
  const labels = {
    demo: 'Demo',
    unavailable: 'Ollama unavailable',
    missing_model: 'Model missing',
    installed: 'Model installed',
    working: model,
    reply_failed: 'Last reply failed',
    unchecked: 'Unverified',
    dashboard_unavailable: 'Dashboard unavailable'
  };
  const badge = document.getElementById('status-engine');
  badge.dataset.state = state || 'unchecked';
  badge.title = model;
  document.getElementById('stat-engine-name').textContent = labels[state] || labels.unchecked;
}

function initChat() {
  const input = document.getElementById('chat-text-input');
  document.getElementById('btn-send-chat').addEventListener('click', sendTextMessage);
  input.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      sendTextMessage();
    }
  });
  document.getElementById('btn-record-chat').addEventListener('click', toggleChatRecording);
}

function lockComposer(locked) {
  document.getElementById('chat-text-input').disabled = locked;
  document.getElementById('btn-send-chat').disabled = locked;
  document.getElementById('btn-record-chat').disabled = locked;
}

async function sendTextMessage() {
  const input = document.getElementById('chat-text-input');
  const status = document.getElementById('chat-input-status');
  const text = input.value.trim();
  if (input.disabled || isChatRecording || !text) return;

  const started = performance.now();
  lockComposer(true);
  status.textContent = 'Generating a reply…';
  resetReplyAudio();
  try {
    const response = await fetch('/api/chat/message', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: text })
    });
    const turn = await response.json();
    if (!response.ok) throw new Error(turn.error || 'Could not generate a reply.');

    input.value = '';
    appendChatMessage('user', text);
    const replyId = appendChatMessage('assistant', turn.response_text);
    renderRecalledMemories(turn.cited_memories, turn.metrics);
    status.textContent = '';
    const reply = document.getElementById(replyId).querySelector('.msg-body');
    if (turn.cited_memories?.length) {
      const sources = document.createElement('details');
      sources.className = 'reply-sources';
      const heading = document.createElement('summary');
      heading.textContent = 'Memory sources';
      sources.append(heading, ...turn.cited_memories.map(memoryCard));
      reply.append(sources);
    }
    if (turn.llm_backend === 'mock') {
      const demoLabel = document.createElement('p');
      demoLabel.className = 'chat-input-status';
      demoLabel.textContent = 'Demo reply';
      reply.append(demoLabel);
    }
    if (turn.audio_url) {
      const audio = {
        url: turn.audio_url,
        label: turn.tts_is_mock ? 'Demo chime' : 'Cloned voice'
      };
      const replay = document.createElement('button');
      replay.type = 'button';
      replay.className = 'btn-secondary reply-replay';
      replay.textContent = 'Replay';
      replay.addEventListener('click', () => playReplyAudio(audio, null));
      reply.append(replay);
      playReplyAudio(audio, started);
    } else {
      const message = turn.tts_message || 'Audio is unavailable for this reply.';
      document.getElementById('audio-engine-badge').textContent = 'Audio unavailable';
      document.getElementById('audio-status').textContent = message;
      const audioNote = document.createElement('p');
      audioNote.className = 'chat-input-status';
      audioNote.textContent = message;
      reply.append(audioNote);
    }
    refreshSystemStatus();
  } catch (error) {
    document.getElementById('audio-engine-badge').textContent = 'No new audio';
    status.textContent = error.message === 'Failed to fetch'
      ? 'Could not reach Digital Self. Your message is ready to retry.'
      : error.message;
    refreshSystemStatus();
  } finally {
    lockComposer(false);
    input.focus();
  }
}

async function toggleChatRecording() {
  const micBtn = document.getElementById('btn-record-chat');
  const micLabel = document.getElementById('mic-label');
  const banner = document.getElementById('recording-banner');
  const timer = document.getElementById('recording-timer');
  const input = document.getElementById('chat-text-input');
  const status = document.getElementById('chat-input-status');

  if (!isChatRecording) {
    document.getElementById('audio-playback').pause();
    lockComposer(true);
    status.textContent = '';
    try {
      chatRecorder = new PcmRecorder();
      await chatRecorder.start();
      isChatRecording = true;
      micBtn.classList.add('recording');
      micLabel.textContent = 'Stop';
      micBtn.disabled = false;
      banner.classList.remove('hidden');

      chatRecordingStartTime = Date.now();
      chatRecordingTimerInterval = setInterval(() => {
        const elapsedSec = Math.floor((Date.now() - chatRecordingStartTime) / 1000);
        const mm = String(Math.floor(elapsedSec / 60)).padStart(2, '0');
        const ss = String(elapsedSec % 60).padStart(2, '0');
        timer.textContent = `${mm}:${ss}`;
      }, 500);

    } catch (err) {
      if (chatRecorder) { await chatRecorder.stop().catch(() => {}); chatRecorder = null; }
      lockComposer(false);
      status.textContent = 'Microphone unavailable. Allow microphone access or type a message.';
    }
  } else {
    // Stop Recording
    isChatRecording = false;
    micBtn.classList.remove('recording');
    micLabel.textContent = 'Record';
    banner.classList.add('hidden');
    clearInterval(chatRecordingTimerInterval);

    micBtn.disabled = true;
    status.textContent = 'Transcribing recording…';
    const recorder = chatRecorder;
    chatRecorder = null;
    try {
      const wav = await recorder.stop(24000);
      if (!wav) throw new Error('No audio captured. Record again or type a message.');
      const form = new FormData();
      form.append('audio', wav, 'mic_input.wav');
      const response = await fetch('/api/chat/transcribe', { method: 'POST', body: form });
      const result = await response.json();
      if (!response.ok) throw new Error(result.error || 'Could not transcribe the recording.');
      input.value = result.user_text;
      status.textContent = 'Review the transcript, then send.';
    } catch (error) {
      status.textContent = error.message;
    } finally {
      lockComposer(false);
      input.focus();
    }
  }
}

function appendChatMessage(role, text) {
  const container = document.getElementById('chat-messages');
  const msgId = 'msg-' + Date.now() + '-' + Math.random().toString(36).substr(2, 4);

  const msgDiv = document.createElement('div');
  msgDiv.className = `message ${role}-msg`;
  msgDiv.id = msgId;

  const authorName = role === 'user' ? 'You' : (document.getElementById('chat-twin-name').textContent || 'Digital Twin');
  const avatarSvg = role === 'user'
    ? `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path><circle cx="12" cy="7" r="4"></circle></svg>`
    : `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="3"></circle><path d="M12 2v2"></path><path d="M12 20v2"></path><path d="m4.93 4.93 1.41 1.41"></path><path d="m17.66 17.66 1.41 1.41"></path><path d="M2 12h2"></path><path d="M20 12h2"></path></svg>`;

  msgDiv.innerHTML = `
    <div class="msg-avatar">${avatarSvg}</div>
    <div class="msg-body">
      <div class="msg-header">
        <span class="msg-author">${escapeHtml(authorName)}</span>
        <span class="msg-time">${new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</span>
      </div>
      <div class="msg-text">${escapeHtml(text)}</div>
    </div>
  `;

  container.appendChild(msgDiv);
  container.scrollTop = container.scrollHeight;
  return msgId;
}

function renderRecalledMemories(memories, metrics) {
  const listContainer = document.getElementById('recalled-memories-container');
  const badge = document.getElementById('recalled-count-badge');
  listContainer.innerHTML = '';

  if (!memories || memories.length === 0) {
    listContainer.innerHTML = '<div class="empty-recalled">The reply cited no approved memories.</div>';
    badge.textContent = '0 cited';
  } else {
    badge.textContent = `${memories.length} cited`;
    listContainer.append(...memories.map(memoryCard));
  }

  document.getElementById('metric-total-time').textContent =
    metrics?.total_sec == null ? '-' : `${metrics.total_sec}s`;
  document.getElementById('metric-tts-time').textContent =
    metrics?.tts_synthesis_sec == null ? '-' : `${metrics.tts_synthesis_sec}s`;
}

function memoryCard(memory) {
  const card = document.createElement('div');
  card.className = 'recalled-card';
  const name = memory.metadata?.slug?.replaceAll('_', ' ') || memory.category || 'General';
  card.innerHTML = `
    <div class="recalled-card-header">
      <span class="recalled-cat">${escapeHtml(name)}</span>
      <span>${escapeHtml(memory.citation)}</span>
    </div>
    <div class="recalled-snippet">${escapeHtml(memory.content)}</div>
  `;
  return card;
}

function resetReplyAudio() {
  const player = document.getElementById('audio-playback');
  player.onplaying = null;
  player.onerror = null;
  player.pause();
  player.removeAttribute('src');
  player.load();
  player.hidden = true;
  document.getElementById('metric-playback-time').textContent = '-';
  document.getElementById('audio-engine-badge').textContent = 'Waiting';
  document.getElementById('audio-status').textContent = '';
}

function playReplyAudio(audio, started) {
  const player = document.getElementById('audio-playback');
  const status = document.getElementById('audio-status');
  status.textContent = '';
  player.onplaying = () => {
    status.textContent = '';
    if (started !== null) {
      document.getElementById('metric-playback-time').textContent =
        `${((performance.now() - started) / 1000).toFixed(2)}s`;
    }
    player.onplaying = null;
  };
  player.onerror = () => { status.textContent = 'Could not load the reply audio.'; };
  player.hidden = false;
  document.getElementById('audio-engine-badge').textContent = audio.label;
  if (player.getAttribute('src') === audio.url) {
    player.currentTime = 0;
  } else {
    player.src = audio.url;
  }
  player.play().catch((error) => {
    if (error.name !== 'AbortError' && player.getAttribute('src') === audio.url) {
      status.textContent = error.name === 'NotAllowedError'
        ? 'Press Play to hear the reply.'
        : 'Could not play the reply audio.';
    }
  });
}

// ========================================================
// 4. LIFE INTERVIEW TAB
// ========================================================
function initInterview() {
  document.getElementById('btn-save-question').addEventListener('click', saveActiveQuestion);
  document.getElementById('btn-next-question').addEventListener('click', () => {
    if (activeQuestionIndex < interviewQuestions.length - 1) {
      selectQuestion(activeQuestionIndex + 1);
    }
  });
}

async function loadQuestions() {
  try {
    const res = await fetch('/api/questions');
    if (!res.ok) return;
    interviewQuestions = await res.json();
    renderQuestionsList();
    if (interviewQuestions.length > 0) {
      selectQuestion(activeQuestionIndex);
    }
  } catch (err) {
    console.error('Failed to load questions:', err);
  }
}

function renderQuestionsList() {
  const list = document.getElementById('questions-list-container');
  list.innerHTML = '';

  let completedCount = 0;
  interviewQuestions.forEach((q, idx) => {
    if (q.answered) completedCount++;

    const item = document.createElement('div');
    item.className = `q-nav-item ${idx === activeQuestionIndex ? 'active' : ''}`;
    item.addEventListener('click', () => selectQuestion(idx));

    const statusBadge = q.answered
      ? `<span class="q-status-badge done">✓ Saved</span>`
      : `<span class="q-status-badge pending">Pending</span>`;

    item.innerHTML = `
      <div>
        <div class="q-nav-title">${idx + 1}. ${escapeHtml(q.slug.replace(/_/g, ' '))}</div>
        <div class="q-nav-cat">${escapeHtml(q.category)}</div>
      </div>
      ${statusBadge}
    `;
    list.appendChild(item);
  });

  const total = interviewQuestions.length;
  const pct = total > 0 ? Math.round((completedCount / total) * 100) : 0;
  document.getElementById('interview-progress-bar').style.width = `${pct}%`;
  document.getElementById('interview-progress-text').textContent = `${completedCount} / ${total} completed (${pct}%)`;
}

function selectQuestion(idx) {
  activeQuestionIndex = idx;
  const q = interviewQuestions[idx];
  if (!q) return;

  document.querySelectorAll('.q-nav-item').forEach((el, i) => {
    el.classList.toggle('active', i === idx);
  });

  document.getElementById('active-q-category').textContent = q.category;
  document.getElementById('active-q-title').textContent = `${idx + 1}. ${q.slug.replace(/_/g, ' ').toUpperCase()}`;
  document.getElementById('active-q-prompt').textContent = `"${q.question}"`;

  const input = document.getElementById('interview-answer-input');
  input.value = q.answer || '';
  document.getElementById('interview-chat-approval').checked = !!q.approved_for_chat;

  const saveStatus = document.getElementById('save-status-indicator');
  saveStatus.textContent = q.answered ? 'Saved' : 'Unsaved';
  saveStatus.style.color = q.answered ? 'var(--accent-emerald)' : 'var(--text-muted)';
}

async function saveActiveQuestion() {
  const q = interviewQuestions[activeQuestionIndex];
  if (!q) return;

  const answer = document.getElementById('interview-answer-input').value.trim();
  const approved = !!answer && document.getElementById('interview-chat-approval').checked;
  const saveStatus = document.getElementById('save-status-indicator');
  saveStatus.textContent = 'Saving...';

  try {
    const res = await fetch('/api/interview/save', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        slug: q.slug,
        category: q.category,
        question: q.question,
        answer: answer,
        approved_for_chat: approved
      })
    });
    const result = await res.json();
    if (result.success) {
      q.answer = answer;
      q.answered = !!answer;
      q.approved_for_chat = approved;
      saveStatus.textContent = 'Saved!';
      saveStatus.style.color = 'var(--accent-emerald)';
      renderQuestionsList();
      refreshSystemStatus();
    }
  } catch (err) {
    saveStatus.textContent = 'Save Failed';
    console.error('Save error:', err);
  }
}

// ========================================================
// 5. VOICE STUDIO TAB
// ========================================================
function initVoiceStudio() {
  document.getElementById('voice-reference-form').addEventListener('submit', saveVoiceReferences);
}

async function loadPrompts() {
  try {
    const res = await fetch('/api/prompts');
    if (!res.ok) return;
    voicePrompts = await res.json();
    renderPromptsGrid();
    loadMasterAudio();
    loadVoiceReferences();
  } catch (err) {
    console.error('Failed to load prompts:', err);
  }
}

async function loadVoiceReferences() {
  const options = document.getElementById('voice-reference-options');
  const status = document.getElementById('voice-reference-status');
  const save = document.querySelector('#voice-reference-form button');
  save.disabled = true;
  try {
    const response = await fetch('/api/voice/references');
    if (!response.ok) throw new Error('Could not load voice references.');
    const data = await response.json();
    options.replaceChildren();
    for (const name of data.available) {
      const label = document.createElement('label');
      label.className = 'choice-field';
      const box = document.createElement('input');
      box.type = 'checkbox';
      box.value = name;
      box.checked = data.selected.includes(name);
      label.append(box, name);
      options.append(label);
    }
    const missing = data.selected.filter((name) => !data.available.includes(name));
    status.textContent = missing.length ? `Selected outside this list: ${missing.join(', ')}` : '';
    if (!data.available.length && !missing.length) {
      status.textContent = 'Record a sample to add a voice reference.';
    }
    save.disabled = !data.available.length && !data.has_local_selection;
  } catch (error) {
    status.textContent = error.message;
  }
}

async function saveVoiceReferences(event) {
  event.preventDefault();
  const form = event.currentTarget;
  const button = form.querySelector('button');
  const status = document.getElementById('voice-reference-status');
  const selected = Array.from(form.querySelectorAll('input:checked'), (box) => box.value);
  button.disabled = true;
  try {
    const response = await fetch('/api/voice/references', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ selected })
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || 'Could not save voice references.');
    status.textContent = result.selected.length ? 'Voice references saved.' : 'Voice references cleared.';
  } catch (error) {
    status.textContent = error.message;
  } finally {
    button.disabled = false;
  }
}

async function loadMasterAudio() {
  const badge = document.getElementById('master-ref-badge');
  const player = document.getElementById('master-audio-player');
  try {
    const res = await fetch('/api/audio/master_ref.wav', { method: 'HEAD' });
    if (res.ok) {
      badge.textContent = 'Master Audio: Ready';
      badge.style.color = 'var(--accent-emerald)';
      player.src = '/api/audio/master_ref.wav';
    } else {
      badge.textContent = 'Master Audio: Pending';
      badge.style.color = 'var(--text-muted)';
    }
  } catch (e) {
    badge.textContent = 'Master Audio: Pending';
  }
}

function renderPromptsGrid() {
  const grid = document.getElementById('voice-prompts-grid');
  grid.innerHTML = '';

  voicePrompts.forEach((p, idx) => {
    const card = document.createElement('div');
    card.className = 'prompt-card';
    card.id = `prompt-card-${p.id}`;

    const isDone = p.recorded;
    const statusText = isDone ? '✓ Recorded & Normalized' : 'Unrecorded';
    const statusColor = isDone ? 'var(--accent-emerald)' : 'var(--text-muted)';

    card.innerHTML = `
      <div class="prompt-card-header">
        <span class="prompt-category">${escapeHtml(p.category)}</span>
        <span style="font-size:0.75rem; font-weight:600; color:${statusColor};" id="p-status-${p.id}">${statusText}</span>
      </div>
      <div class="prompt-text">"${escapeHtml(p.prompt)}"</div>
      <div class="prompt-intent">Intent: ${escapeHtml(p.intent)}</div>

      <div class="prompt-actions">
        <button class="btn-secondary" id="btn-rec-${p.id}" onclick="togglePromptRecording('${p.id}')">
          🎙️ Record Sample
        </button>
        <audio id="audio-prev-${p.id}" controls style="height:32px; max-width:200px; display:${isDone ? 'block' : 'none'};" 
               src="${isDone ? `/api/audio/${p.id}_normalized.wav` : ''}"></audio>
      </div>
    `;
    grid.appendChild(card);
  });
}

async function togglePromptRecording(promptId) {
  const recBtn = document.getElementById(`btn-rec-${promptId}`);
  const statusEl = document.getElementById(`p-status-${promptId}`);
  const audioPrev = document.getElementById(`audio-prev-${promptId}`);

  if (activeRecordingPromptId !== promptId) {
    // Discard any recording still running on a different prompt
    if (promptRecorder) {
      const prevId = activeRecordingPromptId;
      await promptRecorder.stop().catch(() => {});
      promptRecorder = null;
      activeRecordingPromptId = null;
      const prevBtn = document.getElementById(`btn-rec-${prevId}`);
      if (prevBtn) {
        prevBtn.textContent = '🎙️ Record Sample';
        prevBtn.style.background = '';
        prevBtn.style.color = '';
      }
    }

    // Start Recording
    try {
      promptRecorder = new PcmRecorder();
      await promptRecorder.start();
      activeRecordingPromptId = promptId;
      recBtn.textContent = '⏹️ Stop Recording';
      recBtn.style.background = 'var(--accent-rose)';
      recBtn.style.color = '#fff';
      statusEl.textContent = 'Recording now... Speak clearly';
    } catch (err) {
      if (promptRecorder) { await promptRecorder.stop().catch(() => {}); promptRecorder = null; }
      alert('Microphone error: ' + err.message);
    }
    return;
  }

  // Stop recording and upload
  const recorder = promptRecorder;
  promptRecorder = null;
  activeRecordingPromptId = null;
  recBtn.style.background = '';
  recBtn.style.color = '';
  recBtn.textContent = 'Processing...';

  const wavBlob = recorder ? await recorder.stop(24000) : null;
  if (!wavBlob) {
    recBtn.textContent = '🎙️ Record Sample';
    statusEl.textContent = 'Unrecorded';
    alert('No audio was captured from the microphone. Please try again.');
    return;
  }

  recBtn.textContent = 'Uploading...';
  const formData = new FormData();
  formData.append('prompt_id', promptId);
  formData.append('audio', wavBlob, `${promptId}.wav`);

  try {
    const res = await fetch('/api/voice/upload', {
      method: 'POST',
      body: formData
    });
    const result = await res.json();
    if (result.success) {
      statusEl.textContent = '✓ Recorded & Normalized';
      statusEl.style.color = 'var(--accent-emerald)';
      audioPrev.src = `/api/audio/${promptId}_normalized.wav?t=${Date.now()}`;
      audioPrev.style.display = 'block';
      recBtn.textContent = 'Re-Record';
      loadMasterAudio();
      loadVoiceReferences();
      refreshSystemStatus();
    } else {
      recBtn.textContent = '🎙️ Record Sample';
      statusEl.textContent = 'Upload failed';
      alert('Upload error: ' + (result.error || 'Server rejected audio format'));
    }
  } catch (e) {
    recBtn.textContent = '🎙️ Record Sample';
    statusEl.textContent = 'Upload failed';
    alert('Upload failed: ' + e);
  }
}

// ========================================================
// 6. MEMORIES & CHATS TAB
// ========================================================
function initMemories() {
  document.getElementById('btn-search-memories').addEventListener('click', searchMemories);
  document.getElementById('memory-search-input').addEventListener('keydown', (e) => {
    if (e.key === 'Enter') searchMemories();
  });
  document.getElementById('btn-import-chat').addEventListener('click', importChatLogs);
}

async function loadAllMemories() {
  try {
    const res = await fetch('/api/memories');
    if (!res.ok) return;
    const memories = await res.json();
    renderMemoriesList(memories);
    document.getElementById('total-memories-count-badge').textContent = `${memories.length} Cards`;
  } catch (err) {
    console.error('Failed to load memories:', err);
  }
}

async function searchMemories() {
  const query = document.getElementById('memory-search-input').value.trim();
  if (!query) {
    loadAllMemories();
    return;
  }
  try {
    const res = await fetch('/api/memories/search', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query: query })
    });
    const results = await res.json();
    renderMemoriesList(results);
  } catch (err) {
    console.error('Search error:', err);
  }
}

function renderMemoriesList(items) {
  const container = document.getElementById('memory-search-results');
  container.innerHTML = '';
  if (!items || items.length === 0) {
    container.innerHTML = '<div class="empty-recalled">No matching memories found.</div>';
    return;
  }

  items.forEach(item => {
    const div = document.createElement('div');
    div.className = 'recalled-card';
    const cat = item.category || 'General';
    const score = item.relevance_score ? `<span style="color:var(--accent-emerald);">Score: ${item.relevance_score}</span>` : '';
    const content = item.content || '';

    div.innerHTML = `
      <div class="recalled-card-header">
        <span class="recalled-cat">${escapeHtml(cat)}</span>
        <span style="font-size:0.75rem;">${score}</span>
      </div>
      <div style="font-size:0.88rem; line-height:1.5; white-space:pre-wrap;">${escapeHtml(content)}</div>
    `;
    container.appendChild(div);
  });
}

async function importChatLogs() {
  const text = document.getElementById('chat-import-text').value.trim();
  const author = document.getElementById('chat-author-filter').value.trim();
  if (!text) {
    alert('Please paste chat export text.');
    return;
  }

  const btn = document.getElementById('btn-import-chat');
  btn.textContent = 'Parsing & Ingesting...';

  try {
    const res = await fetch('/api/chat/import', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: text, author: author })
    });
    const data = await res.json();
    btn.textContent = 'Parse & Ingest Chat History';

    if (data.success) {
      const box = document.getElementById('chat-import-analytics');
      const content = document.getElementById('analytics-content');
      box.classList.remove('hidden');

      const prof = data.profile || {};
      const topWords = (prof.top_words || []).map(w => w[0]).slice(0, 10).join(', ');

      content.innerHTML = `
        <p><strong>Indexed Chunks:</strong> ${data.chunk_count} conversation blocks</p>
        <p><strong>Total Words:</strong> ${prof.total_words} | <strong>Avg Words/Msg:</strong> ${prof.avg_words_per_message}</p>
        <p><strong>Top Vocabulary:</strong> ${topWords}</p>
      `;

      loadAllMemories();
      refreshSystemStatus();
    }
  } catch (err) {
    btn.textContent = 'Parse & Ingest Chat History';
    alert('Failed to import chat: ' + err);
  }
}

// ========================================================
// 7. PERSONA MATRIX TAB
// ========================================================
function initPersona() {
  document.getElementById('btn-save-persona').addEventListener('click', savePersona);
}

async function loadPersona() {
  try {
    const res = await fetch('/api/persona');
    if (!res.ok) return;
    const p = await res.json();

    document.getElementById('persona-name').value = p.identity?.name || '';
    document.getElementById('persona-fullname').value = p.identity?.full_name || '';
    document.getElementById('persona-openness').value = p.core_traits?.openness || '';
    document.getElementById('persona-conscientiousness').value = p.core_traits?.conscientiousness || '';
    document.getElementById('persona-agreeableness').value = p.core_traits?.agreeableness || '';

    document.getElementById('persona-phrases').value = (p.lexicon?.favorite_phrases || []).join(', ');
    document.getElementById('persona-avoid').value = (p.lexicon?.avoid_words || []).join(', ');
    document.getElementById('persona-constraints').value = (p.communication_style?.spoken_constraints || []).join('\n');
    document.getElementById('persona-approved-replies').value = (p.communication_style?.approved_replies || []).join('\n');

    const twinName = p.identity?.name || 'Digital Twin';
    const twinEl = document.getElementById('chat-twin-name');
    if (twinEl) twinEl.textContent = twinName;
  } catch (err) {
    console.error('Failed to load persona:', err);
  }
}

async function savePersona() {
  const btn = document.getElementById('btn-save-persona');
  btn.textContent = 'Saving...';

  const payload = {
    name: document.getElementById('persona-name').value.trim(),
    full_name: document.getElementById('persona-fullname').value.trim(),
    openness: document.getElementById('persona-openness').value.trim(),
    conscientiousness: document.getElementById('persona-conscientiousness').value.trim(),
    agreeableness: document.getElementById('persona-agreeableness').value.trim(),
    favorite_phrases: document.getElementById('persona-phrases').value.split(',').map(s => s.trim()).filter(Boolean),
    avoid_words: document.getElementById('persona-avoid').value.split(',').map(s => s.trim()).filter(Boolean),
    spoken_constraints: document.getElementById('persona-constraints').value.split('\n').map(s => s.trim()).filter(Boolean),
    approved_replies: document.getElementById('persona-approved-replies').value.split('\n').map(s => s.trim()).filter(Boolean)
  };

  try {
    const res = await fetch('/api/persona', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const result = await res.json();
    if (!res.ok || !result.success) throw new Error(result.error || 'Could not save persona.');
    btn.textContent = 'Saved!';
    const twinName = payload.name || 'Digital Twin';
    const twinEl = document.getElementById('chat-twin-name');
    if (twinEl) twinEl.textContent = twinName;
    setTimeout(() => { btn.textContent = 'Save Changes'; }, 1500);
  } catch (err) {
    btn.textContent = 'Save Failed';
    console.error('Error saving persona:', err);
  }
}

// Utility: Escape HTML
function escapeHtml(str) {
  if (!str) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}

