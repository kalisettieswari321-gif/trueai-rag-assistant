/* ══════════════════════════════════════════════════════════════════
   TrueAI RAG Assistant – Frontend Application Logic
   ══════════════════════════════════════════════════════════════════ */

// ── Constants ────────────────────────────────────────────────────
const API_URL = '/api/chat';
const SESSION_KEY = 'trueai_session_id';
const MAX_TEXTAREA_HEIGHT = 140;

// ── State ────────────────────────────────────────────────────────
let sessionId = getOrCreateSession();
let isLoading = false;

// ── DOM refs ─────────────────────────────────────────────────────
const messageInput    = document.getElementById('messageInput');
const sendBtn         = document.getElementById('sendBtn');
const messagesList    = document.getElementById('messagesList');
const messagesContainer = document.getElementById('messagesContainer');
const welcomeScreen   = document.getElementById('welcomeScreen');
const newChatBtn      = document.getElementById('newChatBtn');
const sidebarToggle   = document.getElementById('sidebarToggle');
const sidebar         = document.getElementById('sidebar');
const sessionBadge    = document.getElementById('sessionBadge');
const charCount       = document.getElementById('charCount');

// ── Initialise ───────────────────────────────────────────────────
sessionBadge.textContent = `Session: ${sessionId.slice(0, 8)}…`;
sessionBadge.title = `Full session ID: ${sessionId}`;

// ── Session management ───────────────────────────────────────────
function getOrCreateSession() {
  let id = localStorage.getItem(SESSION_KEY);
  if (!id) {
    if (typeof crypto !== 'undefined' && crypto.randomUUID) {
      id = crypto.randomUUID();
    } else {
      // Robust standard RFC4122 v4 UUID generator fallback
      id = 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
        const r = Math.random() * 16 | 0;
        const v = c === 'x' ? r : (r & 0x3 | 0x8);
        return v.toString(16);
      });
    }
    localStorage.setItem(SESSION_KEY, id);
  }
  return id;
}

function startNewSession() {
  sessionId = crypto.randomUUID();
  localStorage.setItem(SESSION_KEY, sessionId);
  sessionBadge.textContent = `Session: ${sessionId.slice(0, 8)}…`;
  sessionBadge.title = `Full session ID: ${sessionId}`;

  // Clear UI
  messagesList.innerHTML = '';
  welcomeScreen.style.display = 'flex';

  // Also clear server-side history (best effort)
  fetch(`/api/session/${sessionId}`, { method: 'DELETE' }).catch(() => {});
}

// ── Textarea auto-resize ─────────────────────────────────────────
messageInput.addEventListener('input', () => {
  charCount.textContent = messageInput.value.length;
  messageInput.style.height = 'auto';
  messageInput.style.height = Math.min(messageInput.scrollHeight, MAX_TEXTAREA_HEIGHT) + 'px';
  setSendEnabled(messageInput.value.trim().length > 0 && !isLoading);
});

messageInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    handleSend();
  }
});

// ── Send button ──────────────────────────────────────────────────
sendBtn.addEventListener('click', handleSend);

function setSendEnabled(enabled) {
  sendBtn.disabled = !enabled;
}

setSendEnabled(false);

// ── Suggestion chips ─────────────────────────────────────────────
document.querySelectorAll('.chip').forEach(chip => {
  chip.addEventListener('click', () => {
    const q = chip.dataset.q;
    if (q) {
      messageInput.value = q;
      charCount.textContent = q.length;
      setSendEnabled(true);
      handleSend();
    }
  });
});

// ── New Chat button ──────────────────────────────────────────────
newChatBtn.addEventListener('click', () => {
  startNewSession();
});

// ── Sidebar toggle ───────────────────────────────────────────────
sidebarToggle.addEventListener('click', () => {
  sidebar.classList.toggle('collapsed');
});

// ── Main send handler ────────────────────────────────────────────
async function handleSend() {
  const text = messageInput.value.trim();
  if (!text || isLoading) return;

  // Hide welcome screen
  welcomeScreen.style.display = 'none';

  // Append user message
  appendMessage('user', text);

  // Clear input
  messageInput.value = '';
  messageInput.style.height = 'auto';
  charCount.textContent = '0';
  setSendEnabled(false);

  // Show typing indicator
  const typingEl = showTypingIndicator();
  isLoading = true;

  try {
    const data = await sendChatRequest(text);
    typingEl.remove();
    appendMessage('bot', data.reply, {
      tokensUsed: data.tokensUsed,
      retrievedChunks: data.retrievedChunks,
    });
  } catch (err) {
    typingEl.remove();
    appendErrorMessage(err.message || 'Something went wrong. Please try again.');
  } finally {
    isLoading = false;
    setSendEnabled(messageInput.value.trim().length > 0);
    scrollToBottom();
  }
}

// ── API call ─────────────────────────────────────────────────────
async function sendChatRequest(message) {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 35000);

  try {
    const res = await fetch(API_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ sessionId, message }),
      signal: controller.signal,
    });

    clearTimeout(timeoutId);

    const json = await res.json();

    if (!res.ok) {
      throw new Error(json.detail || json.error || `Error ${res.status}`);
    }

    return json;
  } catch (err) {
    clearTimeout(timeoutId);
    if (err.name === 'AbortError') {
      throw new Error('Request timed out. The server took too long to respond.');
    }
    throw err;
  }
}

// ── Message rendering ─────────────────────────────────────────────
function appendMessage(role, text, meta = {}) {
  const row = document.createElement('div');
  row.className = `message-row ${role}`;

  const avatar = document.createElement('div');
  avatar.className = `avatar ${role === 'user' ? 'user-avatar' : 'bot-avatar'}`;
  avatar.textContent = role === 'user' ? '👤' : '⚡';

  const content = document.createElement('div');
  content.className = 'message-content';

  const bubble = document.createElement('div');
  bubble.className = 'bubble';

  if (role === 'bot') {
    // Render markdown for bot responses
    bubble.innerHTML = marked.parse(text);
  } else {
    bubble.textContent = text;
  }

  const metaDiv = document.createElement('div');
  metaDiv.className = 'message-meta';

  const timeEl = document.createElement('span');
  timeEl.className = 'meta-time';
  timeEl.textContent = formatTime(new Date());
  metaDiv.appendChild(timeEl);

  if (role === 'bot' && meta.retrievedChunks !== undefined) {
    if (meta.retrievedChunks > 0) {
      const chunksEl = document.createElement('span');
      chunksEl.className = 'meta-chunks';
      chunksEl.textContent = `📄 ${meta.retrievedChunks} chunk${meta.retrievedChunks > 1 ? 's' : ''}`;
      metaDiv.appendChild(chunksEl);
    }

    if (meta.tokensUsed > 0) {
      const tokensEl = document.createElement('span');
      tokensEl.className = 'meta-tokens';
      tokensEl.textContent = `${meta.tokensUsed} tokens`;
      metaDiv.appendChild(tokensEl);
    }
  }

  content.appendChild(bubble);
  content.appendChild(metaDiv);
  row.appendChild(avatar);
  row.appendChild(content);

  messagesList.appendChild(row);
  scrollToBottom();
}

function appendErrorMessage(text) {
  const row = document.createElement('div');
  row.className = 'message-row bot';

  const avatar = document.createElement('div');
  avatar.className = 'avatar bot-avatar';
  avatar.textContent = '⚡';

  const content = document.createElement('div');
  content.className = 'message-content';

  const bubble = document.createElement('div');
  bubble.className = 'bubble error-bubble';
  bubble.textContent = `⚠️ ${text}`;

  const metaDiv = document.createElement('div');
  metaDiv.className = 'message-meta';
  const timeEl = document.createElement('span');
  timeEl.className = 'meta-time';
  timeEl.textContent = formatTime(new Date());
  metaDiv.appendChild(timeEl);

  content.appendChild(bubble);
  content.appendChild(metaDiv);
  row.appendChild(avatar);
  row.appendChild(content);
  messagesList.appendChild(row);
  scrollToBottom();
}

// ── Typing indicator ─────────────────────────────────────────────
function showTypingIndicator() {
  const row = document.createElement('div');
  row.className = 'typing-indicator';

  const avatar = document.createElement('div');
  avatar.className = 'avatar bot-avatar';
  avatar.textContent = '⚡';

  const bubble = document.createElement('div');
  bubble.className = 'typing-bubble';
  bubble.innerHTML = '<span></span><span></span><span></span>';

  row.appendChild(avatar);
  row.appendChild(bubble);
  messagesList.appendChild(row);
  scrollToBottom();
  return row;
}

// ── Scroll to bottom ─────────────────────────────────────────────
function scrollToBottom() {
  requestAnimationFrame(() => {
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
  });
}

// ── Time formatter ───────────────────────────────────────────────
function formatTime(date) {
  return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}
