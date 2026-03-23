;(function () {
  'use strict'

  // ── Read config from <script> tag ──────────────────────────────────────────
  const script =
    document.currentScript ||
    document.querySelector('script[data-api-key]')

  const API_URL    = (script?.getAttribute('data-api-url')  || 'http://localhost:8000').replace(/\/$/, '')
  const API_KEY    = script?.getAttribute('data-api-key')   || ''
  const TITLE      = script?.getAttribute('data-title')     || 'AI Assistant'
  const THEME      = script?.getAttribute('data-theme')     || 'dark'   // 'dark' | 'light'
  const POSITION   = script?.getAttribute('data-position')  || 'right'  // 'right' | 'left'

  if (!API_KEY) {
    console.warn('[RAG Widget] No data-api-key provided — widget disabled.')
    return
  }

  // ── Prevent double-init ────────────────────────────────────────────────────
  if (document.getElementById('__rag_widget__')) return

  // ── Styles ─────────────────────────────────────────────────────────────────
  const COLORS = THEME === 'light'
    ? { bg: '#ffffff', surface: '#f5f5f7', border: '#e0e0e0', text: '#1a1a2e', muted: '#777', accent: '#6c63ff', userBubble: '#6c63ff', userText: '#fff', botBubble: '#f0f0f5', botText: '#1a1a2e', input: '#ffffff', inputBorder: '#e0e0e0' }
    : { bg: '#1a1d27', surface: '#21242f', border: '#2e3148', text: '#e8eaf0', muted: '#9499b0', accent: '#6c63ff', userBubble: '#6c63ff', userText: '#fff', botBubble: '#21242f', botText: '#e8eaf0', input: '#21242f', inputBorder: '#2e3148' }

  const side = POSITION === 'left' ? 'left: 24px;' : 'right: 24px;'

  const CSS = `
    #__rag_widget__ * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }
    #__rag_widget__ { position: fixed; bottom: 24px; ${side} z-index: 2147483647; }

    .rw-fab {
      width: 56px; height: 56px; border-radius: 50%;
      background: ${COLORS.accent}; border: none; cursor: pointer;
      display: flex; align-items: center; justify-content: center;
      box-shadow: 0 4px 20px rgba(108,99,255,0.45);
      transition: transform 0.2s, box-shadow 0.2s;
      color: #fff; font-size: 22px;
    }
    .rw-fab:hover { transform: scale(1.08); box-shadow: 0 6px 28px rgba(108,99,255,0.55); }

    .rw-panel {
      position: absolute; bottom: 70px; ${POSITION === 'left' ? 'left: 0;' : 'right: 0;'}
      width: 360px; height: 520px;
      background: ${COLORS.bg}; border: 1px solid ${COLORS.border};
      border-radius: 16px; box-shadow: 0 8px 40px rgba(0,0,0,0.35);
      display: flex; flex-direction: column; overflow: hidden;
      animation: rw-slide-in 0.2s ease;
    }
    @keyframes rw-slide-in { from { opacity:0; transform: translateY(12px); } to { opacity:1; transform:none; } }

    .rw-header {
      padding: 14px 16px; background: ${COLORS.accent};
      display: flex; align-items: center; justify-content: space-between;
      flex-shrink: 0;
    }
    .rw-header-title { color: #fff; font-weight: 600; font-size: 15px; display: flex; align-items: center; gap: 8px; }
    .rw-close { background: transparent; border: none; color: rgba(255,255,255,0.8); font-size: 20px; cursor: pointer; line-height:1; padding: 2px 4px; border-radius: 4px; }
    .rw-close:hover { background: rgba(255,255,255,0.15); color: #fff; }

    .rw-messages {
      flex: 1; overflow-y: auto; padding: 16px; display: flex; flex-direction: column; gap: 10px;
      scrollbar-width: thin; scrollbar-color: ${COLORS.border} transparent;
    }
    .rw-messages::-webkit-scrollbar { width: 4px; }
    .rw-messages::-webkit-scrollbar-thumb { background: ${COLORS.border}; border-radius: 2px; }

    .rw-msg { max-width: 82%; padding: 10px 13px; border-radius: 12px; font-size: 13px; line-height: 1.55; word-break: break-word; }
    .rw-msg.user { align-self: flex-end; background: ${COLORS.userBubble}; color: ${COLORS.userText}; border-bottom-right-radius: 3px; }
    .rw-msg.bot  { align-self: flex-start; background: ${COLORS.botBubble}; color: ${COLORS.botText}; border: 1px solid ${COLORS.border}; border-bottom-left-radius: 3px; }
    .rw-msg.error { background: rgba(255,92,92,0.12); border-color: rgba(255,92,92,0.3); color: #ff5c5c; }

    .rw-sources { margin-top: 7px; padding-top: 7px; border-top: 1px solid ${COLORS.border}; display: flex; flex-wrap: wrap; gap: 5px; }
    .rw-source-chip { font-size: 10px; padding: 2px 7px; background: ${COLORS.surface}; border: 1px solid ${COLORS.border}; border-radius: 20px; color: ${COLORS.muted}; max-width: 160px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }

    .rw-typing { display: inline-flex; gap: 4px; align-items: center; padding: 2px 0; }
    .rw-typing span { width: 6px; height: 6px; border-radius: 50%; background: ${COLORS.muted}; animation: rw-bounce 1.2s ease-in-out infinite; }
    .rw-typing span:nth-child(2) { animation-delay: 0.2s; }
    .rw-typing span:nth-child(3) { animation-delay: 0.4s; }
    @keyframes rw-bounce { 0%,60%,100%{transform:translateY(0);opacity:.4} 30%{transform:translateY(-4px);opacity:1} }

    .rw-cursor { display: inline-block; width: 2px; height: 1em; background: ${COLORS.muted}; margin-left: 1px; vertical-align: text-bottom; animation: rw-blink 0.8s step-end infinite; }
    @keyframes rw-blink { 0%,100%{opacity:1} 50%{opacity:0} }

    .rw-welcome { display:flex; flex-direction:column; align-items:center; justify-content:center; text-align:center; flex:1; color:${COLORS.muted}; gap:10px; padding: 24px; }
    .rw-welcome-icon { font-size: 36px; }
    .rw-welcome h3 { font-size:15px; font-weight:600; color:${COLORS.text}; }
    .rw-welcome p { font-size:12px; line-height:1.5; }

    .rw-input-area {
      padding: 10px 12px; border-top: 1px solid ${COLORS.border};
      display: flex; gap: 8px; align-items: flex-end; flex-shrink: 0;
      background: ${COLORS.bg};
    }
    .rw-input {
      flex: 1; background: ${COLORS.input}; border: 1px solid ${COLORS.inputBorder};
      border-radius: 10px; padding: 9px 12px; color: ${COLORS.text};
      font-size: 13px; resize: none; min-height: 38px; max-height: 100px;
      outline: none; line-height: 1.45; overflow-y: auto;
      transition: border-color 0.2s;
    }
    .rw-input:focus { border-color: ${COLORS.accent}; }
    .rw-input::placeholder { color: ${COLORS.muted}; }

    .rw-send {
      width: 36px; height: 36px; border-radius: 10px; flex-shrink: 0;
      background: ${COLORS.accent}; border: none; cursor: pointer;
      color: #fff; display: flex; align-items: center; justify-content: center;
      transition: opacity 0.15s;
    }
    .rw-send:disabled { opacity: 0.35; cursor: not-allowed; }
    .rw-send:not(:disabled):hover { opacity: 0.85; }

    .rw-send-spinner { width:14px; height:14px; border:2px solid rgba(255,255,255,0.3); border-top-color:#fff; border-radius:50%; animation: rw-spin 0.7s linear infinite; }
    @keyframes rw-spin { to { transform: rotate(360deg); } }

    .rw-branding { text-align:center; font-size:10px; color:${COLORS.muted}; padding: 5px 0 8px; flex-shrink:0; }
    .rw-branding a { color:${COLORS.muted}; text-decoration:none; }
    .rw-branding a:hover { text-decoration:underline; }
  `

  // ── HTML ───────────────────────────────────────────────────────────────────
  const HTML = `
    <button class="rw-fab" id="rw-fab" aria-label="Open chat">
      <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
      </svg>
    </button>

    <div class="rw-panel" id="rw-panel" style="display:none" role="dialog" aria-label="${TITLE}">
      <div class="rw-header">
        <div class="rw-header-title">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
          </svg>
          ${TITLE}
        </div>
        <button class="rw-close" id="rw-close" aria-label="Close">&#x2715;</button>
      </div>

      <div class="rw-messages" id="rw-messages">
        <div class="rw-welcome">
          <div class="rw-welcome-icon">💬</div>
          <h3>${TITLE}</h3>
          <p>Ask me anything about the available documents.</p>
        </div>
      </div>

      <div class="rw-input-area">
        <textarea class="rw-input" id="rw-input" placeholder="Ask a question…" rows="1"></textarea>
        <button class="rw-send" id="rw-send" aria-label="Send">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
            <line x1="22" y1="2" x2="11" y2="13"/>
            <polygon points="22 2 15 22 11 13 2 9 22 2"/>
          </svg>
        </button>
      </div>
      <div class="rw-branding">Powered by <a href="#" target="_blank">RAG Chatbot</a></div>
    </div>
  `

  // ── Mount into Shadow DOM ──────────────────────────────────────────────────
  const host = document.createElement('div')
  host.id = '__rag_widget__'
  document.body.appendChild(host)

  const shadow = host.attachShadow({ mode: 'closed' })
  const styleEl = document.createElement('style')
  styleEl.textContent = CSS
  shadow.appendChild(styleEl)

  const wrapper = document.createElement('div')
  wrapper.innerHTML = HTML
  shadow.appendChild(wrapper)

  // ── Refs ───────────────────────────────────────────────────────────────────
  const fab      = shadow.getElementById('rw-fab')
  const panel    = shadow.getElementById('rw-panel')
  const closeBtn = shadow.getElementById('rw-close')
  const messages = shadow.getElementById('rw-messages')
  const input    = shadow.getElementById('rw-input')
  const sendBtn  = shadow.getElementById('rw-send')

  let isOpen    = false
  let sessionId = null
  let sending   = false
  let welcomed  = false  // whether the welcome placeholder has been removed

  // ── Toggle panel ──────────────────────────────────────────────────────────
  function togglePanel() {
    isOpen = !isOpen
    panel.style.display = isOpen ? 'flex' : 'none'
    panel.style.flexDirection = isOpen ? 'column' : ''
    if (isOpen) { input.focus() }
  }

  fab.addEventListener('click', togglePanel)
  closeBtn.addEventListener('click', togglePanel)

  // Close on Escape
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && isOpen) togglePanel()
  })

  // ── Message helpers ────────────────────────────────────────────────────────
  function clearWelcome() {
    if (!welcomed) {
      messages.innerHTML = ''
      welcomed = true
    }
  }

  function addMessage(role, text = '', opts = {}) {
    clearWelcome()
    const div = document.createElement('div')
    div.className = `rw-msg ${role}${opts.error ? ' error' : ''}`

    if (opts.typing) {
      div.innerHTML = `<span class="rw-typing"><span></span><span></span><span></span></span>`
    } else {
      div.textContent = text
    }

    messages.appendChild(div)
    messages.scrollTop = messages.scrollHeight
    return div
  }

  function updateMessage(el, text, sources) {
    el.textContent = text
    if (sources?.length) {
      const sourcesEl = document.createElement('div')
      sourcesEl.className = 'rw-sources'
      sources.forEach((s) => {
        const chip = document.createElement('span')
        chip.className = 'rw-source-chip'
        chip.title = s
        chip.textContent = '📄 ' + s
        sourcesEl.appendChild(chip)
      })
      el.appendChild(sourcesEl)
    }
    messages.scrollTop = messages.scrollHeight
  }

  function setSending(val) {
    sending = val
    sendBtn.disabled = val
    if (val) {
      sendBtn.innerHTML = '<span class="rw-send-spinner"></span>'
    } else {
      sendBtn.innerHTML = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/></svg>`
    }
  }

  // ── Send message (streaming) ───────────────────────────────────────────────
  async function send() {
    const query = input.value.trim()
    if (!query || sending) return

    input.value = ''
    input.style.height = 'auto'

    addMessage('user', query)
    const botEl = addMessage('bot', '', { typing: true })
    setSending(true)

    try {
      const res = await fetch(`${API_URL}/api/v1/chat/stream`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-API-Key': API_KEY,
        },
        body: JSON.stringify({ query, session_id: sessionId }),
      })

      if (!res.ok) {
        const err = await res.json().catch(() => ({}))
        updateMessage(botEl, err.detail || `Error ${res.status}`, null)
        botEl.classList.add('error')
        return
      }

      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''
      let content = ''
      let sources = []

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })

        const lines = buffer.split('\n')
        buffer = lines.pop()

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue
          try {
            const ev = JSON.parse(line.slice(6))
            if (ev.type === 'meta') {
              if (ev.session_id) sessionId = ev.session_id
              if (ev.sources) sources = ev.sources
            } else if (ev.type === 'chunk') {
              content += ev.content
              // Show streaming content with cursor
              botEl.innerHTML = escapeHtml(content) + '<span class="rw-cursor"></span>'
              messages.scrollTop = messages.scrollHeight
            } else if (ev.type === 'done') {
              updateMessage(botEl, content, sources)
            } else if (ev.type === 'error') {
              updateMessage(botEl, ev.detail || 'An error occurred', null)
              botEl.classList.add('error')
            }
          } catch (_) {}
        }
      }
    } catch (err) {
      updateMessage(botEl, 'Connection failed. Is the server running?', null)
      botEl.classList.add('error')
    } finally {
      setSending(false)
      input.focus()
    }
  }

  function escapeHtml(str) {
    return str
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/\n/g, '<br>')
  }

  // ── Auto-resize textarea ───────────────────────────────────────────────────
  input.addEventListener('input', () => {
    input.style.height = 'auto'
    input.style.height = Math.min(input.scrollHeight, 100) + 'px'
  })

  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      send()
    }
  })

  sendBtn.addEventListener('click', send)

})()
