import { useState, useRef, useEffect } from 'react'
import MessageBubble from './MessageBubble.jsx'
import './ChatWindow.css'

const SUGGESTED = [
  'What is this document about?',
  'Summarize the key points',
  'What are the main topics covered?',
]

export default function ChatWindow({ messages, onSend, sessionId, hasDocuments }) {
  const [input, setInput] = useState('')
  const [useStream, setUseStream] = useState(true)
  const [sending, setSending] = useState(false)
  const bottomRef = useRef(null)
  const textareaRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  async function submit(query) {
    const q = query ?? input.trim()
    if (!q || sending) return
    setInput('')
    setSending(true)
    await onSend(q, useStream)
    setSending(false)
  }

  function onKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      submit()
    }
  }

  const isEmpty = messages.length === 0

  return (
    <div className="chat-window">
      <div className="messages-area">
        {isEmpty ? (
          <div className="chat-welcome">
            <div className="welcome-icon">
              <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="var(--accent)" strokeWidth="1.5">
                <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
              </svg>
            </div>
            <h2>Chat with your documents</h2>
            <p>
              {hasDocuments
                ? 'Ask a question about your indexed documents.'
                : 'Upload a PDF in the sidebar to get started.'}
            </p>
            {hasDocuments && (
              <div className="suggestions">
                {SUGGESTED.map((s) => (
                  <button key={s} className="suggestion-chip" onClick={() => submit(s)}>
                    {s}
                  </button>
                ))}
              </div>
            )}
          </div>
        ) : (
          messages.map((msg) => <MessageBubble key={msg.id} message={msg} />)
        )}
        <div ref={bottomRef} />
      </div>

      <div className="input-area">
        <div className="input-row">
          <textarea
            ref={textareaRef}
            className="chat-input"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={onKeyDown}
            placeholder="Ask a question about your documents…"
            rows={1}
            disabled={sending}
          />
          <button
            className="send-btn"
            onClick={() => submit()}
            disabled={!input.trim() || sending}
            title="Send (Enter)"
          >
            {sending ? (
              <div className="send-spinner" />
            ) : (
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <line x1="22" y1="2" x2="11" y2="13"/>
                <polygon points="22 2 15 22 11 13 2 9 22 2"/>
              </svg>
            )}
          </button>
        </div>
        <div className="input-footer">
          <label className="stream-toggle" title="Stream tokens as they are generated">
            <input
              type="checkbox"
              checked={useStream}
              onChange={(e) => setUseStream(e.target.checked)}
            />
            <span>Streaming</span>
          </label>
          {sessionId && (
            <span className="session-id" title={`Session: ${sessionId}`}>
              Session active
            </span>
          )}
          <span className="input-hint">Enter to send · Shift+Enter for newline</span>
        </div>
      </div>
    </div>
  )
}
