import './MessageBubble.css'

function TypingDots() {
  return (
    <span className="typing-dots">
      <span />
      <span />
      <span />
    </span>
  )
}

function SourcesChips({ sources }) {
  if (!sources?.length) return null
  return (
    <div className="sources">
      <span className="sources-label">Sources:</span>
      {sources.map((s) => (
        <span key={s} className="source-chip" title={s}>
          <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
            <polyline points="14 2 14 8 20 8"/>
          </svg>
          {s}
        </span>
      ))}
    </div>
  )
}

export default function MessageBubble({ message }) {
  const isUser = message.role === 'user'

  return (
    <div className={`message-row ${isUser ? 'user' : 'bot'}`}>
      <div className="avatar">{isUser ? 'U' : 'AI'}</div>
      <div className={`bubble ${isUser ? 'user-bubble' : 'bot-bubble'} ${message.error ? 'error-bubble' : ''}`}>
        {message.loading ? (
          <TypingDots />
        ) : (
          <>
            <p className="bubble-text">{message.content}</p>
            {message.streaming && !message.content && <TypingDots />}
            {message.streaming && (
              <span className="streaming-cursor" aria-hidden="true" />
            )}
            {!message.streaming && <SourcesChips sources={message.sources} />}
          </>
        )}
      </div>
    </div>
  )
}
