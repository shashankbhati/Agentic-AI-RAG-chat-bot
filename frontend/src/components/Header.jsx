import { Link } from 'react-router-dom'
import { auth } from '../services/authApi.js'
import './Header.css'

export default function Header({ health, sidebarOpen, onToggleSidebar, onClearChat }) {
  const statusColor = health?.status === 'ok' ? '#4caf87' : health?.status === 'degraded' ? '#f0a500' : '#ff5c5c'
  const statusLabel = health?.status ?? 'connecting...'

  return (
    <header className="header">
      <div className="header-left">
        <button className="icon-btn" onClick={onToggleSidebar} title="Toggle document panel">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <rect x="3" y="3" width="18" height="18" rx="2"/>
            <line x1="9" y1="3" x2="9" y2="21"/>
          </svg>
        </button>
        <div className="logo">
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="var(--accent)" strokeWidth="2">
            <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
          </svg>
          <span className="logo-text">RAG Chatbot</span>
        </div>
      </div>

      <div className="header-right">
        <div className="status-badge" title={`LLM: ${health?.llm_provider ?? '—'} | Model: ${health?.embed_model ?? '—'}`}>
          <span className="status-dot" style={{ background: statusColor }} />
          <span className="status-text">{statusLabel}</span>
        </div>
        {health?.llm_provider && (
          <span className="model-chip">{health.llm_provider}</span>
        )}
        {auth.isLoggedIn()
          ? <Link to="/dashboard" className="icon-btn dash-link" title="Dashboard">⚙️</Link>
          : <Link to="/login" className="icon-btn dash-link" title="Sign in">Sign in</Link>
        }
        <button className="icon-btn" onClick={onClearChat} title="New conversation">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M12 5v14M5 12h14"/>
          </svg>
        </button>
      </div>
    </header>
  )
}
