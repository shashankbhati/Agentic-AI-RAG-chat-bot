import { useState, useEffect } from 'react'
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import ChatWindow from './components/ChatWindow.jsx'
import DocumentPanel from './components/DocumentPanel.jsx'
import Header from './components/Header.jsx'
import Login from './pages/Login.jsx'
import Register from './pages/Register.jsx'
import Dashboard from './pages/Dashboard.jsx'
import { api } from './services/api.js'
import { auth } from './services/authApi.js'
import './App.css'

function ChatApp() {
  const [messages, setMessages] = useState([])
  const [sessionId, setSessionId] = useState(null)
  const [documents, setDocuments] = useState([])
  const [health, setHealth] = useState(null)
  const [sidebarOpen, setSidebarOpen] = useState(true)

  useEffect(() => {
    fetchHealth()
    fetchDocuments()
  }, [])

  async function fetchHealth() {
    try { setHealth(await api.health()) }
    catch { setHealth({ status: 'error' }) }
  }

  async function fetchDocuments() {
    try {
      const apiKey = auth.apiKey()
      const headers = apiKey ? { 'X-API-Key': apiKey } : {}
      const res = await fetch('/api/v1/documents', { headers })
      const data = await res.json()
      setDocuments(data.documents || [])
    } catch { setDocuments([]) }
  }

  async function handleSend(query, useStream) {
    const apiKey = auth.apiKey()
    const userMsg = { role: 'user', content: query, id: Date.now() }
    setMessages((prev) => [...prev, userMsg])

    if (useStream) {
      const botMsg = { role: 'assistant', content: '', sources: [], streaming: true, id: Date.now() + 1 }
      setMessages((prev) => [...prev, botMsg])
      const botId = botMsg.id

      const BASE = import.meta.env.VITE_API_URL || '/api/v1'
      const res = await fetch(`${BASE}/chat/stream`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(apiKey ? { 'X-API-Key': apiKey } : {}),
        },
        body: JSON.stringify({ query, session_id: sessionId, stream: true }),
      })

      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop()
        for (const line of lines) {
          if (!line.startsWith('data: ')) continue
          try {
            const event = JSON.parse(line.slice(6))
            if (event.type === 'meta') {
              if (!sessionId) setSessionId(event.session_id)
              setMessages((prev) => prev.map((m) => m.id === botId ? { ...m, sources: event.sources } : m))
            } else if (event.type === 'chunk') {
              setMessages((prev) => prev.map((m) => m.id === botId ? { ...m, content: m.content + event.content } : m))
            } else if (event.type === 'done') {
              setMessages((prev) => prev.map((m) => m.id === botId ? { ...m, streaming: false } : m))
            } else if (event.type === 'error') {
              setMessages((prev) => prev.map((m) => m.id === botId ? { ...m, content: `Error: ${event.detail}`, streaming: false, error: true } : m))
            }
          } catch (_) {}
        }
      }
    } else {
      const loadingMsg = { role: 'assistant', content: '', loading: true, id: Date.now() + 1 }
      setMessages((prev) => [...prev, loadingMsg])
      const loadId = loadingMsg.id
      try {
        const res = await api.chat(query, sessionId)
        if (!sessionId) setSessionId(res.session_id)
        setMessages((prev) => prev.map((m) => m.id === loadId
          ? { ...m, content: res.answer, sources: res.sources, loading: false } : m))
      } catch (err) {
        setMessages((prev) => prev.map((m) => m.id === loadId
          ? { ...m, content: `Error: ${err.message}`, loading: false, error: true } : m))
      }
    }
  }

  async function handleClearChat() {
    if (sessionId) { try { await api.clearSession(sessionId) } catch {} }
    setMessages([])
    setSessionId(null)
  }

  return (
    <div className="app-layout">
      <Header health={health} sidebarOpen={sidebarOpen}
        onToggleSidebar={() => setSidebarOpen((o) => !o)} onClearChat={handleClearChat} />
      <div className="app-body">
        {sidebarOpen && <DocumentPanel documents={documents} onRefresh={fetchDocuments} />}
        <ChatWindow messages={messages} onSend={handleSend} sessionId={sessionId} hasDocuments={documents.length > 0} />
      </div>
    </div>
  )
}

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<ChatApp />} />
        <Route path="/login" element={<Login />} />
        <Route path="/register" element={<Register />} />
        <Route path="/dashboard" element={
          auth.isLoggedIn() ? <Dashboard /> : <Navigate to="/login" replace />
        } />
      </Routes>
    </BrowserRouter>
  )
}
