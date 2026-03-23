import { useState, useEffect, useRef } from 'react'
import { useNavigate, Link } from 'react-router-dom'
import { authApi, auth } from '../services/authApi.js'
import './Dashboard.css'

function StatCard({ label, value, icon }) {
  return (
    <div className="stat-card">
      <span className="stat-icon">{icon}</span>
      <div>
        <div className="stat-value">{value}</div>
        <div className="stat-label">{label}</div>
      </div>
    </div>
  )
}

function CopyButton({ value }) {
  const [copied, setCopied] = useState(false)
  function copy() {
    navigator.clipboard.writeText(value)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }
  return (
    <button className="copy-btn" onClick={copy} title="Copy to clipboard">
      {copied ? '✓ Copied' : 'Copy'}
    </button>
  )
}

export default function Dashboard() {
  const navigate = useNavigate()
  const [stats, setStats] = useState(null)
  const [documents, setDocuments] = useState([])
  const [uploading, setUploading] = useState(false)
  const [uploadError, setUploadError] = useState(null)
  const [dragOver, setDragOver] = useState(false)
  const [activeTab, setActiveTab] = useState('documents')
  const fileRef = useRef(null)

  const apiKey = auth.apiKey()
  const email = auth.email()

  useEffect(() => {
    if (!auth.isLoggedIn()) { navigate('/login'); return }
    fetchAll()
  }, [])

  async function fetchAll() {
    try {
      const [s, d] = await Promise.all([authApi.getStats(), authApi.getDashboardDocuments()])
      setStats(s)
      setDocuments(d.documents)
    } catch (err) {
      if (err.message.includes('401')) { auth.clear(); navigate('/login') }
    }
  }

  async function handleUpload(file) {
    if (!file?.name.endsWith('.pdf')) { setUploadError('Only PDF files supported'); return }
    setUploadError(null)
    setUploading(true)
    try {
      await authApi.uploadDocument(file)
      await fetchAll()
    } catch (err) {
      setUploadError(err.message)
    } finally {
      setUploading(false)
    }
  }

  async function handleDelete(filename) {
    if (!confirm(`Delete "${filename}" and all its indexed chunks?`)) return
    try {
      await authApi.deleteDocument(filename)
      await fetchAll()
    } catch (err) {
      alert(`Delete failed: ${err.message}`)
    }
  }

  function onDrop(e) {
    e.preventDefault(); setDragOver(false)
    const file = e.dataTransfer.files[0]
    if (file) handleUpload(file)
  }

  const embedCode = `<script
  src="https://your-domain.com/rag-widget.js"
  data-api-url="https://your-domain.com"
  data-api-key="${apiKey || 'YOUR_API_KEY'}"
  data-title="AI Assistant">
</script>`

  return (
    <div className="dashboard">
      {/* Sidebar */}
      <aside className="dash-sidebar">
        <div className="dash-logo">
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="var(--accent)" strokeWidth="1.5">
            <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
          </svg>
          RAG Chatbot
        </div>

        <nav className="dash-nav">
          {[
            { id: 'documents', label: 'Documents', icon: '📄' },
            { id: 'stats', label: 'Usage Stats', icon: '📊' },
            { id: 'embed', label: 'Embed Code', icon: '🔌' },
            { id: 'key', label: 'API Key', icon: '🔑' },
          ].map((item) => (
            <button
              key={item.id}
              className={`dash-nav-item ${activeTab === item.id ? 'active' : ''}`}
              onClick={() => setActiveTab(item.id)}
            >
              <span>{item.icon}</span> {item.label}
            </button>
          ))}
        </nav>

        <div className="dash-footer">
          <span className="dash-email">{email}</span>
          <button className="logout-btn" onClick={() => { auth.clear(); navigate('/login') }}>
            Sign out
          </button>
          <Link to="/" className="back-to-chat">← Back to chat</Link>
        </div>
      </aside>

      {/* Main content */}
      <main className="dash-main">

        {/* ── Documents tab ── */}
        {activeTab === 'documents' && (
          <div className="dash-section">
            <div className="section-header">
              <h2>Documents</h2>
              <span className="badge">{documents.length} indexed</span>
            </div>

            {/* Upload zone */}
            <div
              className={`upload-zone ${dragOver ? 'drag-over' : ''} ${uploading ? 'uploading' : ''}`}
              onClick={() => !uploading && fileRef.current?.click()}
              onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
              onDragLeave={() => setDragOver(false)}
              onDrop={onDrop}
            >
              <input ref={fileRef} type="file" accept=".pdf" style={{ display: 'none' }}
                onChange={(e) => e.target.files[0] && handleUpload(e.target.files[0])} />
              {uploading
                ? <><div className="spinner" /> <span>Processing document…</span></>
                : <>
                    <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                      <polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/>
                    </svg>
                    <span>Drop PDF here or click to upload</span>
                    <span className="upload-hint">PDF files only · Max {50}MB</span>
                  </>
              }
            </div>
            {uploadError && <div className="error-banner">{uploadError}</div>}

            {/* Document table */}
            {documents.length === 0
              ? <div className="empty-state">No documents yet. Upload your first PDF above.</div>
              : (
                <table className="doc-table">
                  <thead>
                    <tr><th>Filename</th><th>Chunks</th><th>Indexed at</th><th></th></tr>
                  </thead>
                  <tbody>
                    {documents.map((doc) => (
                      <tr key={doc.filename}>
                        <td>
                          <span className="doc-icon">📄</span>
                          <span className="doc-filename">{doc.filename}</span>
                        </td>
                        <td><span className="badge">{doc.chunk_count} chunks</span></td>
                        <td className="muted">{doc.ingested_at ? new Date(doc.ingested_at).toLocaleDateString() : '—'}</td>
                        <td>
                          <button className="delete-btn" onClick={() => handleDelete(doc.filename)}>
                            Delete
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )
            }
          </div>
        )}

        {/* ── Stats tab ── */}
        {activeTab === 'stats' && stats && (
          <div className="dash-section">
            <div className="section-header"><h2>Usage Statistics</h2></div>
            <div className="stats-grid">
              <StatCard label="Chat queries" value={stats.totals.chat} icon="💬" />
              <StatCard label="Searches" value={stats.totals.search} icon="🔍" />
              <StatCard label="Uploads" value={stats.totals.upload} icon="📤" />
              <StatCard label="Deletes" value={stats.totals.delete} icon="🗑️" />
            </div>

            <h3 className="sub-heading">Recent Activity</h3>
            {stats.recent_activity.length === 0
              ? <div className="empty-state">No activity yet.</div>
              : (
                <table className="doc-table">
                  <thead><tr><th>Event</th><th>Document</th><th>Time</th></tr></thead>
                  <tbody>
                    {stats.recent_activity.map((log, i) => (
                      <tr key={i}>
                        <td><span className={`event-badge ${log.event_type}`}>{log.event_type}</span></td>
                        <td className="muted">{log.document_name || '—'}</td>
                        <td className="muted">{log.timestamp ? new Date(log.timestamp).toLocaleString() : '—'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )
            }
          </div>
        )}

        {/* ── Embed tab ── */}
        {activeTab === 'embed' && (
          <div className="dash-section">
            <div className="section-header"><h2>Embed on Your Website</h2></div>
            <p className="section-desc">
              Add this snippet to any HTML page to embed an AI chat widget powered by your documents.
            </p>
            <div className="code-block">
              <pre>{embedCode}</pre>
              <CopyButton value={embedCode} />
            </div>
            <div className="info-box">
              <strong>How it works:</strong>
              <ol>
                <li>Replace <code>your-domain.com</code> with your server URL</li>
                <li>Paste the snippet before <code>&lt;/body&gt;</code> on any page</li>
                <li>A chat bubble will appear in the bottom-right corner</li>
                <li>Your visitors can ask questions about the documents you indexed above</li>
              </ol>
            </div>
          </div>
        )}

        {/* ── API Key tab ── */}
        {activeTab === 'key' && (
          <div className="dash-section">
            <div className="section-header"><h2>Your API Key</h2></div>
            <p className="section-desc">Use this key in the <code>X-API-Key</code> header for all API calls.</p>
            <div className="key-display">
              <code>{apiKey}</code>
              <CopyButton value={apiKey} />
            </div>
            <div className="info-box">
              <strong>Keep this key secret.</strong> It controls access to your document collection.
              Anyone with this key can query and upload documents to your collection.
            </div>
            {stats && (
              <div className="key-meta">
                <span>Collection: <code>{stats.collection_name}</code></span>
              </div>
            )}
          </div>
        )}

      </main>
    </div>
  )
}
