import { useState, useRef } from 'react'
import { api } from '../services/api.js'
import './DocumentPanel.css'

export default function DocumentPanel({ documents, onRefresh }) {
  const [uploading, setUploading] = useState(false)
  const [uploadError, setUploadError] = useState(null)
  const [dragOver, setDragOver] = useState(false)
  const fileRef = useRef(null)

  async function handleUpload(file) {
    if (!file || !file.name.endsWith('.pdf')) {
      setUploadError('Only PDF files are supported')
      return
    }
    setUploadError(null)
    setUploading(true)
    try {
      await api.uploadDocument(file)
      await onRefresh()
    } catch (err) {
      setUploadError(err.message)
    } finally {
      setUploading(false)
    }
  }

  async function handleDelete(filename) {
    if (!confirm(`Delete "${filename}" and all its indexed chunks?`)) return
    try {
      await api.deleteDocument(filename)
      await onRefresh()
    } catch (err) {
      alert(`Failed to delete: ${err.message}`)
    }
  }

  function onDrop(e) {
    e.preventDefault()
    setDragOver(false)
    const file = e.dataTransfer.files[0]
    if (file) handleUpload(file)
  }

  return (
    <aside className="doc-panel">
      <div className="doc-panel-header">
        <span className="doc-panel-title">Documents</span>
        <span className="doc-count">{documents.length}</span>
      </div>

      {/* Upload zone */}
      <div
        className={`upload-zone ${dragOver ? 'drag-over' : ''} ${uploading ? 'uploading' : ''}`}
        onClick={() => !uploading && fileRef.current?.click()}
        onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
        onDragLeave={() => setDragOver(false)}
        onDrop={onDrop}
      >
        <input
          ref={fileRef}
          type="file"
          accept=".pdf"
          style={{ display: 'none' }}
          onChange={(e) => e.target.files[0] && handleUpload(e.target.files[0])}
        />
        {uploading ? (
          <div className="upload-spinner" />
        ) : (
          <>
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
              <polyline points="17 8 12 3 7 8"/>
              <line x1="12" y1="3" x2="12" y2="15"/>
            </svg>
            <span>Drop PDF or click to upload</span>
          </>
        )}
      </div>
      {uploadError && <p className="upload-error">{uploadError}</p>}

      {/* Document list */}
      <div className="doc-list">
        {documents.length === 0 ? (
          <div className="doc-empty">
            <p>No documents indexed yet.</p>
            <p>Upload a PDF to get started.</p>
          </div>
        ) : (
          documents.map((doc) => (
            <div key={doc.filename} className="doc-item">
              <div className="doc-icon">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
                  <polyline points="14 2 14 8 20 8"/>
                </svg>
              </div>
              <div className="doc-info">
                <span className="doc-name" title={doc.filename}>{doc.filename}</span>
                <span className="doc-meta">{doc.chunk_count} chunks</span>
              </div>
              <button
                className="doc-delete-btn"
                onClick={() => handleDelete(doc.filename)}
                title="Delete document"
              >
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <polyline points="3 6 5 6 21 6"/>
                  <path d="M19 6l-1 14H6L5 6"/>
                  <path d="M10 11v6M14 11v6"/>
                  <path d="M9 6V4h6v2"/>
                </svg>
              </button>
            </div>
          ))
        )}
      </div>
    </aside>
  )
}
