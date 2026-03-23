const BASE_URL = import.meta.env.VITE_API_URL || '/api/v1'

async function request(path, options = {}) {
  const res = await fetch(`${BASE_URL}${path}`, {
    headers: { 'Content-Type': 'application/json', ...options.headers },
    ...options,
  })
  if (!res.ok) {
    const error = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(error.detail || `Request failed: ${res.status}`)
  }
  return res.json()
}

export const api = {
  health: () => request('/health'),

  chat: (query, sessionId = null) =>
    request('/chat', {
      method: 'POST',
      body: JSON.stringify({ query, session_id: sessionId }),
    }),

  /**
   * Streaming chat — calls onChunk(text) for each token,
   * onMeta({ session_id, sources }) once at the start,
   * onDone() when finished, onError(err) on failure.
   */
  chatStream: async (query, sessionId, { onMeta, onChunk, onDone, onError }) => {
    try {
      const res = await fetch(`${BASE_URL}/chat/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, session_id: sessionId, stream: true }),
      })
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }))
        throw new Error(err.detail || `Stream failed: ${res.status}`)
      }

      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })

        const lines = buffer.split('\n')
        buffer = lines.pop() // keep incomplete line

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue
          try {
            const event = JSON.parse(line.slice(6))
            if (event.type === 'meta') onMeta?.(event)
            else if (event.type === 'chunk') onChunk?.(event.content)
            else if (event.type === 'done') onDone?.()
            else if (event.type === 'error') onError?.(new Error(event.detail))
          } catch (_) {}
        }
      }
    } catch (err) {
      onError?.(err)
    }
  },

  search: (query, topK = 3) =>
    request('/search', {
      method: 'POST',
      body: JSON.stringify({ query, top_k: topK }),
    }),

  listDocuments: () => request('/documents'),

  uploadDocument: (file) => {
    const form = new FormData()
    form.append('file', file)
    return fetch(`${BASE_URL}/documents/upload`, { method: 'POST', body: form })
      .then(async (res) => {
        if (!res.ok) {
          const err = await res.json().catch(() => ({ detail: res.statusText }))
          throw new Error(err.detail || `Upload failed: ${res.status}`)
        }
        return res.json()
      })
  },

  deleteDocument: (filename) =>
    request(`/documents/${encodeURIComponent(filename)}`, { method: 'DELETE' }),

  getHistory: (sessionId) => request(`/chat/${sessionId}/history`),

  clearSession: (sessionId) =>
    request(`/chat/${sessionId}`, { method: 'DELETE' }),
}
