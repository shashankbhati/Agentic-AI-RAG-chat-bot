const BASE_URL = import.meta.env.VITE_API_URL || '/api/v1'

// ── Token storage ─────────────────────────────────────────────────────────────
export const auth = {
  save(token, email, apiKey, collectionName) {
    localStorage.setItem('rag_token', token)
    localStorage.setItem('rag_email', email)
    localStorage.setItem('rag_api_key', apiKey)
    localStorage.setItem('rag_collection', collectionName)
  },
  clear() {
    localStorage.removeItem('rag_token')
    localStorage.removeItem('rag_email')
    localStorage.removeItem('rag_api_key')
    localStorage.removeItem('rag_collection')
  },
  token: () => localStorage.getItem('rag_token'),
  email: () => localStorage.getItem('rag_email'),
  apiKey: () => localStorage.getItem('rag_api_key'),
  collection: () => localStorage.getItem('rag_collection'),
  isLoggedIn: () => !!localStorage.getItem('rag_token'),
}

// ── API helpers ───────────────────────────────────────────────────────────────
async function request(path, options = {}) {
  const token = auth.token()
  const res = await fetch(`${BASE_URL}${path}`, {
    headers: {
      'Content-Type': 'application/json',
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
      ...options.headers,
    },
    ...options,
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail || `Request failed: ${res.status}`)
  }
  return res.json()
}

export const authApi = {
  register: (email, password) =>
    request('/auth/register', {
      method: 'POST',
      body: JSON.stringify({ email, password }),
    }),

  login: (email, password) =>
    request('/auth/login', {
      method: 'POST',
      body: JSON.stringify({ email, password }),
    }),

  me: () => request('/auth/me'),

  // Dashboard (uses API key header)
  getStats: () =>
    request('/dashboard/stats', {
      headers: { 'X-API-Key': auth.apiKey() },
    }),

  getDashboardDocuments: () =>
    request('/dashboard/documents', {
      headers: { 'X-API-Key': auth.apiKey() },
    }),

  uploadDocument: (file) => {
    const form = new FormData()
    form.append('file', file)
    return fetch(`${BASE_URL}/documents/upload`, {
      method: 'POST',
      headers: { 'X-API-Key': auth.apiKey() },
      body: form,
    }).then(async (res) => {
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }))
        throw new Error(err.detail || `Upload failed: ${res.status}`)
      }
      return res.json()
    })
  },

  deleteDocument: (filename) =>
    request(`/documents/${encodeURIComponent(filename)}`, {
      method: 'DELETE',
      headers: { 'X-API-Key': auth.apiKey() },
    }),
}
