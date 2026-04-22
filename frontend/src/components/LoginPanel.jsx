import React, { useState } from 'react'
import axios from 'axios'

export default function LoginPanel({ onLogin, onClose }) {
  const [isRegister, setIsRegister] = useState(false)
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  const handleSubmit = async (e) => {
    e.preventDefault()
    setError('')
    setLoading(true)
    const endpoint = isRegister ? '/api/auth/register' : '/api/auth/login'
    
    try {
      const res = await axios.post(endpoint, { email, password })
      if (res.data.token) {
        onLogin(res.data.token)
      }
    } catch (err) {
      setError(err.response?.data?.error || 'Authentication failed. Please check credentials.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div style={{ position: 'fixed', top: 0, left: 0, width: '100%', height: '100%', backgroundColor: 'rgba(0,0,0,0.7)', display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 1000 }}>
      <div className="card" style={{ width: '400px', backgroundColor: '#1e293b', border: '1px solid #334155', padding: '24px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
          <h2 style={{ margin: 0, color: '#f8fafc' }}>{isRegister ? 'DriveIQ Register' : 'DriveIQ Login'}</h2>
          <button onClick={onClose} style={{ background: 'none', border: 'none', color: '#94a3b8', cursor: 'pointer', fontSize: '20px' }}>✕</button>
        </div>
        
        {error && <div style={{ padding: '10px', backgroundColor: '#cf222e20', border: '1px solid #ef4444', borderRadius: '4px', color: '#ff7b72', marginBottom: '15px' }}>{error}</div>}
        
        <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '15px' }}>
          <div>
            <label style={{ display: 'block', marginBottom: '5px', color: '#94a3b8' }}>Email Address</label>
            <input 
              type="email" 
              value={email} 
              onChange={e => setEmail(e.target.value)} 
              required 
              style={{ width: '100%', padding: '10px', backgroundColor: '#0f172a', border: '1px solid #475569', borderRadius: '4px', color: 'white', boxSizing: 'border-box' }}
            />
          </div>
          <div>
            <label style={{ display: 'block', marginBottom: '5px', color: '#94a3b8' }}>Password</label>
            <input 
              type="password" 
              value={password} 
              onChange={e => setPassword(e.target.value)} 
              required 
              style={{ width: '100%', padding: '10px', backgroundColor: '#0f172a', border: '1px solid #475569', borderRadius: '4px', color: 'white', boxSizing: 'border-box' }}
            />
          </div>
          <button type="submit" disabled={loading} style={{ padding: '12px', backgroundColor: '#3b82f6', color: 'white', border: 'none', borderRadius: '4px', cursor: loading ? 'not-allowed' : 'pointer', fontWeight: 'bold', marginTop: '10px' }}>
            {loading ? 'Processing...' : (isRegister ? 'Create Account' : 'Sign In')}
          </button>
        </form>
        
        <div style={{ marginTop: '20px', textAlign: 'center' }}>
          <button type="button" onClick={() => setIsRegister(!isRegister)} style={{ background: 'none', border: 'none', color: '#60a5fa', cursor: 'pointer', textDecoration: 'underline' }}>
            {isRegister ? 'Already have an account? Login here.' : 'Need an account? Register here.'}
          </button>
        </div>
      </div>
    </div>
  )
}
