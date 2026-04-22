import { useState, useEffect, useRef, useCallback, useMemo } from 'react'
import axios from 'axios'

import ScoreGauge         from './components/ScoreGauge'
import CoachingPanel      from './components/CoachingPanel'
import TrendChart         from './components/TrendChart'
import BehaviourVisualiser from './components/BehaviourVisualiser'
import FeatureTable        from './components/FeatureTable'
import ReviewPanel         from './components/ReviewPanel'
import LoginPanel          from './components/LoginPanel'
import { Bar }   from 'react-chartjs-2'
import { Chart as ChartJS, BarElement, CategoryScale, LinearScale, Tooltip, Legend } from 'chart.js'

ChartJS.register(BarElement, CategoryScale, LinearScale, Tooltip, Legend)

const API = ''   // empty = same origin (Vite proxy to Flask)
const POLL_MS = 2500
const HEALTH_POLL_MS = 15000
const BACKEND_FAIL_THRESHOLD = 3

// Simulate telemetry that changes over time for demo
function generateTelemetry(t) {
  return {
    speed:            55 + Math.sin(t * 0.3) * 30,
    rpm:              2000 + Math.sin(t * 0.5) * 800,
    throttle_position: 30 + Math.sin(t * 0.4) * 20,
    gear:              Math.floor(3 + Math.sin(t * 0.2) * 1.5),
    acceleration:      Math.sin(t * 0.7) * 2,
    fuel_rate:         7 + Math.sin(t * 0.3) * 2,
  }
}

function generateSyntheticFrameB64(t, telemetry) {
  const canvas = document.createElement('canvas')
  canvas.width = 320
  canvas.height = 180
  const ctx = canvas.getContext('2d')
  if (!ctx) return null

  // Sky + road background
  const grad = ctx.createLinearGradient(0, 0, 0, 180)
  grad.addColorStop(0, '#60a5fa')
  grad.addColorStop(0.55, '#93c5fd')
  grad.addColorStop(0.56, '#334155')
  grad.addColorStop(1, '#0f172a')
  ctx.fillStyle = grad
  ctx.fillRect(0, 0, 320, 180)

  // Lane markers with subtle motion
  const laneShift = Math.sin(t * 1.1) * 8
  ctx.strokeStyle = 'rgba(255,255,255,0.7)'
  ctx.setLineDash([10, 10])
  ctx.lineWidth = 2
  ctx.beginPath()
  ctx.moveTo(160 + laneShift - 20, 180)
  ctx.lineTo(145 + laneShift, 95)
  ctx.moveTo(160 + laneShift + 20, 180)
  ctx.lineTo(175 + laneShift, 95)
  ctx.stroke()
  ctx.setLineDash([])

  // Front vehicle proxy based on speed/proximity-like movement
  const speed = Number(telemetry?.speed ?? 50)
  const carW = Math.max(18, Math.min(48, 18 + speed * 0.2))
  const carH = Math.max(10, Math.min(28, 10 + speed * 0.11))
  const carX = 160 + Math.sin(t * 0.7) * 12 - carW / 2
  const carY = 120 - Math.sin(t * 0.45) * 6
  ctx.fillStyle = '#ef4444'
  ctx.fillRect(carX, carY, carW, carH)

  // Export as base64 payload body only (without data URL prefix)
  const dataUrl = canvas.toDataURL('image/jpeg', 0.75)
  return dataUrl.split(',')[1]
}

const ZERO_FEATURES = {
  rpm: 0,
  throttle_position: 0,
  braking_flag: 0,
  lane_change_flag: 0,
  proximity_score: 0,
  mean_flow: 0,
  flow_variance: 0,
}

function severityClass(score) {
  if (score === 'green' || score === 'yellow' || score === 'red') {
    return `severity-${score}`
  }
  if (score >= 75) return 'severity-green'
  if (score >= 50) return 'severity-yellow'
  return 'severity-red'
}

function severityLabel(score) {
  if (score >= 75) return 'green'
  if (score >= 50) return 'yellow'
  return 'red'
}

function getLiveInsights(features, score) {
  const insights = []
  
  if (score >= 80) {
    insights.push({ label: 'Excellent driving behavior', type: 'green' })
  } else if (score < 50) {
    insights.push({ label: 'Eco score is critical', type: 'red' })
  }

  if (features?.braking_flag === 1 || features?.braking_flag_ratio > 0) {
    insights.push({ label: 'Harsh braking detected (-score)', type: 'red' })
  }
  if (features?.lane_change_flag === 1 || features?.lane_change_flag_ratio > 0) {
    insights.push({ label: 'Erratic swerving / lane changes', type: 'yellow' })
  }
  if (Number(features?.proximity_score) > 0.15) {
    insights.push({ label: 'Following distance too close (tailgating)', type: 'red' })
  } else if (Number(features?.proximity_score) > 0.05) {
    insights.push({ label: 'Moderate following distance', type: 'yellow' })
  }
  if (Number(features?.flow_variance) > 5.0) {
    insights.push({ label: 'High optical velocity changes', type: 'yellow' })
  }
  
  if (insights.length === 0 || (insights.length === 1 && insights[0].type === 'green')) {
    insights.push({ label: 'Maintaining smooth, safe flow (+score)', type: 'green' })
  }
  
  // Deduplicate using Map
  const unique = new Map()
  insights.forEach(i => unique.set(i.label, i))
  return Array.from(unique.values())
}

function decodeJwtPayload(token) {
  try {
    const payloadPart = String(token || '').split('.')[1]
    if (!payloadPart) return null
    const base64 = payloadPart.replace(/-/g, '+').replace(/_/g, '/')
    const padded = base64 + '='.repeat((4 - (base64.length % 4)) % 4)
    return JSON.parse(atob(padded))
  } catch {
    return null
  }
}

function isJwtExpired(token) {
  const payload = decodeJwtPayload(token)
  if (!payload || typeof payload.exp !== 'number') return true
  return (Date.now() / 1000) >= Number(payload.exp)
}

export default function App() {
  const [token,             setToken]             = useState(localStorage.getItem('driveiq_token') || '')
  const [showAuthDialog,    setShowAuthDialog]    = useState(false)
  
  const [isLiveMode,        setIsLiveMode]        = useState(false)
  const [liveScore,         setLiveScore]         = useState(0)
  const [liveFeatures,      setLiveFeatures]      = useState({ ...ZERO_FEATURES })
  const [reviewResult,      setReviewResult]      = useState(null)
  const [selectedWindow,    setSelectedWindow]    = useState(null)
  const [healthState,       setHealthState]       = useState('checking')
  const [healthMessage,     setHealthMessage]     = useState('Checking backend health...')
  const [offlineMode,       setOfflineMode]       = useState(false)
  const [sessionSaveWarning, setSessionSaveWarning] = useState('')
  const [healthMeta,        setHealthMeta]        = useState({ schema_valid: false, core_models_loaded: false })
  const [livePoints,        setLivePoints]        = useState([])
  const [liveEvents,        setLiveEvents]        = useState([])
  const [liveVideoFile,     setLiveVideoFile]     = useState(null)
  const [streamActive,      setStreamActive]      = useState(false)
  
  const liveVideoUrl = useMemo(() => {
    if (!liveVideoFile) return null
    return URL.createObjectURL(liveVideoFile)
  }, [liveVideoFile])
  
  const clockRef = useRef(0)
  const sessionIdRef = useRef(`sess-${Math.random().toString(36).slice(2)}`)
  const sessionStartedAtRef = useRef(Date.now())
  const prevFrameRef = useRef(null)
  const healthFailCountRef = useRef(0)
  const liveVideoRef = useRef(null)
  const streamCompleteRef = useRef(false)

  // Cleanup object URLs
  useEffect(() => {
    return () => {
      if (liveVideoUrl) URL.revokeObjectURL(liveVideoUrl)
    }
  }, [liveVideoUrl])

  const fetchHealth = useCallback(async () => {
    if (offlineMode) {
      return
    }

    try {
      const { data } = await axios.get(`${API}/api/health`)
      healthFailCountRef.current = 0
      setOfflineMode(false)
      setHealthMeta({
        schema_valid: Boolean(data?.schema_valid),
        core_models_loaded: Boolean(data?.core_models_loaded),
      })
      const scoreReady = data?.score_ready === true
      const ready = data?.ready === true || scoreReady
      if (ready) {
        setHealthState('ready')
        setHealthMessage('Backend ready for review and scoring.')
      } else {
        setHealthState('degraded')
        const reasonParts = []
        if (data?.schema_error) reasonParts.push(data.schema_error)
        if (data?.coach_status === 'loading') reasonParts.push('Coach model warming up')
        if (data?.coach_error) reasonParts.push(`Coach error: ${data.coach_error}`)
        const reason = reasonParts.length ? ` ${reasonParts.join(' | ')}` : ''
        setHealthMessage(`Backend reachable but degraded.${reason}`.trim())
      }
    } catch {
      healthFailCountRef.current += 1
      if (healthFailCountRef.current >= BACKEND_FAIL_THRESHOLD) {
        setOfflineMode(true)
        setHealthState('degraded')
        setHealthMessage('Backend unavailable. UI is in offline placeholder mode.')
      } else {
        setHealthState('error')
        setHealthMessage('Cannot reach backend health endpoint.')
      }
    }
  }, [offlineMode])

  // ── /api/score polling (Live Mode only) ──────────────────────────────────
  const fetchScore = useCallback(async () => {
    if (!isLiveMode) {
      return
    }

    if (offlineMode || healthState === 'error') {
      return
    }

    // If stream already completed full pass, don't score again
    if (streamCompleteRef.current) {
      return
    }

    let frameB64 = null
    const v = liveVideoRef.current
    if (v && v.readyState >= 2 && !v.paused && !v.ended) {
      const canvas = document.createElement('canvas')
      canvas.width = 480
      canvas.height = 270
      const ctx = canvas.getContext('2d')
      if (!ctx) return
      ctx.drawImage(v, 0, 0, 480, 270)
      frameB64 = canvas.toDataURL('image/jpeg', 0.8).split(',')[1]
      clockRef.current = v.currentTime
    } else if (v) {
      // Video element exists but is paused/ended — stop scoring
      return
    } else if (streamActive) {
      // No video element at all, but stream active — use synthetic fallback
      clockRef.current += POLL_MS / 1000
      frameB64 = generateSyntheticFrameB64(clockRef.current, generateTelemetry(clockRef.current))
    } else {
      return // Not active
    }

    // When real video is playing, send neutral/empty telemetry so the backend
    // relies purely on CV-extracted features (not fake sine-wave acceleration).
    const hasRealVideo = liveVideoRef.current && liveVideoRef.current.readyState >= 2
    const telemetry = hasRealVideo
      ? { speed: 0, rpm: 0, throttle_position: 0, gear: 0, acceleration: 0, fuel_rate: 0 }
      : generateTelemetry(clockRef.current)
    const prevFrameB64 = prevFrameRef.current
    const tokenExpired = Boolean(token) && isJwtExpired(token)
    const authHeaders = {}
    if (token && !tokenExpired) {
      authHeaders.Authorization = `Bearer ${token}`
    } else if (token && tokenExpired) {
      setSessionSaveWarning('Session expired. Live scoring continues, but this drive is not being saved. Please log in again.')
    }

    try {
      const { data } = await axios.post(`${API}/api/score`, {
        telemetry,
        session_id: sessionIdRef.current,
        session_started_at: sessionStartedAtRef.current,
        frame_b64: frameB64,
        prev_frame_b64: prevFrameB64,
      }, {
        headers: authHeaders
      })
      prevFrameRef.current = frameB64

      if (token && !tokenExpired) {
        if (data?.auth_failed) {
          setSessionSaveWarning('Authentication failed. Live scoring continues, but this drive is not being saved. Please log in again.')
        } else if (data?.session_saved === false) {
          setSessionSaveWarning('Live scoring is active, but session saving is currently unavailable.')
        } else if (data?.session_saved === true) {
          setSessionSaveWarning('')
        }
      }
      
      const s = Number(data.score ?? 0)
      setLiveScore(s)
      
      const featuresMerged = { ...data.features }
      setLiveFeatures(featuresMerged)
      
      const insights = getLiveInsights(featuresMerged, s)
      const severity = severityLabel(s)
      
      setLivePoints(prev => {
        const next = [...prev, { timestamp_sec: clockRef.current, score: s, severity }]
        if (next.length > 60) next.shift()
        return next
      })

      // Pin one consolidated event per timestamp to the live log
      const warnings = insights.filter((i) => i.type !== 'green')
      if (warnings.length > 0) {
        // Pick the worst severity and join all labels
        const worst = warnings.some(w => w.type === 'red') ? 'red' : 'yellow'
        const summary = warnings.map(w => w.label).join(' · ')
        setLiveEvents(prev => {
          const entry = { label: summary, type: worst, timestamp_sec: clockRef.current }
          const next = [entry, ...prev]
          if (next.length > 30) next.length = 30
          return next
        })
      }

    } catch {
      setHealthState('degraded')
      setHealthMessage('Backend unstable. Retrying score stream with backoff...')
    }
  }, [healthState, offlineMode, isLiveMode])

  const reconnectBackend = useCallback(async () => {
    healthFailCountRef.current = 0
    setOfflineMode(false)
    await fetchHealth()
  }, [fetchHealth])

  // Start polling
  useEffect(() => {
    fetchHealth()
    const healthId = setInterval(fetchHealth, HEALTH_POLL_MS)
    let scoreId = null
    if (isLiveMode) {
      fetchScore()
      scoreId = setInterval(fetchScore, POLL_MS)
    }
    return () => {
      if (scoreId) clearInterval(scoreId)
      clearInterval(healthId)
    }
  }, [fetchHealth, fetchScore, isLiveMode])

  const reviewPoints = useMemo(() => {
    if (isLiveMode) return livePoints
    return reviewResult?.segments || []
  }, [isLiveMode, livePoints, reviewResult])
  const displayedScore = isLiveMode
    ? liveScore
    : Number(selectedWindow?.avg_score ?? selectedWindow?.score ?? 0)
  const displayedFeatures = isLiveMode
    ? {
        rpm: liveFeatures?.rpm ?? 0,
        throttle_position: liveFeatures?.throttle_position ?? 0,
        braking_flag: liveFeatures?.braking_flag ?? 0,
        lane_change_flag: liveFeatures?.lane_change_flag ?? 0,
        proximity_score: liveFeatures?.proximity_score ?? 0,
        mean_flow: liveFeatures?.mean_flow ?? 0,
        flow_variance: liveFeatures?.flow_variance ?? 0,
      }
    : {
        rpm: Number(selectedWindow?.rpm ?? 0),
        throttle_position: Number(selectedWindow?.throttle_position ?? 0),
        braking_flag: Number(selectedWindow?.braking_flag_ratio ?? 0) > 0 ? 1 : 0,
        lane_change_flag: Number(selectedWindow?.lane_change_flag_ratio ?? 0) > 0 ? 1 : 0,
        proximity_score: Number(selectedWindow?.proximity_score_mean ?? 0),
        mean_flow: Number(selectedWindow?.mean_flow_mean ?? 0),
        flow_variance: Number(selectedWindow?.flow_variance ?? 0),
      }

  const selectedCoach = selectedWindow?.coach_note || 'Select a segment to view coaching note.'
  const selectedSeverity = selectedWindow?.severity || 'yellow'

  const healthClass = `health-banner health-${healthState}`
  const schemaOk = healthMeta.schema_valid
  const modelsOk = healthMeta.core_models_loaded

  return (
    <>
      <header className="topbar">
        <div className="header-left">
          <div className="topbar-logo">
            <span>DriveIQ</span>
          </div>
          <div className="status-pill neutral-pill">
            <span className={`severity-badge ${schemaOk && modelsOk ? 'severity-green' : 'severity-red'}`}>
              schema_valid: {String(schemaOk)} | core_models_loaded: {String(modelsOk)}
            </span>
          </div>
        </div>
        <div className="header-right">
          {token ? (
            <div className="driver-profile card" style={{ display: 'flex', alignItems: 'center', gap: '15px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <div className="driver-avatar" style={{ background: '#22c55e' }}>✓</div>
                <div>
                  <div style={{ fontWeight: 700, color: '#4ade80' }}>Verified Driver</div>
                  <div style={{ fontSize: 12, color: '#94a3b8' }}>Session actively saved</div>
                </div>
              </div>
              <button 
                onClick={() => { localStorage.removeItem('driveiq_token'); setToken(''); setSessionSaveWarning(''); }} 
                className="scenario-btn" 
                style={{ background: '#ef4444', padding: '6px 12px', minHeight: 'auto', fontSize: '12px' }}>
                Logout
              </button>
            </div>
          ) : (
            <div className="driver-profile card" style={{ display: 'flex', alignItems: 'center', gap: '15px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <div className="driver-avatar" style={{ background: '#64748b' }}>?</div>
                <div>
                  <div style={{ fontWeight: 700, color: '#94a3b8' }}>Guest Driver</div>
                  <div style={{ fontSize: 12, color: '#64748b' }}>Data won't be saved</div>
                </div>
              </div>
              <button 
                onClick={() => setShowAuthDialog(true)} 
                className="scenario-btn" 
                style={{ background: '#3b82f6', padding: '6px 12px', minHeight: 'auto', fontSize: '12px' }}>
                Login / Register
              </button>
            </div>
          )}
        </div>
      </header>

      {showAuthDialog && (
        <LoginPanel 
          onClose={() => setShowAuthDialog(false)} 
          onLogin={(t) => { localStorage.setItem('driveiq_token', t); setToken(t); setSessionSaveWarning(''); setShowAuthDialog(false); }} 
        />
      )}

      <div className={healthClass}>{healthMessage}</div>
      {sessionSaveWarning ? <div className="health-banner health-degraded">{sessionSaveWarning}</div> : null}
      {offlineMode ? (
        <div className="offline-controls">
          <button className="scenario-btn" onClick={reconnectBackend}>Retry Backend Connection</button>
        </div>
      ) : null}

      <section className="stats-bar">
        <div className="card stat-card"><div className="card-title">Today&apos;s Score</div><strong>67</strong></div>
        <div className="card stat-card"><div className="card-title">Best Score</div><strong>82</strong></div>
        <div className="card stat-card"><div className="card-title">Trips This Week</div><strong>4</strong></div>
        <div className="card stat-card"><div className="card-title">Fuel Saved</div><strong>2.3L</strong></div>
      </section>

      <main className="demo-layout">
        <div style={{ display: 'flex', gap: 8, marginBottom: 16 }}>
          <button 
            className="scenario-btn" 
            style={{ flex: 1, background: !isLiveMode ? '#3b82f6' : 'rgba(255,255,255,0.05)' }}
            onClick={() => setIsLiveMode(false)}
          >
            Post-Drive Full Analysis
          </button>
          <button 
            className="scenario-btn" 
            style={{ flex: 1, background: isLiveMode ? '#3b82f6' : 'rgba(255,255,255,0.05)', borderColor: isLiveMode ? '#60a5fa' : 'transparent' }}
            onClick={() => {
              setIsLiveMode(true)
              setLivePoints([])
              setLiveEvents([])
              clockRef.current = 0
              sessionStartedAtRef.current = Date.now()
              prevFrameRef.current = null
              streamCompleteRef.current = false
            }}
          >
            Real-Time Live Mode
          </button>
        </div>

        {!isLiveMode ? (
          <>
            <section className="card review-primary">
              <div className="review-head">
                <div className="card-title">Trip Review (Full Analysis)</div>
                <button className="scenario-btn" onClick={() => window.alert('Report export coming soon')}>
                  Export Report
                </button>
              </div>
              <ReviewPanel
                onAnalysisComplete={setReviewResult}
                onWindowSelect={setSelectedWindow}
                selectedTimestampSec={selectedWindow?.start_sec ?? selectedWindow?.timestamp_sec}
              />
            </section>
            
            {reviewResult && (
              <section style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 14 }}>
                <div className="card">
                  <div className="card-title">Drive Segment Breakdown</div>
                  <div style={{ height: 220, padding: 10 }}>
                    <Bar 
                      data={{
                        labels: reviewResult.segments?.map((_, i) => `Segment ${i+1}`) || [],
                        datasets: [{
                          label: 'Score per Segment',
                          data: reviewResult.segments?.map(s => s.avg_score) || [],
                          backgroundColor: reviewResult.segments?.map(s => severityClass(s.severity || s.avg_score) === 'severity-green' ? '#10b981' : severityClass(s.severity || s.avg_score) === 'severity-yellow' ? '#f59e0b' : '#ef4444'),
                          borderRadius: 4
                        }]
                      }} 
                      options={{ maintainAspectRatio: false, plugins: { legend: { display: false } } }} 
                    />
                  </div>
                </div>
                
                <div className="card">
                  <div className="card-title">Detailed Trip Report</div>
                  <div style={{ color: '#e2e8f0', fontSize: 14, lineHeight: '1.6', marginTop: 10 }}>
                    <p><strong>Overall Journey Score:</strong> {reviewResult.avg_batch_score?.toFixed(1) || 'N/A'}</p>
                    <p><strong>Total Duration:</strong> {Math.floor((reviewResult.duration_sec || 0)/60)}m {Math.floor((reviewResult.duration_sec || 0)%60)}s</p>
                    <p style={{ marginTop: 10, color: '#94a3b8' }}>
                      This trip consisted of {reviewResult.window_count} analyzed extraction windows. 
                      {reviewResult.segments?.some(s => s.severity === 'red') 
                        ? ' There were critical drops in Eco Score, primarily due to instances of tailgating and erratic velocity shifts across specific timeline segments. Review the red segments mapped above.'
                        : ' Flow and speed variation remained smooth. The XGBoost framework confidently categorized your driving consistency as safe.'}
                    </p>
                  </div>
                  <TrendChart points={reviewPoints} emptyMessage="Upload a clip to see score trend" />
                </div>
              </section>
            )}
          </>
        ) : (
          <>
            <section className="card review-primary" style={{ border: '2px solid #3b82f6' }}>
              <div className="review-head">
                <div className="card-title">Live Dynamic Streaming</div>
                <div style={{ display: 'flex', gap: 10, alignItems: 'center' }}>
                  <span style={{ fontSize: 13, color: '#94a3b8' }}>Load Video to Stream:</span>
                  <input type="file" accept="video/mp4" onChange={(e) => {
                    setLiveVideoFile(e.target.files?.[0])
                    setLivePoints([])
                    setLiveEvents([])
                    setLiveScore(0)
                    setLiveFeatures({ ...ZERO_FEATURES })
                    setStreamActive(false)
                    clockRef.current = 0
                    sessionStartedAtRef.current = Date.now()
                    prevFrameRef.current = null
                    streamCompleteRef.current = false
                    sessionIdRef.current = `sess-${Math.random().toString(36).slice(2)}`
                  }} />
                  {liveVideoUrl && !streamActive && (
                    <button className="scenario-btn" style={{ background: '#10b981' }} onClick={() => {
                      sessionStartedAtRef.current = Date.now()
                      setStreamActive(true)
                      liveVideoRef.current?.play()
                    }}>
                      Start Stream
                    </button>
                  )}
                </div>
              </div>
              {liveVideoUrl ? (
                <video 
                  ref={liveVideoRef}
                  src={liveVideoUrl} 
                  controls
                  muted
                  onEnded={() => {
                    setStreamActive(false)
                    streamCompleteRef.current = true
                    // Calculate mean score
                    if (livePoints.length > 0) {
                      const avg = livePoints.reduce((acc, p) => acc + p.score, 0) / livePoints.length
                      setLiveScore(Math.round(avg))
                      setLiveEvents(prev => [{ label: `Stream Complete. Mean Eco Score: ${Math.round(avg)}`, type: 'green', timestamp_sec: clockRef.current }, ...prev])
                    }
                  }}
                  style={{ width: '100%', maxHeight: 400, borderRadius: 10, marginTop: 14, background: '#020617' }} 
                />
              ) : (
                <div style={{ padding: 40, textAlign: 'center', background: 'rgba(255,255,255,0.02)', borderRadius: 10, marginTop: 14, color: '#94a3b8' }}>
                  Upload a mp4 file above to stream real video frames to the backend...
                </div>
              )}
            </section>

            <section className="layout-two-col">
              <TrendChart points={reviewPoints} emptyMessage="Stream a video to generate live trend mapping" />

              <div className="trip-history card">
                <div className="card-title">Live Insights (Real-Time)</div>
                <div className="history-list">
                  {liveEvents.length > 0 ? (
                    liveEvents.map((insight, idx) => {
                      const m = Math.floor(insight.timestamp_sec / 60)
                      const s = Math.floor(insight.timestamp_sec % 60)
                      const tLabel = `${m}:${String(s).padStart(2, '0')}`
                      return (
                        <button
                          key={idx}
                          className="history-item"
                          onClick={() => {
                            if (liveVideoRef.current) {
                              liveVideoRef.current.currentTime = insight.timestamp_sec
                              liveVideoRef.current.play()
                            }
                          }}
                          style={{ display: 'flex', justifyContent: 'space-between', textAlign: 'left' }}
                        >
                          <span><strong style={{ color: '#94a3b8' }}>[{tLabel}]</strong> {insight.label}</span>
                          <span className={`severity-badge severity-${insight.type}`}>
                            {insight.type.toUpperCase()}
                          </span>
                        </button>
                      )
                    })
                  ) : (
                    <div style={{ padding: 20, textAlign: 'center', color: '#94a3b8', fontSize: 13 }}>
                      {streamActive ? 'Analysing stream... no major infractions.' : 'Start stream to generate insights.'}
                    </div>
                  )}
                </div>
              </div>
            </section>

            <section className="score-feature-grid">
              <ScoreGauge score={displayedScore} />
              <FeatureTable features={displayedFeatures} />
            </section>

            <section>
              <CoachingPanel
                tips={[selectedCoach]}
                loading={false}
                message={isLiveMode ? 'Live mode active. Real-time insights appear above.' : 'Coaching for selected review segment'}
                severity={selectedSeverity}
                source={selectedWindow?.score_source || 'review'}
                fallback={false}
                debugReason=""
                warning=""
                topIssue={selectedWindow?.dominant_issue || selectedWindow?.top_issue || ''}
              />
            </section>
          </>
        )}

        <section className="card">
          <div className="card-title">Behaviour Visualiser (Secondary)</div>
          <div style={{ marginTop: 12 }}>
            <BehaviourVisualiser features={displayedFeatures} />
          </div>
        </section>
      </main>
    </>
  )
}
