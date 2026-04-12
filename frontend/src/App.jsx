import { useState, useEffect, useRef, useCallback } from 'react'
import axios from 'axios'

import ScoreGauge         from './components/ScoreGauge'
import CoachingPanel      from './components/CoachingPanel'
import TrendChart         from './components/TrendChart'
import BehaviourVisualiser from './components/BehaviourVisualiser'
import FeatureTable        from './components/FeatureTable'

const API = ''   // empty = same origin (Vite proxy to Flask)
const POLL_MS = 2500
const HEALTH_POLL_MS = 15000
const SCORE_FAIL_BACKOFF_MS = 8000
const COACH_RETRY_MIN_MS = 10000
const BACKEND_FAIL_THRESHOLD = 3
const HISTORY_MAX = 30

const DEMO_SCENARIOS = {
  green: {
    score: 84,
    features: {
      braking_flag: 0,
      lane_change_flag: 0,
      proximity_score: 0.04,
      mean_flow: 1.2,
      flow_variance: 0.5,
      vehicle_count: 2,
      weather_id: 0,
      road_type_id: 1,
    },
    predictedFuelRate: 5.9,
    historySummary: 'Smooth throttle and stable lane discipline in light traffic.',
  },
  yellow: {
    score: 62,
    features: {
      braking_flag: 1,
      lane_change_flag: 0,
      proximity_score: 0.17,
      mean_flow: 2.0,
      flow_variance: 1.0,
      vehicle_count: 5,
      weather_id: 1,
      road_type_id: 2,
    },
    predictedFuelRate: 7.8,
    historySummary: 'Moderate stop-and-go traffic with a few abrupt decelerations.',
  },
  red: {
    score: 34,
    features: {
      braking_flag: 1,
      lane_change_flag: 1,
      proximity_score: 0.28,
      mean_flow: 3.1,
      flow_variance: 1.8,
      vehicle_count: 9,
      weather_id: 2,
      road_type_id: 3,
    },
    predictedFuelRate: 10.9,
    historySummary: 'Aggressive speed variation, dense traffic, and repeated harsh maneuvers.',
  },
}

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

function synthFeaturesFromTelemetry(telemetry) {
  const speed = Number(telemetry?.speed ?? 0)
  const accel = Number(telemetry?.acceleration ?? 0)
  const throttle = Number(telemetry?.throttle_position ?? 0)

  const proximity = Math.max(0, Math.min(1, (speed - 35) / 100))
  const meanFlow = Math.abs(accel) * 0.8 + throttle / 120
  const flowVariance = Math.abs(accel) * 0.5

  return {
    vehicle_count: Math.max(0, Math.min(12, Math.round(1 + speed / 30))),
    proximity_score: Number(proximity.toFixed(4)),
    pedestrian_flag: 0,
    mean_flow: Number(meanFlow.toFixed(4)),
    flow_variance: Number(flowVariance.toFixed(4)),
    braking_flag: accel < -0.8 ? 1 : 0,
    lane_change_flag: Math.abs(accel) > 1.6 ? 1 : 0,
    road_type_id: speed > 70 ? 1 : 0,
    weather_id: 0,
  }
}

function heuristicScore(features) {
  let s = 85
  s -= Number(features?.proximity_score ?? 0) * 30
  s -= Number(features?.mean_flow ?? 0) * 8
  s -= Number(features?.flow_variance ?? 0) * 10
  s -= Number(features?.braking_flag ?? 0) * 12
  s -= Number(features?.lane_change_flag ?? 0) * 10
  return Math.max(0, Math.min(100, s))
}

function severityFromScore(s) {
  if (s >= 75) return 'green'
  if (s >= 50) return 'yellow'
  return 'red'
}

function localTips(features) {
  const tips = []
  if (features?.braking_flag) tips.push('Anticipate traffic earlier and reduce abrupt braking events.')
  if (features?.lane_change_flag) tips.push('Avoid unnecessary lane changes to maintain smooth momentum.')
  if (Number(features?.proximity_score ?? 0) > 0.15) tips.push('Increase following distance to smooth acceleration and deceleration.')
  const defaults = [
    'Keep throttle inputs gentle and progressive.',
    'Maintain a steady cruising speed whenever possible.',
    'Check tire pressure regularly to reduce rolling resistance.',
  ]
  for (const d of defaults) {
    if (tips.length >= 3) break
    if (!tips.includes(d)) tips.push(d)
  }
  return tips.slice(0, 3)
}

export default function App() {
  const [score,             setScore]             = useState(72)
  const [features,          setFeatures]          = useState({})
  const [tips,              setTips]              = useState([])
  const [coachMessage,      setCoachMessage]      = useState('Monitoring driving behaviour...')
  const [coachSeverity,     setCoachSeverity]     = useState('yellow')
  const [tipsLoading,       setTipsLoading]       = useState(false)
  const [predictedFuelRate, setPredictedFuelRate] = useState(null)
  const [scoreHistory,      setScoreHistory]      = useState([72])
  const [connected,         setConnected]         = useState(false)
  const [lastTipScore,      setLastTipScore]      = useState(null)
  const [coachSource,       setCoachSource]       = useState('rules')
  const [coachFallback,     setCoachFallback]     = useState(false)
  const [coachDebugReason,  setCoachDebugReason]  = useState('')
  const [coachWarning,      setCoachWarning]      = useState('')
  const [scoreLatencyMs,    setScoreLatencyMs]    = useState(null)
  const [coachLatencyMs,    setCoachLatencyMs]    = useState(null)
  const [healthState,       setHealthState]       = useState('checking')
  const [healthMessage,     setHealthMessage]     = useState('Checking backend health...')
  const [offlineMode,       setOfflineMode]       = useState(false)

  const clockRef = useRef(0)
  const sessionIdRef = useRef(`sess-${Math.random().toString(36).slice(2)}`)
  const prevFrameRef = useRef(null)
  const lastScoreFailureAtRef = useRef(0)
  const lastCoachAttemptAtRef = useRef(0)
  const healthFailCountRef = useRef(0)

  const historySummaryFromState = useCallback(() => {
    const recent = scoreHistory.slice(-5)
    if (!recent.length) {
      return 'No recent score history.'
    }
    const avg = recent.reduce((a, b) => a + b, 0) / recent.length
    return `Recent average score ${avg.toFixed(1)} across ${recent.length} samples.`
  }, [scoreHistory])

  const fetchHealth = useCallback(async () => {
    if (offlineMode) {
      return
    }

    try {
      const { data } = await axios.get(`${API}/api/health`)
      healthFailCountRef.current = 0
      setOfflineMode(false)
      const scoreReady = data?.score_ready === true
      const coachReady = data?.coach_ready === true
      const ready = data?.ready === true || (scoreReady && (coachReady || data?.coach_disabled === true))
      if (ready) {
        setHealthState('ready')
        setHealthMessage('Backend ready: scoring and coaching services online.')
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
        setHealthMessage('Backend unavailable. Running local offline simulation mode.')
        setConnected(true)
      } else {
        setHealthState('error')
        setHealthMessage('Cannot reach backend health endpoint.')
      }
    }
  }, [offlineMode])

  // ── /api/score polling ───────────────────────────────────────────────────
  const fetchScore = useCallback(async () => {
    if (offlineMode) {
      clockRef.current += POLL_MS / 1000
      const telemetry = generateTelemetry(clockRef.current)
      const vision = synthFeaturesFromTelemetry(telemetry)
      const merged = { ...vision, ...telemetry }
      const s = Number(heuristicScore(vision).toFixed(2))
      setScore(s)
      setFeatures(merged)
      setPredictedFuelRate(Number(telemetry.fuel_rate.toFixed(2)))
      setScoreLatencyMs(0)
      setConnected(true)
      setScoreHistory(prev => {
        const next = [...prev, s]
        return next.length > HISTORY_MAX ? next.slice(-HISTORY_MAX) : next
      })
      return
    }

    if (healthState === 'error') {
      return
    }

    if (Date.now() - lastScoreFailureAtRef.current < SCORE_FAIL_BACKOFF_MS) {
      return
    }

    clockRef.current += POLL_MS / 1000
    const telemetry = generateTelemetry(clockRef.current)
    const frameB64 = generateSyntheticFrameB64(clockRef.current, telemetry)
    const prevFrameB64 = prevFrameRef.current

    try {
      const start = performance.now()
      const { data } = await axios.post(`${API}/api/score`, {
        telemetry,
        session_id: sessionIdRef.current,
        frame_b64: frameB64,
        prev_frame_b64: prevFrameB64,
      })
      setScoreLatencyMs(Math.round(performance.now() - start))
      prevFrameRef.current = frameB64
      setScore(data.score)
      const merged = { ...data.features, ...telemetry }
      setFeatures(merged)
      setConnected(true)
      setCoachWarning('')
      setScoreHistory(prev => {
        const next = [...prev, data.score]
        return next.length > HISTORY_MAX ? next.slice(-HISTORY_MAX) : next
      })

      // Keep a stable fuel metric for coaching payload and UI while /api/predict is removed.
      const fuelRate = Number(telemetry?.fuel_rate ?? 0)
      setPredictedFuelRate(Number.isFinite(fuelRate) ? fuelRate : null)

    } catch {
      setConnected(false)
      setScoreLatencyMs(null)
      lastScoreFailureAtRef.current = Date.now()
      setHealthState('degraded')
      setHealthMessage('Backend unstable. Retrying score stream with backoff...')
    }
  }, [healthState, offlineMode])

  // ── /api/coach — only re-fetch when score changes by ≥5 ─────────────────
  const fetchTips = useCallback(async (currentScore, currentFeatures, fuelRate, historySummary) => {
    if (offlineMode) {
      setTipsLoading(true)
      const local = localTips(currentFeatures)
      setTips(local)
      setCoachSeverity(severityFromScore(currentScore))
      setCoachMessage('Offline mode coaching based on local driving heuristics.')
      setCoachSource('local_rules')
      setCoachFallback(true)
      setCoachDebugReason('offline_demo_mode')
      setCoachWarning('Backend unavailable. Showing local simulated coaching.')
      setCoachLatencyMs(0)
      setLastTipScore(currentScore)
      setTipsLoading(false)
      return
    }

    const now = Date.now()
    if (now - lastCoachAttemptAtRef.current < COACH_RETRY_MIN_MS) {
      return
    }
    lastCoachAttemptAtRef.current = now

    setTipsLoading(true)
    try {
      const start = performance.now()
      const { data } = await axios.post(`${API}/api/coach`, {
        session_id: sessionIdRef.current,
        score: currentScore,
        features: currentFeatures,
        predicted_fuel_rate: fuelRate ?? 0,
        history_summary: historySummary || 'Live session in progress.',
      })
      setCoachLatencyMs(Math.round(performance.now() - start))
      setTips(data.tips || [])
      setCoachMessage(data.message || 'Focus on smooth and consistent driving.')
      setCoachSeverity(data.severity || 'yellow')
      setCoachSource(data.source || 'rules')
      setCoachFallback(Boolean(data.fallback))
      setCoachDebugReason(data.debug_flan_reason || '')
      setCoachWarning('')
      setLastTipScore(currentScore)
    } catch {
      setCoachLatencyMs(null)
      if (!tips.length) {
        setTips(['Maintain steady speed.', 'Anticipate traffic ahead.', 'Keep tyre pressure optimal.'])
        setCoachMessage('Coaching service unavailable; showing fallback tips.')
        setCoachSeverity('yellow')
      }
      setCoachSource('rules')
      setCoachFallback(true)
      setCoachWarning('Coaching request failed. Showing latest available tips.')
      setLastTipScore(currentScore)
    } finally {
      setTipsLoading(false)
    }
  }, [tips.length, offlineMode])

  const runScenario = useCallback((key) => {
    const scenario = DEMO_SCENARIOS[key]
    if (!scenario) return

    const mergedFeatures = {
      ...features,
      ...scenario.features,
      speed: scenario.score >= 75 ? 72 : scenario.score >= 50 ? 88 : 114,
      rpm: scenario.score >= 75 ? 1850 : scenario.score >= 50 ? 2350 : 2950,
      throttle_position: scenario.score >= 75 ? 28 : scenario.score >= 50 ? 42 : 61,
      gear: scenario.score >= 75 ? 5 : scenario.score >= 50 ? 4 : 3,
      acceleration: scenario.score >= 75 ? 0.3 : scenario.score >= 50 ? 1.1 : 2.0,
    }

    setScore(scenario.score)
    setFeatures(mergedFeatures)
    setPredictedFuelRate(scenario.predictedFuelRate)
    setScoreHistory(prev => {
      const next = [...prev, scenario.score]
      return next.length > HISTORY_MAX ? next.slice(-HISTORY_MAX) : next
    })
    fetchTips(scenario.score, mergedFeatures, scenario.predictedFuelRate, scenario.historySummary)
  }, [features, fetchTips])

  const reconnectBackend = useCallback(async () => {
    healthFailCountRef.current = 0
    setOfflineMode(false)
    await fetchHealth()
  }, [fetchHealth])

  // Start polling
  useEffect(() => {
    fetchHealth()
    const healthId = setInterval(fetchHealth, HEALTH_POLL_MS)
    fetchScore()
    const scoreId = setInterval(fetchScore, POLL_MS)
    return () => {
      clearInterval(scoreId)
      clearInterval(healthId)
    }
  }, [fetchHealth, fetchScore])

  // Fetch tips when score shifts enough
  useEffect(() => {
    if (connected && (lastTipScore === null || Math.abs(score - lastTipScore) >= 5)) {
      fetchTips(score, features, predictedFuelRate, historySummaryFromState())
    }
  }, [score, connected]) // eslint-disable-line

  const healthClass = `health-banner health-${healthState}`

  return (
    <>
      {/* ── Top bar ── */}
      <header className="topbar">
        <div className="topbar-logo">
          <span>DriveIQ</span>
        </div>
        <div className="status-cluster">
          <div className="status-pill">
            <div className="status-dot" style={{ background: connected ? '#10b981' : '#ef4444' }} />
            <span style={{ fontSize: 12, color: '#94a3b8' }}>
              {connected ? 'Live score stream' : 'Reconnecting score stream'}
            </span>
          </div>
          <div className="status-pill neutral-pill">
            <span className="mini-label">Score</span>
            <span>{scoreLatencyMs != null ? `${scoreLatencyMs}ms` : '—'}</span>
          </div>
          <div className="status-pill neutral-pill">
            <span className="mini-label">Coach</span>
            <span>{coachLatencyMs != null ? `${coachLatencyMs}ms` : '—'}</span>
          </div>
        </div>
      </header>

      <div className={healthClass}>{healthMessage}</div>
      {offlineMode ? (
        <div className="offline-controls">
          <button className="scenario-btn" onClick={reconnectBackend}>Retry Backend Connection</button>
        </div>
      ) : null}

      {/* ── Dashboard grid ── */}
      <main className="dashboard">
        {/* Left column */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
          <ScoreGauge   score={score} />
          <div className="card">
            <div className="card-title">Demo Scenarios</div>
            <div className="scenario-row">
              <button className="scenario-btn scenario-green" onClick={() => runScenario('green')}>Green</button>
              <button className="scenario-btn scenario-yellow" onClick={() => runScenario('yellow')}>Yellow</button>
              <button className="scenario-btn scenario-red" onClick={() => runScenario('red')}>Red</button>
            </div>
            <p className="scenario-note">Use these presets to showcase severity transitions and coaching updates on demand.</p>
          </div>
          <CoachingPanel
            tips={tips}
            loading={tipsLoading}
            message={coachMessage}
            severity={coachSeverity}
            source={coachSource}
            fallback={coachFallback}
            debugReason={coachDebugReason}
            warning={coachWarning}
          />
        </div>

        {/* Right column */}
        <div className="right-col">
          <BehaviourVisualiser features={features} />
          <TrendChart          history={scoreHistory} />
          <FeatureTable        features={features} predictedFuelRate={predictedFuelRate} />
        </div>
      </main>
    </>
  )
}
