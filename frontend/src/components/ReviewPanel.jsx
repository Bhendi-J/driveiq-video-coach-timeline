import { useEffect, useMemo, useRef, useState } from 'react'
import axios from 'axios'

function formatTime(sec) {
  const total = Math.max(0, Math.floor(Number(sec) || 0))
  const m = Math.floor(total / 60)
  const s = total % 60
  return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`
}

function formatRangeTime(sec) {
  const total = Math.max(0, Math.floor(Number(sec) || 0))
  const m = Math.floor(total / 60)
  const s = total % 60
  return `${m}:${String(s).padStart(2, '0')}`
}

function scoreColor(severity) {
  if (severity === 'green') return '#10b981'
  if (severity === 'yellow') return '#f59e0b'
  return '#ef4444'
}

function severityRowBg(severity, isSelected) {
  if (isSelected) return '#1e293b'
  if (severity === 'green') return 'rgba(16,185,129,0.10)'
  if (severity === 'yellow') return 'rgba(245,158,11,0.10)'
  return 'rgba(239,68,68,0.10)'
}

function quantile(values, q) {
  const sorted = [...values].sort((a, b) => a - b)
  if (sorted.length === 0) return 0
  const pos = (sorted.length - 1) * q
  const base = Math.floor(pos)
  const rest = pos - base
  const next = sorted[Math.min(base + 1, sorted.length - 1)]
  return sorted[base] + (next - sorted[base]) * rest
}

function deriveSeverityThresholds(windows = []) {
  const scores = windows
    .map((w) => toNum(w?.score, Number.NaN))
    .filter((v) => Number.isFinite(v))

  if (scores.length < 3) {
    return { yellowMin: 50, greenMin: 75, mode: 'fixed_75_50_fallback' }
  }

  const yellowMin = quantile(scores, 1 / 3)
  const greenMin = quantile(scores, 2 / 3)

  if (!Number.isFinite(yellowMin) || !Number.isFinite(greenMin) || Math.abs(greenMin - yellowMin) < 1e-6) {
    return { yellowMin: 50, greenMin: 75, mode: 'fixed_75_50_fallback' }
  }

  return { yellowMin, greenMin, mode: 'dynamic_tertiles' }
}

function classifySeverity(score, thresholds) {
  const yellowMin = Number(thresholds?.yellowMin ?? 50)
  const greenMin = Number(thresholds?.greenMin ?? 75)
  if (score >= greenMin) return 'green'
  if (score >= yellowMin) return 'yellow'
  return 'red'
}

function toNum(value, fallback = 0) {
  const n = Number(value)
  return Number.isFinite(n) ? n : fallback
}

function mostFrequentValue(items, key, fallback = '') {
  const counts = new Map()
  const firstIndex = new Map()
  items.forEach((item, idx) => {
    const value = String(item?.[key] ?? fallback)
    counts.set(value, (counts.get(value) || 0) + 1)
    if (!firstIndex.has(value)) firstIndex.set(value, idx)
  })

  let best = fallback
  let bestCount = -1
  let bestIdx = Number.MAX_SAFE_INTEGER
  counts.forEach((count, value) => {
    const idx = firstIndex.get(value) ?? Number.MAX_SAFE_INTEGER
    if (count > bestCount || (count === bestCount && idx < bestIdx)) {
      best = value
      bestCount = count
      bestIdx = idx
    }
  })
  return best
}

function buildSegment(windows, thresholds) {
  const first = windows[0]
  const last = windows[windows.length - 1]
  const scoreSum = windows.reduce((acc, w) => acc + toNum(w?.score), 0)
  const avgScore = scoreSum / Math.max(1, windows.length)
  const dominantIssue = mostFrequentValue(windows, 'top_issue', 'smooth_driving')
  const severity = classifySeverity(avgScore, thresholds)
  const worstWindow = windows.reduce(
    (worst, w) => (toNum(w?.score) < toNum(worst?.score, Infinity) ? w : worst),
    windows[0],
  )
  const startSec = toNum(first?.timestamp_sec)
  const endSec = toNum(last?.timestamp_sec)

  const mean = (key, fallback = 0) =>
    windows.reduce((acc, w) => acc + toNum(w?.[key], fallback), 0) / Math.max(1, windows.length)

  return {
    start_sec: startSec,
    end_sec: endSec,
    avg_score: Number(avgScore.toFixed(2)),
    dominant_issue: dominantIssue,
    severity,
    coach_note: worstWindow?.coach_note || 'Maintain smooth, consistent driving.',
    window_count: windows.length,

    // Compatibility fields for existing selected state consumers.
    timestamp_sec: startSec,
    score: Number(avgScore.toFixed(2)),
    top_issue: dominantIssue,
    score_source: mostFrequentValue(windows, 'score_source', 'xgb'),
    braking_flag_ratio: mean('braking_flag_ratio', mean('braking_flag')),
    lane_change_flag_ratio: mean('lane_change_flag_ratio', mean('lane_change_flag')),
    proximity_score_mean: mean('proximity_score_mean', mean('proximity_score')),
    mean_flow_mean: mean('mean_flow_mean', mean('mean_flow')),
    flow_variance: mean('flow_variance'),
    _windows: windows,
  }
}

export function groupSegments(windows = []) {
  if (!Array.isArray(windows) || windows.length === 0) return []
  const ordered = [...windows].sort((a, b) => toNum(a?.timestamp_sec) - toNum(b?.timestamp_sec))
  const thresholds = deriveSeverityThresholds(ordered)

  const segments = []
  let current = [ordered[0]]
  let runningScoreSum = toNum(ordered[0]?.score)

  for (let i = 1; i < ordered.length; i += 1) {
    const prev = ordered[i - 1]
    const win = ordered[i]
    const runningAvg = runningScoreSum / Math.max(1, current.length)

    const issueChanged = String(win?.top_issue ?? '') !== String(prev?.top_issue ?? '')
    const scoreJumped = Math.abs(toNum(win?.score) - runningAvg) > 10 // Relaxed sensitivity to avoid fragmentation
    const timeBucketFull = (toNum(win?.timestamp_sec) - toNum(current[0]?.timestamp_sec)) >= 5 // Max 5 seconds per block

    if (issueChanged || scoreJumped || timeBucketFull) {
      segments.push(buildSegment(current, thresholds))
      current = [win]
      runningScoreSum = toNum(win?.score)
      continue
    }

    current.push(win)
    runningScoreSum += toNum(win?.score)
  }

  if (current.length) segments.push(buildSegment(current, thresholds))
  return segments
}

export default function ReviewPanel({ onAnalysisComplete, onWindowSelect, selectedTimestampSec }) {
  const [file, setFile] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [result, setResult] = useState(null)
  const videoRef = useRef(null)

  const videoUrl = useMemo(() => {
    if (!file) return null
    return URL.createObjectURL(file)
  }, [file])

  useEffect(() => {
    return () => {
      if (videoUrl) URL.revokeObjectURL(videoUrl)
    }
  }, [videoUrl])

  const onFileChange = (e) => {
    const picked = e.target.files?.[0] || null
    setFile(picked)
    setResult(null)
    setError('')
    if (onAnalysisComplete) onAnalysisComplete(null)
  }

  const analyse = async () => {
    if (!file) {
      setError('Please choose an mp4 file first.')
      return
    }
    setLoading(true)
    setError('')
    setResult(null)

    try {
      const form = new FormData()
      form.append('video', file)
      const { data } = await axios.post('/api/review', form, {
        headers: { 'Content-Type': 'multipart/form-data' },
      })
      const segments = groupSegments(data?.windows || [])
      const nextResult = { ...data, segments }
      setResult(nextResult)
      if (onAnalysisComplete) onAnalysisComplete(nextResult)
      if (onWindowSelect && segments.length) onWindowSelect(segments[0])
    } catch (e) {
      const msg = e?.response?.data?.message || e?.response?.data?.error || 'Review failed.'
      setError(msg)
    } finally {
      setLoading(false)
    }
  }

  const seekTo = (windowItem) => {
    if (!videoRef.current) return
    const sec = windowItem?.start_sec ?? windowItem?.timestamp_sec
    videoRef.current.currentTime = Number(sec) || 0
    videoRef.current.play().catch(() => {})
    if (onWindowSelect) onWindowSelect(windowItem)
  }

  const isFlatResult = useMemo(() => {
    const segments = result?.segments || []
    if (segments.length < 2) return false
    const scores = segments.map((s) => Number(s?.avg_score ?? 0))
    const minScore = Math.min(...scores)
    const maxScore = Math.max(...scores)
    return (maxScore - minScore) <= 5
  }, [result])

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
      <div style={{ display: 'flex', gap: 10, alignItems: 'center', flexWrap: 'wrap' }}>
        <input type="file" accept="video/mp4" onChange={onFileChange} />
        <button className="scenario-btn" onClick={analyse} disabled={loading || !file}>
          {loading ? 'Analysing...' : 'Analyse'}
        </button>
        {loading ? <span style={{ color: '#94a3b8', fontSize: 13 }}>Processing video windows...</span> : null}
      </div>

      {error ? <div style={{ color: '#fca5a5', fontSize: 13 }}>{error}</div> : null}

      {result ? (
        <div style={{ display: 'grid', gap: 14, gridTemplateColumns: '1.2fr 1fr' }}>
          <div className="card" style={{ margin: 0 }}>
            <div className="card-title">Uploaded Clip</div>
            {videoUrl ? (
              <video
                ref={videoRef}
                src={videoUrl}
                controls
                style={{ width: '100%', maxHeight: 360, borderRadius: 10, background: '#020617' }}
              />
            ) : null}
            <p className="scenario-note" style={{ marginTop: 10 }}>
              Duration: {formatTime(result.duration_sec)} • Windows: {result.window_count}
            </p>
          </div>

          <div className="card" style={{ margin: 0 }}>
            <div className="card-title">Segment Review</div>
            <div style={{ maxHeight: 360, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: 8 }}>
              {(result.segments || []).map((segment, idx) => {
                const isSelected = Number(selectedTimestampSec) === Number(segment.start_sec)
                const durationSec = Math.max(1, Math.round(Number(segment.end_sec) - Number(segment.start_sec)))
                return (
                <button
                  key={`${segment.start_sec}-${segment.end_sec}-${idx}`}
                  onClick={() => seekTo(segment)}
                  style={{
                    textAlign: 'left',
                    border: '1px solid rgba(148,163,184,0.2)',
                    background: severityRowBg(segment.severity, isSelected),
                    color: '#e2e8f0',
                    borderRadius: 10,
                    padding: '10px 12px',
                    cursor: 'pointer',
                  }}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 8 }}>
                    <strong>
                      {formatRangeTime(segment.start_sec)}
                      {formatRangeTime(segment.start_sec) !== formatRangeTime(segment.end_sec)
                        ? ` - ${formatRangeTime(segment.end_sec)}`
                        : ` - ${formatRangeTime(segment.start_sec + 2)}`}
                    </strong>
                    <span
                      style={{
                        fontSize: 12,
                        padding: '2px 8px',
                        borderRadius: 999,
                        background: scoreColor(segment.severity),
                        color: '#041014',
                        fontWeight: 700,
                      }}
                    >
                      {segment.avg_score}
                    </span>
                  </div>
                  <div style={{ fontSize: 12, color: '#94a3b8', marginTop: 4 }}>
                    Issue: {String(segment.dominant_issue || 'smooth_driving').replaceAll('_', ' ')}
                  </div>
                  <div style={{ fontSize: 12, color: '#94a3b8', marginTop: 2 }}>
                    {durationSec} seconds | {segment.window_count} windows
                  </div>
                  <p className="coach-note">{segment.coach_note}</p>
                </button>
                )
              })}
            </div>
            {isFlatResult ? (
              <p className="timeline-disclaimer">
                Score variation is low for this clip — try a clip with mixed traffic conditions for full timeline spread.
              </p>
            ) : null}
          </div>
        </div>
      ) : null}
    </div>
  )
}
