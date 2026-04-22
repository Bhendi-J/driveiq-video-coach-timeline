export default function CoachingPanel({ tips, loading, message, severity, source, fallback, debugReason, warning, topIssue }) {
  const sev = ['green', 'yellow', 'red'].includes(severity) ? severity : 'yellow'
  const coachSource = source || 'rules'
  const issueLabel = topIssue ? String(topIssue).replaceAll('_', ' ') : null

  return (
    <div className="card">
      <div className="coach-head">
        <div className="card-title" style={{ marginBottom: 0 }}>Coaching Tips</div>
        <span className={`severity-badge severity-${sev}`}>{sev.toUpperCase()}</span>
      </div>
      <div className="coach-meta-row">
        <span className="meta-chip">source: {coachSource}</span>
        {fallback ? <span className="meta-chip meta-chip-warn">fallback</span> : null}
      </div>
      {issueLabel ? <div className="issue-pill">Issue: {issueLabel}</div> : null}
      {warning ? <p className="coach-warning">{warning}</p> : null}
      <p className="coach-message">{message || 'Keep your driving smooth and consistent.'}</p>
      {loading ? (
        <p className="loading-tips">Generating personalised tips…</p>
      ) : (
        <ul className="tip-list">
          {(tips || []).map((tip, i) => (
            <li key={i} className="tip-item">
              <span className="tip-num">{i + 1}</span>
              <span className="tip-text">{tip}</span>
            </li>
          ))}
        </ul>
      )}
      {debugReason ? (
        <details className="debug-box">
          <summary>Diagnostics</summary>
          <p>{debugReason}</p>
        </details>
      ) : null}
    </div>
  )
}
