const FLAG_LABEL = { 0: '✅ None', 1: '⚠️ Yes' }

function fmt(val, digits = 2) {
  const n = Number(val)
  return Number.isFinite(n) ? n.toFixed(digits) : '—'
}

export default function FeatureTable({ features }) {
  const rows = [
    { label: 'RPM',             value: fmt(features?.rpm ?? 0, 0) },
    { label: 'Throttle Position', value: `${fmt(features?.throttle_position ?? 0, 1)}%` },
    { label: 'Braking Flag',    value: FLAG_LABEL[Number(features?.braking_flag ?? 0)] ?? '✅ None' },
    { label: 'Lane Change Flag', value: FLAG_LABEL[Number(features?.lane_change_flag ?? 0)] ?? '✅ None' },
    { label: 'Proximity Score', value: fmt(features?.proximity_score ?? 0, 4) },
    { label: 'Mean Flow',       value: fmt(features?.mean_flow ?? 0, 4) },
    { label: 'Flow Variance',   value: fmt(features?.flow_variance ?? 0, 4) },
  ]

  return (
    <div className="card">
      <div className="card-title">Feature Snapshot</div>
      <table className="feat-table">
        <thead>
          <tr>
            <th>Metric</th>
            <th style={{ textAlign: 'right' }}>Value</th>
          </tr>
        </thead>
        <tbody>
          {rows.map(({ label, value }) => (
            <tr key={label}>
              <td style={{ color: '#94a3b8' }}>{label}</td>
              <td>{value}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
