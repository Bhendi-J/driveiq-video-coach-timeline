const FLAG_LABEL = { 0: 'None', 1: 'Detected' }

function fmt(val, digits = 2) {
  const n = Number(val)
  return Number.isFinite(n) ? n.toFixed(digits) : '-'
}

export default function FeatureTable({ features }) {
  const rows = [
    { label: 'RPM',              value: fmt(features?.rpm ?? 0, 0) },
    { label: 'Throttle',         value: `${fmt(features?.throttle_position ?? 0, 1)}%` },
    { label: 'Braking',          value: FLAG_LABEL[Number(features?.braking_flag ?? 0)] ?? 'None',
      flag: Number(features?.braking_flag ?? 0) === 1 },
    { label: 'Lane Change',      value: FLAG_LABEL[Number(features?.lane_change_flag ?? 0)] ?? 'None',
      flag: Number(features?.lane_change_flag ?? 0) === 1 },
    { label: 'Proximity',        value: fmt(features?.proximity_score ?? 0, 4) },
    { label: 'Mean Flow',        value: fmt(features?.mean_flow ?? 0, 4) },
    { label: 'Flow Variance',    value: fmt(features?.flow_variance ?? 0, 4) },
  ]

  return (
    <div className="card">
      <div className="card-title">Feature Snapshot</div>
      <table className="feat-table">
        <tbody>
          {rows.map(({ label, value, flag }) => (
            <tr key={label}>
              <td>{label}</td>
              <td className={flag ? 'feature-flag' : ''}>{value}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
