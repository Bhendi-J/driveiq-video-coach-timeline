const FLAG_LABEL = { 0: '✅ None', 1: '⚠️ Yes' }

function fmt(val, digits = 2) {
  const n = Number(val)
  return Number.isFinite(n) ? n.toFixed(digits) : '—'
}

export default function FeatureTable({ features, predictedFuelRate }) {
  const rows = [
    { label: 'Speed',           value: features?.speed != null ? `${fmt(features.speed, 1)} km/h` : '—' },
    { label: 'RPM',             value: features?.rpm != null ? fmt(features.rpm, 0) : '—' },
    { label: 'Throttle',        value: features?.throttle_position != null ? `${fmt(features.throttle_position, 1)}%` : '—' },
    { label: 'Gear',            value: features?.gear ?? '—' },
    { label: 'Vehicles Nearby', value: features?.vehicle_count ?? '—' },
    { label: 'Proximity',       value: features?.proximity_score != null ? `${fmt(features.proximity_score * 100, 1)}%` : '—' },
    { label: 'Pedestrian',      value: FLAG_LABEL[features?.pedestrian_flag] ?? '—' },
    { label: 'Braking',         value: FLAG_LABEL[features?.braking_flag]    ?? '—' },
    { label: 'Lane Change',     value: FLAG_LABEL[features?.lane_change_flag] ?? '—' },
    { label: 'Road Type ID',    value: features?.road_type_id ?? '—' },
    { label: 'Weather ID',      value: features?.weather_id   ?? '—' },
    { label: 'Pred. Fuel Rate', value: predictedFuelRate != null ? `${predictedFuelRate.toFixed(2)} L/100` : '—' },
  ]

  return (
    <div className="card">
      <div className="card-title">📋 Live Telemetry</div>
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
