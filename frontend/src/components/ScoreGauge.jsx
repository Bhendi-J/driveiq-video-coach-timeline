import { Doughnut } from 'react-chartjs-2'
import { Chart as ChartJS, ArcElement, Tooltip } from 'chart.js'

ChartJS.register(ArcElement, Tooltip)

const scoreColor = (s) =>
  s >= 80 ? '#10b981' : s >= 50 ? '#f59e0b' : '#ef4444'

export default function ScoreGauge({ score }) {
  const s = Math.round(score ?? 0)
  const color = scoreColor(s)

  const data = {
    datasets: [{
      data: [s, 100 - s],
      backgroundColor: [color, 'rgba(255,255,255,0.05)'],
      borderWidth: 0,
      borderRadius: 6,
      circumference: 270,
      rotation: -135,
    }]
  }

  const options = {
    cutout: '80%',
    plugins: { tooltip: { enabled: false } },
    animation: { animateRotate: true, duration: 600 },
    responsive: true,
    maintainAspectRatio: false,
  }

  return (
    <div className="card">
      <div className="card-title">Eco Score</div>
      <div className="gauge-wrap">
        <Doughnut data={data} options={options} />
        <div className="gauge-center">
          <span className="gauge-score" style={{ color }}>{s}</span>
          <span className="gauge-label">/ 100</span>
        </div>
      </div>
    </div>
  )
}
