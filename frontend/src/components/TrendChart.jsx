import { Line } from 'react-chartjs-2'
import {
  Chart as ChartJS,
  LineElement,
  PointElement,
  LinearScale,
  CategoryScale,
  Filler,
  Tooltip,
} from 'chart.js'

ChartJS.register(LineElement, PointElement, LinearScale, CategoryScale, Filler, Tooltip)

export default function TrendChart({ history }) {
  const labels = history.map((_, i) => `T-${history.length - i}`)

  const data = {
    labels,
    datasets: [{
      label: 'Eco Score',
      data: history,
      borderColor: '#3b82f6',
      backgroundColor: 'rgba(59,130,246,0.08)',
      fill: true,
      tension: 0.4,
      pointRadius: 3,
      pointBackgroundColor: '#3b82f6',
    }]
  }

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    scales: {
      y: { min: 0, max: 100, grid: { color: 'rgba(255,255,255,0.05)' },
           ticks: { color: '#64748b', font: { size: 11 } } },
      x: { grid: { display: false }, ticks: { color: '#64748b', font: { size: 10 } } },
    },
    plugins: { legend: { display: false }, tooltip: { mode: 'index', intersect: false } },
    animation: { duration: 300 },
  }

  return (
    <div className="card">
      <div className="card-title">📈 Score Trend</div>
      <div className="trend-wrap">
        <Line data={data} options={options} />
      </div>
    </div>
  )
}
