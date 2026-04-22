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

const referenceLinesPlugin = {
  id: 'referenceLines',
  afterDraw(chart, _args, pluginOptions) {
    const yScale = chart.scales?.y
    const chartArea = chart.chartArea
    if (!yScale || !chartArea) return

    const lines = pluginOptions?.lines || []
    const { ctx } = chart
    ctx.save()
    ctx.font = '11px Space Grotesk, sans-serif'
    lines.forEach((line) => {
      const y = yScale.getPixelForValue(line.value)
      ctx.strokeStyle = line.color
      ctx.setLineDash([6, 6])
      ctx.lineWidth = 1
      ctx.beginPath()
      ctx.moveTo(chartArea.left, y)
      ctx.lineTo(chartArea.right, y)
      ctx.stroke()

      ctx.setLineDash([])
      ctx.fillStyle = line.color
      ctx.textAlign = 'right'
      ctx.textBaseline = 'bottom'
      ctx.fillText(line.label, chartArea.right - 4, y - 4)
    })
    ctx.restore()
  },
}

ChartJS.register(referenceLinesPlugin)

function formatTime(sec) {
  const total = Math.max(0, Math.floor(Number(sec) || 0))
  const m = Math.floor(total / 60)
  const s = total % 60
  return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`
}

function severityPointColor(severity) {
  if (severity === 'green') return '#10b981'
  if (severity === 'yellow') return '#f59e0b'
  return '#ef4444'
}

export default function TrendChart({ points = [], emptyMessage = 'Upload a clip to see score trend' }) {
  if (!points.length) {
    return (
      <div className="card">
        <div className="card-title">Score Trend</div>
        <div className="trend-empty">{emptyMessage}</div>
      </div>
    )
  }

  const labels = points.map((p) => formatTime(p.start_sec ?? p.timestamp_sec))

  const data = {
    labels,
    datasets: [{
      label: 'Segment Avg Eco Score',
      data: points.map((p) => Number(p.avg_score ?? p.score ?? 0)),
      borderColor: '#3b82f6',
      backgroundColor: 'rgba(59,130,246,0.08)',
      fill: true,
      tension: 0.4,
      pointRadius: 3,
      pointBackgroundColor: points.map((p) => severityPointColor(p.severity)),
      pointBorderColor: points.map((p) => severityPointColor(p.severity)),
      pointHoverRadius: 5,
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
    plugins: {
      legend: { display: false },
      tooltip: { mode: 'index', intersect: false },
      referenceLines: {
        lines: [
          { value: 65, label: 'Good', color: '#10b981' },
          { value: 40, label: 'Poor', color: '#ef4444' },
        ],
      },
    },
    animation: { duration: 300 },
  }

  return (
    <div className="card">
      <div className="card-title">Score Trend</div>
      <div className="trend-wrap">
        <Line data={data} options={options} />
      </div>
    </div>
  )
}
