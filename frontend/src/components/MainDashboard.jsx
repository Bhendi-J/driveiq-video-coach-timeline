import { useState, useEffect } from 'react'
import axios from 'axios'

export default function MainDashboard({ token, offlineMode }) {
  const [metrics, setMetrics] = useState({
    mean_eco_score: 0,
    lowest_eco_score: 0,
    total_trips: 0,
    trips_this_week: 0
  })

  useEffect(() => {
    if (!token || offlineMode) return

    axios.get('/api/dashboard/metrics', {
      headers: { Authorization: `Bearer ${token}` }
    })
    .then(res => {
      setMetrics(res.data)
    })
    .catch(err => {
      console.error('Failed to fetch dashboard metrics', err)
    })
  }, [token, offlineMode])

  // If no token, show placeholder metrics
  const displayMetrics = token ? metrics : {
    mean_eco_score: '—',
    lowest_eco_score: '—',
    total_trips: '—',
    trips_this_week: '—'
  }

  return (
    <section className="stat-strip" id="main-dashboard">
      <article className="card stat-card">
        <span className="card-title">Mean Eco Score</span>
        <strong className="card-value">{displayMetrics.mean_eco_score}</strong>
        <span className="card-sub">Overall account average</span>
      </article>
      <article className="card stat-card">
        <span className="card-title">Lowest Score</span>
        <strong className="card-value">{displayMetrics.lowest_eco_score}</strong>
        <span className="card-sub">Lowest recorded session</span>
      </article>
      <article className="card stat-card">
        <span className="card-title">Total Trips</span>
        <strong className="card-value">{displayMetrics.total_trips}</strong>
        <span className="card-sub">Lifetime drive sessions</span>
      </article>
      <article className="card stat-card">
        <span className="card-title">Trips This Week</span>
        <strong className="card-value">{displayMetrics.trips_this_week}</strong>
        <span className="card-sub">Last 7 days</span>
      </article>
    </section>
  )
}
