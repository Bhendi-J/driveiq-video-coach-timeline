import { useState, useEffect } from 'react'
import axios from 'axios'
import TrendChart from './TrendChart'

export default function MainDashboard({ token, offlineMode }) {
  const [metrics, setMetrics] = useState({
    mean_eco_score: 0,
    lowest_eco_score: 0,
    total_trips: 0,
    trips_this_week: 0
  })
  
  const [tripHistory, setTripHistory] = useState([])
  const [selectedTrip, setSelectedTrip] = useState(null)
  const [selectedTripTimeline, setSelectedTripTimeline] = useState([])
  const [selectedTripCoach, setSelectedTripCoach] = useState(null)

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

    axios.get('/api/trips/history', {
      headers: { Authorization: `Bearer ${token}` }
    })
    .then(res => {
      setTripHistory(res.data)
    })
    .catch(err => {
      console.error('Failed to fetch trip history', err)
    })
  }, [token, offlineMode])

  // If no token, show placeholder metrics
  const displayMetrics = token ? metrics : {
    mean_eco_score: '—',
    lowest_eco_score: '—',
    total_trips: '—',
    trips_this_week: '—'
  }

  const loadTripDetails = (sessionId) => {
    const trip = tripHistory.find(t => t.session_id === sessionId)
    if (trip) {
      setSelectedTrip(trip)
      setSelectedTripTimeline([])
      setSelectedTripCoach(null)
      
      axios.get(`/api/trips/${sessionId}/timeline`, {
        headers: { Authorization: `Bearer ${token}` }
      })
      .then(res => {
        const timeline = res.data
        setSelectedTripTimeline(timeline)
        
        if (timeline.length > 0) {
          let totalVehicles = 0
          let totalPedestrians = 0
          
          timeline.forEach(f => {
            const feats = f.features || {}
            totalVehicles += Number(feats.vehicle_density || feats.vehicle_count || 0)
            if (Number(feats.pedestrian_ratio || feats.pedestrian_flag || 0) > 0) {
              totalPedestrians += 1
            }
          })
          
          const avgVehicles = totalVehicles / timeline.length
          const aggFeatures = {
            vehicle_density: avgVehicles,
            pedestrian_flag: totalPedestrians > 0 ? 1 : 0
          }
          
          axios.post('/api/coach', {
            score: trip.final_score,
            features: aggFeatures,
            events: [trip.top_event],
            session_id: sessionId,
            is_summary: true
          }, {
            headers: { Authorization: `Bearer ${token}` }
          })
          .then(coachRes => {
            const tips = coachRes.data.tips || []
            setSelectedTripCoach({
              tips: tips,
              pedestrians: totalPedestrians,
              avgVehicles: avgVehicles
            })
          })
          .catch(err => console.error('Coach fetch error', err))
        }
      })
      .catch(err => {
        console.error('Failed to fetch trip timeline', err)
      })
    }
  }

  return (
    <>
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

      {token && !offlineMode && (
        <section className="card" style={{ marginTop: '16px' }}>
          <div className="card-title">Trip History</div>
          <div style={{ overflowX: 'auto', marginTop: '16px' }}>
            <table className="feat-table">
              <thead>
                <tr>
                  <th style={{ textAlign: 'left', paddingBottom: '8px' }}>Date</th>
                  <th style={{ textAlign: 'left', paddingBottom: '8px' }}>Score</th>
                  <th style={{ textAlign: 'left', paddingBottom: '8px' }}>Top Event</th>
                  <th style={{ textAlign: 'left', paddingBottom: '8px' }}>Duration (frames)</th>
                  <th style={{ textAlign: 'left', paddingBottom: '8px' }}>Actions</th>
                </tr>
              </thead>
              <tbody>
                {tripHistory.map(trip => (
                  <tr key={trip.session_id}>
                    <td>{new Date(trip.date).toLocaleDateString()}</td>
                    <td>{typeof trip.final_score === 'number' ? trip.final_score.toFixed(1) : trip.final_score}</td>
                    <td style={{ textTransform: 'capitalize' }}>{trip.top_event.replace(/_/g, ' ')}</td>
                    <td>{trip.frame_count}</td>
                    <td>
                      <button className="btn btn-ghost" style={{ padding: '4px 8px' }} onClick={() => loadTripDetails(trip.session_id)}>View</button>
                    </td>
                  </tr>
                ))}
                {tripHistory.length === 0 && (
                  <tr>
                    <td colSpan="5" style={{ textAlign: 'center', padding: '16px', color: 'var(--c-white-46)' }}>No trips recorded yet.</td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </section>
      )}

      {selectedTrip && (
        <div className="modal-backdrop" onClick={() => setSelectedTrip(null)}>
          <div className="modal" onClick={(e) => e.stopPropagation()} style={{ width: '600px', maxWidth: 'calc(100vw - 32px)' }}>
            <div className="modal-header">
              <h3 className="modal-title">Trip Details</h3>
              <button className="modal-close" onClick={() => setSelectedTrip(null)}>&times;</button>
            </div>
            <div className="form-group" style={{ gap: '12px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid var(--c-white-08)', paddingBottom: '8px' }}>
                <span className="label" style={{ marginBottom: 0 }}>Trip ID</span>
                <span style={{ fontSize: '12px', color: 'var(--c-white-72)' }}>{selectedTrip.session_id}</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid var(--c-white-08)', paddingBottom: '8px' }}>
                <span className="label" style={{ marginBottom: 0 }}>Date</span>
                <span style={{ fontSize: '12px', color: 'var(--c-white-72)' }}>{new Date(selectedTrip.date).toLocaleString()}</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid var(--c-white-08)', paddingBottom: '8px' }}>
                <span className="label" style={{ marginBottom: 0 }}>Final Score</span>
                <span style={{ fontSize: '14px', fontWeight: 600, color: 'var(--c-white-92)' }}>{typeof selectedTrip.final_score === 'number' ? selectedTrip.final_score.toFixed(1) : selectedTrip.final_score}</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid var(--c-white-08)', paddingBottom: '8px' }}>
                <span className="label" style={{ marginBottom: 0 }}>Duration</span>
                <span style={{ fontSize: '12px', color: 'var(--c-white-72)' }}>{selectedTrip.frame_count} frames</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span className="label" style={{ marginBottom: 0 }}>Dominant Event</span>
                <span style={{ fontSize: '12px', color: 'var(--c-white-72)', textTransform: 'capitalize' }}>{selectedTrip.top_event.replace(/_/g, ' ')}</span>
              </div>
              
              {selectedTripCoach && (
                <div style={{ marginTop: '16px', borderTop: '1px solid var(--c-white-08)', paddingTop: '16px' }}>
                  <div style={{ fontSize: '11px', textTransform: 'uppercase', color: 'var(--c-white-46)', marginBottom: '12px', letterSpacing: '0.06em' }}>CV Feature Snapshot</div>
                  
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
                    <span style={{ fontSize: '12px', color: 'var(--c-white-72)' }}>Pedestrian Frames Detected</span>
                    <span style={{ fontSize: '12px', color: 'var(--c-white-92)' }}>{selectedTripCoach.pedestrians}</span>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '16px' }}>
                    <span style={{ fontSize: '12px', color: 'var(--c-white-72)' }}>Avg Vehicle Density</span>
                    <span style={{ fontSize: '12px', color: 'var(--c-white-92)' }}>{selectedTripCoach.avgVehicles.toFixed(1)}</span>
                  </div>

                  <div style={{ fontSize: '11px', textTransform: 'uppercase', color: 'var(--c-white-46)', marginBottom: '12px', letterSpacing: '0.06em' }}>AI Coaching Insights</div>
                  <ul style={{ margin: 0, paddingLeft: '16px', fontSize: '12px', color: 'var(--c-white-92)' }}>
                    {selectedTripCoach.tips.map((tip, idx) => (
                      <li key={idx} style={{ marginBottom: '6px' }}>{tip}</li>
                    ))}
                  </ul>
                </div>
              )}
              
              {selectedTripTimeline.length > 0 ? (
                <div style={{ marginTop: '16px', borderTop: '1px solid var(--c-white-08)', paddingTop: '16px' }}>
                  <TrendChart points={selectedTripTimeline} emptyMessage="No timeline data available" />
                </div>
              ) : (
                <div style={{ marginTop: '16px', textAlign: 'center', color: 'var(--c-white-46)', fontSize: '12px' }}>
                  Loading timeline graph...
                </div>
              )}
              
              <div className="modal-footer" style={{ marginTop: '16px' }}>
                <button className="btn btn-primary btn-full" onClick={() => setSelectedTrip(null)}>Close</button>
              </div>
            </div>
          </div>
        </div>
      )}
    </>
  )
}
