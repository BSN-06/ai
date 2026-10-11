import { useState, useEffect } from 'react'
import {
  listObservations, runCorrelation, autoGenerateEvents,
  listEvents, getCorrelationHistory, updateEventReview,
} from '../utils/api'
import { formatTs, priorityColour, corrColour, sensorIcon, fmtConf, trunc } from '../utils/format'
import type { Observation, Event, CorrelationDecision } from '../types'

export default function SensorFusion() {
  const [observations, setObservations] = useState<Observation[]>([])
  const [events, setEvents] = useState<Event[]>([])
  const [correlations, setCorrelations] = useState<CorrelationDecision[]>([])
  const [selected, setSelected] = useState<Set<string>>(new Set())
  const [selectedEvent, setSelectedEvent] = useState<Event | null>(null)
  const [loading, setLoading] = useState(false)
  const [message, setMessage] = useState<string | null>(null)
  const [reviewNotes, setReviewNotes] = useState('')

  useEffect(() => { refresh() }, [])

  const refresh = async () => {
    const [o, e, c] = await Promise.all([
      listObservations(undefined, 100),
      listEvents(undefined, undefined, 50),
      getCorrelationHistory(50),
    ])
    setObservations(o.observations)
    setEvents(e.events)
    setCorrelations(c.decisions ?? [])
  }

  const toggle = (id: string) => {
    setSelected(prev => {
      const n = new Set(prev)
      n.has(id) ? n.delete(id) : n.add(id)
      return n
    })
  }

  const runCorr = async () => {
    setLoading(true); setMessage(null)
    try {
      const ids = selected.size > 0 ? Array.from(selected) : []
      await runCorrelation(ids)
      setMessage('Correlation complete.')
      await refresh()
    } catch { setMessage('Correlation failed.') }
    finally { setLoading(false) }
  }

  const generateEvents = async () => {
    setLoading(true); setMessage(null)
    try {
      const ids = selected.size > 0 ? Array.from(selected) : []
      const r = await autoGenerateEvents(ids)
      setMessage(`Event created — priority: ${r.review_priority}`)
      await refresh()
    } catch { setMessage('Event generation failed.') }
    finally { setLoading(false) }
  }

  const submitReview = async (status: string) => {
    if (!selectedEvent) return
    await updateEventReview(selectedEvent.id, status, reviewNotes || undefined)
    setMessage(`Event marked as ${status}`)
    setSelectedEvent(null); setReviewNotes('')
    await refresh()
  }

  return (
    <div style={s.page}>
      <h1 style={s.title}>🔗 Sensor Fusion & Event Prioritization</h1>
      <p style={s.sub}>
        Rule-based, uncertainty-aware correlation pipeline — NOT a trained fusion model.
        Select observations below, run correlation, then generate prioritized review events.
      </p>

      <div style={s.infoBox}>
        <strong>Correlation rules:</strong> Both timestamp AND geographic coordinates are required.
        Pixel bounding boxes are not treated as geographic coordinates. Missing metadata returns
        INSUFFICIENT_METADATA — the system never guesses.
        Priority is a <em>human review recommendation</em>, not a determination of hostile intent.
      </div>

      <div style={s.row}>
        {/* Observations selector */}
        <div style={{ ...s.card, flex: 2 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
            <h3 style={s.cardTitle}>Observations ({observations.length})</h3>
            <div style={{ display: 'flex', gap: 8 }}>
              <button style={s.btnSm} onClick={() => setSelected(new Set(observations.map(o => o.id)))}>Select All</button>
              <button style={s.btnSm} onClick={() => setSelected(new Set())}>Clear</button>
              <button style={{ ...s.btn, opacity: loading ? 0.5 : 1 }} disabled={loading} onClick={runCorr}>
                {loading ? '⏳' : '🔗 Run Correlation'}
              </button>
              <button style={{ ...s.btn, background: '#7c3aed', opacity: loading ? 0.5 : 1 }} disabled={loading} onClick={generateEvents}>
                ⚡ Generate Event
              </button>
            </div>
          </div>
          {message && <div style={s.msgBox}>{message}</div>}
          {observations.length === 0
            ? <p style={s.noData}>No observations. Run inference or synthetic scenarios first.</p>
            : (
              <div style={{ maxHeight: 320, overflowY: 'auto' }}>
                <table style={s.table}>
                  <thead><tr>
                    <th style={s.th}></th>
                    {['Sensor', 'Class', 'Conf', 'Lat', 'Lon', 'Timestamp', 'Provenance'].map(h => <th key={h} style={s.th}>{h}</th>)}
                  </tr></thead>
                  <tbody>
                    {observations.map(o => (
                      <tr key={o.id} style={{ background: selected.has(o.id) ? '#1e3a5f' : undefined, cursor: 'pointer' }}
                        onClick={() => toggle(o.id)}>
                        <td style={s.td}><input type="checkbox" checked={selected.has(o.id)} onChange={() => toggle(o.id)} /></td>
                        <td style={s.td}>{sensorIcon(o.sensor_type)} {o.sensor_type}</td>
                        <td style={s.td}>{o.object_class ?? '—'}</td>
                        <td style={s.td}>{fmtConf(o.confidence)}</td>
                        <td style={s.td}>{o.latitude?.toFixed(4) ?? <span style={{ color: '#ef4444' }}>missing</span>}</td>
                        <td style={s.td}>{o.longitude?.toFixed(4) ?? <span style={{ color: '#ef4444' }}>missing</span>}</td>
                        <td style={s.td}>{o.observation_timestamp ? formatTs(o.observation_timestamp) : <span style={{ color: '#ef4444' }}>missing</span>}</td>
                        <td style={s.td}>
                          <span style={{ color: o.data_provenance === 'synthetic' ? '#a78bfa' : '#60a5fa', fontSize: 11 }}>{o.data_provenance}</span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
        </div>
      </div>

      {/* Correlations */}
      <div style={s.card}>
        <h3 style={s.cardTitle}>Correlation Decisions ({correlations.length})</h3>
        {correlations.length === 0
          ? <p style={s.noData}>No correlation decisions yet — select observations and run correlation.</p>
          : (
            <div style={{ maxHeight: 260, overflowY: 'auto' }}>
              <table style={s.table}>
                <thead><tr>{['Status', 'Obs IDs', 'Temporal Δ', 'Spatial Δ', 'Missing Metadata', 'Reason'].map(h => <th key={h} style={s.th}>{h}</th>)}</tr></thead>
                <tbody>
                  {correlations.map(c => (
                    <tr key={c.id}>
                      <td style={s.td}>
                        <span style={{ color: corrColour(c.decision_status), fontWeight: 600, fontSize: 12 }}>{c.decision_status}</span>
                      </td>
                      <td style={{ ...s.td, fontFamily: 'monospace', fontSize: 11 }}>
                        {(c.observation_ids ?? []).map(id => id.slice(0, 8)).join(', ')}
                      </td>
                      <td style={s.td}>{c.temporal_delta_s != null ? `${c.temporal_delta_s.toFixed(0)}s` : '—'}</td>
                      <td style={s.td}>{c.spatial_distance_m != null ? `${c.spatial_distance_m.toFixed(0)}m` : '—'}</td>
                      <td style={s.td}>
                        {(c.missing_metadata ?? []).length > 0
                          ? <span style={{ color: '#fbbf24', fontSize: 11 }}>{(c.missing_metadata ?? []).join(', ')}</span>
                          : <span style={{ color: '#22c55e', fontSize: 11 }}>none</span>}
                      </td>
                      <td style={{ ...s.td, fontSize: 11, maxWidth: 260 }}>{trunc(c.reason, 80)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
      </div>

      {/* Events */}
      <div style={s.card}>
        <h3 style={s.cardTitle}>Prioritized Events ({events.length})</h3>
        {events.length === 0
          ? <p style={s.noData}>No events yet.</p>
          : (
            <div style={{ maxHeight: 300, overflowY: 'auto' }}>
              <table style={s.table}>
                <thead><tr>{['Priority', 'Review Status', 'Sensors', 'Evidence', 'Provenance', 'Time', 'Actions'].map(h => <th key={h} style={s.th}>{h}</th>)}</tr></thead>
                <tbody>
                  {events.map(e => (
                    <tr key={e.id}>
                      <td style={s.td}>
                        <span style={{ color: priorityColour(e.review_priority), fontWeight: 700 }}>{e.review_priority}</span>
                      </td>
                      <td style={s.td}>{e.review_status}</td>
                      <td style={s.td}>{(e.source_sensors ?? []).join(', ') || '—'}</td>
                      <td style={{ ...s.td, fontSize: 11, maxWidth: 240 }}>{trunc(e.evidence_summary, 70)}</td>
                      <td style={s.td}><span style={{ color: e.data_provenance === 'synthetic' ? '#a78bfa' : '#60a5fa', fontSize: 11 }}>{e.data_provenance}</span></td>
                      <td style={s.td}>{formatTs(e.created_at)}</td>
                      <td style={s.td}>
                        <button style={s.btnXs} onClick={() => setSelectedEvent(e)}>🔍 Review</button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
      </div>

      {/* Event review panel */}
      {selectedEvent && (
        <div style={s.reviewPanel}>
          <div style={{ display: 'flex', justifyContent: 'space-between' }}>
            <h3 style={{ ...s.cardTitle, color: priorityColour(selectedEvent.review_priority) }}>
              {selectedEvent.review_priority} PRIORITY EVENT — {selectedEvent.id.slice(0, 12)}
            </h3>
            <button style={s.btnXs} onClick={() => setSelectedEvent(null)}>✕ Close</button>
          </div>

          <div style={s.evidenceGrid}>
            <div><span style={s.evLabel}>Review Status</span><br />{selectedEvent.review_status}</div>
            <div><span style={s.evLabel}>Source Sensors</span><br />{(selectedEvent.source_sensors ?? []).join(', ') || '—'}</div>
            <div><span style={s.evLabel}>Observations</span><br />{(selectedEvent.observation_ids ?? []).length}</div>
            <div><span style={s.evLabel}>Uncertainty</span><br />{selectedEvent.uncertainty?.toFixed(2) ?? '—'}</div>
            <div><span style={s.evLabel}>Provenance</span><br /><span style={{ color: selectedEvent.data_provenance === 'synthetic' ? '#a78bfa' : '#60a5fa' }}>{selectedEvent.data_provenance}</span></div>
          </div>

          <div style={{ marginTop: 12 }}>
            <span style={s.evLabel}>Evidence Summary</span>
            <p style={{ color: '#cbd5e1', fontSize: 13, margin: '4px 0 0', lineHeight: 1.6 }}>{selectedEvent.evidence_summary}</p>
          </div>
          <div style={{ marginTop: 10 }}>
            <span style={s.evLabel}>Priority Explanation</span>
            <p style={{ color: '#94a3b8', fontSize: 12, margin: '4px 0 0', lineHeight: 1.6 }}>{selectedEvent.priority_explanation}</p>
          </div>
          {(selectedEvent.missing_information ?? []).length > 0 && (
            <div style={{ marginTop: 10 }}>
              <span style={s.evLabel}>Missing Information</span>
              <ul style={{ color: '#fbbf24', fontSize: 12, margin: '4px 0 0' }}>
                {(selectedEvent.missing_information ?? []).map((m, i) => <li key={i}>{m}</li>)}
              </ul>
            </div>
          )}

          <div style={{ marginTop: 14 }}>
            <span style={s.evLabel}>Reviewer Notes</span>
            <textarea value={reviewNotes} onChange={e => setReviewNotes(e.target.value)}
              placeholder="Add analyst notes…" style={s.textarea} />
          </div>
          <div style={{ display: 'flex', gap: 8, marginTop: 10, flexWrap: 'wrap' }}>
            {(['REVIEWING', 'REVIEWED', 'DISMISSED'] as const).map(status => (
              <button key={status} style={{ ...s.btn, background: status === 'DISMISSED' ? '#7f1d1d' : status === 'REVIEWED' ? '#166534' : '#1e3a8a' }}
                onClick={() => submitReview(status)}>
                {status}
              </button>
            ))}
          </div>
          <p style={{ color: '#475569', fontSize: 11, marginTop: 8 }}>
            ⚠️ Priority is a <strong>review recommendation</strong> only. Not a determination of hostile
            intent or an autonomous operational decision.
          </p>
        </div>
      )}
    </div>
  )
}

const s: Record<string, React.CSSProperties> = {
  page: { padding: 24, color: '#e2e8f0', maxWidth: 1400 },
  title: { margin: '0 0 6px', fontSize: 22, fontWeight: 700, color: '#f1f5f9' },
  sub: { margin: '0 0 12px', color: '#64748b', fontSize: 13 },
  infoBox: { background: '#0f172a', border: '1px solid #1e3a8a', borderRadius: 8, padding: '10px 14px', color: '#93c5fd', fontSize: 12, marginBottom: 20, lineHeight: 1.6 },
  row: { display: 'flex', gap: 16, marginBottom: 16, flexWrap: 'wrap' },
  card: { background: '#1e293b', border: '1px solid #334155', borderRadius: 10, padding: 20, marginBottom: 16 },
  cardTitle: { margin: '0 0 4px', fontSize: 13, fontWeight: 600, color: '#94a3b8', textTransform: 'uppercase', letterSpacing: 1 },
  btn: { background: '#2563eb', color: '#fff', border: 'none', borderRadius: 7, padding: '8px 14px', fontSize: 13, fontWeight: 600, cursor: 'pointer' },
  btnSm: { background: '#1e293b', color: '#64748b', border: '1px solid #334155', borderRadius: 6, padding: '5px 10px', fontSize: 12, cursor: 'pointer' },
  btnXs: { background: '#1e293b', color: '#94a3b8', border: '1px solid #334155', borderRadius: 5, padding: '4px 8px', fontSize: 11, cursor: 'pointer' },
  table: { width: '100%', borderCollapse: 'collapse' },
  th: { textAlign: 'left', padding: '6px 10px', fontSize: 11, color: '#64748b', borderBottom: '1px solid #334155', textTransform: 'uppercase' },
  td: { padding: '8px 10px', fontSize: 13, color: '#cbd5e1', borderBottom: '1px solid #1e293b' },
  noData: { color: '#475569', fontStyle: 'italic', fontSize: 13 },
  msgBox: { background: '#0f2a1a', border: '1px solid #166534', borderRadius: 6, padding: '8px 12px', color: '#86efac', fontSize: 13, marginBottom: 10 },
  reviewPanel: { background: '#162032', border: '1px solid #1e3a5f', borderRadius: 10, padding: 20, marginBottom: 16 },
  evidenceGrid: { display: 'flex', gap: 24, flexWrap: 'wrap', marginTop: 8 },
  evLabel: { fontSize: 11, color: '#64748b', textTransform: 'uppercase' },
  textarea: { width: '100%', minHeight: 80, background: '#0f172a', border: '1px solid #334155', borderRadius: 6, color: '#e2e8f0', padding: 8, fontSize: 13, resize: 'vertical', boxSizing: 'border-box', marginTop: 4 },
}
