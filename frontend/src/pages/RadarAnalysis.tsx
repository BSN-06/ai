import { useState, useRef } from 'react'
import { runRadarImport, getRadarTracks, resetRadarTracks } from '../utils/api'
import { formatTs } from '../utils/format'
import type { RadarTrack, RadarProcessResult } from '../types'
import {
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, ScatterChart, Scatter, CartesianGrid, Legend,
} from 'recharts'

export default function RadarAnalysis() {
  const [file, setFile] = useState<File | null>(null)
  const [result, setResult] = useState<RadarProcessResult | null>(null)
  const [tracks, setTracks] = useState<RadarTrack[]>([])
  const [selectedTrack, setSelectedTrack] = useState<RadarTrack | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const fileRef = useRef<HTMLInputElement>(null)

  const handleFile = (f: File) => { setFile(f); setResult(null); setError(null); setSelectedTrack(null) }

  const runImport = async () => {
    if (!file) return
    setLoading(true); setError(null)
    try {
      const r = await runRadarImport(file)
      setResult(r)
      setTracks(r.tracks ?? [])
    } catch (e: unknown) {
      const msg = (e as { response?: { data?: { detail?: string } } })?.response?.data?.detail
      setError(msg ?? 'Radar import failed — check file format.')
    } finally { setLoading(false) }
  }

  const loadTracks = async () => {
    try {
      const r = await getRadarTracks()
      setTracks(r.tracks ?? [])
    } catch { setError('Could not load tracks') }
  }

  const handleReset = async () => {
    await resetRadarTracks()
    setTracks([]); setResult(null); setSelectedTrack(null)
  }

  // Build scatter data for all tracks
  const scatterData = tracks.map(t => ({
    name: t.track_id,
    data: t.history.map(h => ({ x: h.estimated_x, y: h.estimated_y, measured: h.measured })),
  }))

  const TRACK_COLOURS = ['#3b82f6','#10b981','#f59e0b','#ef4444','#8b5cf6','#ec4899','#06b6d4']

  return (
    <div style={s.page}>
      <h1 style={s.title}>📡 Radar Track Analysis</h1>
      <p style={s.sub}>
        Import a CSV or JSON file containing radar detections (columns: timestamp, x, y).
        A Kalman filter tracks targets using a constant-velocity motion model.
        Domain: maritime radar data — NOT aircraft radar.
      </p>

      <div style={s.domainNote}>
        📌 <strong>Domain:</strong> This tracker processes structured radar detections. It is designed
        for maritime radar data (e.g., DLR Marine Radar dataset). It does NOT claim aircraft tracking
        capability. State vector: [x, y, vx, vy] — constant velocity model.
      </div>

      {/* Upload */}
      <div style={s.row}>
        <div style={s.card}>
          <h3 style={s.cardTitle}>Upload Radar Data</h3>
          <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', alignItems: 'center' }}>
            <button style={s.btnSecondary} onClick={() => fileRef.current?.click()}>
              📂 Choose File
            </button>
            <input ref={fileRef} type="file" accept=".csv,.json" style={{ display: 'none' }}
              onChange={e => { if (e.target.files?.[0]) handleFile(e.target.files[0]) }} />
            {file && <span style={{ color: '#94a3b8', fontSize: 13 }}>📄 {file.name}</span>}
            <button style={{ ...s.btn, opacity: (!file || loading) ? 0.5 : 1 }}
              disabled={!file || loading} onClick={runImport}>
              {loading ? '⏳ Processing…' : '▶ Import & Track'}
            </button>
            <button style={s.btnDanger} onClick={loadTracks}>🔄 Reload Tracks</button>
            <button style={s.btnDanger} onClick={handleReset}>🗑️ Reset Tracks</button>
          </div>

          <div style={s.formatNote}>
            <strong>Expected CSV format:</strong> <code>timestamp,x,y</code> (also accepts: time/datetime,
            easting/longitude, northing/latitude, plus optional id/track_id/mmsi columns).<br />
            <strong>JSON format:</strong> list or <code>{`{"detections":[...]}`}</code> with timestamp, x, y keys.
          </div>

          {error && <div style={s.errBox}>⚠️ {error}</div>}
        </div>

        {result && (
          <div style={s.statsCard}>
            <h3 style={s.cardTitle}>Processing Summary</h3>
            {[
              ['Detections processed', result.total_detections_processed],
              ['Active tracks', result.active_tracks],
              ['Terminated tracks', result.terminated_tracks],
              ['New tracks created', result.new_tracks_created],
              ['Tracks updated', result.tracks_updated],
              ['Processing time', `${result.processing_time_ms?.toFixed(1)} ms`],
            ].map(([k, v]) => (
              <div key={String(k)} style={s.statRow}>
                <span style={{ color: '#64748b', fontSize: 13 }}>{k}</span>
                <strong style={{ color: '#f1f5f9' }}>{String(v)}</strong>
              </div>
            ))}
            <p style={{ color: '#475569', fontSize: 11, marginTop: 8 }}>
              {result.timestamp && `Processed at ${formatTs(result.timestamp)}`}
            </p>
          </div>
        )}
      </div>

      {/* Position plot */}
      {tracks.length > 0 && (
        <div style={s.card}>
          <h3 style={s.cardTitle}>Track Position Plot (Estimated Positions)</h3>
          <p style={{ color: '#64748b', fontSize: 12, margin: '0 0 12px' }}>
            Solid points = estimated positions (Kalman filter output). Tracks are colour-coded.
            Units depend on your input data — no geographic map is shown unless coordinates are validated as WGS84 lat/lon.
          </p>
          <ResponsiveContainer width="100%" height={320}>
            <ScatterChart margin={{ top: 8, right: 16, left: 0, bottom: 8 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis dataKey="x" type="number" name="X" tick={{ fill: '#64748b', fontSize: 11 }} label={{ value: 'X (units depend on data)', fill: '#475569', fontSize: 11, position: 'insideBottom', offset: -4 }} />
              <YAxis dataKey="y" type="number" name="Y" tick={{ fill: '#64748b', fontSize: 11 }} />
              <Tooltip cursor={{ strokeDasharray: '3 3' }} contentStyle={{ background: '#0f172a', border: '1px solid #334155', color: '#e2e8f0' }} />
              <Legend wrapperStyle={{ fontSize: 12, color: '#94a3b8' }} />
              {scatterData.map((t, i) => (
                <Scatter key={t.name} name={t.name} data={t.data}
                  fill={TRACK_COLOURS[i % TRACK_COLOURS.length]} />
              ))}
            </ScatterChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Track list */}
      {tracks.length > 0 && (
        <div style={s.card}>
          <h3 style={s.cardTitle}>Tracks ({tracks.length})</h3>
          <table style={s.table}>
            <thead><tr>{['Track ID', 'Status', 'Observations', 'Duration (s)', 'Pos (x,y)', 'Vel (vx,vy)', 'Uncertainty', 'Created'].map(h =>
              <th key={h} style={s.th}>{h}</th>)}</tr></thead>
            <tbody>
              {tracks.map(t => (
                <tr key={t.track_id} style={{ cursor: 'pointer', background: selectedTrack?.track_id === t.track_id ? '#1e3a5f' : undefined }}
                  onClick={() => setSelectedTrack(t)}>
                  <td style={s.td}><code style={{ color: '#60a5fa' }}>{t.track_id}</code></td>
                  <td style={s.td}>
                    <span style={{ color: t.status === 'ACTIVE' ? '#22c55e' : '#94a3b8', fontWeight: 600 }}>{t.status}</span>
                  </td>
                  <td style={s.td}>{t.num_observations}</td>
                  <td style={s.td}>{t.duration_s?.toFixed(1) ?? '—'}</td>
                  <td style={{ ...s.td, fontFamily: 'monospace', fontSize: 12 }}>
                    {t.current_position.x.toFixed(2)}, {t.current_position.y.toFixed(2)}
                  </td>
                  <td style={{ ...s.td, fontFamily: 'monospace', fontSize: 12 }}>
                    {t.current_velocity.x.toFixed(2)}, {t.current_velocity.y.toFixed(2)}
                  </td>
                  <td style={s.td}>{t.uncertainty?.toFixed(2)}</td>
                  <td style={s.td}>{formatTs(t.created_at)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Selected track history */}
      {selectedTrack && (
        <div style={s.card}>
          <h3 style={s.cardTitle}>Track History — {selectedTrack.track_id}</h3>
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={selectedTrack.history} margin={{ top: 8, right: 16, left: 0, bottom: 0 }}>
              <XAxis dataKey="timestamp" tick={false} />
              <YAxis tick={{ fill: '#64748b', fontSize: 11 }} />
              <Tooltip contentStyle={{ background: '#0f172a', border: '1px solid #334155', color: '#e2e8f0', fontSize: 12 }} />
              <Legend wrapperStyle={{ fontSize: 12, color: '#94a3b8' }} />
              <Line type="monotone" dataKey="estimated_x" stroke="#3b82f6" dot={false} name="Est X" />
              <Line type="monotone" dataKey="estimated_y" stroke="#10b981" dot={false} name="Est Y" />
              <Line type="monotone" dataKey="uncertainty" stroke="#f59e0b" dot={false} name="Uncertainty" strokeDasharray="4 2" />
            </LineChart>
          </ResponsiveContainer>
          <p style={{ color: '#475569', fontSize: 12, marginTop: 8 }}>
            Solid = estimated position (Kalman output). Dashed = position uncertainty.
            Measured positions shown where available. Click another track to compare.
          </p>
        </div>
      )}

      {tracks.length === 0 && !loading && (
        <div style={s.emptyState}>
          <span style={{ fontSize: 40 }}>📡</span>
          <p>No tracks loaded. Upload a CSV or JSON radar detection file to begin.</p>
          <p style={{ fontSize: 12, color: '#475569' }}>
            You can also generate a sample CSV: timestamp,x,y rows with ISO timestamps.
          </p>
        </div>
      )}
    </div>
  )
}

const s: Record<string, React.CSSProperties> = {
  page: { padding: 24, color: '#e2e8f0', maxWidth: 1300 },
  title: { margin: '0 0 6px', fontSize: 22, fontWeight: 700, color: '#f1f5f9' },
  sub: { margin: '0 0 12px', color: '#64748b', fontSize: 13 },
  domainNote: { background: '#0f172a', border: '1px solid #1e3a8a', borderRadius: 8, padding: '10px 14px', color: '#93c5fd', fontSize: 13, marginBottom: 20 },
  row: { display: 'flex', gap: 16, marginBottom: 16, flexWrap: 'wrap' },
  card: { background: '#1e293b', border: '1px solid #334155', borderRadius: 10, padding: 20, marginBottom: 16, flex: 1 },
  statsCard: { background: '#1e293b', border: '1px solid #334155', borderRadius: 10, padding: 20, minWidth: 220 },
  cardTitle: { margin: '0 0 12px', fontSize: 13, fontWeight: 600, color: '#94a3b8', textTransform: 'uppercase', letterSpacing: 1 },
  btn: { background: '#2563eb', color: '#fff', border: 'none', borderRadius: 8, padding: '9px 16px', fontSize: 13, fontWeight: 600, cursor: 'pointer' },
  btnSecondary: { background: '#1e293b', color: '#94a3b8', border: '1px solid #334155', borderRadius: 8, padding: '9px 16px', fontSize: 13, cursor: 'pointer' },
  btnDanger: { background: '#1e293b', color: '#64748b', border: '1px solid #334155', borderRadius: 8, padding: '9px 16px', fontSize: 13, cursor: 'pointer' },
  formatNote: { marginTop: 12, color: '#475569', fontSize: 12, lineHeight: 1.8 },
  errBox: { marginTop: 10, background: '#1c080822', border: '1px solid #7f1d1d', borderRadius: 6, padding: '8px 12px', color: '#f87171', fontSize: 13 },
  statRow: { display: 'flex', justifyContent: 'space-between', padding: '4px 0', borderBottom: '1px solid #0f172a' },
  table: { width: '100%', borderCollapse: 'collapse' },
  th: { textAlign: 'left', padding: '6px 10px', fontSize: 11, color: '#64748b', borderBottom: '1px solid #334155', textTransform: 'uppercase' },
  td: { padding: '8px 10px', fontSize: 13, color: '#cbd5e1', borderBottom: '1px solid #1e293b' },
  emptyState: { textAlign: 'center', color: '#475569', padding: 48 },
}
