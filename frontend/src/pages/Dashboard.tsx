import { useHealth } from '../hooks/useHealth'
import { formatTs, priorityColour, statusColour, sensorIcon } from '../utils/format'
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, Legend,
} from 'recharts'

const PRIORITY_COLOURS: Record<string, string> = {
  HIGH: '#ef4444', MEDIUM: '#f59e0b', LOW: '#22c55e',
}

export default function Dashboard() {
  const { health, stats, error } = useHealth(4000)

  if (error) {
    return (
      <div style={styles.errorBox}>
        <span style={{ fontSize: 24 }}>⚠️</span>
        <div>
          <strong>Backend unreachable</strong>
          <p style={{ margin: '4px 0 0', color: '#94a3b8' }}>{error}</p>
          <p style={{ margin: '4px 0 0', color: '#94a3b8', fontSize: 12 }}>
            Start the backend: <code style={styles.code}>python -m uvicorn app.main:app --reload</code>
          </p>
        </div>
      </div>
    )
  }

  if (!health || !stats) {
    return <div style={styles.loading}>Connecting to backend…</div>
  }

  const sensorData = Object.entries(stats.observations.by_sensor).map(([k, v]) => ({
    name: k.charAt(0).toUpperCase() + k.slice(1), value: v,
  }))

  const priorityData = Object.entries(stats.events.by_priority).map(([k, v]) => ({
    name: k, value: v, fill: PRIORITY_COLOURS[k],
  }))

  return (
    <div style={styles.page}>
      {/* Header */}
      <div style={styles.header}>
        <div>
          <h1 style={styles.title}>System Overview</h1>
          <p style={styles.subtitle}>
            AI-Assisted Multi-Sensor Surveillance — Offline Research Prototype
          </p>
        </div>
        <div style={styles.statusPill}>
          <span style={{ ...styles.dot, background: health.status === 'operational' ? '#22c55e' : '#ef4444' }} />
          {health.status.toUpperCase()} · {health.device.toUpperCase()} ·{' '}
          {health.gpu_available ? '🟢 GPU' : '🔵 CPU'}
        </div>
      </div>

      {/* KPI row */}
      <div style={styles.kpiRow}>
        {[
          { label: 'Total Observations', value: stats.observations.total, icon: '👁️' },
          { label: 'Total Events', value: stats.events.total, icon: '📋' },
          { label: 'Pending Review', value: stats.events.pending, icon: '⏳' },
          { label: 'Models Registered', value: stats.models.total, icon: '🧠' },
          { label: 'Datasets', value: stats.datasets.total, icon: '🗄️' },
          { label: 'Models Trained', value: stats.models.trained, icon: '✅' },
        ].map(kpi => (
          <div key={kpi.label} style={styles.kpiCard}>
            <span style={styles.kpiIcon}>{kpi.icon}</span>
            <span style={styles.kpiValue}>{kpi.value}</span>
            <span style={styles.kpiLabel}>{kpi.label}</span>
          </div>
        ))}
      </div>

      {/* Charts row */}
      <div style={styles.chartsRow}>
        <div style={styles.chartCard}>
          <h3 style={styles.cardTitle}>Observations by Sensor</h3>
          {sensorData.every(d => d.value === 0)
            ? <p style={styles.noData}>No observations yet — run inference or synthetic scenarios.</p>
            : (
              <ResponsiveContainer width="100%" height={200}>
                <BarChart data={sensorData} margin={{ top: 8, right: 16, left: -16, bottom: 0 }}>
                  <XAxis dataKey="name" tick={{ fill: '#94a3b8', fontSize: 12 }} />
                  <YAxis tick={{ fill: '#94a3b8', fontSize: 12 }} allowDecimals={false} />
                  <Tooltip contentStyle={styles.tooltip} />
                  <Bar dataKey="value" fill="#3b82f6" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            )}
        </div>

        <div style={styles.chartCard}>
          <h3 style={styles.cardTitle}>Event Priority Distribution</h3>
          {priorityData.every(d => d.value === 0)
            ? <p style={styles.noData}>No events yet — run correlation or synthetic scenarios.</p>
            : (
              <ResponsiveContainer width="100%" height={200}>
                <PieChart>
                  <Pie data={priorityData} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={70} label>
                    {priorityData.map((entry) => (
                      <Cell key={entry.name} fill={entry.fill} />
                    ))}
                  </Pie>
                  <Tooltip contentStyle={styles.tooltip} />
                  <Legend wrapperStyle={{ fontSize: 12, color: '#94a3b8' }} />
                </PieChart>
              </ResponsiveContainer>
            )}
        </div>
      </div>

      {/* Recent rows */}
      <div style={styles.chartsRow}>
        <div style={{ ...styles.chartCard, flex: 1 }}>
          <h3 style={styles.cardTitle}>Recent Observations</h3>
          {stats.recent_observations.length === 0
            ? <p style={styles.noData}>None yet.</p>
            : (
              <table style={styles.table}>
                <thead>
                  <tr>{['Sensor', 'Class', 'Confidence', 'Provenance', 'Time'].map(h => (
                    <th key={h} style={styles.th}>{h}</th>
                  ))}</tr>
                </thead>
                <tbody>
                  {stats.recent_observations.map(o => (
                    <tr key={o.id} style={styles.tr}>
                      <td style={styles.td}>{sensorIcon(o.sensor_type)} {o.sensor_type}</td>
                      <td style={styles.td}>{o.object_class ?? '—'}</td>
                      <td style={styles.td}>
                        {o.confidence != null ? `${(o.confidence * 100).toFixed(1)}%` : '—'}
                      </td>
                      <td style={styles.td}>
                        <span style={{
                          ...styles.badge,
                          background: o.data_provenance === 'synthetic' ? '#7c3aed22' : '#1e40af22',
                          color: o.data_provenance === 'synthetic' ? '#a78bfa' : '#60a5fa',
                        }}>
                          {o.data_provenance}
                        </span>
                      </td>
                      <td style={styles.td}>{formatTs(o.created_at)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
        </div>
      </div>

      <div style={styles.chartsRow}>
        <div style={{ ...styles.chartCard, flex: 1 }}>
          <h3 style={styles.cardTitle}>Recent Events</h3>
          {stats.recent_events.length === 0
            ? <p style={styles.noData}>None yet.</p>
            : (
              <table style={styles.table}>
                <thead>
                  <tr>{['Priority', 'Status', 'Sensors', 'Evidence', 'Time'].map(h => (
                    <th key={h} style={styles.th}>{h}</th>
                  ))}</tr>
                </thead>
                <tbody>
                  {stats.recent_events.map(e => (
                    <tr key={e.id} style={styles.tr}>
                      <td style={styles.td}>
                        <span style={{ ...styles.badge, background: priorityColour(e.review_priority) + '22', color: priorityColour(e.review_priority) }}>
                          {e.review_priority}
                        </span>
                      </td>
                      <td style={styles.td}>{e.review_status}</td>
                      <td style={styles.td}>{(e.source_sensors ?? []).join(', ') || '—'}</td>
                      <td style={{ ...styles.td, maxWidth: 300, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                        {e.evidence_summary ?? '—'}
                      </td>
                      <td style={styles.td}>{formatTs(e.created_at)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
        </div>
      </div>

      {/* Training jobs */}
      {stats.training_jobs.length > 0 && (
        <div style={styles.chartCard}>
          <h3 style={styles.cardTitle}>Active Training Jobs</h3>
          {stats.training_jobs.map(j => (
            <div key={j.id} style={styles.jobRow}>
              <span style={{ color: '#94a3b8', fontSize: 12 }}>{j.id.slice(0, 8)}</span>
              <span style={{ color: statusColour(j.status), fontWeight: 600 }}>{j.status}</span>
              <span style={{ color: '#cbd5e1' }}>
                Epoch {j.current_epoch}/{j.total_epochs}
              </span>
              {j.status === 'RUNNING' && (
                <div style={styles.progressBar}>
                  <div style={{
                    ...styles.progressFill,
                    width: `${j.total_epochs > 0 ? (j.current_epoch / j.total_epochs) * 100 : 0}%`,
                  }} />
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      {/* Errors */}
      {stats.errors.length > 0 && (
        <div style={{ ...styles.chartCard, borderColor: '#7f1d1d' }}>
          <h3 style={{ ...styles.cardTitle, color: '#f87171' }}>⚠️ Recent Errors</h3>
          {stats.errors.map((e, i) => (
            <div key={i} style={styles.errorRow}>
              <span style={{ color: '#f87171', fontWeight: 600, fontSize: 12 }}>{e.level}</span>
              <span style={{ color: '#94a3b8', fontSize: 12 }}>{e.component}</span>
              <span style={{ color: '#e2e8f0' }}>{e.message}</span>
            </div>
          ))}
        </div>
      )}

      <p style={styles.footer}>
        Last updated: {formatTs(health.timestamp)} · Database: {health.database}
      </p>
    </div>
  )
}

const styles: Record<string, React.CSSProperties> = {
  page: { padding: 24, color: '#e2e8f0', maxWidth: 1400 },
  header: { display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 24 },
  title: { margin: 0, fontSize: 24, fontWeight: 700, color: '#f1f5f9' },
  subtitle: { margin: '4px 0 0', color: '#64748b', fontSize: 13 },
  statusPill: { display: 'flex', alignItems: 'center', gap: 8, background: '#1e293b', border: '1px solid #334155', borderRadius: 8, padding: '6px 14px', fontSize: 13, color: '#94a3b8' },
  dot: { width: 8, height: 8, borderRadius: '50%', display: 'inline-block' },
  kpiRow: { display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: 24 },
  kpiCard: { background: '#1e293b', border: '1px solid #334155', borderRadius: 10, padding: '16px 20px', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 4, minWidth: 120, flex: 1 },
  kpiIcon: { fontSize: 22 },
  kpiValue: { fontSize: 28, fontWeight: 700, color: '#f1f5f9' },
  kpiLabel: { fontSize: 11, color: '#64748b', textAlign: 'center' },
  chartsRow: { display: 'flex', gap: 16, marginBottom: 16, flexWrap: 'wrap' },
  chartCard: { background: '#1e293b', border: '1px solid #334155', borderRadius: 10, padding: 20, flex: 1, minWidth: 280 },
  cardTitle: { margin: '0 0 12px', fontSize: 14, fontWeight: 600, color: '#94a3b8', textTransform: 'uppercase', letterSpacing: 1 },
  noData: { color: '#475569', fontSize: 13, fontStyle: 'italic' },
  tooltip: { background: '#0f172a', border: '1px solid #334155', borderRadius: 6, color: '#e2e8f0' },
  table: { width: '100%', borderCollapse: 'collapse' },
  th: { textAlign: 'left', padding: '6px 10px', fontSize: 11, color: '#64748b', borderBottom: '1px solid #334155', textTransform: 'uppercase' },
  td: { padding: '8px 10px', fontSize: 13, color: '#cbd5e1', borderBottom: '1px solid #1e293b' },
  tr: { transition: 'background 0.15s' },
  badge: { padding: '2px 8px', borderRadius: 4, fontSize: 11, fontWeight: 600 },
  jobRow: { display: 'flex', gap: 16, alignItems: 'center', padding: '8px 0', borderBottom: '1px solid #334155' },
  progressBar: { flex: 1, height: 6, background: '#0f172a', borderRadius: 3, overflow: 'hidden' },
  progressFill: { height: '100%', background: '#3b82f6', borderRadius: 3, transition: 'width 0.4s' },
  errorBox: { background: '#1c0a0a', border: '1px solid #7f1d1d', borderRadius: 10, padding: 24, display: 'flex', gap: 16, alignItems: 'flex-start', margin: 24, color: '#f87171' },
  errorRow: { display: 'flex', gap: 12, alignItems: 'center', padding: '6px 0', borderBottom: '1px solid #1e293b' },
  loading: { color: '#64748b', padding: 48, textAlign: 'center' },
  footer: { margin: '16px 0 0', color: '#334155', fontSize: 11 },
  code: { background: '#0f172a', padding: '2px 6px', borderRadius: 4, fontSize: 12 },
}
