import { useState, useEffect } from 'react'
import { listDatasets, validateDataset, updateDatasetPath } from '../utils/api'
import { formatTs } from '../utils/format'
import type { Dataset } from '../types'

const STATUS_COLOUR: Record<string, string> = {
  VALIDATED: '#22c55e', NOT_INSTALLED: '#f59e0b', NOT_FOUND: '#ef4444',
  NOT_VALIDATED: '#94a3b8', EMPTY: '#f87171',
}

export default function Datasets() {
  const [datasets, setDatasets] = useState<Dataset[]>([])
  const [expanded, setExpanded] = useState<string | null>(null)
  const [paths, setPaths] = useState<Record<string, string>>({})
  const [messages, setMessages] = useState<Record<string, string>>({})

  useEffect(() => { listDatasets().then(setDatasets) }, [])

  const validate = async (id: string) => {
    try {
      const r = await validateDataset(id)
      setMessages(p => ({ ...p, [id]: `${r.status}: ${r.message}` }))
      listDatasets().then(setDatasets)
    } catch { setMessages(p => ({ ...p, [id]: 'Validation request failed.' })) }
  }

  const savePath = async (id: string) => {
    const path = paths[id]
    if (!path) return
    try {
      await updateDatasetPath(id, path)
      setMessages(p => ({ ...p, [id]: 'Path saved. Click Validate to check.' }))
      listDatasets().then(setDatasets)
    } catch { setMessages(p => ({ ...p, [id]: 'Failed to save path.' })) }
  }

  return (
    <div style={s.page}>
      <h1 style={s.title}>🗄️ Dataset Management</h1>
      <p style={s.sub}>
        Registered datasets for all sensor modules. None of these datasets are automatically
        downloaded — each requires registration and manual download due to size and licence requirements.
      </p>

      {datasets.map(d => (
        <div key={d.id} style={{ ...s.card, borderColor: expanded === d.id ? '#3b82f6' : '#334155' }}>
          <div style={s.header} onClick={() => setExpanded(expanded === d.id ? null : d.id)}>
            <div style={s.headerLeft}>
              <span style={{ fontSize: 20 }}>
                {d.sensor_type === 'satellite' ? '🛰️' : d.sensor_type === 'thermal' ? '🌡️' : '📡'}
              </span>
              <div>
                <strong style={{ color: '#f1f5f9', fontSize: 15 }}>{d.name}</strong>
                <span style={{ color: '#64748b', fontSize: 12, marginLeft: 10 }}>{d.sensor_type} · {d.annotation_format ?? 'unknown format'}</span>
              </div>
            </div>
            <div style={s.headerRight}>
              <span style={{ color: STATUS_COLOUR[d.validation_status] ?? '#94a3b8', fontWeight: 600, fontSize: 13 }}>
                {d.validation_status}
              </span>
              {d.num_classes && (
                <span style={{ color: '#64748b', fontSize: 12 }}>{d.num_classes} classes</span>
              )}
              <span style={{ color: '#475569', fontSize: 14 }}>{expanded === d.id ? '▲' : '▼'}</span>
            </div>
          </div>

          {expanded === d.id && (
            <div style={s.details}>
              <div style={s.detailGrid}>
                {/* Info */}
                <div style={s.section}>
                  <h4 style={s.sectionTitle}>Dataset Info</h4>
                  <div style={s.row}><span>Sensor Type</span><strong>{d.sensor_type}</strong></div>
                  <div style={s.row}><span>Format</span><strong>{d.annotation_format ?? '—'}</strong></div>
                  <div style={s.row}><span>Classes</span><strong>{d.num_classes ?? '—'}</strong></div>
                  <div style={s.row}><span>Files found</span><strong>{d.file_count ?? 'Not checked'}</strong></div>
                  <div style={s.row}><span>Train</span><strong>{d.train_count ?? '—'}</strong></div>
                  <div style={s.row}><span>Val</span><strong>{d.val_count ?? '—'}</strong></div>
                  <div style={s.row}><span>Test</span><strong>{d.test_count ?? '—'}</strong></div>
                  {d.source_url && (
                    <div style={{ marginTop: 8 }}>
                      <a href={d.source_url} target="_blank" rel="noreferrer" style={s.link}>
                        🔗 Official Dataset Page ↗
                      </a>
                    </div>
                  )}
                </div>

                {/* Classes */}
                {d.classes && d.classes.length > 0 && (
                  <div style={s.section}>
                    <h4 style={s.sectionTitle}>Classes ({d.classes.length})</h4>
                    <div style={s.classGrid}>
                      {d.classes.map((c, i) => (
                        <span key={i} style={s.classBadge}>{c}</span>
                      ))}
                    </div>
                  </div>
                )}

                {/* Local path */}
                <div style={s.section}>
                  <h4 style={s.sectionTitle}>Local Path</h4>
                  <div style={{ display: 'flex', gap: 8 }}>
                    <input
                      style={s.input}
                      placeholder={d.local_path ?? 'e.g. C:\\datasets\\xview\\'}
                      value={paths[d.id] ?? d.local_path ?? ''}
                      onChange={e => setPaths(p => ({ ...p, [d.id]: e.target.value }))}
                    />
                    <button style={s.btnSm} onClick={() => savePath(d.id)}>Save</button>
                    <button style={s.btn} onClick={() => validate(d.id)}>Validate</button>
                  </div>
                  {messages[d.id] && (
                    <p style={{ color: STATUS_COLOUR[d.validation_status] ?? '#94a3b8', fontSize: 12, marginTop: 6 }}>
                      {messages[d.id]}
                    </p>
                  )}
                </div>
              </div>

              {/* Access requirements */}
              {d.access_requirements && (
                <div style={s.accessBox}>
                  <strong>⚠️ Access Requirements:</strong>
                  <p style={{ margin: '4px 0 0', fontSize: 12, lineHeight: 1.6 }}>{d.access_requirements}</p>
                </div>
              )}

              {/* Setup instructions */}
              {d.setup_instructions && (
                <div style={s.setupBox}>
                  <strong>📋 Setup Instructions:</strong>
                  <pre style={s.pre}>{d.setup_instructions}</pre>
                </div>
              )}

              <p style={{ color: '#334155', fontSize: 11, marginTop: 8 }}>
                ID: {d.id} · Registered: {formatTs(d.created_at)}
              </p>
            </div>
          )}
        </div>
      ))}

      <div style={s.disclaimer}>
        <strong>⚠️ No datasets are automatically downloaded.</strong> Each requires registration
        or request through the official source. Check the licence terms before use.
        Dataset sizes range from ~5 GB (FLIR ADAS) to ~20 GB (xView). Sufficient disk space
        and a stable internet connection are required.
      </div>
    </div>
  )
}

const s: Record<string, React.CSSProperties> = {
  page: { padding: 24, color: '#e2e8f0', maxWidth: 1100 },
  title: { margin: '0 0 6px', fontSize: 22, fontWeight: 700, color: '#f1f5f9' },
  sub: { margin: '0 0 20px', color: '#64748b', fontSize: 13 },
  card: { background: '#1e293b', border: '1px solid', borderRadius: 10, marginBottom: 12, overflow: 'hidden' },
  header: { display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '14px 20px', cursor: 'pointer' },
  headerLeft: { display: 'flex', gap: 12, alignItems: 'center' },
  headerRight: { display: 'flex', gap: 12, alignItems: 'center' },
  details: { padding: '0 20px 20px', borderTop: '1px solid #334155', paddingTop: 16 },
  detailGrid: { display: 'flex', gap: 20, flexWrap: 'wrap' },
  section: { flex: 1, minWidth: 200 },
  sectionTitle: { margin: '0 0 8px', fontSize: 12, color: '#64748b', textTransform: 'uppercase', fontWeight: 600 },
  row: { display: 'flex', justifyContent: 'space-between', fontSize: 13, color: '#94a3b8', padding: '3px 0', borderBottom: '1px solid #0f172a' },
  classGrid: { display: 'flex', flexWrap: 'wrap', gap: 6, maxHeight: 200, overflowY: 'auto' },
  classBadge: { background: '#1e3a8a33', color: '#93c5fd', padding: '2px 8px', borderRadius: 4, fontSize: 11 },
  input: { flex: 1, background: '#0f172a', border: '1px solid #334155', borderRadius: 6, color: '#e2e8f0', padding: '7px 10px', fontSize: 13 },
  btn: { background: '#2563eb', color: '#fff', border: 'none', borderRadius: 6, padding: '7px 14px', fontSize: 13, fontWeight: 600, cursor: 'pointer' },
  btnSm: { background: '#1e293b', color: '#94a3b8', border: '1px solid #334155', borderRadius: 6, padding: '7px 12px', fontSize: 13, cursor: 'pointer' },
  link: { color: '#60a5fa', fontSize: 13 },
  accessBox: { background: '#1c1a0a', border: '1px solid #92400e', borderRadius: 6, padding: '10px 14px', color: '#fbbf24', fontSize: 13, marginTop: 16 },
  setupBox: { background: '#0f172a', border: '1px solid #334155', borderRadius: 6, padding: '10px 14px', marginTop: 10 },
  pre: { margin: '4px 0 0', fontSize: 12, color: '#94a3b8', whiteSpace: 'pre-wrap', lineHeight: 1.7 },
  disclaimer: { background: '#1c1a0a', border: '1px solid #92400e', borderRadius: 8, padding: '12px 16px', color: '#fbbf24', fontSize: 13, marginTop: 8 },
}
