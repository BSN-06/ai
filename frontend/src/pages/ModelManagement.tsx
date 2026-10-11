import { useState, useEffect } from 'react'
import { listModels, listDatasets, getModelLoadStatus } from '../utils/api'
import { formatTs, statusColour } from '../utils/format'
import type { ModelRecord, Dataset } from '../types'

export default function ModelManagement() {
  const [models, setModels] = useState<ModelRecord[]>([])
  const [datasets, setDatasets] = useState<Dataset[]>([])
  const [expanded, setExpanded] = useState<string | null>(null)
  const [loadStatus, setLoadStatus] = useState<Record<string, unknown>>({})

  useEffect(() => {
    listModels().then(setModels)
    listDatasets().then(setDatasets)
  }, [])

  const toggleExpand = async (id: string) => {
    setExpanded(prev => prev === id ? null : id)
    if (!loadStatus[id]) {
      try {
        const s = await getModelLoadStatus(id)
        setLoadStatus(prev => ({ ...prev, [id]: s }))
      } catch { /* ignore */ }
    }
  }

  const datasetFor = (name: string | null) => datasets.find(d => d.name === name)

  return (
    <div style={s.page}>
      <h1 style={s.title}>🧠 Model Management</h1>
      <p style={s.sub}>
        Registered model architectures, training status, checkpoints, and known limitations.
        Each sensor module uses its own independent model.
      </p>

      {models.length === 0
        ? <p style={s.noData}>No models registered. Backend may still be starting up.</p>
        : models.map(m => {
          const ds = datasetFor(m.dataset_name)
          const ls = loadStatus[m.id] as { in_memory_loaded?: boolean } | undefined
          return (
            <div key={m.id} style={{ ...s.card, borderColor: expanded === m.id ? '#3b82f6' : '#334155' }}>
              <div style={s.cardHeader} onClick={() => toggleExpand(m.id)}>
                <div style={s.cardLeft}>
                  <span style={{ fontSize: 20 }}>{m.sensor_type === 'satellite' ? '🛰️' : m.sensor_type === 'thermal' ? '🌡️' : '📡'}</span>
                  <div>
                    <span style={s.modelName}>{m.name}</span>
                    <span style={{ color: '#64748b', fontSize: 12, marginLeft: 10 }}>{m.architecture} · {m.variant}</span>
                  </div>
                </div>
                <div style={s.cardRight}>
                  <span style={{ ...s.badge, color: statusColour(m.status), borderColor: statusColour(m.status) }}>
                    {m.status}
                  </span>
                  {m.trained
                    ? <span style={{ ...s.badge, color: '#22c55e', borderColor: '#22c55e' }}>TRAINED</span>
                    : <span style={{ ...s.badge, color: '#f59e0b', borderColor: '#f59e0b' }}>NOT TRAINED</span>}
                  <span style={{ ...s.badge, color: m.evaluation_status === 'EVALUATED' ? '#22c55e' : '#64748b', borderColor: '#334155' }}>
                    {m.evaluation_status}
                  </span>
                  {ls?.in_memory_loaded !== undefined && (
                    <span style={{ ...s.badge, color: ls.in_memory_loaded ? '#22c55e' : '#475569', borderColor: '#334155' }}>
                      {ls.in_memory_loaded ? 'LOADED' : 'NOT IN MEMORY'}
                    </span>
                  )}
                  <span style={{ color: '#475569', fontSize: 14 }}>{expanded === m.id ? '▲' : '▼'}</span>
                </div>
              </div>

              {expanded === m.id && (
                <div style={s.details}>
                  <div style={s.detailGrid}>
                    <div style={s.detailSection}>
                      <h4 style={s.detailTitle}>Architecture</h4>
                      <div style={s.detailRow}><span>Framework</span><strong>{m.architecture}</strong></div>
                      <div style={s.detailRow}><span>Variant</span><strong>{m.variant ?? '—'}</strong></div>
                      <div style={s.detailRow}><span>Backbone</span><strong>{m.backbone ?? '—'}</strong></div>
                      <div style={s.detailRow}><span>Neck</span><strong>{m.neck ?? '—'}</strong></div>
                      <div style={s.detailRow}><span>Head</span><strong>{m.head ?? '—'}</strong></div>
                      <div style={s.detailRow}><span>Input Size</span><strong>{m.input_size}px</strong></div>
                      <div style={s.detailRow}><span>Num Classes</span><strong>{m.num_classes ?? '—'}</strong></div>
                    </div>

                    <div style={s.detailSection}>
                      <h4 style={s.detailTitle}>Training Status</h4>
                      <div style={s.detailRow}><span>Pretrained</span><strong>{m.pretrained ? 'Yes (COCO)' : 'No'}</strong></div>
                      <div style={s.detailRow}><span>Fine-tuned</span><strong style={{ color: m.trained ? '#22c55e' : '#f59e0b' }}>{m.trained ? 'YES' : 'NO — pretrained only'}</strong></div>
                      <div style={s.detailRow}><span>Checkpoint</span><strong style={{ fontSize: 11, wordBreak: 'break-all' }}>{m.checkpoint_path ?? 'Not set (using default weights)'}</strong></div>
                      <div style={s.detailRow}><span>Last Training</span><strong>{formatTs(m.last_training_time)}</strong></div>
                      <div style={s.detailRow}><span>Dataset</span><strong>{m.dataset_name ?? '—'}</strong></div>
                    </div>

                    <div style={s.detailSection}>
                      <h4 style={s.detailTitle}>Evaluation Metrics</h4>
                      {m.evaluation_status !== 'EVALUATED' || !m.metrics
                        ? <p style={{ color: '#475569', fontSize: 12, fontStyle: 'italic' }}>
                            Not evaluated. Requires: dataset installed + model fine-tuned + evaluation job run.
                          </p>
                        : <>
                            <div style={s.detailRow}><span>Precision</span><strong>{m.metrics.precision != null ? (m.metrics.precision * 100).toFixed(1) + '%' : '—'}</strong></div>
                            <div style={s.detailRow}><span>Recall</span><strong>{m.metrics.recall != null ? (m.metrics.recall * 100).toFixed(1) + '%' : '—'}</strong></div>
                            <div style={s.detailRow}><span>mAP@0.5</span><strong>{m.metrics.map50 != null ? (m.metrics.map50 * 100).toFixed(1) + '%' : '—'}</strong></div>
                            <div style={s.detailRow}><span>mAP@0.5:0.95</span><strong>{m.metrics.map50_95 != null ? (m.metrics.map50_95 * 100).toFixed(1) + '%' : '—'}</strong></div>
                          </>}
                    </div>
                  </div>

                  {m.known_limitations && (
                    <div style={s.limitBox}>
                      <strong>⚠️ Known Limitations:</strong>
                      <p style={{ margin: '4px 0 0', fontSize: 12, lineHeight: 1.6 }}>{m.known_limitations}</p>
                    </div>
                  )}

                  {ds && (
                    <div style={s.dsBox}>
                      <strong>Dataset: {ds.name}</strong>
                      <span style={{ ...s.badge, marginLeft: 10, color: ds.validation_status === 'VALIDATED' ? '#22c55e' : '#f59e0b', borderColor: '#334155' }}>
                        {ds.validation_status}
                      </span>
                      {ds.validation_status !== 'VALIDATED' && (
                        <p style={{ margin: '6px 0 0', color: '#94a3b8', fontSize: 12, lineHeight: 1.6 }}>
                          {ds.access_requirements}
                        </p>
                      )}
                    </div>
                  )}

                  <p style={{ color: '#334155', fontSize: 11, marginTop: 12 }}>
                    Model ID: {m.id} · Created: {formatTs(m.created_at)}
                  </p>
                </div>
              )}
            </div>
          )
        })}

      <div style={s.infoBox}>
        Models are labelled <strong>PRETRAINED</strong> until fine-tuning on the target dataset completes
        successfully. Do not interpret pretrained COCO metrics as satellite or thermal performance.
        Use the Training page to start fine-tuning once a dataset is installed.
      </div>
    </div>
  )
}

const s: Record<string, React.CSSProperties> = {
  page: { padding: 24, color: '#e2e8f0', maxWidth: 1100 },
  title: { margin: '0 0 6px', fontSize: 22, fontWeight: 700, color: '#f1f5f9' },
  sub: { margin: '0 0 20px', color: '#64748b', fontSize: 13 },
  card: { background: '#1e293b', border: '1px solid', borderRadius: 10, marginBottom: 12, overflow: 'hidden' },
  cardHeader: { display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '14px 20px', cursor: 'pointer' },
  cardLeft: { display: 'flex', gap: 12, alignItems: 'center' },
  cardRight: { display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap' },
  modelName: { fontWeight: 600, fontSize: 15, color: '#f1f5f9' },
  badge: { padding: '2px 8px', borderRadius: 4, fontSize: 11, fontWeight: 600, border: '1px solid' },
  details: { padding: '0 20px 20px', borderTop: '1px solid #334155', paddingTop: 16 },
  detailGrid: { display: 'flex', gap: 20, flexWrap: 'wrap' },
  detailSection: { flex: 1, minWidth: 220 },
  detailTitle: { margin: '0 0 8px', fontSize: 12, color: '#64748b', textTransform: 'uppercase', fontWeight: 600 },
  detailRow: { display: 'flex', justifyContent: 'space-between', fontSize: 13, color: '#94a3b8', padding: '3px 0', borderBottom: '1px solid #0f172a' },
  limitBox: { background: '#1c1a0a', border: '1px solid #92400e', borderRadius: 6, padding: '10px 14px', color: '#fbbf24', fontSize: 13, marginTop: 16 },
  dsBox: { background: '#0f172a', border: '1px solid #1e3a8a', borderRadius: 6, padding: '10px 14px', marginTop: 12 },
  noData: { color: '#475569', fontStyle: 'italic' },
  infoBox: { background: '#0f2a1a', border: '1px solid #166534', borderRadius: 8, padding: '12px 16px', color: '#86efac', fontSize: 13, marginTop: 8 },
}
