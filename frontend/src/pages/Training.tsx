import { useState, useEffect } from 'react'
import { listModels, listDatasets, startTraining, listTrainingJobs, cancelTrainingJob } from '../utils/api'
import { formatTs, statusColour } from '../utils/format'
import type { ModelRecord, Dataset, TrainingJob } from '../types'
import { usePolling } from '../hooks/usePolling'

export default function Training() {
  const [models, setModels] = useState<ModelRecord[]>([])
  const [datasets, setDatasets] = useState<Dataset[]>([])
  const [jobs, setJobs] = useState<TrainingJob[]>([])
  const [selectedModel, setSelectedModel] = useState('')
  const [selectedDataset, setSelectedDataset] = useState('')
  const [epochs, setEpochs] = useState(50)
  const [batchSize, setBatchSize] = useState(16)
  const [imageSize, setImageSize] = useState(640)
  const [device, setDevice] = useState('cpu')
  const [message, setMessage] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [submitting, setSubmitting] = useState(false)

  useEffect(() => {
    listModels().then(setModels)
    listDatasets().then(setDatasets)
    listTrainingJobs().then(data => setJobs(Array.isArray(data) ? data : []))
  }, [])

  // Poll job status every 5s
  usePolling(async () => {
    const data = await listTrainingJobs()
    setJobs(Array.isArray(data) ? data : [])
  }, 5000)

  const submit = async () => {
    if (!selectedModel || !selectedDataset) {
      setError('Select a model and dataset.')
      return
    }
    setSubmitting(true); setError(null); setMessage(null)
    try {
      const r = await startTraining({
        model_id: selectedModel,
        dataset_name: selectedDataset,
        epochs, batch_size: batchSize, image_size: imageSize, device,
      })
      setMessage(`Training job started — ID: ${r.job_id}`)
    } catch (e: unknown) {
      const msg = (e as { response?: { data?: { detail?: string } } })?.response?.data?.detail
      setError(msg ?? 'Failed to start training job.')
    } finally { setSubmitting(false) }
  }

  const cancel = async (id: string) => {
    await cancelTrainingJob(id)
    setMessage('Cancellation requested.')
    const data = await listTrainingJobs()
    setJobs(Array.isArray(data) ? data : [])
  }

  const runningJob = jobs.find(j => j.status === 'RUNNING')

  return (
    <div style={s.page}>
      <h1 style={s.title}>🎓 Model Training</h1>
      <p style={s.sub}>
        Start a training job to fine-tune a YOLO model on a downloaded dataset.
        Training runs as a background task — you can navigate away and return.
        Jobs run on CPU by default; select CUDA if a compatible GPU is available.
      </p>

      <div style={s.warnBox}>
        ⚠️ <strong>Prerequisites:</strong> The dataset must be downloaded, extracted, and its local path
        set in the Datasets page before training can proceed. Training large datasets requires substantial
        storage (10–20 GB), RAM, and time (hours on CPU, minutes on GPU). Do not start multiple jobs
        simultaneously.
      </div>

      <div style={s.row}>
        {/* Config form */}
        <div style={s.card}>
          <h3 style={s.cardTitle}>New Training Job</h3>
          <div style={s.formGrid}>
            <div style={s.formGroup}>
              <label style={s.label}>Model</label>
              <select style={s.select} value={selectedModel} onChange={e => setSelectedModel(e.target.value)}>
                <option value="">— select model —</option>
                {models.map(m => <option key={m.id} value={m.id}>{m.name} ({m.sensor_type})</option>)}
              </select>
            </div>

            <div style={s.formGroup}>
              <label style={s.label}>Dataset</label>
              <select style={s.select} value={selectedDataset} onChange={e => setSelectedDataset(e.target.value)}>
                <option value="">— select dataset —</option>
                {datasets.map(d => (
                  <option key={d.id} value={d.name}>
                    {d.name} [{d.validation_status}]
                  </option>
                ))}
              </select>
            </div>

            <div style={s.formGroup}>
              <label style={s.label}>Epochs: <strong>{epochs}</strong></label>
              <input type="range" min={1} max={300} value={epochs}
                onChange={e => setEpochs(parseInt(e.target.value))} style={s.slider} />
            </div>

            <div style={s.formGroup}>
              <label style={s.label}>Batch Size: <strong>{batchSize}</strong></label>
              <input type="range" min={1} max={64} step={1} value={batchSize}
                onChange={e => setBatchSize(parseInt(e.target.value))} style={s.slider} />
            </div>

            <div style={s.formGroup}>
              <label style={s.label}>Image Size: <strong>{imageSize}px</strong></label>
              <select style={s.select} value={imageSize} onChange={e => setImageSize(parseInt(e.target.value))}>
                {[320, 416, 512, 640, 768, 1024].map(sz => <option key={sz} value={sz}>{sz}px</option>)}
              </select>
            </div>

            <div style={s.formGroup}>
              <label style={s.label}>Device</label>
              <select style={s.select} value={device} onChange={e => setDevice(e.target.value)}>
                <option value="cpu">CPU</option>
                <option value="cuda">CUDA (GPU)</option>
                <option value="0">GPU 0</option>
              </select>
            </div>
          </div>

          {error && <div style={s.errBox}>⚠️ {error}</div>}
          {message && <div style={s.msgBox}>✅ {message}</div>}

          {runningJob && (
            <div style={s.runningWarn}>
              ⏳ A training job is already running (ID: {runningJob.id.slice(0, 8)}…).
              Wait for it to finish or cancel it before starting a new one.
            </div>
          )}

          <button
            style={{ ...s.btn, opacity: (submitting || !!runningJob) ? 0.5 : 1 }}
            disabled={submitting || !!runningJob}
            onClick={submit}
          >
            {submitting ? '⏳ Submitting…' : '▶ Start Training'}
          </button>
        </div>

        {/* Quick guide */}
        <div style={{ ...s.card, minWidth: 260, maxWidth: 320 }}>
          <h3 style={s.cardTitle}>Quick Guide</h3>
          {[
            ['1', 'Download dataset (see Datasets page)'],
            ['2', 'Set dataset local path in Datasets page'],
            ['3', 'Select model + dataset + settings above'],
            ['4', 'Click Start Training'],
            ['5', 'Monitor progress in the Jobs table'],
            ['6', 'Run Evaluation after training completes'],
          ].map(([n, t]) => (
            <div key={n} style={s.step}>
              <span style={s.stepNum}>{n}</span>
              <span style={{ fontSize: 13, color: '#94a3b8' }}>{t}</span>
            </div>
          ))}
          <p style={{ color: '#475569', fontSize: 11, marginTop: 12, lineHeight: 1.6 }}>
            Training on CPU is slow but works. A GPU with 8+ GB VRAM is recommended for
            YOLOv8n/s. YOLOv8m/l requires 16+ GB VRAM.
          </p>
        </div>
      </div>

      {/* Jobs table */}
      <div style={s.card}>
        <h3 style={s.cardTitle}>Training Jobs</h3>
        {jobs.length === 0
          ? <p style={s.noData}>No training jobs yet.</p>
          : (
            <table style={s.table}>
              <thead>
                <tr>{['ID', 'Model', 'Dataset', 'Status', 'Epoch', 'Device', 'mAP@50', 'Started', 'Completed', ''].map(h =>
                  <th key={h} style={s.th}>{h}</th>)}</tr>
              </thead>
              <tbody>
                {jobs.map(j => (
                  <tr key={j.id}>
                    <td style={{ ...s.td, fontFamily: 'monospace', fontSize: 11 }}>{j.id.slice(0, 8)}</td>
                    <td style={s.td}>{j.model_id.slice(0, 8)}</td>
                    <td style={s.td}>{j.dataset_name}</td>
                    <td style={s.td}>
                      <span style={{ color: statusColour(j.status), fontWeight: 600 }}>{j.status}</span>
                    </td>
                    <td style={s.td}>
                      {j.status === 'RUNNING'
                        ? <>
                            {j.current_epoch}/{j.total_epochs}
                            <div style={s.miniBar}>
                              <div style={{ ...s.miniFill, width: `${j.total_epochs > 0 ? (j.current_epoch / j.total_epochs) * 100 : 0}%` }} />
                            </div>
                          </>
                        : `${j.current_epoch}/${j.total_epochs}`}
                    </td>
                    <td style={s.td}>{j.device}</td>
                    <td style={s.td}>{j.best_map50 != null ? `${(j.best_map50 * 100).toFixed(1)}%` : '—'}</td>
                    <td style={s.td}>{formatTs(j.started_at)}</td>
                    <td style={s.td}>{formatTs(j.completed_at)}</td>
                    <td style={s.td}>
                      {j.status === 'RUNNING' && (
                        <button style={s.btnXs} onClick={() => cancel(j.id)}>Cancel</button>
                      )}
                      {j.error_message && (
                        <span style={{ color: '#ef4444', fontSize: 11 }} title={j.error_message}>⚠️ Error</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
      </div>
    </div>
  )
}

const s: Record<string, React.CSSProperties> = {
  page: { padding: 24, color: '#e2e8f0', maxWidth: 1200 },
  title: { margin: '0 0 6px', fontSize: 22, fontWeight: 700, color: '#f1f5f9' },
  sub: { margin: '0 0 12px', color: '#64748b', fontSize: 13 },
  warnBox: { background: '#1c1a0a', border: '1px solid #92400e', borderRadius: 8, padding: '10px 14px', color: '#fbbf24', fontSize: 13, marginBottom: 20 },
  row: { display: 'flex', gap: 16, marginBottom: 16, flexWrap: 'wrap' },
  card: { background: '#1e293b', border: '1px solid #334155', borderRadius: 10, padding: 20, flex: 1, marginBottom: 16 },
  cardTitle: { margin: '0 0 16px', fontSize: 13, fontWeight: 600, color: '#94a3b8', textTransform: 'uppercase', letterSpacing: 1 },
  formGrid: { display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 16 },
  formGroup: { display: 'flex', flexDirection: 'column', gap: 6 },
  label: { fontSize: 13, color: '#94a3b8' },
  select: { background: '#0f172a', border: '1px solid #334155', borderRadius: 6, color: '#e2e8f0', padding: '8px 10px', fontSize: 13 },
  slider: { accentColor: '#3b82f6' },
  btn: { background: '#2563eb', color: '#fff', border: 'none', borderRadius: 8, padding: '10px 20px', fontSize: 14, fontWeight: 600, cursor: 'pointer' },
  btnXs: { background: '#7f1d1d', color: '#fca5a5', border: 'none', borderRadius: 4, padding: '3px 8px', fontSize: 11, cursor: 'pointer' },
  errBox: { background: '#1c080822', border: '1px solid #7f1d1d', borderRadius: 6, padding: '8px 12px', color: '#f87171', fontSize: 13, marginBottom: 10 },
  msgBox: { background: '#0f2a1a', border: '1px solid #166534', borderRadius: 6, padding: '8px 12px', color: '#86efac', fontSize: 13, marginBottom: 10 },
  runningWarn: { background: '#1c1a0a', border: '1px solid #92400e', borderRadius: 6, padding: '8px 12px', color: '#fbbf24', fontSize: 13, marginBottom: 10 },
  step: { display: 'flex', gap: 10, alignItems: 'flex-start', padding: '6px 0', borderBottom: '1px solid #0f172a' },
  stepNum: { background: '#1e3a8a', color: '#93c5fd', borderRadius: '50%', width: 20, height: 20, display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 11, fontWeight: 700, flexShrink: 0 },
  table: { width: '100%', borderCollapse: 'collapse' },
  th: { textAlign: 'left', padding: '6px 10px', fontSize: 11, color: '#64748b', borderBottom: '1px solid #334155', textTransform: 'uppercase' },
  td: { padding: '8px 10px', fontSize: 13, color: '#cbd5e1', borderBottom: '1px solid #1e293b' },
  noData: { color: '#475569', fontStyle: 'italic', fontSize: 13 },
  miniBar: { height: 4, background: '#0f172a', borderRadius: 2, marginTop: 2, overflow: 'hidden', width: 80 },
  miniFill: { height: '100%', background: '#3b82f6', borderRadius: 2, transition: 'width 0.4s' },
}
