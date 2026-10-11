import { useState, useEffect } from 'react'
import {
  listModels, listDatasets, startEvaluation, listEvaluationJobs,
  runSyntheticScenarios, getReplayStatus, controlReplay, loadReplayFile,
} from '../utils/api'
import { formatTs, statusColour } from '../utils/format'
import type { ModelRecord, Dataset, EvaluationJob, SyntheticRunResult } from '../types'
import { usePolling } from '../hooks/usePolling'

type Tab = 'real' | 'replay' | 'synthetic'

export default function Evaluation() {
  const [tab, setTab] = useState<Tab>('synthetic')
  const [models, setModels] = useState<ModelRecord[]>([])
  const [datasets, setDatasets] = useState<Dataset[]>([])
  const [evalJobs, setEvalJobs] = useState<EvaluationJob[]>([])
  const [selectedModel, setSelectedModel] = useState('')
  const [selectedDataset, setSelectedDataset] = useState('')
  const [split, setSplit] = useState('val')
  const [synResult, setSynResult] = useState<SyntheticRunResult | null>(null)
  const [replayStatus, setReplayStatus] = useState<{ status: string; processed_records: number; total_records: number; label?: string } | null>(null)
  const [replayFile, setReplayFile] = useState<File | null>(null)
  const [replaySpeed, setReplaySpeed] = useState(5)
  const [loading, setLoading] = useState(false)
  const [message, setMessage] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    listModels().then(setModels)
    listDatasets().then(setDatasets)
    listEvaluationJobs().then(data => setEvalJobs(Array.isArray(data) ? data : []))
    getReplayStatus().then(setReplayStatus).catch(() => {})
  }, [])

  usePolling(async () => {
    const data = await listEvaluationJobs()
    setEvalJobs(Array.isArray(data) ? data : [])
  }, 5000)

  usePolling(async () => {
    if (tab === 'replay') {
      const s = await getReplayStatus()
      setReplayStatus(s)
    }
  }, 2000, tab === 'replay')

  const startEval = async () => {
    if (!selectedModel || !selectedDataset) { setError('Select model and dataset.'); return }
    setLoading(true); setError(null); setMessage(null)
    try {
      const r = await startEvaluation(selectedModel, selectedDataset, split)
      setMessage(`Evaluation job started — ID: ${r.job_id}`)
    } catch (e: unknown) {
      const msg = (e as { response?: { data?: { detail?: string } } })?.response?.data?.detail
      setError(msg ?? 'Failed to start evaluation.')
    } finally { setLoading(false) }
  }

  const runSynthetic = async () => {
    setLoading(true); setError(null)
    try {
      const r = await runSyntheticScenarios()
      setSynResult(r)
    } catch { setError('Synthetic scenario run failed.') }
    finally { setLoading(false) }
  }

  const handleReplayLoad = async () => {
    if (!replayFile) return
    setLoading(true)
    try {
      const r = await loadReplayFile(replayFile)
      setReplayStatus(r as { status: string; processed_records: number; total_records: number; label?: string })
      setMessage('File loaded. Press Play to start replay.')
    } catch { setError('Failed to load replay file.') }
    finally { setLoading(false) }
  }

  const replayControl = async (action: string) => {
    try {
      const r = await controlReplay(action, action === 'start' || action === 'resume' ? replaySpeed : undefined)
      setReplayStatus(r)
    } catch { setError('Replay control failed.') }
  }

  const TABS: { key: Tab; label: string }[] = [
    { key: 'synthetic', label: '🧪 Synthetic Integration Test' },
    { key: 'real', label: '📊 Real Dataset Evaluation' },
    { key: 'replay', label: '⏯️ Recorded Data Replay' },
  ]

  return (
    <div style={s.page}>
      <h1 style={s.title}>🔬 Test & Evaluation</h1>
      <p style={s.sub}>Three separate evaluation modes — results are kept independent.</p>

      <div style={s.tabs}>
        {TABS.map(t => (
          <button key={t.key} style={{ ...s.tab, ...(tab === t.key ? s.tabActive : {}) }}
            onClick={() => setTab(t.key)}>{t.label}</button>
        ))}
      </div>

      {message && <div style={s.msgBox}>✅ {message}</div>}
      {error && <div style={s.errBox}>⚠️ {error}</div>}

      {/* ── Synthetic ── */}
      {tab === 'synthetic' && (
        <div>
          <div style={s.synBanner}>
            🧪 SYNTHETIC INTEGRATION TEST — NOT LIVE SENSOR DATA
          </div>
          <p style={s.bodyText}>
            Runs deterministic scenarios through the full software pipeline: observation creation →
            metadata validation → correlation → event prioritization → event review.
            Shows expected vs actual results. Does NOT use real model inference.
          </p>
          <button style={{ ...s.btn, background: '#7c3aed', marginBottom: 20 }}
            disabled={loading} onClick={runSynthetic}>
            {loading ? '⏳ Running…' : '▶ Run All Synthetic Scenarios'}
          </button>

          {synResult && (
            <div>
              <div style={s.scoreRow}>
                <div style={{ ...s.scoreCard, borderColor: '#22c55e' }}>
                  <span style={s.scoreNum}>{synResult.passed}</span>
                  <span style={s.scoreLabel}>PASSED</span>
                </div>
                <div style={{ ...s.scoreCard, borderColor: '#ef4444' }}>
                  <span style={{ ...s.scoreNum, color: '#f87171' }}>{synResult.failed}</span>
                  <span style={s.scoreLabel}>FAILED</span>
                </div>
                <div style={{ ...s.scoreCard, borderColor: '#334155' }}>
                  <span style={s.scoreNum}>{synResult.total_scenarios}</span>
                  <span style={s.scoreLabel}>TOTAL</span>
                </div>
              </div>

              {Object.entries(synResult.scenarios).map(([name, sc]) => (
                <div key={name} style={{ ...s.scenarioCard, borderColor: sc.pass ? '#166534' : '#7f1d1d' }}>
                  <div style={s.scenarioHeader}>
                    <span style={{ color: sc.pass ? '#22c55e' : '#ef4444', fontSize: 16 }}>
                      {sc.pass ? '✅' : '❌'}
                    </span>
                    <strong style={{ color: '#f1f5f9' }}>{name}</strong>
                    <span style={{ color: '#64748b', fontSize: 12 }}>{sc.description}</span>
                  </div>
                  <div style={s.scenarioBody}>
                    <div style={s.scenarioCol}>
                      <span style={s.evLabel}>Expected</span>
                      <pre style={s.pre}>{JSON.stringify(sc.expected, null, 2)}</pre>
                    </div>
                    <div style={s.scenarioCol}>
                      <span style={s.evLabel}>Actual</span>
                      <pre style={s.pre}>{JSON.stringify(sc.actual, null, 2)}</pre>
                    </div>
                  </div>
                  <p style={{ margin: '4px 0 0', fontSize: 11, color: '#475569' }}>{sc.label}</p>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* ── Real Evaluation ── */}
      {tab === 'real' && (
        <div>
          <p style={s.bodyText}>
            Runs model validation on a downloaded dataset split. Requires the dataset to be installed
            and the model to have a checkpoint. Metrics are computed from actual predictions vs ground truth.
          </p>
          <div style={s.formRow}>
            <div style={s.formGroup}>
              <label style={s.label}>Model</label>
              <select style={s.select} value={selectedModel} onChange={e => setSelectedModel(e.target.value)}>
                <option value="">— select model —</option>
                {models.map(m => <option key={m.id} value={m.id}>{m.name}</option>)}
              </select>
            </div>
            <div style={s.formGroup}>
              <label style={s.label}>Dataset</label>
              <select style={s.select} value={selectedDataset} onChange={e => setSelectedDataset(e.target.value)}>
                <option value="">— select dataset —</option>
                {datasets.map(d => <option key={d.id} value={d.name}>{d.name} [{d.validation_status}]</option>)}
              </select>
            </div>
            <div style={s.formGroup}>
              <label style={s.label}>Split</label>
              <select style={s.select} value={split} onChange={e => setSplit(e.target.value)}>
                {['val', 'test', 'train'].map(sp => <option key={sp} value={sp}>{sp}</option>)}
              </select>
            </div>
            <button style={{ ...s.btn, alignSelf: 'flex-end' }} disabled={loading} onClick={startEval}>
              {loading ? '⏳' : '▶ Start Evaluation'}
            </button>
          </div>

          <div style={s.card}>
            <h3 style={s.cardTitle}>Evaluation Jobs</h3>
            {evalJobs.length === 0
              ? <p style={s.noData}>No evaluation jobs yet.</p>
              : (
                <table style={s.table}>
                  <thead><tr>{['Model', 'Dataset', 'Split', 'Status', 'Precision', 'Recall', 'mAP@50', 'mAP@50:95', 'Latency', 'Time'].map(h =>
                    <th key={h} style={s.th}>{h}</th>)}</tr></thead>
                  <tbody>
                    {evalJobs.map(j => (
                      <tr key={j.id}>
                        <td style={{ ...s.td, fontSize: 11 }}>{j.model_id.slice(0, 8)}</td>
                        <td style={s.td}>{j.dataset_name}</td>
                        <td style={s.td}>{j.dataset_split}</td>
                        <td style={s.td}><span style={{ color: statusColour(j.status), fontWeight: 600 }}>{j.status}</span></td>
                        <td style={s.td}>{j.precision != null ? `${(j.precision * 100).toFixed(1)}%` : '—'}</td>
                        <td style={s.td}>{j.recall != null ? `${(j.recall * 100).toFixed(1)}%` : '—'}</td>
                        <td style={s.td}>{j.map50 != null ? `${(j.map50 * 100).toFixed(1)}%` : '—'}</td>
                        <td style={s.td}>{j.map50_95 != null ? `${(j.map50_95 * 100).toFixed(1)}%` : '—'}</td>
                        <td style={s.td}>{j.avg_inference_ms != null ? `${j.avg_inference_ms.toFixed(1)}ms` : '—'}</td>
                        <td style={s.td}>{formatTs(j.completed_at ?? j.started_at)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            {evalJobs.length > 0 && (
              <p style={{ color: '#475569', fontSize: 11, marginTop: 8 }}>
                Metrics are computed from actual model predictions vs ground-truth labels.
                FAILED jobs indicate missing dataset, missing checkpoint, or an error — check backend logs.
              </p>
            )}
          </div>
        </div>
      )}

      {/* ── Replay ── */}
      {tab === 'replay' && (
        <div>
          <div style={s.replayBanner}>
            ⏯️ RECORDED DATA REPLAY — NOT A LIVE SENSOR FEED
          </div>
          <p style={s.bodyText}>
            Upload a previously recorded JSON or CSV observation file to process in chronological order.
            Timestamps from the recorded file are used, not the current wall clock. Replay speed is adjustable.
          </p>
          <div style={s.replayControls}>
            <input type="file" accept=".json,.csv"
              onChange={e => { if (e.target.files?.[0]) setReplayFile(e.target.files[0]) }} />
            {replayFile && <span style={{ color: '#94a3b8', fontSize: 13 }}>📄 {replayFile.name}</span>}
            <button style={{ ...s.btn, background: '#0e7490' }} disabled={!replayFile || loading} onClick={handleReplayLoad}>
              📂 Load File
            </button>
          </div>

          <div style={s.replaySpeedRow}>
            <label style={s.label}>Replay Speed: <strong>{replaySpeed}×</strong></label>
            <input type="range" min={1} max={50} value={replaySpeed}
              onChange={e => setReplaySpeed(parseInt(e.target.value))} style={s.slider} />
          </div>

          <div style={s.replayButtons}>
            <button style={s.btn} onClick={() => replayControl('start')}>▶ Play</button>
            <button style={s.btn} onClick={() => replayControl('pause')}>⏸ Pause</button>
            <button style={s.btn} onClick={() => replayControl('resume')}>▶▶ Resume</button>
            <button style={{ ...s.btn, background: '#374151' }} onClick={() => replayControl('reset')}>⏹ Reset</button>
          </div>

          {replayStatus && (
            <div style={s.replayStatusCard}>
              <div style={s.replayRow}><span>Status</span><strong style={{ color: replayStatus.status === 'PLAYING' ? '#22c55e' : '#94a3b8' }}>{replayStatus.status}</strong></div>
              <div style={s.replayRow}><span>Progress</span><strong>{replayStatus.processed_records} / {replayStatus.total_records} records</strong></div>
              {replayStatus.total_records > 0 && (
                <div style={s.progressBar}>
                  <div style={{ ...s.progressFill, width: `${(replayStatus.processed_records / replayStatus.total_records) * 100}%` }} />
                </div>
              )}
              <p style={{ color: '#475569', fontSize: 11, margin: '8px 0 0' }}>
                {replayStatus.label ?? 'RECORDED DATA REPLAY — NOT A LIVE SENSOR FEED'}
              </p>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

const s: Record<string, React.CSSProperties> = {
  page: { padding: 24, color: '#e2e8f0', maxWidth: 1200 },
  title: { margin: '0 0 6px', fontSize: 22, fontWeight: 700, color: '#f1f5f9' },
  sub: { margin: '0 0 16px', color: '#64748b', fontSize: 13 },
  tabs: { display: 'flex', gap: 4, marginBottom: 20, borderBottom: '1px solid #334155' },
  tab: { background: 'none', border: 'none', borderBottom: '2px solid transparent', color: '#64748b', padding: '8px 16px', fontSize: 13, cursor: 'pointer', marginBottom: -1 },
  tabActive: { color: '#f1f5f9', borderBottom: '2px solid #3b82f6' },
  bodyText: { color: '#64748b', fontSize: 13, marginBottom: 16 },
  synBanner: { background: '#2d1b69', border: '1px solid #7c3aed', borderRadius: 8, padding: '10px 14px', color: '#c4b5fd', fontSize: 13, fontWeight: 600, marginBottom: 12 },
  replayBanner: { background: '#0c1a2e', border: '1px solid #0e7490', borderRadius: 8, padding: '10px 14px', color: '#67e8f9', fontSize: 13, fontWeight: 600, marginBottom: 12 },
  btn: { background: '#2563eb', color: '#fff', border: 'none', borderRadius: 8, padding: '9px 16px', fontSize: 13, fontWeight: 600, cursor: 'pointer' },
  msgBox: { background: '#0f2a1a', border: '1px solid #166534', borderRadius: 6, padding: '8px 12px', color: '#86efac', fontSize: 13, marginBottom: 12 },
  errBox: { background: '#1c080822', border: '1px solid #7f1d1d', borderRadius: 6, padding: '8px 12px', color: '#f87171', fontSize: 13, marginBottom: 12 },
  card: { background: '#1e293b', border: '1px solid #334155', borderRadius: 10, padding: 20, marginBottom: 16 },
  cardTitle: { margin: '0 0 12px', fontSize: 13, fontWeight: 600, color: '#94a3b8', textTransform: 'uppercase' },
  formRow: { display: 'flex', gap: 12, marginBottom: 16, flexWrap: 'wrap', alignItems: 'flex-end' },
  formGroup: { display: 'flex', flexDirection: 'column', gap: 6, minWidth: 160 },
  label: { fontSize: 13, color: '#94a3b8' },
  select: { background: '#0f172a', border: '1px solid #334155', borderRadius: 6, color: '#e2e8f0', padding: '8px 10px', fontSize: 13 },
  slider: { accentColor: '#3b82f6' },
  table: { width: '100%', borderCollapse: 'collapse' },
  th: { textAlign: 'left', padding: '6px 10px', fontSize: 11, color: '#64748b', borderBottom: '1px solid #334155', textTransform: 'uppercase' },
  td: { padding: '8px 10px', fontSize: 13, color: '#cbd5e1', borderBottom: '1px solid #1e293b' },
  noData: { color: '#475569', fontStyle: 'italic', fontSize: 13 },
  scoreRow: { display: 'flex', gap: 12, marginBottom: 20 },
  scoreCard: { background: '#1e293b', border: '1px solid', borderRadius: 8, padding: '12px 20px', display: 'flex', flexDirection: 'column', alignItems: 'center', minWidth: 80 },
  scoreNum: { fontSize: 28, fontWeight: 700, color: '#22c55e' },
  scoreLabel: { fontSize: 11, color: '#64748b', marginTop: 2 },
  scenarioCard: { background: '#1e293b', border: '1px solid', borderRadius: 8, padding: 16, marginBottom: 10 },
  scenarioHeader: { display: 'flex', gap: 10, alignItems: 'center', marginBottom: 8 },
  scenarioBody: { display: 'flex', gap: 16 },
  scenarioCol: { flex: 1 },
  evLabel: { fontSize: 11, color: '#64748b', textTransform: 'uppercase' },
  pre: { background: '#0f172a', padding: 8, borderRadius: 4, fontSize: 11, color: '#94a3b8', overflow: 'auto', margin: '4px 0 0' },
  replayControls: { display: 'flex', gap: 12, alignItems: 'center', marginBottom: 12, flexWrap: 'wrap' },
  replaySpeedRow: { display: 'flex', gap: 12, alignItems: 'center', marginBottom: 12, maxWidth: 400 },
  replayButtons: { display: 'flex', gap: 8, marginBottom: 16, flexWrap: 'wrap' },
  replayStatusCard: { background: '#1e293b', border: '1px solid #334155', borderRadius: 8, padding: 16 },
  replayRow: { display: 'flex', justifyContent: 'space-between', padding: '4px 0', borderBottom: '1px solid #0f172a' },
  progressBar: { height: 8, background: '#0f172a', borderRadius: 4, overflow: 'hidden', margin: '8px 0 0' },
  progressFill: { height: '100%', background: '#0e7490', borderRadius: 4, transition: 'width 0.4s' },
}
