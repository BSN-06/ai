import { useEffect, useMemo, useState } from 'react'
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import {
  controlReplay, exportReport, getReplayProcessed, getReplayStatus, listDatasets, listEvaluationJobs, listModels, listReports,
  loadReplayFile, runSyntheticScenarios, startEvaluation,
} from '../utils/api'
import type { Dataset, EvaluationJob, ModelRecord, ReplayState, Report, SyntheticRunResult } from '../types'
import { formatTs, sensorIcon, statusColour } from '../utils/format'
import { usePolling } from '../hooks/usePolling'
import {
  Badge, Banner, Btn, C, Card, Empty, Field, Grid, KV, Page, PRIORITY_COL, Progress, Stat, Table, downloadJson, errMsg, inputStyle, tooltipStyle,
} from '../components/ui'

type Tab = 'real' | 'replay' | 'synthetic' | 'reports'
const TABS: [Tab, string, string][] = [
  ['real', 'Real dataset evaluation', C.blue], ['replay', 'Recorded-data replay', C.cyan],
  ['synthetic', 'Synthetic integration tests', C.purple], ['reports', 'Reports', C.muted],
]

export default function Evaluation() {
  const [tab, setTab] = useState<Tab>('real')
  return (
    <Page title="Test & Evaluation" subtitle="Three separate execution modes with separate reports. Results from one mode are never presented as another.">
      <div style={{ display: 'flex', gap: 6, marginBottom: 16, flexWrap: 'wrap' }}>
        {TABS.map(([k, label, col]) => (
          <button key={k} onClick={() => setTab(k)} style={{
            padding: '8px 14px', borderRadius: 6, border: `1px solid ${tab === k ? col : C.border}`, cursor: 'pointer',
            background: tab === k ? `${col}22` : 'transparent', color: tab === k ? '#fff' : C.muted, fontWeight: 600, fontSize: 13,
          }}>{label}</button>
        ))}
      </div>
      {tab === 'real' && <RealEval />}
      {tab === 'replay' && <Replay />}
      {tab === 'synthetic' && <Synthetic />}
      {tab === 'reports' && <Reports />}
    </Page>
  )
}

// ───────────────────────────── Real evaluation ─────────────────────────────
function RealEval() {
  const [models, setModels] = useState<ModelRecord[]>([])
  const [datasets, setDatasets] = useState<Dataset[]>([])
  const [jobs, setJobs] = useState<EvaluationJob[]>([])
  const [modelId, setModelId] = useState('')
  const [ds, setDs] = useState('')
  const [split, setSplit] = useState('test')
  const [maxImages, setMaxImages] = useState('')
  const [sel, setSel] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    listModels().then(ms => { const t = ms.filter(m => m.sensor_type !== 'radar'); setModels(t); if (t[0]) setModelId(t[0].id) })
    listDatasets().then(setDatasets)
  }, [])
  usePolling(async () => setJobs(await listEvaluationJobs()), 3000)
  const model = models.find(m => m.id === modelId)
  const dsOpts = datasets.filter(d => d.sensor_type === model?.sensor_type)
  useEffect(() => { if (dsOpts.length && !dsOpts.some(d => d.name === ds)) setDs(dsOpts[0].name) }, [dsOpts, ds])
  const dsRec = datasets.find(d => d.name === ds)
  const job = jobs.find(j => j.id === sel) ?? jobs.find(j => j.status === 'COMPLETED') ?? jobs[0]

  const start = async () => {
    setError(null)
    try { const r = await startEvaluation(modelId, ds, split, maxImages ? +maxImages : undefined); setSel(r.job_id) } catch (e) { setError(errMsg(e)) }
  }
  const perClass = useMemo(() => Object.entries(job?.per_class_metrics ?? {})
    .filter(([, v]) => typeof v.ap50 === 'number').map(([k, v]) => ({ name: k, ap50: v.ap50 as number, ap: v.ap50_95 as number })), [job])

  return (
    <>
      <Banner kind="info">Metrics are computed from the model's actual predictions against the dataset's ground-truth labels
        (precision/recall at conf 0.25, IoU 0.5; COCO-style mAP@0.5 and mAP@0.5:0.95). Without an installed, validated dataset nothing is computed.</Banner>
      <Grid cols={3}>
        <Card title="Configure">
          <Field label="Model"><select value={modelId} onChange={e => setModelId(e.target.value)} style={inputStyle}>
            {models.map(m => <option key={m.id} value={m.id}>{m.name} · {m.status}</option>)}</select></Field>
          <Field label="Dataset" hint={dsRec ? `${dsRec.validation_status}` : ''}><select value={ds} onChange={e => setDs(e.target.value)} style={inputStyle}>
            {dsOpts.map(d => <option key={d.id} value={d.name}>{d.name} · {d.validation_status}</option>)}</select></Field>
          <div style={{ display: 'flex', gap: 8 }}>
            <Field label="Split"><select value={split} onChange={e => setSplit(e.target.value)} style={inputStyle}>
              <option value="test">test (held out)</option><option value="val">val</option></select></Field>
            <Field label="Max images"><input value={maxImages} onChange={e => setMaxImages(e.target.value)} placeholder="all" style={inputStyle} /></Field>
          </div>
          {dsRec?.validation_status !== 'VALIDATED' && <Banner kind="warn">Dataset not installed/validated — evaluation is unavailable. See the Datasets page.</Banner>}
          <Btn onClick={start} disabled={dsRec?.validation_status !== 'VALIDATED'}>Start evaluation</Btn>
          {error && <p style={{ color: C.red, fontSize: 12, marginTop: 8 }}>{error}</p>}
        </Card>
        <Card title="Evaluation jobs" style={{ gridColumn: 'span 2' }}>
          <Table head={['Job', 'Dataset / split', 'Model status', 'Images', 'mAP@0.5', 'mAP@0.5:0.95', 'Status', 'Completed']} empty="No evaluation jobs yet."
            rows={jobs.map(j => [<a onClick={() => setSel(j.id)} style={{ cursor: 'pointer', color: job?.id === j.id ? C.cyan : C.blue }}>{j.id.slice(0, 8)}</a>,
              `${j.dataset_name} / ${j.dataset_split}`, j.model_status_at_eval ?? '—', j.num_images ?? '—', j.map50?.toFixed(4) ?? '—',
              j.map50_95?.toFixed(4) ?? '—', <Badge color={statusColour(j.status)}>{j.status}</Badge>, formatTs(j.completed_at)])} />
        </Card>
      </Grid>
      {job && job.status === 'FAILED' && <Banner kind="error">Evaluation {job.id.slice(0, 8)} failed: {job.error_message}</Banner>}
      {job && job.status === 'COMPLETED' && (
        <>
          {job.warnings?.map((w, i) => <Banner key={i} kind="warn">{w}</Banner>)}
          <Grid cols={4}>
            <Stat label="Precision" value={job.precision?.toFixed(4) ?? '—'} />
            <Stat label="Recall" value={job.recall?.toFixed(4) ?? '—'} />
            <Stat label="mAP@0.5" value={job.map50?.toFixed(4) ?? '—'} color={C.blue} />
            <Stat label="mAP@0.5:0.95" value={job.map50_95?.toFixed(4) ?? '—'} color={C.blue} />
            <Stat label="Images evaluated" value={job.num_images ?? '—'} sub={`${job.dataset_name} / ${job.dataset_split}`} />
            <Stat label="Mean latency" value={job.avg_inference_ms != null ? `${job.avg_inference_ms} ms` : '—'} />
          </Grid>
          <Grid cols={2}>
            <Card title="Per-class AP">
              {perClass.length ? (
                <ResponsiveContainer width="100%" height={Math.max(200, perClass.length * 18)}>
                  <BarChart data={perClass} layout="vertical" margin={{ left: 40 }}><CartesianGrid stroke="#1f2b45" strokeDasharray="3 3" />
                    <XAxis type="number" domain={[0, 1]} tick={{ fill: C.muted, fontSize: 11 }} />
                    <YAxis type="category" dataKey="name" width={130} tick={{ fill: C.muted, fontSize: 10 }} />
                    <Tooltip contentStyle={tooltipStyle} /><Bar dataKey="ap50" name="AP@0.5" fill={C.blue} /><Bar dataKey="ap" name="AP@0.5:0.95" fill={C.purple} />
                  </BarChart>
                </ResponsiveContainer>) : <Empty>No classes with ground truth.</Empty>}
            </Card>
            <Card title="Run details">
              <KV items={[['Checkpoint', job.checkpoint_path], ['Model status at evaluation', job.model_status_at_eval],
                ['Evaluated at', formatTs(job.completed_at)], ['Report file', job.report_path]]} />
              <Table head={['Class', 'GT', 'Preds', 'P', 'R', 'AP50']}
                rows={Object.entries(job.per_class_metrics ?? {}).map(([k, v]) => [k, String(v.gt_instances), String(v.predictions),
                  String(v.precision ?? '—'), String(v.recall ?? '—'), String(v.ap50 ?? v.note ?? '—')])} />
            </Card>
          </Grid>
          {job.confusion_matrix && job.confusion_matrix.labels.length <= 25 && (
            <Card title={`Confusion matrix (IoU 0.5, conf 0.25 · ${job.confusion_matrix.axes})`}>
              <Table head={['pred \\ true', ...job.confusion_matrix.labels]}
                rows={job.confusion_matrix.matrix.map((r, i) => [<strong>{job.confusion_matrix!.labels[i]}</strong>, ...r.map(v => v || '')])} />
            </Card>
          )}
        </>
      )}
    </>
  )
}

// ───────────────────────────── Replay ─────────────────────────────
function Replay() {
  const [st, setSt] = useState<ReplayState | null>(null)
  const [file, setFile] = useState<File | null>(null)
  const [speed, setSpeed] = useState(10)
  const [recent, setRecent] = useState<Record<string, unknown>[]>([])
  const [error, setError] = useState<string | null>(null)
  const [loadInfo, setLoadInfo] = useState<Record<string, unknown> | null>(null)

  usePolling(async () => {
    try {
      const s = await getReplayStatus(); setSt(s)
      const p = await getReplayProcessed(500, 0); setRecent((p.records as Record<string, unknown>[]).slice(-12).reverse())
    } catch (e) { setError(errMsg(e)) }
  }, 1000)

  const load = async () => {
    if (!file) return
    setError(null)
    try { setLoadInfo(await loadReplayFile(file) as unknown as Record<string, unknown>) } catch (e) { setError(errMsg(e)) }
  }
  const ctl = async (action: string) => { try { await controlReplay(action, speed) } catch (e) { setError(errMsg(e)) } }

  return (
    <>
      <Banner kind="replay"><strong>RECORDED DATA REPLAY — NOT A LIVE SENSOR FEED.</strong> Records are replayed in recorded-timestamp order;
        the clock shown is the recorded time, not wall-clock time. Each record is validated, stored (provenance <em>recorded</em>) and
        correlated with earlier replayed records to create/update events. Replayed records are previously generated observations — no new model inference is executed.</Banner>
      {error && <Banner kind="error">{error}</Banner>}
      <Grid cols={3}>
        <Card title="Recorded file">
          <input type="file" accept=".json,.csv" onChange={e => setFile(e.target.files?.[0] ?? null)} style={{ ...inputStyle, padding: 6 }} />
          <div style={{ marginTop: 8 }}><Btn onClick={load} disabled={!file}>Load</Btn></div>
          {loadInfo && <p style={{ fontSize: 12, color: C.muted, marginTop: 8 }}>{String(loadInfo.total_records)} records loaded · {String(loadInfo.skipped_records)} skipped (no valid timestamp)</p>}
          <p style={{ fontSize: 11, color: C.dim, marginTop: 8 }}>Sample: <code>python generate_sample_replay.py</code></p>
        </Card>
        <Card title="Controls">
          <Field label={`Replay speed ×${speed}`}><input type="range" min={0.5} max={100} step={0.5} value={speed} onChange={e => setSpeed(+e.target.value)} style={{ width: '100%' }} /></Field>
          <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
            <Btn small onClick={() => ctl('start')} disabled={!st?.total_records || st.status === 'PLAYING'}>Start</Btn>
            <Btn small kind="ghost" onClick={() => ctl('pause')} disabled={st?.status !== 'PLAYING'}>Pause</Btn>
            <Btn small kind="ghost" onClick={() => ctl('resume')} disabled={st?.status !== 'PAUSED'}>Resume</Btn>
            <Btn small kind="danger" onClick={() => ctl('reset')} disabled={!st?.total_records}>Reset</Btn>
            <Btn small kind="ghost" onClick={() => ctl('speed')}>Apply speed</Btn>
          </div>
          <p style={{ fontSize: 11, color: C.dim, marginTop: 8 }}>Gaps between records are capped at 5 s of wall time.</p>
        </Card>
        <Card title="Status">
          {st ? (
            <>
              <KV items={[['State', <Badge color={st.status === 'PLAYING' ? C.cyan : st.status === 'COMPLETED' ? C.green : C.muted}>{st.status}</Badge>],
                ['Recorded clock', formatTs(st.current_timestamp)], ['Recorded range', `${formatTs(st.start_timestamp)} → ${formatTs(st.end_timestamp)}`],
                ['Speed', `×${st.replay_speed}`], ['Source', st.source_file ?? '—']]} />
              <div style={{ marginTop: 8 }}><Progress value={st.progress ?? 0} color={C.cyan} /></div>
            </>
          ) : <Empty>—</Empty>}
        </Card>
      </Grid>
      {st && (
        <>
          <Grid cols={4}>
            <Stat label="Processed" value={`${st.processed_records}/${st.total_records}`} />
            <Stat label="Queue" value={st.queue_remaining ?? 0} />
            <Stat label="Stored / rejected" value={`${st.stats?.saved ?? 0} / ${st.stats?.rejected ?? 0}`} />
            <Stat label="Events created / updated" value={`${st.stats?.events_created ?? 0} / ${st.stats?.events_updated ?? 0}`} color={C.cyan} />
          </Grid>
          <Grid cols={3}>
            <Card title="Incoming queue (next records)">
              <Table head={['Recorded time', 'Sensor', 'ID']} empty="Queue empty."
                rows={(st.queue_preview ?? []).map(q => [formatTs(q.observation_timestamp), `${sensorIcon(q.sensor_type ?? '')} ${q.sensor_type ?? '—'}`, (q.id ?? '—').slice(0, 10)])} />
            </Card>
            <Card title="Recently processed">
              <Table head={['Recorded time', 'Sensor', 'Outcome']} empty="Nothing processed yet."
                rows={recent.map(r => {
                  const o = (r._outcome ?? {}) as Record<string, unknown>
                  return [formatTs(String(r.observation_timestamp)), `${sensorIcon(String(r.sensor_type))} ${String(r.sensor_type)}`,
                    <span>{String(o.status ?? '—')}{o.event_action ? <Badge color={C.cyan}>event {String(o.event_action)}</Badge> : null}</span>]
                })} />
            </Card>
            <Card title="Event timeline (recorded time)">
              <Table head={['Recorded time', 'Action', 'Priority', 'Obs.']} empty="No events yet."
                rows={[...(st.event_timeline ?? [])].reverse().map(e => [formatTs(e.recorded_timestamp), e.action,
                  <Badge color={PRIORITY_COL[e.review_priority]}>{e.review_priority}</Badge>, e.observation_count])} />
            </Card>
          </Grid>
          {!!st.errors?.length && <Banner kind="error">{st.errors.join(' · ')}</Banner>}
          {st.status === 'COMPLETED' && <Banner kind="ok">Replay completed — a recorded-replay report was saved (see Reports tab).</Banner>}
        </>
      )}
    </>
  )
}

// ───────────────────────────── Synthetic ─────────────────────────────
function Synthetic() {
  const [res, setRes] = useState<SyntheticRunResult | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [shown, setShown] = useState(0)
  useEffect(() => {  // reveal scenario results one by one (animation only; results are already final)
    if (!res) return
    setShown(0)
    const n = Object.keys(res.scenarios).length
    const id = setInterval(() => setShown(s => { if (s >= n) { clearInterval(id); return s } return s + 1 }), 120)
    return () => clearInterval(id)
  }, [res])
  const run = async () => {
    setBusy(true); setError(null)
    try { setRes(await runSyntheticScenarios()) } catch (e) { setError(errMsg(e)) } finally { setBusy(false) }
  }
  const entries = Object.entries(res?.scenarios ?? {})
  return (
    <>
      <Banner kind="synthetic"><strong>SYNTHETIC DEMONSTRATION — NOT LIVE SENSOR DATA.</strong> Deterministic scenarios exercise the real software path
        (validation → storage → correlation → duplicates → prioritization → events → review). PASS/FAIL compares expected with actual software outcomes.
        These results say nothing about model accuracy or real-world fusion performance.</Banner>
      {error && <Banner kind="error">{error}</Banner>}
      <div style={{ display: 'flex', gap: 12, alignItems: 'center', marginBottom: 14 }}>
        <Btn onClick={run} disabled={busy}>{busy ? 'Running…' : 'Run synthetic integration suite'}</Btn>
        {res && <><Badge color={res.failed ? C.red : C.green}>{res.passed}/{res.total_scenarios} PASS</Badge>
          <span style={{ fontSize: 12, color: C.muted }}>run {res.run_tag} · report {res.report_id?.slice(0, 8)}</span></>}
      </div>
      {entries.slice(0, shown).map(([name, sc]) => (
        <Card key={name} title={<span><Badge color={sc.pass ? C.green : C.red}>{sc.pass ? 'PASS' : 'FAIL'}</Badge> &nbsp;{name.replace(/_/g, ' ')}</span>}
          style={{ marginBottom: 10, borderColor: sc.pass ? C.border : C.red }}>
          <p style={{ fontSize: 12.5, color: C.muted, marginBottom: 8 }}>{sc.description}</p>
          <Table head={['Check', 'Expected', 'Actual', 'Result']}
            rows={sc.checks.map(c => [c.check, String(c.expected), String(c.actual), <Badge color={c.pass ? C.green : C.red}>{c.pass ? 'PASS' : 'FAIL'}</Badge>])} />
          <div style={{ fontSize: 11.5, color: C.dim, marginTop: 6 }}>
            {sc.actual.validation.map(v => `${v.id}: ${v.status}${v.errors.length ? ' — ' + v.errors.join('; ') : ''}${v.warnings.length ? ' — ' + v.warnings.join('; ') : ''}`).join(' | ')}
            {sc.actual.event_priority && <> · event priority <strong>{sc.actual.event_priority}</strong></>}
          </div>
        </Card>
      ))}
    </>
  )
}

// ───────────────────────────── Reports ─────────────────────────────
function Reports() {
  const [reports, setReports] = useState<Report[]>([])
  const [type, setType] = useState('')
  const [open, setOpen] = useState<Report | null>(null)
  useEffect(() => { listReports(type || undefined).then(setReports) }, [type])
  return (
    <Grid cols={2}>
      <Card title={`Reports (${reports.length})`} right={
        <select value={type} onChange={e => setType(e.target.value)} style={{ ...inputStyle, width: 200 }}>
          <option value="">all types</option>
          {['satellite_detection', 'thermal_detection', 'radar_tracking', 'training', 'dataset_validation', 'fusion_tests', 'recorded_replay', 'synthetic_integration']
            .map(t => <option key={t} value={t}>{t}</option>)}
        </select>}>
        <div style={{ maxHeight: 560, overflowY: 'auto' }}>
          <Table head={['Type', 'Mode', 'Source', 'Samples', 'Status', 'Time']} empty="No reports yet."
            rows={reports.map(r => [<a onClick={async () => setOpen(await exportReport(r.id))} style={{ cursor: 'pointer', color: C.blue }}>{r.report_type}</a>,
              <Badge color={r.test_mode === 'synthetic' ? C.purple : r.test_mode === 'recorded_replay' ? C.cyan : C.blue}>{r.test_mode}</Badge>,
              (r.source ?? '—').slice(0, 30), r.num_samples, r.completion_status, formatTs(r.timestamp)])} />
        </div>
      </Card>
      <Card title="Report" right={open && <div style={{ display: 'flex', gap: 6 }}>
        <Btn small kind="ghost" onClick={() => downloadJson(`${open.report_type}_${open.id}.json`, open)}>Download JSON</Btn></div>}>
        {open ? (
          <>
            <KV items={[['Report ID', open.id], ['Type', open.report_type], ['Mode', open.test_mode], ['Source', open.source],
              ['Model / checkpoint', open.model_checkpoint], ['Samples', open.num_samples], ['Status', open.completion_status], ['Time', formatTs(open.timestamp)]]} />
            {!!open.warnings?.length && <Banner kind="warn">{open.warnings.join(' · ')}</Banner>}
            {!!open.limitations?.length && <Banner kind="info"><strong>Limitations:</strong> {open.limitations.join(' · ')}</Banner>}
            <h4 style={{ fontSize: 11, color: C.muted, margin: '10px 0 4px', textTransform: 'uppercase' }}>Metrics</h4>
            <pre style={{ fontSize: 11, color: '#cbd5e1', maxHeight: 300, overflow: 'auto' }}>{JSON.stringify(open.metrics, null, 2)}</pre>
          </>
        ) : <Empty>Select a report.</Empty>}
      </Card>
    </Grid>
  )
}
