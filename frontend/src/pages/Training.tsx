import { useEffect, useMemo, useState } from 'react'
import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import {
  cancelTrainingJob, getHealth, getTrainingJob, listDatasets, listModels, listTrainingJobs, startTraining, trainingPreflight,
} from '../utils/api'
import type { Dataset, ModelRecord, TrainingJob } from '../types'
import { formatTs, statusColour } from '../utils/format'
import { usePolling } from '../hooks/usePolling'
import {
  Badge, Banner, Btn, C, Card, Empty, Field, Grid, KV, Page, Progress, Table, errMsg, inputStyle, tooltipStyle,
} from '../components/ui'

const LOSS_KEYS = ['train/box_loss', 'train/cls_loss', 'train/dfl_loss', 'val/box_loss', 'val/cls_loss', 'val/dfl_loss']
const METRIC_KEYS = ['metrics/mAP50(B)', 'metrics/mAP50-95(B)', 'metrics/precision(B)', 'metrics/recall(B)']
const LINE_COLS = [C.blue, C.amber, C.purple, C.cyan, C.red, C.green]

export default function Training() {
  const [models, setModels] = useState<ModelRecord[]>([])
  const [datasets, setDatasets] = useState<Dataset[]>([])
  const [jobs, setJobs] = useState<TrainingJob[]>([])
  const [modelId, setModelId] = useState('')
  const [dataset, setDataset] = useState('')
  const [epochs, setEpochs] = useState(50)
  const [batch, setBatch] = useState(16)
  const [imgsz, setImgsz] = useState(640)
  const [lr, setLr] = useState('')
  const [device, setDevice] = useState('cpu')
  const [gpu, setGpu] = useState(false)
  const [preflight, setPreflight] = useState<{ ok: boolean; problems: string[]; running_jobs: number } | null>(null)
  const [selJob, setSelJob] = useState<string | null>(null)
  const [jobDetail, setJobDetail] = useState<TrainingJob | null>(null)
  const [msg, setMsg] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    listModels().then(ms => { const t = ms.filter(m => m.sensor_type !== 'radar'); setModels(t); if (t[0]) setModelId(t[0].id) })
    listDatasets().then(setDatasets)
    getHealth().then(h => setGpu(h.gpu_available))
  }, [])

  const model = models.find(m => m.id === modelId)
  const dsOptions = useMemo(() => datasets.filter(d => d.sensor_type === model?.sensor_type), [datasets, model])
  useEffect(() => { if (dsOptions.length && !dsOptions.some(d => d.name === dataset)) setDataset(dsOptions[0].name) }, [dsOptions, dataset])
  useEffect(() => {
    if (modelId && dataset) trainingPreflight(modelId, dataset, device).then(setPreflight).catch(e => setError(errMsg(e)))
  }, [modelId, dataset, device])

  usePolling(async () => {
    const js = await listTrainingJobs()
    setJobs(js)
    const id = selJob ?? js[0]?.id
    if (id) setJobDetail(await getTrainingJob(id))
  }, 3000)

  const submit = async () => {
    setError(null); setMsg(null)
    try {
      const r = await startTraining({ model_id: modelId, dataset_name: dataset, epochs, batch_size: batch, image_size: imgsz,
        learning_rate: lr ? Number(lr) : undefined, device })
      setMsg(`Training job ${r.job_id} started in the background.`); setSelJob(r.job_id)
    } catch (e) { setError(errMsg(e)) }
  }

  const hist = jobDetail?.epoch_history ?? []
  const ds = datasets.find(d => d.name === dataset)

  return (
    <Page title="Model Training" subtitle="Fine-tune the satellite (xView) or thermal (FLIR ADAS) YOLO detector. Jobs run in a background worker; progress is persisted.">
      <Banner kind="warn">
        Training on full datasets needs substantial disk (xView ≈ 20 GB + tiles, FLIR ≈ 14 GB), RAM and time — days on CPU, hours on a GPU.
        Only one training job runs at a time. A model is marked TRAINED only after a run completes and produces a checkpoint.
      </Banner>
      <Grid cols={3}>
        <Card title="Configuration">
          <Field label="Model">
            <select value={modelId} onChange={e => setModelId(e.target.value)} style={inputStyle}>
              {models.map(m => <option key={m.id} value={m.id}>{m.name} ({m.sensor_type}) · {m.status}</option>)}
            </select>
          </Field>
          <Field label="Dataset" hint={ds ? `status: ${ds.validation_status} · train ${ds.train_count ?? '—'} / val ${ds.val_count ?? '—'} / test ${ds.test_count ?? '—'}` : undefined}>
            <select value={dataset} onChange={e => setDataset(e.target.value)} style={inputStyle}>
              {dsOptions.map(d => <option key={d.id} value={d.name}>{d.name} · {d.validation_status}</option>)}
            </select>
          </Field>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
            <Field label="Epochs"><input type="number" min={1} max={1000} value={epochs} onChange={e => setEpochs(+e.target.value)} style={inputStyle} /></Field>
            <Field label="Batch size"><input type="number" min={1} max={128} value={batch} onChange={e => setBatch(+e.target.value)} style={inputStyle} /></Field>
            <Field label="Image size"><input type="number" min={32} max={2048} step={32} value={imgsz} onChange={e => setImgsz(+e.target.value)} style={inputStyle} /></Field>
            <Field label="Learning rate (lr0)"><input value={lr} placeholder="default 0.01" onChange={e => setLr(e.target.value)} style={inputStyle} /></Field>
          </div>
          <Field label="Device" hint={gpu ? 'CUDA GPU detected.' : 'No CUDA GPU detected — CPU only.'}>
            <select value={device} onChange={e => setDevice(e.target.value)} style={inputStyle}>
              <option value="cpu">CPU</option>
              <option value="cuda" disabled={!gpu}>CUDA {gpu ? '' : '(unavailable)'}</option>
            </select>
          </Field>
        </Card>
        <Card title="Pre-flight checks">
          {preflight ? (
            preflight.ok
              ? <Banner kind="ok">All prerequisites met — dataset validated, dependencies installed.</Banner>
              : <Banner kind="error"><strong>Training cannot start:</strong><ul style={{ paddingLeft: 18 }}>{preflight.problems.map((p, i) => <li key={i}>{p}</li>)}</ul></Banner>
          ) : <Empty>Select a model and dataset.</Empty>}
          {preflight && preflight.running_jobs > 0 && <Banner kind="warn">A training job is already running.</Banner>}
          <Btn onClick={submit} disabled={!preflight?.ok || (preflight?.running_jobs ?? 0) > 0}>Start training</Btn>
          {msg && <p style={{ color: C.green, fontSize: 12, marginTop: 8 }}>{msg}</p>}
          {error && <p style={{ color: C.red, fontSize: 12, marginTop: 8 }}>{error}</p>}
          <h4 style={{ fontSize: 11, color: C.muted, margin: '14px 0 6px', textTransform: 'uppercase' }}>Equivalent command</h4>
          <code style={{ fontSize: 11, color: '#cbd5e1', display: 'block', background: C.panel2, padding: 8, borderRadius: 6, wordBreak: 'break-all' }}>
            python scripts/train.py --model {model?.name} --dataset {dataset} --epochs {epochs} --batch {batch} --imgsz {imgsz} --device {device}
          </code>
        </Card>
        <Card title="Jobs">
          <div style={{ maxHeight: 360, overflowY: 'auto' }}>
            <Table head={['Job', 'Dataset', 'Status', 'Epoch']} empty="No training jobs yet."
              rows={jobs.map(j => [<a onClick={() => setSelJob(j.id)} style={{ cursor: 'pointer', color: (selJob ?? jobs[0]?.id) === j.id ? C.cyan : C.blue }}>{j.id.slice(0, 8)}</a>,
                j.dataset_name, <Badge color={statusColour(j.status)}>{j.status}</Badge>, `${j.current_epoch}/${j.total_epochs}`])} />
          </div>
        </Card>
      </Grid>

      {jobDetail && (
        <>
          <Card title={`Job ${jobDetail.id.slice(0, 8)} · ${jobDetail.dataset_name}`} style={{ marginBottom: 14 }} right={
            (jobDetail.status === 'RUNNING' || jobDetail.status === 'PENDING') &&
            <Btn small kind="danger" disabled={jobDetail.cancel_requested} onClick={() => cancelTrainingJob(jobDetail.id)}>
              {jobDetail.cancel_requested ? 'Cancelling…' : 'Cancel job'}</Btn>}>
            <div style={{ marginBottom: 10 }}><Progress value={jobDetail.total_epochs ? jobDetail.current_epoch / jobDetail.total_epochs : 0} color={statusColour(jobDetail.status)} /></div>
            <Grid cols={2}>
              <KV items={[['Status', <Badge color={statusColour(jobDetail.status)}>{jobDetail.status}</Badge>],
                ['Epoch', `${jobDetail.current_epoch}/${jobDetail.total_epochs}`], ['Device', jobDetail.device],
                ['Batch / imgsz', `${jobDetail.batch_size} / ${jobDetail.image_size}`], ['Train loss (sum)', jobDetail.train_loss ?? '—'],
                ['Val loss (sum)', jobDetail.val_loss ?? '—'], ['Best val mAP@0.5', jobDetail.best_map50?.toFixed(4) ?? '—']]} />
              <KV items={[['Started', formatTs(jobDetail.started_at)], ['Completed', formatTs(jobDetail.completed_at)],
                ['Checkpoint', jobDetail.checkpoint_path ?? '—'], ['Error', jobDetail.error_message ?? '—']]} />
            </Grid>
          </Card>
          <Grid cols={2}>
            <Card title="Loss per epoch">
              {hist.length ? (
                <ResponsiveContainer width="100%" height={240}>
                  <LineChart data={hist}><CartesianGrid stroke="#1f2b45" strokeDasharray="3 3" />
                    <XAxis dataKey="epoch" tick={{ fill: C.muted, fontSize: 11 }} /><YAxis tick={{ fill: C.muted, fontSize: 11 }} />
                    <Tooltip contentStyle={tooltipStyle} /><Legend wrapperStyle={{ fontSize: 11 }} />
                    {LOSS_KEYS.filter(k => k in hist[0]).map((k, i) => <Line key={k} dataKey={k} stroke={LINE_COLS[i]} dot={false} strokeDasharray={k.startsWith('val') ? '4 3' : undefined} />)}
                  </LineChart>
                </ResponsiveContainer>) : <Empty>No completed epochs yet.</Empty>}
            </Card>
            <Card title="Validation metrics per epoch (Ultralytics, val split)">
              {hist.length ? (
                <ResponsiveContainer width="100%" height={240}>
                  <LineChart data={hist}><CartesianGrid stroke="#1f2b45" strokeDasharray="3 3" />
                    <XAxis dataKey="epoch" tick={{ fill: C.muted, fontSize: 11 }} /><YAxis domain={[0, 1]} tick={{ fill: C.muted, fontSize: 11 }} />
                    <Tooltip contentStyle={tooltipStyle} /><Legend wrapperStyle={{ fontSize: 11 }} />
                    {METRIC_KEYS.filter(k => k in hist[0]).map((k, i) => <Line key={k} dataKey={k} stroke={LINE_COLS[i]} dot={false} />)}
                  </LineChart>
                </ResponsiveContainer>) : <Empty>No completed epochs yet.</Empty>}
            </Card>
          </Grid>
          <Card title="Job log (tail)">
            <pre style={{ fontSize: 11, color: '#94a3b8', maxHeight: 260, overflow: 'auto', whiteSpace: 'pre-wrap' }}>{jobDetail.log_tail || 'No log yet.'}</pre>
          </Card>
        </>
      )}
    </Page>
  )
}
