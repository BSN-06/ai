import { useEffect, useMemo, useState } from 'react'
import { listModels, runSatelliteInference, runThermalInference } from '../utils/api'
import type { InferenceResult, ModelRecord } from '../types'
import {
  Badge, Banner, Btn, C, Card, Field, Grid, KV, Page, Stat, Table, downloadCsv, downloadJson, errMsg, inputStyle,
} from './ui'
import { fmtConf } from '../utils/format'

type Sensor = 'satellite' | 'thermal'

const COPY: Record<Sensor, { title: string; subtitle: string; dataset: string; note: string }> = {
  satellite: {
    title: 'Satellite / Aerial Analysis',
    subtitle: 'Module A — YOLO object detection on satellite or aerial imagery (real model inference).',
    dataset: 'xView (fine-tuning target)',
    note: 'Use tiled inference for large scenes: boxes are detected per tile and mapped back to full-image pixel coordinates.',
  },
  thermal: {
    title: 'Thermal / Infrared Analysis',
    subtitle: 'Module B — separate YOLO detector for thermal imagery (own checkpoint and class mapping).',
    dataset: 'FLIR ADAS thermal (fine-tuning target)',
    note: 'FLIR ADAS is automotive data (people, vehicles). Results do not demonstrate aircraft or military-target detection.',
  },
}

export default function ImageAnalysis({ sensor }: { sensor: Sensor }) {
  const copy = COPY[sensor]
  const [models, setModels] = useState<ModelRecord[]>([])
  const [modelId, setModelId] = useState('')
  const [file, setFile] = useState<File | null>(null)
  const [preview, setPreview] = useState<string | null>(null)
  const [conf, setConf] = useState(0.25)
  const [iou, setIou] = useState(0.45)
  const [tile, setTile] = useState(0)
  const [capturedAt, setCapturedAt] = useState('')
  const [lat, setLat] = useState('')
  const [lon, setLon] = useState('')
  const [unc, setUnc] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<InferenceResult | null>(null)

  useEffect(() => {
    listModels().then(ms => {
      const mine = ms.filter(m => m.sensor_type === sensor)
      setModels(mine)
      if (mine.length) setModelId(mine[0].id)
    }).catch(e => setError(errMsg(e)))
  }, [sensor])

  useEffect(() => {
    if (!file) { setPreview(null); return }
    const url = URL.createObjectURL(file)
    setPreview(url)
    return () => URL.revokeObjectURL(url)
  }, [file])

  const model = useMemo(() => models.find(m => m.id === modelId), [models, modelId])

  const run = async () => {
    if (!file) return
    setBusy(true); setError(null); setResult(null)
    try {
      const opts = {
        confidence: conf, iou, modelId: modelId || undefined, tileSize: tile || undefined,
        capturedAt: capturedAt ? new Date(capturedAt).toISOString() : undefined,
        latitude: lat !== '' ? Number(lat) : undefined, longitude: lon !== '' ? Number(lon) : undefined,
        positionUncertaintyM: unc !== '' ? Number(unc) : undefined,
      }
      const r = sensor === 'satellite' ? await runSatelliteInference(file, opts) : await runThermalInference(file, opts)
      setResult(r)
    } catch (e) {
      setError(errMsg(e))
    } finally {
      setBusy(false)
    }
  }

  const classCounts = useMemo(() => {
    const m: Record<string, number> = {}
    result?.detections.forEach(d => { m[d.class_name] = (m[d.class_name] ?? 0) + 1 })
    return Object.entries(m).sort((a, b) => b[1] - a[1])
  }, [result])

  return (
    <Page title={copy.title} subtitle={copy.subtitle}
      right={<Badge color={C.blue}>MODE A · REAL MODEL INFERENCE</Badge>}>
      <Banner kind="info">{copy.note}</Banner>
      <Grid cols={3}>
        <Card title="1 · Input image">
          <input type="file" accept=".jpg,.jpeg,.png,.tif,.tiff,.bmp" onChange={e => { setFile(e.target.files?.[0] ?? null); setResult(null) }}
            style={{ ...inputStyle, padding: 6 }} />
          {preview && <img src={preview} alt="preview" style={{ width: '100%', marginTop: 10, borderRadius: 6, maxHeight: 220, objectFit: 'contain', background: '#000' }} />}
          {file && <p style={{ fontSize: 11, color: C.dim, marginTop: 6 }}>{file.name} · {(file.size / 1024).toFixed(0)} KB</p>}
        </Card>

        <Card title="2 · Model & thresholds">
          <Field label="Model">
            <select value={modelId} onChange={e => setModelId(e.target.value)} style={inputStyle}>
              {models.map(m => <option key={m.id} value={m.id}>{m.name} · {m.status}</option>)}
            </select>
          </Field>
          {model && (
            <div style={{ marginBottom: 10 }}>
              <Badge color={model.trained ? C.green : C.amber}>{model.trained ? 'FINE-TUNED' : 'PRETRAINED (COCO)'}</Badge>{' '}
              <span style={{ fontSize: 11, color: C.dim }}>{model.trained ? `checkpoint: ${model.checkpoint_path}` : `Not fine-tuned on ${copy.dataset}`}</span>
            </div>
          )}
          <Field label={`Confidence threshold: ${conf.toFixed(2)}`}>
            <input type="range" min={0.01} max={0.95} step={0.01} value={conf} onChange={e => setConf(+e.target.value)} style={{ width: '100%' }} />
          </Field>
          <Field label={`NMS IoU threshold: ${iou.toFixed(2)}`}>
            <input type="range" min={0.1} max={0.9} step={0.05} value={iou} onChange={e => setIou(+e.target.value)} style={{ width: '100%' }} />
          </Field>
          {sensor === 'satellite' && (
            <Field label="Tiled inference" hint="Splits large scenes into overlapping tiles (64 px overlap).">
              <select value={tile} onChange={e => setTile(+e.target.value)} style={inputStyle}>
                <option value={0}>Off (whole image)</option>
                <option value={512}>512 px tiles</option>
                <option value={640}>640 px tiles</option>
                <option value={1024}>1024 px tiles</option>
              </select>
            </Field>
          )}
        </Card>

        <Card title="3 · Optional metadata (operator supplied)">
          <Field label="Capture time" hint="Leave empty if unknown — no timestamp will be invented.">
            <input type="datetime-local" value={capturedAt} onChange={e => setCapturedAt(e.target.value)} style={inputStyle} />
          </Field>
          <div style={{ display: 'flex', gap: 8 }}>
            <Field label="Scene latitude"><input value={lat} onChange={e => setLat(e.target.value)} placeholder="e.g. 51.5" style={inputStyle} /></Field>
            <Field label="Scene longitude"><input value={lon} onChange={e => setLon(e.target.value)} placeholder="e.g. -0.12" style={inputStyle} /></Field>
          </div>
          <Field label="Position uncertainty (m)" hint="Footprint radius of the scene location.">
            <input value={unc} onChange={e => setUnc(e.target.value)} placeholder="e.g. 250" style={inputStyle} />
          </Field>
          <Btn onClick={run} disabled={!file || busy}>{busy ? 'Running inference…' : 'Run inference'}</Btn>
        </Card>
      </Grid>

      {error && <Banner kind="error"><strong>Inference failed:</strong> {error}</Banner>}

      {result && (
        <>
          {result.warnings?.map((w, i) => <Banner key={i} kind={w.includes('MODALITY') ? 'error' : 'warn'}>{w}</Banner>)}
          {result.model_note && <Banner kind="warn">{result.model_note}</Banner>}
          <Grid cols={4}>
            <Stat label="Detections" value={result.num_detections} />
            <Stat label="Inference latency" value={`${result.inference_time_ms.toFixed(0)} ms`} sub={`device: ${result.device}`} />
            <Stat label="Model status" value={result.model_status ?? '—'} color={result.model_status === 'FINE_TUNED' ? C.green : C.amber} />
            <Stat label="Image" value={`${result.image_width}×${result.image_height}`} sub={result.tiled ? `tiled · ${result.tile_size}px` : 'single pass'} />
          </Grid>
          <Grid cols={2}>
            <Card title="Original">
              <img src={result.image_url} alt="original" style={{ width: '100%', borderRadius: 6, background: '#000' }} />
            </Card>
            <Card title="Annotated prediction">
              <img src={result.annotated_image_url} alt="annotated" style={{ width: '100%', borderRadius: 6, background: '#000' }} />
            </Card>
          </Grid>
          <Grid cols={2}>
            <Card title={`Detections (${result.detections.length})`} right={
              <div style={{ display: 'flex', gap: 6 }}>
                <Btn small kind="ghost" onClick={() => downloadJson(`${sensor}_inference.json`, result)}>JSON</Btn>
                <Btn small kind="ghost" onClick={() => downloadCsv(`${sensor}_detections.csv`, result.detections as unknown as Record<string, unknown>[])}>CSV</Btn>
              </div>}>
              <div style={{ maxHeight: 360, overflowY: 'auto' }}>
                <Table head={['#', 'Class', 'Confidence', 'x1', 'y1', 'x2', 'y2']}
                  empty="No detections above the threshold."
                  rows={result.detections.map((d, i) => [i + 1, d.class_name, fmtConf(d.confidence),
                    d.bbox_x1.toFixed(0), d.bbox_y1.toFixed(0), d.bbox_x2.toFixed(0), d.bbox_y2.toFixed(0)])} />
              </div>
            </Card>
            <Card title="Run metadata">
              <KV items={[
                ['Model', result.model_name], ['Checkpoint', result.checkpoint], ['Status', result.model_status],
                ['Processed at (UTC)', result.timestamp], ['Confidence / IoU', `${result.confidence_threshold} / ${iou}`],
                ['Classes in model', `${result.class_names?.length ?? '—'}`], ['Observations saved', `${result.observation_ids?.length ?? 0}`],
                ['Provenance', 'real_inference'],
                ...(result.modality_check ? [['Modality check', `${result.modality_check.thermal_like ? 'thermal-like' : 'NOT thermal-like'} (mean saturation ${result.modality_check.mean_saturation})`] as [string, string]] : []),
              ]} />
              <h4 style={{ fontSize: 11, color: C.muted, margin: '14px 0 6px', textTransform: 'uppercase' }}>Detections per class</h4>
              {classCounts.length ? classCounts.map(([k, v]) => (
                <div key={k} style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12.5, padding: '2px 0' }}>
                  <span>{k}</span><span style={{ color: C.muted }}>{v}</span>
                </div>
              )) : <span style={{ fontSize: 12, color: C.dim }}>—</span>}
            </Card>
          </Grid>
        </>
      )}
    </Page>
  )
}
