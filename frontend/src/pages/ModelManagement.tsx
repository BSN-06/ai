import { useEffect, useState } from 'react'
import { getModelLoadStatus, listDatasets, listModels } from '../utils/api'
import type { Dataset, ModelRecord } from '../types'
import { formatTs, statusColour } from '../utils/format'
import { Badge, Banner, C, Card, Grid, KV, Page } from '../components/ui'

interface LoadStatus { checkpoint_exists: boolean; in_memory_loaded: boolean; runtime_status: Record<string, unknown> }

export default function ModelManagement() {
  const [models, setModels] = useState<ModelRecord[]>([])
  const [datasets, setDatasets] = useState<Dataset[]>([])
  const [load, setLoad] = useState<Record<string, LoadStatus>>({})

  useEffect(() => {
    listModels().then(ms => {
      setModels(ms)
      ms.forEach(m => getModelLoadStatus(m.id).then(s => setLoad(p => ({ ...p, [m.id]: s }))).catch(() => undefined))
    })
    listDatasets().then(setDatasets)
  }, [])

  return (
    <Page title="Model Management" subtitle="Each sensor has its own independently configured, trained and evaluated model. Statuses are taken from the registry and the filesystem.">
      <Banner kind="info">
        <strong>PRETRAINED</strong> = COCO-pretrained Ultralytics weights (COCO classes; not fine-tuned on the sensor dataset).
        <strong> TRAINED</strong> = a fine-tuning run completed and produced a checkpoint. Metrics appear only after an evaluation job
        computed them from predictions vs ground-truth labels.
      </Banner>
      <Grid cols={2}>
        {models.map(m => {
          const ds = datasets.find(d => d.name === m.dataset_name)
          const ls = load[m.id]
          const metrics = m.metrics as Record<string, unknown> | null
          return (
            <Card key={m.id} title={m.name} right={<div style={{ display: 'flex', gap: 6 }}>
              <Badge color={m.trained ? C.green : C.amber}>{m.status}</Badge>
              <Badge color={statusColour(m.evaluation_status)}>{m.evaluation_status}</Badge></div>}>
              <KV items={[
                ['Sensor', m.sensor_type], ['Architecture', `${m.architecture} · ${m.variant ?? ''}`], ['Backbone', m.backbone],
                ['Neck', m.neck], ['Head', m.head], ['Input size', `${m.input_size}px`],
                ['Target dataset', `${m.dataset_name ?? '—'}${ds ? ` (${ds.validation_status})` : ''}`],
                ['Pretrained init', m.pretrained ? 'Yes — COCO (Ultralytics)' : 'No'], ['Fine-tuned', m.trained ? 'Yes' : 'No'],
                ['Checkpoint', m.checkpoint_path ? `${m.checkpoint_path} ${ls ? (ls.checkpoint_exists ? '✔' : '✖ missing on disk') : ''}` : 'none (COCO base weights)'],
                ['Classes', m.class_names?.length ? `${m.class_names.length}: ${m.class_names.slice(0, 8).join(', ')}${m.class_names.length > 8 ? '…' : ''}` : 'COCO (80) until fine-tuned'],
                ['Last training', formatTs(m.last_training_time)], ['Loaded in memory', ls ? (ls.in_memory_loaded ? 'yes' : 'no') : '—'],
              ]} />
              <h4 style={h4}>Evaluation metrics</h4>
              {metrics ? (
                <KV items={[['mAP@0.5', String(metrics.map50 ?? '—')], ['mAP@0.5:0.95', String(metrics.map50_95 ?? '—')],
                  ['Precision / Recall', `${metrics.precision ?? '—'} / ${metrics.recall ?? '—'}`],
                  ['Latency', metrics.avg_inference_ms != null ? `${metrics.avg_inference_ms} ms/img` : '—'],
                  ['Dataset / split', `${metrics.dataset ?? '—'} / ${metrics.split ?? '—'} (${metrics.num_images ?? '—'} images)`],
                  ['Model status at eval', String(metrics.model_status_at_eval ?? '—')], ['Evaluated', formatTs(String(metrics.evaluated_at ?? ''))]]} />
              ) : <p style={{ fontSize: 12, color: C.dim }}>Not evaluated — no metrics are shown until an evaluation job completes.</p>}
              <h4 style={h4}>Known limitations</h4>
              <p style={{ fontSize: 12, color: '#cbd5e1' }}>{m.known_limitations}</p>
            </Card>
          )
        })}
        <Card title="radar_tracker (Module C)" right={<Badge color={C.green}>CLASSICAL · NO TRAINING</Badge>}>
          <KV items={[['Algorithm', 'Linear Kalman filter, constant-velocity state [x, y, vx, vy]'],
            ['Association', 'Global nearest neighbour (Hungarian assignment) with distance gate'],
            ['Track management', 'Create on unassigned detection; terminate after N consecutive misses'],
            ['Input', 'Structured detections: timestamp + x/y (m) or lat/lon (projected to local ENU metres)'],
            ['Target dataset', 'DLR real-world marine radar / AIS (maritime)']]} />
          <h4 style={h4}>Known limitations</h4>
          <p style={{ fontSize: 12, color: '#cbd5e1' }}>Not a neural model. Maritime data does not demonstrate aircraft tracking. Raw radar
            signals/images are not supported. Accuracy metrics require reference identities/positions in the input.</p>
        </Card>
      </Grid>
    </Page>
  )
}

const h4: React.CSSProperties = { fontSize: 11, color: C.muted, margin: '14px 0 6px', textTransform: 'uppercase', letterSpacing: 0.6 }
