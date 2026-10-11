import { useState } from 'react'
import { convertDataset, inspectDataset, listDatasets, updateDatasetPath, validateDataset } from '../utils/api'
import type { Dataset } from '../types'
import { formatTs } from '../utils/format'
import { usePolling } from '../hooks/usePolling'
import { Badge, Banner, Btn, C, Card, Field, Grid, KV, Page, Table, errMsg, inputStyle } from '../components/ui'

const COL: Record<string, string> = {
  VALIDATED: C.green, NOT_INSTALLED: C.dim, NOT_FOUND: C.red, NOT_VALIDATED: C.muted, EMPTY: C.red, INVALID: C.red,
  RAW_NEEDS_CONVERSION: C.amber, CONVERTING: C.blue, CONVERSION_FAILED: C.red,
}

export default function Datasets() {
  const [datasets, setDatasets] = useState<Dataset[]>([])
  const [paths, setPaths] = useState<Record<string, string>>({})
  const [raw, setRaw] = useState<Record<string, string>>({})
  const [msgs, setMsgs] = useState<Record<string, { kind: 'ok' | 'error' | 'warn'; text: string }>>({})
  const [reports, setReports] = useState<Record<string, Record<string, unknown>>>({})

  usePolling(async () => setDatasets(await listDatasets()), 4000)
  const say = (id: string, kind: 'ok' | 'error' | 'warn', text: string) => setMsgs(p => ({ ...p, [id]: { kind, text } }))

  const savePath = async (d: Dataset) => {
    try { const r = await updateDatasetPath(d.id, paths[d.id]); say(d.id, 'ok', `Path set: ${r.local_path}`); setDatasets(await listDatasets()) }
    catch (e) { say(d.id, 'error', errMsg(e)) }
  }
  const validate = async (d: Dataset) => {
    try {
      const r = await validateDataset(d.id)
      setReports(p => ({ ...p, [d.id]: r.report ?? r }))
      say(d.id, r.status === 'VALIDATED' ? 'ok' : 'warn', `${r.status}: ${r.message}`); setDatasets(await listDatasets())
    } catch (e) { say(d.id, 'error', errMsg(e)) }
  }
  const inspect = async (d: Dataset) => {
    try { const r = await inspectDataset(d.id); say(d.id, 'ok', `Detected format: ${r.format ?? '—'} · images ${r.image_count ?? 0} · ${((r.size_bytes ?? 0) / 1e9).toFixed(2)} GB`) }
    catch (e) { say(d.id, 'error', errMsg(e)) }
  }
  const convert = async (d: Dataset) => {
    try { const r = await convertDataset(d.id, raw[d.id]); say(d.id, 'ok', `Conversion started (${r.format}) → ${r.output}`) }
    catch (e) { say(d.id, 'error', errMsg(e)) }
  }

  return (
    <Page title="Dataset Management" subtitle="No dataset is downloaded automatically. Each requires registration/licence acceptance and manual download.">
      <Banner kind="info">
        Place datasets under <code>backend/data/datasets/</code> (or add a folder to the <code>ALLOWED_DATASET_ROOTS</code> environment variable).
        Paths outside these roots are rejected. Raw xView (GeoJSON) and FLIR ADAS (COCO JSON) are converted to YOLO format, then validated:
        label syntax, class ids, image/label pairing, split presence and cross-split leakage.
      </Banner>
      <Grid cols={2}>
        {datasets.map(d => {
          const rep = reports[d.id] as { splits?: Record<string, Record<string, number>>; class_instances?: Record<string, number> } | undefined
          return (
            <Card key={d.id} title={`${d.name} · ${d.sensor_type}`} right={<Badge color={COL[d.validation_status] ?? C.muted}>{d.validation_status}</Badge>}>
              <KV items={[
                ['Source', d.source_url ? <a href={d.source_url} target="_blank" rel="noreferrer" style={{ color: C.blue }}>{d.source_url}</a> : '—'],
                ['Local path', d.local_path ?? 'not set'], ['Annotation format', d.annotation_format], ['Files', d.file_count ?? '—'],
                ['Splits (train / val / test)', `${d.train_count ?? '—'} / ${d.val_count ?? '—'} / ${d.test_count ?? '—'}`],
                ['Classes', d.classes ? `${d.classes.length}: ${d.classes.slice(0, 10).join(', ')}${d.classes.length > 10 ? '…' : ''}` : '—'],
                ['Access requirements', d.access_requirements], ['Registered', formatTs(d.created_at)],
              ]} />
              {d.validation_message && <p style={{ fontSize: 12, color: d.validation_status === 'VALIDATED' ? C.green : C.amber, marginTop: 8 }}>{d.validation_message}</p>}
              <details style={{ marginTop: 10 }}>
                <summary style={{ cursor: 'pointer', fontSize: 12, color: C.muted }}>Setup instructions</summary>
                <pre style={{ fontSize: 11.5, whiteSpace: 'pre-wrap', color: '#cbd5e1', marginTop: 6 }}>{d.setup_instructions}</pre>
              </details>
              {d.sensor_type !== 'radar' && (
                <div style={{ marginTop: 10 }}>
                  <Field label="Raw dataset folder (convert → YOLO)">
                    <div style={{ display: 'flex', gap: 6 }}>
                      <input value={raw[d.id] ?? ''} onChange={e => setRaw(p => ({ ...p, [d.id]: e.target.value }))}
                        placeholder={d.name === 'xView' ? 'xview_raw' : d.name === 'FLIR_ADAS' ? 'flir_adas_raw' : 'dota_raw'} style={inputStyle} />
                      <Btn small onClick={() => convert(d)} disabled={!raw[d.id] || d.validation_status === 'CONVERTING'}>Convert</Btn>
                    </div>
                  </Field>
                </div>
              )}
              <Field label="Converted / prepared dataset folder">
                <div style={{ display: 'flex', gap: 6 }}>
                  <input value={paths[d.id] ?? ''} onChange={e => setPaths(p => ({ ...p, [d.id]: e.target.value }))}
                    placeholder={d.local_path ?? `${d.name.toLowerCase()}_yolo`} style={inputStyle} />
                  <Btn small kind="ghost" onClick={() => savePath(d)} disabled={!paths[d.id]}>Set path</Btn>
                </div>
              </Field>
              <div style={{ display: 'flex', gap: 6 }}>
                <Btn small onClick={() => validate(d)}>Validate</Btn>
                <Btn small kind="ghost" onClick={() => inspect(d)} disabled={!d.local_path}>Inspect</Btn>
              </div>
              {msgs[d.id] && <div style={{ marginTop: 10 }}><Banner kind={msgs[d.id].kind}>{msgs[d.id].text}</Banner></div>}
              {rep?.splits && Object.keys(rep.splits).length > 0 && (
                <Table head={['Split', 'Images', 'Labels', 'No label', 'Orphan labels', 'Bad lines']}
                  rows={Object.entries(rep.splits).map(([k, v]) => [k, v.images, v.labels, v.images_without_labels, v.orphan_labels, v.invalid_label_lines])} />
              )}
            </Card>
          )
        })}
      </Grid>
    </Page>
  )
}
