import { useState, useRef } from 'react'
import { runThermalInference } from '../utils/api'
import { formatTs, fmtConf } from '../utils/format'
import type { InferenceResult } from '../types'

export default function ThermalAnalysis() {
  const [file, setFile] = useState<File | null>(null)
  const [preview, setPreview] = useState<string | null>(null)
  const [conf, setConf] = useState(0.25)
  const [iou, setIou] = useState(0.45)
  const [result, setResult] = useState<InferenceResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const fileRef = useRef<HTMLInputElement>(null)

  const handleFile = (f: File) => { setFile(f); setResult(null); setError(null); setPreview(URL.createObjectURL(f)) }
  const handleDrop = (e: React.DragEvent) => { e.preventDefault(); const f = e.dataTransfer.files[0]; if (f) handleFile(f) }

  const runInference = async () => {
    if (!file) return
    setLoading(true); setError(null)
    try {
      const r = await runThermalInference(file, conf, iou)
      setResult(r)
    } catch (e: unknown) {
      const msg = (e as { response?: { data?: { detail?: string } } })?.response?.data?.detail
      setError(msg ?? 'Thermal inference failed — check backend logs.')
    } finally { setLoading(false) }
  }

  const annotatedUrl = result?.annotated_image_path
    ? `/api/files/uploads/${result.annotated_image_path.split(/[\\/]/).pop()}`
    : null

  return (
    <div style={s.page}>
      <h1 style={s.title}>🌡️ Thermal / Infrared Analysis</h1>
      <p style={s.sub}>
        Upload a thermal or infrared image for object detection. The thermal model is a separate
        YOLO instance from the satellite model. Dataset: FLIR ADAS (requires registration at FLIR website).
      </p>

      <div style={s.syntheticBanner}>
        ⚠️ <strong>Domain Note:</strong> The FLIR ADAS dataset contains automotive thermal data (cars,
        pedestrians, cyclists). Detection classes reflect that dataset. This does NOT represent
        military-target or aircraft detection capability.
      </div>

      <div style={s.row}>
        <div
          style={{ ...s.dropZone, borderColor: file ? '#10b981' : '#334155' }}
          onDragOver={e => e.preventDefault()} onDrop={handleDrop}
          onClick={() => fileRef.current?.click()}
        >
          <input ref={fileRef} type="file" accept=".jpg,.jpeg,.png,.tif,.tiff,.bmp"
            style={{ display: 'none' }} onChange={e => { if (e.target.files?.[0]) handleFile(e.target.files[0]) }} />
          {preview
            ? <img src={preview} alt="preview" style={s.previewImg} />
            : <div style={s.dropHint}><span style={{ fontSize: 36 }}>🌡️</span><br />Drop thermal image here or click<br /><span style={{ fontSize: 12, color: '#475569' }}>JPG · PNG · TIF</span></div>}
        </div>

        <div style={s.settingsCard}>
          <h3 style={s.cardTitle}>Inference Settings</h3>
          <label style={s.label}>Confidence: <strong>{conf.toFixed(2)}</strong></label>
          <input type="range" min={0.05} max={0.95} step={0.05} value={conf}
            onChange={e => setConf(parseFloat(e.target.value))} style={s.slider} />
          <label style={s.label}>IoU: <strong>{iou.toFixed(2)}</strong></label>
          <input type="range" min={0.1} max={0.9} step={0.05} value={iou}
            onChange={e => setIou(parseFloat(e.target.value))} style={s.slider} />
          <button style={{ ...s.btn, opacity: (!file || loading) ? 0.5 : 1, background: '#059669' }}
            disabled={!file || loading} onClick={runInference}>
            {loading ? '⏳ Running…' : '▶ Run Thermal Inference'}
          </button>
          {error && <div style={s.errBox}>⚠️ {error}</div>}
          {result && (
            <div style={s.metaBox}>
              <div style={s.metaRow}><span>Model</span><strong>{result.model_name}</strong></div>
              <div style={s.metaRow}><span>Device</span><strong>{result.device?.toUpperCase()}</strong></div>
              <div style={s.metaRow}><span>Latency</span><strong>{result.inference_time_ms?.toFixed(1)} ms</strong></div>
              <div style={s.metaRow}><span>Detections</span><strong>{result.num_detections}</strong></div>
              {result.warning && <div style={{ ...s.errBox, borderColor: '#92400e', color: '#fbbf24', marginTop: 8 }}>⚠️ {result.warning}</div>}
            </div>
          )}
        </div>
      </div>

      {result && (
        <>
          <div style={s.row}>
            <div style={s.imgPane}><h3 style={s.cardTitle}>Original</h3>
              {preview && <img src={preview} alt="original" style={s.resultImg} />}
            </div>
            <div style={s.imgPane}><h3 style={s.cardTitle}>Annotated ({result.num_detections} detections)</h3>
              {annotatedUrl
                ? <img src={annotatedUrl} alt="annotated" style={s.resultImg} />
                : <p style={s.noData}>Annotated image unavailable.</p>}
            </div>
          </div>

          <div style={s.card}>
            <h3 style={s.cardTitle}>Detection Results — Thermal Model</h3>
            {result.detections.length === 0
              ? <p style={s.noData}>No detections above threshold.</p>
              : (
                <table style={s.table}>
                  <thead><tr>{['#', 'Class', 'Confidence', 'BBox'].map(h => <th key={h} style={s.th}>{h}</th>)}</tr></thead>
                  <tbody>
                    {result.detections.map((d, i) => (
                      <tr key={i}>
                        <td style={s.td}>{i + 1}</td>
                        <td style={s.td}><span style={{ ...s.cls, background: '#05966922', color: '#6ee7b7' }}>{d.class_name}</span></td>
                        <td style={s.td}>{fmtConf(d.confidence)}</td>
                        <td style={{ ...s.td, fontFamily: 'monospace', fontSize: 12 }}>
                          {d.bbox_x1.toFixed(0)}, {d.bbox_y1.toFixed(0)}, {d.bbox_x2.toFixed(0)}, {d.bbox_y2.toFixed(0)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
          </div>
          <p style={s.evalNote}>
            📊 Evaluation metrics (precision, recall, mAP) require the FLIR ADAS dataset to be downloaded
            and a model fine-tuned. Use the Training and Evaluation pages once the dataset is available.
          </p>
        </>
      )}

      <div style={s.infoBox}>
        <strong>📌 Model Status:</strong> PRETRAINED on COCO — NOT fine-tuned on FLIR ADAS thermal data.
        Classes shown are COCO classes. Download FLIR ADAS from flir.com/oem/adas/ and fine-tune via the
        Training page to obtain thermal-specific detection performance.
      </div>
    </div>
  )
}

const s: Record<string, React.CSSProperties> = {
  page: { padding: 24, color: '#e2e8f0', maxWidth: 1300 },
  title: { margin: '0 0 6px', fontSize: 22, fontWeight: 700, color: '#f1f5f9' },
  sub: { margin: '0 0 12px', color: '#64748b', fontSize: 13 },
  syntheticBanner: { background: '#1c1a0a', border: '1px solid #92400e', borderRadius: 8, padding: '10px 14px', color: '#fbbf24', fontSize: 13, marginBottom: 20 },
  row: { display: 'flex', gap: 16, marginBottom: 16, flexWrap: 'wrap' },
  dropZone: { flex: 1, minWidth: 260, minHeight: 220, border: '2px dashed', borderRadius: 10, display: 'flex', alignItems: 'center', justifyContent: 'center', cursor: 'pointer', overflow: 'hidden', background: '#0f172a' },
  dropHint: { textAlign: 'center', color: '#475569', padding: 24, lineHeight: 1.8 },
  previewImg: { width: '100%', objectFit: 'contain', maxHeight: 280 },
  settingsCard: { background: '#1e293b', border: '1px solid #334155', borderRadius: 10, padding: 20, minWidth: 240, display: 'flex', flexDirection: 'column', gap: 10 },
  card: { background: '#1e293b', border: '1px solid #334155', borderRadius: 10, padding: 20, marginBottom: 16 },
  imgPane: { background: '#1e293b', border: '1px solid #334155', borderRadius: 10, padding: 16, flex: 1 },
  cardTitle: { margin: '0 0 12px', fontSize: 13, fontWeight: 600, color: '#94a3b8', textTransform: 'uppercase', letterSpacing: 1 },
  label: { fontSize: 13, color: '#94a3b8' },
  slider: { width: '100%', accentColor: '#059669' },
  btn: { border: 'none', borderRadius: 8, padding: '10px 18px', fontSize: 14, fontWeight: 600, cursor: 'pointer', color: '#fff' },
  errBox: { background: '#1c080822', border: '1px solid #7f1d1d', borderRadius: 6, padding: '8px 12px', color: '#f87171', fontSize: 13 },
  metaBox: { background: '#0f172a', borderRadius: 6, padding: 12, display: 'flex', flexDirection: 'column', gap: 6 },
  metaRow: { display: 'flex', justifyContent: 'space-between', fontSize: 13, color: '#94a3b8' },
  table: { width: '100%', borderCollapse: 'collapse' },
  th: { textAlign: 'left', padding: '6px 10px', fontSize: 11, color: '#64748b', borderBottom: '1px solid #334155', textTransform: 'uppercase' },
  td: { padding: '8px 10px', fontSize: 13, color: '#cbd5e1', borderBottom: '1px solid #1e293b' },
  cls: { padding: '2px 8px', borderRadius: 4, fontSize: 12 },
  resultImg: { width: '100%', objectFit: 'contain', borderRadius: 6, maxHeight: 320 },
  noData: { color: '#475569', fontStyle: 'italic', fontSize: 13 },
  evalNote: { color: '#64748b', fontSize: 12, fontStyle: 'italic', marginBottom: 12 },
  infoBox: { background: '#0f2a1a', border: '1px solid #166534', borderRadius: 8, padding: '12px 16px', color: '#86efac', fontSize: 13 },
}
