import { useEffect, useMemo, useState } from 'react'
import {
  ScatterChart, Scatter, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Legend, LineChart, Line,
} from 'recharts'
import { getRadarTracks, resetRadarTracks, runRadarImport } from '../utils/api'
import type { RadarProcessResult, RadarTrack } from '../types'
import { formatTs } from '../utils/format'
import {
  Badge, Banner, Btn, C, Card, Empty, Field, Grid, KV, Page, Stat, Table, downloadJson, errMsg, inputStyle, tooltipStyle,
} from '../components/ui'

const COLS = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#a855f7', '#ec4899', '#06b6d4', '#84cc16']

export default function RadarAnalysis() {
  const [file, setFile] = useState<File | null>(null)
  const [gate, setGate] = useState(50)
  const [maxMisses, setMaxMisses] = useState(5)
  const [result, setResult] = useState<RadarProcessResult | null>(null)
  const [tracks, setTracks] = useState<RadarTrack[]>([])
  const [evaluation, setEvaluation] = useState<RadarProcessResult['evaluation'] | null>(null)
  const [sel, setSel] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [showMeasured, setShowMeasured] = useState(true)

  useEffect(() => {
    getRadarTracks().then(r => {
      setTracks(r.tracks ?? [])
      setEvaluation((r as unknown as { evaluation: RadarProcessResult['evaluation'] }).evaluation ?? null)
    }).catch(() => undefined)
  }, [])

  const run = async () => {
    if (!file) return
    setBusy(true); setError(null)
    try {
      const r = await runRadarImport(file, { associationThreshold: gate, maxMisses })
      setResult(r); setTracks(r.tracks); setEvaluation(r.evaluation ?? null); setSel(r.tracks[0]?.track_id ?? null)
    } catch (e) { setError(errMsg(e)) } finally { setBusy(false) }
  }

  const reset = async () => { await resetRadarTracks(); setTracks([]); setResult(null); setEvaluation(null); setSel(null) }

  const selected = useMemo(() => tracks.find(t => t.track_id === sel) ?? null, [tracks, sel])
  const ev = evaluation as Record<string, unknown> | null

  return (
    <Page title="Radar Track Analysis"
      subtitle="Module C — recorded radar detections → timestamp/coordinate validation → Kalman filter (constant velocity) + global-nearest-neighbour association."
      right={<Badge color={C.blue}>MODE A · REAL TRACKING ALGORITHM</Badge>}>
      <Banner kind="info">
        Classical tracker — no neural radar model. Designed for structured maritime radar/AIS detections (e.g. DLR datasets); it does
        <strong> not</strong> demonstrate aircraft tracking. Accepted columns: <code>timestamp</code> + (<code>x, y</code> in metres) or
        (<code>latitude, longitude</code> → projected to local metres). Optional reference columns <code>truth_id / mmsi / target_id</code>
        and <code>truth_x, truth_y</code> enable accuracy metrics.
      </Banner>

      <Grid cols={3}>
        <Card title="Recorded data">
          <input type="file" accept=".csv,.json" onChange={e => setFile(e.target.files?.[0] ?? null)} style={{ ...inputStyle, padding: 6 }} />
          <div style={{ display: 'flex', gap: 8, marginTop: 10 }}>
            <Field label="Association gate (m)"><input type="number" value={gate} min={1} onChange={e => setGate(+e.target.value)} style={inputStyle} /></Field>
            <Field label="Max missed scans"><input type="number" value={maxMisses} min={1} max={100} onChange={e => setMaxMisses(+e.target.value)} style={inputStyle} /></Field>
          </div>
          <div style={{ display: 'flex', gap: 8 }}>
            <Btn onClick={run} disabled={!file || busy}>{busy ? 'Processing…' : 'Validate & track'}</Btn>
            <Btn kind="ghost" onClick={reset}>Reset tracks</Btn>
          </div>
          <p style={{ fontSize: 11, color: C.dim, marginTop: 8 }}>Sample (simulated) file: <code>python generate_sample_radar.py</code></p>
        </Card>
        <Card title="Processing summary">
          {result ? (
            <KV items={[
              ['Detections processed', result.total_detections_processed], ['Tracks created', result.new_tracks_created],
              ['Active / terminated', `${result.active_tracks} / ${result.terminated_tracks}`],
              ['Processing latency', `${result.processing_time_ms} ms`], ['Coordinate mode', result.coordinate_mode],
              ['Units', result.units], ['Observations saved', result.observation_ids?.length ?? 0], ['Report', result.report_id],
            ]} />
          ) : <Empty>Upload a file to run the tracker.</Empty>}
        </Card>
        <Card title="Evaluation (reference data only)">
          {ev ? (
            <>
              {(ev.metrics_available as string[]).length === 0 && <Banner kind="warn">{String(ev.note ?? 'No reference data — no accuracy metrics.')}</Banner>}
              <KV items={[
                ['Association accuracy', ev.association_accuracy != null ? `${((ev.association_accuracy as number) * 100).toFixed(1)}%` : 'n/a'],
                ['Track purity', ev.mean_track_purity != null ? `${((ev.mean_track_purity as number) * 100).toFixed(1)}%` : 'n/a'],
                ['Continuity', ev.mean_track_continuity != null ? `${((ev.mean_track_continuity as number) * 100).toFixed(1)}%` : 'n/a'],
                ['Fragmentation (extra tracks / target)', String(ev.fragmentation ?? 'n/a')],
                ['ID switches', String(ev.id_switches ?? 'n/a')],
                ['Position RMSE', ev.position_rmse != null ? `${ev.position_rmse} (input units)` : 'n/a'],
                ['Reference targets', String(ev.reference_targets ?? 'n/a')],
                ['Mean measurement residual', String(ev.mean_measurement_residual ?? '—')],
              ]} />
              {(ev.metrics_unavailable as string[]).map((m, i) => <p key={i} style={{ fontSize: 11, color: C.dim, marginTop: 4 }}>Not computed: {m}</p>)}
            </>
          ) : <Empty>No tracks.</Empty>}
        </Card>
      </Grid>

      {error && <Banner kind="error">{error}</Banner>}
      {result?.warnings?.map((w, i) => <Banner key={i} kind="warn">Missing/invalid records: {w}</Banner>)}

      {tracks.length > 0 && (
        <>
          <Grid cols={4}>
            <Stat label="Tracks" value={tracks.length} />
            <Stat label="Active" value={tracks.filter(t => t.status === 'ACTIVE').length} color={C.green} />
            <Stat label="Terminated" value={tracks.filter(t => t.status === 'TERMINATED').length} />
            <Stat label="Track points" value={tracks.reduce((a, t) => a + t.history.length, 0)} />
          </Grid>
          <Card title="Track plot (local plane)" right={
            <label style={{ fontSize: 12, color: C.muted }}><input type="checkbox" checked={showMeasured} onChange={e => setShowMeasured(e.target.checked)} /> show measured points</label>}
            style={{ marginBottom: 14 }}>
            <p style={{ fontSize: 11, color: C.dim, marginBottom: 6 }}>Lines/filled dots = Kalman <strong>estimated</strong> positions · hollow dots = raw <strong>measured</strong> detections. No basemap is drawn (coordinates are local).</p>
            <ResponsiveContainer width="100%" height={380}>
              <ScatterChart margin={{ top: 10, right: 20, bottom: 10, left: 0 }}>
                <CartesianGrid stroke="#1f2b45" strokeDasharray="3 3" />
                <XAxis type="number" dataKey="x" name="x" tick={{ fill: C.muted, fontSize: 11 }} domain={['auto', 'auto']} />
                <YAxis type="number" dataKey="y" name="y" tick={{ fill: C.muted, fontSize: 11 }} domain={['auto', 'auto']} />
                <Tooltip contentStyle={tooltipStyle} cursor={{ strokeDasharray: '3 3' }} />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                {tracks.slice(0, 20).map((t, i) => (
                  <Scatter key={t.track_id} name={`${t.track_id} (est.)`} line fill={COLS[i % COLS.length]} shape="circle"
                    data={t.history.map(h => ({ x: +h.estimated_x.toFixed(2), y: +h.estimated_y.toFixed(2) }))} />
                ))}
                {showMeasured && tracks.slice(0, 20).map((t, i) => (
                  <Scatter key={`${t.track_id}-m`} name={`${t.track_id} (meas.)`} legendType="none" fill="none" stroke={COLS[i % COLS.length]}
                    data={t.history.filter(h => h.measured_x != null).map(h => ({ x: h.measured_x, y: h.measured_y }))} />
                ))}
              </ScatterChart>
            </ResponsiveContainer>
          </Card>
          <Grid cols={2}>
            <Card title="Tracks">
              <div style={{ maxHeight: 340, overflowY: 'auto' }}>
                <Table head={['Track', 'Status', 'Points', 'Position (est.)', 'Velocity (est.)', 'σ pos', 'Last update']}
                  rows={tracks.map(t => [
                    <a onClick={() => setSel(t.track_id)} style={{ color: sel === t.track_id ? C.cyan : C.blue, cursor: 'pointer' }}>{t.track_id}</a>,
                    <Badge color={t.status === 'ACTIVE' ? C.green : C.muted}>{t.status}</Badge>, t.num_observations,
                    `${t.current_position.x.toFixed(1)}, ${t.current_position.y.toFixed(1)}`,
                    `${t.current_velocity.x.toFixed(2)}, ${t.current_velocity.y.toFixed(2)}`,
                    t.uncertainty.toFixed(2), formatTs(t.last_update)])} />
              </div>
            </Card>
            <Card title={selected ? `Track history · ${selected.track_id}` : 'Track history'} right={
              selected && <Btn small kind="ghost" onClick={() => downloadJson(`${selected.track_id}.json`, selected)}>Export</Btn>}>
              {selected ? (
                <>
                  <ResponsiveContainer width="100%" height={140}>
                    <LineChart data={selected.history.map((h, i) => ({ i, sigma: +h.uncertainty.toFixed(3), speed: +Math.hypot(h.velocity_x, h.velocity_y).toFixed(3) }))}>
                      <XAxis dataKey="i" tick={{ fill: C.muted, fontSize: 10 }} /><YAxis tick={{ fill: C.muted, fontSize: 10 }} />
                      <Tooltip contentStyle={tooltipStyle} /><Legend wrapperStyle={{ fontSize: 11 }} />
                      <Line dataKey="sigma" name="position σ" stroke={C.amber} dot={false} />
                      <Line dataKey="speed" name="speed (units/s)" stroke={C.cyan} dot={false} />
                    </LineChart>
                  </ResponsiveContainer>
                  <div style={{ maxHeight: 200, overflowY: 'auto' }}>
                    <Table head={['Recorded time', 'Measured', 'Estimated', 'Reference id']}
                      rows={selected.history.map(h => [formatTs(h.timestamp),
                        h.measured_x != null ? `${h.measured_x.toFixed(1)}, ${h.measured_y?.toFixed(1)}` : '—',
                        `${h.estimated_x.toFixed(1)}, ${h.estimated_y.toFixed(1)}`, h.truth_id ?? '—'])} />
                  </div>
                </>
              ) : <Empty>Select a track.</Empty>}
            </Card>
          </Grid>
        </>
      )}
    </Page>
  )
}
