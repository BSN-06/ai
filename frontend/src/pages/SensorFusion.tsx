import { useCallback, useEffect, useState } from 'react'
import {
  autoGenerateEvents, exportEvent, getCorrelationHistory, getEvent, listEvents, listObservations, runCorrelation, updateEventReview,
} from '../utils/api'
import type { CorrelationDecision, CorrelationResult, Event, Observation } from '../types'
import { corrColour, fmtConf, formatTs, sensorIcon, trunc } from '../utils/format'
import { usePolling } from '../hooks/usePolling'
import {
  Badge, Banner, Btn, C, Card, Empty, Field, Grid, KV, Page, PRIORITY_COL, ProvBadge, Table, downloadJson, errMsg, inputStyle,
} from '../components/ui'

export default function SensorFusion() {
  const [obs, setObs] = useState<Observation[]>([])
  const [obsTotal, setObsTotal] = useState(0)
  const [sensor, setSensor] = useState('')
  const [prov, setProv] = useState('')
  const [selected, setSelected] = useState<string[]>([])
  const [temporal, setTemporal] = useState(300)
  const [spatial, setSpatial] = useState(500)
  const [corr, setCorr] = useState<CorrelationResult | null>(null)
  const [history, setHistory] = useState<CorrelationDecision[]>([])
  const [events, setEvents] = useState<Event[]>([])
  const [prioFilter, setPrioFilter] = useState('')
  const [statusFilter, setStatusFilter] = useState('')
  const [detail, setDetail] = useState<Event | null>(null)
  const [notes, setNotes] = useState('')
  const [msg, setMsg] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  const refresh = useCallback(async () => {
    try {
      const [o, e, h] = await Promise.all([
        listObservations(sensor, 100, 0, prov), listEvents(prioFilter || undefined, statusFilter || undefined, 100),
        getCorrelationHistory(30)])
      setObs(o.observations); setObsTotal(o.total); setEvents(e.events); setHistory(h.decisions)
    } catch (err) { setError(errMsg(err)) }
  }, [sensor, prov, prioFilter, statusFilter])

  useEffect(() => { refresh() }, [refresh])
  usePolling(refresh, 5000)

  const toggle = (id: string) => setSelected(s => s.includes(id) ? s.filter(x => x !== id) : [...s, id])

  const doCorrelate = async () => {
    setError(null); setMsg(null)
    try {
      const r = await runCorrelation(selected, temporal, spatial)
      if (r.status === 'insufficient_data') setMsg(r.message ?? 'Need at least two observations.')
      setCorr(r); refresh()
    } catch (e) { setError(errMsg(e)) }
  }
  const doEvents = async (singletons: boolean) => {
    setError(null)
    try {
      const r = await autoGenerateEvents(selected, singletons)
      setMsg(`${r.events_created} event(s) created from ${r.observations_considered ?? selected.length} observation(s). ` +
        (r.decisions_summary ? Object.entries(r.decisions_summary).map(([k, v]) => `${k}: ${v}`).join(' · ') : ''))
      refresh()
    } catch (e) { setError(errMsg(e)) }
  }
  const openEvent = async (id: string) => {
    try { const e = await getEvent(id); setDetail(e); setNotes(e.reviewer_notes ?? '') } catch (err) { setError(errMsg(err)) }
  }
  const review = async (status: string) => {
    if (!detail) return
    await updateEventReview(detail.id, status, notes || undefined)
    openEvent(detail.id); refresh()
  }

  return (
    <Page title="Sensor Fusion & Event Prioritization"
      subtitle="Module D — rule-based, uncertainty-aware correlation (not a trained fusion model). Priority is a recommendation for human review only.">
      <Banner kind="info">
        Correlation requires valid timestamps and geographic positions in a common CRS. Pixel boxes are never treated as
        geographic coordinates and matching class labels alone never correlate observations; otherwise the decision is
        <strong> INSUFFICIENT_METADATA</strong>.
      </Banner>
      {error && <Banner kind="error">{error}</Banner>}
      {msg && <Banner kind="ok">{msg}</Banner>}

      <Card title={`Observations (${obsTotal}) · ${selected.length} selected`} style={{ marginBottom: 14 }} right={
        <div style={{ display: 'flex', gap: 6, alignItems: 'center', flexWrap: 'wrap' }}>
          <select value={sensor} onChange={e => setSensor(e.target.value)} style={{ ...inputStyle, width: 120 }}>
            <option value="">all sensors</option><option>satellite</option><option>thermal</option><option>radar</option>
          </select>
          <select value={prov} onChange={e => setProv(e.target.value)} style={{ ...inputStyle, width: 140 }}>
            <option value="">all provenance</option><option value="real_inference">real inference</option>
            <option value="recorded">recorded</option><option value="synthetic">synthetic</option>
          </select>
          <Btn small kind="ghost" onClick={() => setSelected(obs.map(o => o.id))}>select page</Btn>
          <Btn small kind="ghost" onClick={() => setSelected([])}>clear</Btn>
        </div>}>
        <div style={{ maxHeight: 300, overflowY: 'auto' }}>
          <Table head={['', 'Sensor', 'Class', 'Conf.', 'Observed (UTC)', 'Lat / Lon', 'Track', 'σ', 'Provenance']}
            empty="No observations. Run inference, radar import, replay or the synthetic tests first."
            rows={obs.map(o => [
              <input type="checkbox" checked={selected.includes(o.id)} onChange={() => toggle(o.id)} />,
              `${sensorIcon(o.sensor_type)} ${o.sensor_type}`, o.object_class ?? '—', fmtConf(o.confidence),
              o.observation_timestamp ? formatTs(o.observation_timestamp) : <span style={{ color: C.amber }}>missing</span>,
              o.latitude != null ? `${o.latitude.toFixed(5)}, ${o.longitude?.toFixed(5)}` : <span style={{ color: C.amber }}>{o.bbox_x1 != null ? 'pixel bbox only' : 'missing'}</span>,
              o.radar_track_id ?? '—', o.uncertainty != null ? o.uncertainty.toFixed(1) : '—', <ProvBadge p={o.data_provenance} />])} />
        </div>
        <div style={{ display: 'flex', gap: 10, alignItems: 'flex-end', marginTop: 12, flexWrap: 'wrap' }}>
          <div style={{ width: 170 }}><Field label="Temporal window (s)"><input type="number" value={temporal} onChange={e => setTemporal(+e.target.value)} style={inputStyle} /></Field></div>
          <div style={{ width: 170 }}><Field label="Spatial threshold (m)"><input type="number" value={spatial} onChange={e => setSpatial(+e.target.value)} style={inputStyle} /></Field></div>
          <div style={{ marginBottom: 10, display: 'flex', gap: 8, flexWrap: 'wrap' }}>
            <Btn onClick={doCorrelate}>Run correlation {selected.length ? `(${selected.length})` : '(latest 20)'}</Btn>
            <Btn kind="ok" onClick={() => doEvents(false)}>Create events from correlated groups</Btn>
            <Btn kind="ghost" onClick={() => doEvents(true)}>…including single observations</Btn>
          </div>
        </div>
      </Card>

      {corr && corr.correlations && (
        <Card title={`Correlation decisions · ${corr.total_pairs_checked} pairs`} style={{ marginBottom: 14 }} right={
          <div style={{ display: 'flex', gap: 6 }}>{Object.entries(corr.summary ?? {}).map(([k, v]) => <Badge key={k} color={corrColour(k)}>{k} {v}</Badge>)}</div>}>
          <div style={{ maxHeight: 300, overflowY: 'auto' }}>
            <Table head={['Decision', 'Pair', 'Δt (s)', 'Distance (m)', 'Missing metadata', 'Reason']}
              rows={corr.correlations.map(d => [<Badge color={corrColour(d.decision_status)}>{d.decision_status}</Badge>,
                <span style={{ fontFamily: 'monospace', fontSize: 11 }}>{d.observation_ids.map(i => i.slice(0, 8)).join(' ↔ ')}</span>,
                d.temporal_delta_s?.toFixed(1) ?? '—', d.spatial_distance_m?.toFixed(1) ?? '—',
                (d.missing_metadata ?? []).join(', ') || '—', <span style={{ fontSize: 11.5 }}>{d.reason}</span>])} />
          </div>
          {!!corr.duplicates?.length && (
            <p style={{ fontSize: 12, color: C.amber, marginTop: 8 }}>
              {corr.duplicates.length} potential duplicate pair(s): {corr.duplicates.map(d => `${d.observation_ids.map(i => i.slice(0, 8)).join('/')} (${d.rule})`).join('; ')} — excluded from supporting evidence.
            </p>)}
          {corr.groups && <p style={{ fontSize: 12, color: C.muted, marginTop: 6 }}>Groups: {corr.groups.filter(g => g.observation_ids.length > 1).length} multi-observation · {corr.groups.filter(g => g.observation_ids.length === 1).length} single.</p>}
        </Card>
      )}

      <Grid cols={2}>
        <Card title={`Events (${events.length})`} right={
          <div style={{ display: 'flex', gap: 6 }}>
            <select value={prioFilter} onChange={e => setPrioFilter(e.target.value)} style={{ ...inputStyle, width: 110 }}>
              <option value="">priority</option><option>HIGH</option><option>MEDIUM</option><option>LOW</option></select>
            <select value={statusFilter} onChange={e => setStatusFilter(e.target.value)} style={{ ...inputStyle, width: 120 }}>
              <option value="">status</option><option>PENDING</option><option>REVIEWING</option><option>REVIEWED</option><option>DISMISSED</option></select>
          </div>}>
          <div style={{ maxHeight: 520, overflowY: 'auto' }}>
            <Table head={['Priority', 'Sensors', 'Obs.', 'Status', 'Created', 'Source']} empty="No events."
              rows={events.map(e => [
                <a onClick={() => openEvent(e.id)} style={{ cursor: 'pointer' }}><Badge color={PRIORITY_COL[e.review_priority]}>{e.review_priority}</Badge></a>,
                <a onClick={() => openEvent(e.id)} style={{ cursor: 'pointer', color: detail?.id === e.id ? C.cyan : C.text }}>{(e.source_sensors ?? []).map(s => `${sensorIcon(s)} ${s}`).join(', ')}</a>,
                e.observation_ids?.length ?? 0, e.review_status, formatTs(e.created_at), <ProvBadge p={e.data_provenance} />])} />
          </div>
        </Card>

        <Card title="Event detail & human review" right={detail && (
          <Btn small kind="ghost" onClick={async () => downloadJson(`event_${detail.id}.json`, await exportEvent(detail.id))}>Export evidence</Btn>)}>
          {!detail ? <Empty>Select an event to inspect its evidence.</Empty> : (
            <>
              {detail.data_provenance === 'synthetic' && <Banner kind="synthetic">SYNTHETIC DEMONSTRATION — NOT LIVE SENSOR DATA</Banner>}
              {detail.data_provenance === 'recorded' && <Banner kind="replay">From RECORDED DATA REPLAY — not a live feed.</Banner>}
              <KV items={[
                ['Event ID', detail.id], ['Review priority', <Badge color={PRIORITY_COL[detail.review_priority]}>{detail.review_priority}</Badge>],
                ['Review status', detail.review_status], ['Source sensors', (detail.source_sensors ?? []).join(', ')],
                ['Uncertainty', detail.uncertainty != null ? detail.uncertainty.toFixed(2) : '—'], ['Created', formatTs(detail.created_at)],
              ]} />
              <h4 style={h4}>Evidence summary</h4>
              <p style={{ fontSize: 12.5 }}>{detail.evidence_summary}</p>
              <h4 style={h4}>Why this priority</h4>
              <ul style={{ fontSize: 12, paddingLeft: 18, color: '#cbd5e1' }}>
                {(detail.priority_explanation ?? '').split(' | ').map((f, i) => <li key={i}>{f}</li>)}
              </ul>
              {!!detail.missing_information?.length && (<>
                <h4 style={h4}>Missing information</h4>
                <ul style={{ fontSize: 12, paddingLeft: 18, color: C.amber }}>{detail.missing_information.map((m, i) => <li key={i}>{m}</li>)}</ul>
              </>)}
              <h4 style={h4}>Linked observations (traceable to source)</h4>
              <Table head={['Sensor', 'Class', 'Conf.', 'Observed', 'Source record', 'Model']}
                rows={(detail.observations ?? []).map(o => [`${sensorIcon(o.sensor_type)} ${o.sensor_type}`, o.object_class ?? '—', fmtConf(o.confidence),
                  formatTs(o.observation_timestamp), trunc(`${o.source_dataset ?? ''} ${o.source_record_id ?? o.id}`, 40),
                  o.image_url ? <a href={o.image_url} target="_blank" rel="noreferrer" style={{ color: C.blue }}>{o.model_name ?? 'image'}</a> : (o.model_name ?? '—')])} />
              <h4 style={h4}>Correlation decisions</h4>
              <Table head={['Decision', 'Reason']} empty="No decisions linked (single-observation event)."
                rows={(detail.correlations ?? []).map(d => [<Badge color={corrColour(d.decision_status)}>{d.decision_status}</Badge>, <span style={{ fontSize: 11.5 }}>{d.reason}</span>])} />
              <h4 style={h4}>Review</h4>
              <textarea value={notes} onChange={e => setNotes(e.target.value)} placeholder="Reviewer notes…" rows={3} style={{ ...inputStyle, resize: 'vertical' }} />
              <div style={{ display: 'flex', gap: 6, marginTop: 8, flexWrap: 'wrap' }}>
                <Btn small kind="ghost" onClick={() => review('REVIEWING')}>Mark reviewing</Btn>
                <Btn small kind="ok" onClick={() => review('REVIEWED')}>Mark reviewed</Btn>
                <Btn small kind="danger" onClick={() => review('DISMISSED')}>Dismiss</Btn>
                <Btn small kind="ghost" onClick={() => review('PENDING')}>Reset to pending</Btn>
              </div>
              <h4 style={h4}>Review history</h4>
              <Table head={['Time', 'From', 'To', 'Notes']} empty="No review actions yet."
                rows={(detail.review_history ?? []).map(h => [formatTs(h.timestamp), h.previous_status ?? '—', h.new_status, h.notes ?? '—'])} />
            </>
          )}
        </Card>
      </Grid>

      <Card title="Recent correlation decisions (stored)">
        <Table head={['Time', 'Decision', 'Δt (s)', 'Distance (m)', 'Event', 'Reason']} empty="No decisions yet."
          rows={history.map(d => [formatTs(d.created_at), <Badge color={corrColour(d.decision_status)}>{d.decision_status}</Badge>,
            d.temporal_delta_s?.toFixed(1) ?? '—', d.spatial_distance_m?.toFixed(1) ?? '—', d.event_id ? d.event_id.slice(0, 8) : '—', trunc(d.reason, 110)])} />
      </Card>
    </Page>
  )
}

const h4: React.CSSProperties = { fontSize: 11, color: C.muted, margin: '14px 0 6px', textTransform: 'uppercase', letterSpacing: 0.6 }
