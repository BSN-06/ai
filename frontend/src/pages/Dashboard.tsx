import { Link } from 'react-router-dom'
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, LineChart, Line, CartesianGrid, Cell,
} from 'recharts'
import { useHealth } from '../hooks/useHealth'
import { formatTs, sensorIcon, statusColour, trunc } from '../utils/format'
import {
  Badge, Banner, C, Card, Empty, Grid, Page, PRIORITY_COL, ProvBadge, Progress, Stat, Table, tooltipStyle,
} from '../components/ui'

const DS_COL: Record<string, string> = { VALIDATED: C.green, NOT_INSTALLED: C.dim, INVALID: C.red, CONVERTING: C.blue }

export default function Dashboard() {
  const { health, stats, error } = useHealth(3000)

  if (error) {
    return (
      <Page title="System Overview">
        <Banner kind="error">
          <strong>Backend unreachable.</strong> {error}<br />
          Start it with <code>start_backend.bat</code> (or <code>python -m uvicorn app.main:app --port 8000</code> in <code>backend/</code>).
        </Banner>
      </Page>
    )
  }
  if (!health || !stats) return <Page title="System Overview"><Empty>Connecting to backend…</Empty></Page>

  const sensorData = Object.entries(stats.observations.by_sensor).map(([k, v]) => ({ name: k, value: v }))
  const provData = Object.entries(stats.observations.by_provenance).map(([k, v]) => ({ name: k, value: v }))
  const prioData = (['HIGH', 'MEDIUM', 'LOW'] as const).map(p => ({ name: p, value: stats.events.by_priority[p] ?? 0 }))
  const latency = stats.latency.filter(l => l.latency_ms != null).map((l, i) => ({ i: i + 1, ms: l.latency_ms, c: l.component }))
  const replay = stats.replay
  const replayActive = replay.status === 'PLAYING' || replay.status === 'PAUSED'

  return (
    <Page title="System Overview" subtitle="Monitoring console — all figures come from stored records and service state."
      right={
        <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap' }}>
          <Badge color={C.green}>● {health.status.toUpperCase()}</Badge>
          <Badge color={C.blue}>{health.device.toUpperCase()}{health.gpu_available ? ' · GPU' : ' · CPU-only'}</Badge>
          <Badge color={replayActive ? C.cyan : C.muted}>MODE: {replayActive ? 'RECORDED REPLAY' : 'IDLE / ON-DEMAND'}</Badge>
        </div>
      }>
      {replayActive && (
        <Banner kind="replay">
          <strong>RECORDED DATA REPLAY — NOT A LIVE SENSOR FEED.</strong> {replay.processed_records}/{replay.total_records} records ·
          recorded time {formatTs(replay.current_timestamp)} · speed ×{replay.replay_speed} · queue {replay.queue_remaining}
          <div style={{ marginTop: 6 }}><Progress value={replay.total_records ? replay.processed_records / replay.total_records : 0} color={C.cyan} /></div>
        </Banner>
      )}

      <Grid cols={4}>
        <Stat label="Observations" value={stats.observations.total}
          sub={`real ${stats.observations.by_provenance.real_inference ?? 0} · recorded ${stats.observations.by_provenance.recorded ?? 0} · synthetic ${stats.observations.by_provenance.synthetic ?? 0}`} />
        <Stat label="Events awaiting review" value={stats.events.pending} color={stats.events.pending ? C.amber : undefined}
          sub={`${stats.events.total} total`} />
        <Stat label="High-priority events" value={stats.events.by_priority.HIGH ?? 0} color={C.red} sub="review recommendation only" />
        <Stat label="Models fine-tuned" value={`${stats.models.trained}/${stats.models.total}`} />
        <Stat label="Datasets validated" value={`${stats.datasets.items.filter(d => d.validation_status === 'VALIDATED').length}/${stats.datasets.total}`} />
        <Stat label="Processing errors" value={stats.errors.filter(e => e.level === 'ERROR').length} color={stats.errors.some(e => e.level === 'ERROR') ? C.red : undefined} />
      </Grid>

      <Grid cols={3}>
        <Card title="Observations by sensor">
          {sensorData.every(d => !d.value) ? <Empty>No observations yet — run inference, radar import, replay or synthetic tests.</Empty> : (
            <ResponsiveContainer width="100%" height={180}>
              <BarChart data={sensorData}><XAxis dataKey="name" tick={{ fill: C.muted, fontSize: 12 }} />
                <YAxis allowDecimals={false} tick={{ fill: C.muted, fontSize: 12 }} /><Tooltip contentStyle={tooltipStyle} />
                <Bar dataKey="value" fill={C.blue} radius={[4, 4, 0, 0]} /></BarChart>
            </ResponsiveContainer>)}
          <div style={{ display: 'flex', gap: 6, marginTop: 6, flexWrap: 'wrap' }}>
            {provData.map(p => <span key={p.name}><ProvBadge p={p.name} /> <span style={{ fontSize: 11, color: C.muted }}>{p.value}</span></span>)}
          </div>
        </Card>
        <Card title="Event review-priority distribution">
          {prioData.every(d => !d.value) ? <Empty>No events yet.</Empty> : (
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={prioData} layout="vertical"><XAxis type="number" allowDecimals={false} tick={{ fill: C.muted, fontSize: 12 }} />
                <YAxis type="category" dataKey="name" tick={{ fill: C.muted, fontSize: 12 }} width={70} /><Tooltip contentStyle={tooltipStyle} />
                <Bar dataKey="value" radius={[0, 4, 4, 0]}>{prioData.map(d => <Cell key={d.name} fill={PRIORITY_COL[d.name]} />)}</Bar>
              </BarChart>
            </ResponsiveContainer>)}
          <div style={{ fontSize: 11, color: C.dim }}>
            {Object.entries(stats.events.by_status).map(([k, v]) => `${k} ${v}`).join(' · ')}
          </div>
        </Card>
        <Card title="Processing latency (last runs)">
          {!latency.length ? <Empty>No inference or tracking runs recorded yet.</Empty> : (
            <ResponsiveContainer width="100%" height={200}>
              <LineChart data={latency}><CartesianGrid stroke="#1f2b45" strokeDasharray="3 3" />
                <XAxis dataKey="i" tick={{ fill: C.muted, fontSize: 11 }} /><YAxis unit="ms" tick={{ fill: C.muted, fontSize: 11 }} />
                <Tooltip contentStyle={tooltipStyle} formatter={(v, _n, p) => [`${Number(v).toFixed(0)} ms`, (p.payload as { c: string }).c]} />
                <Line dataKey="ms" stroke={C.cyan} dot={{ r: 2 }} /></LineChart>
            </ResponsiveContainer>)}
        </Card>
      </Grid>

      <Grid cols={3}>
        <Card title="Model availability">
          <Table head={['Model', 'Status', 'Eval', 'Checkpoint']} rows={[
            ...stats.models.items.map(m => [
              <span>{sensorIcon(m.sensor_type)} {m.name}</span>,
              <Badge color={m.status === 'TRAINED' ? C.green : m.status === 'TRAINING' ? C.blue : C.amber}>{m.status}</Badge>,
              <Badge color={statusColour(m.evaluation_status)}>{m.evaluation_status}</Badge>,
              m.status === 'TRAINED' ? (m.checkpoint_exists ? '✔ on disk' : <span style={{ color: C.red }}>missing</span>) : 'COCO base',
            ]),
            [<span>📡 radar tracker</span>, <Badge color={C.green}>AVAILABLE</Badge>, '—', 'classical (no weights)'],
          ]} />
        </Card>
        <Card title="Sensor data availability">
          <Table head={['Dataset', 'Sensor', 'Status']} rows={stats.datasets.items.map(d => [
            d.name, d.sensor_type, <Badge color={DS_COL[d.validation_status] ?? C.amber}>{d.validation_status}</Badge>])} />
          <p style={{ fontSize: 11, color: C.dim, marginTop: 8 }}>Datasets require registration and manual download — see <Link to="/datasets" style={{ color: C.blue }}>Datasets</Link>.</p>
        </Card>
        <Card title="Training & evaluation jobs">
          {!stats.training_jobs.length && !stats.evaluation_jobs.length ? <Empty>No jobs yet.</Empty> : (
            <>
              {stats.training_jobs.map(j => (
                <div key={j.id} style={{ marginBottom: 8 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12 }}>
                    <span>train · {j.dataset_name ?? ''} · {j.current_epoch}/{j.total_epochs}</span>
                    <Badge color={statusColour(j.status)}>{j.status}</Badge>
                  </div>
                  <Progress value={j.total_epochs ? j.current_epoch / j.total_epochs : 0} />
                </div>
              ))}
              {stats.evaluation_jobs.map(j => (
                <div key={j.id} style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, padding: '3px 0' }}>
                  <span>eval · {j.dataset_name} {j.map50 != null ? `· mAP50 ${j.map50.toFixed(3)}` : ''}</span>
                  <Badge color={statusColour(j.status)}>{j.status}</Badge>
                </div>
              ))}
            </>
          )}
        </Card>
      </Grid>

      <Grid cols={2}>
        <Card title="Recent observations">
          <Table head={['Sensor', 'Class', 'Conf.', 'Observed (UTC)', 'Provenance']} empty="None yet."
            rows={stats.recent_observations.map(o => [
              `${sensorIcon(o.sensor_type)} ${o.sensor_type}`, o.object_class ?? '—',
              o.confidence != null ? `${(o.confidence * 100).toFixed(0)}%` : '—',
              o.observation_timestamp ? formatTs(o.observation_timestamp) : <span style={{ color: C.amber }}>no timestamp</span>,
              <ProvBadge p={o.data_provenance} />])} />
        </Card>
        <Card title="Recent events" right={<Link to="/fusion" style={{ fontSize: 12, color: C.blue }}>Open fusion →</Link>}>
          <Table head={['Priority', 'Sensors', 'Status', 'Summary', 'Source']} empty="No events yet."
            rows={stats.recent_events.map(e => [
              <Badge color={PRIORITY_COL[e.review_priority]}>{e.review_priority}</Badge>,
              (e.source_sensors ?? []).map(sensorIcon).join(' '), e.review_status, trunc(e.evidence_summary, 70),
              <ProvBadge p={e.data_provenance} />])} />
        </Card>
      </Grid>

      <Grid cols={2}>
        <Card title="Recent activity">
          <Table head={['Time', 'Component', 'Message']} empty="No activity logged yet."
            rows={stats.activity.map(a => [formatTs(a.timestamp), a.component ?? '—',
              <span style={{ color: a.level === 'ERROR' ? C.red : a.level === 'WARNING' ? C.amber : undefined }}>{trunc(a.message, 90)}</span>])} />
        </Card>
        <Card title="Warnings & errors">
          <Table head={['Level', 'Component', 'Message', 'Time']} empty="No warnings or errors."
            rows={stats.errors.map(e => [<Badge color={e.level === 'ERROR' ? C.red : C.amber}>{e.level}</Badge>,
              e.component ?? '—', trunc(e.message, 80), formatTs(e.timestamp)])} />
          <h4 style={{ fontSize: 11, color: C.muted, margin: '12px 0 4px', textTransform: 'uppercase' }}>Dependencies</h4>
          <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
            {Object.entries(stats.dependencies).map(([k, v]) => <Badge key={k} color={v ? C.green : C.red}>{k} {v ?? 'missing'}</Badge>)}
          </div>
        </Card>
      </Grid>
      <p style={{ fontSize: 11, color: C.dim }}>Updated {formatTs(health.timestamp)} · polling every 3 s · database {health.database}</p>
    </Page>
  )
}
