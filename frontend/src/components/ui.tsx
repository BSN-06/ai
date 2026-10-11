/**
 * Small shared UI kit for the dark control-room theme.
 * Every component only renders what it is given — no fabricated values.
 */
import type { CSSProperties, ReactNode } from 'react'

export const C = {
  bg: '#0a0f1e', panel: '#111a2e', panel2: '#0f172a', border: '#1f2b45', text: '#e2e8f0',
  muted: '#7c8aa5', dim: '#475569', blue: '#3b82f6', green: '#22c55e', amber: '#f59e0b',
  red: '#ef4444', purple: '#a855f7', cyan: '#06b6d4',
}

export const tooltipStyle: CSSProperties = {
  background: '#0f172a', border: `1px solid ${C.border}`, borderRadius: 6, color: C.text, fontSize: 12,
}

export function Page({ title, subtitle, right, children }: {
  title: string; subtitle?: string; right?: ReactNode; children: ReactNode
}) {
  return (
    <div style={{ padding: 24, maxWidth: 1500, color: C.text }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: 16, marginBottom: 18, flexWrap: 'wrap' }}>
        <div>
          <h1 style={{ fontSize: 22, fontWeight: 700, color: '#f1f5f9' }}>{title}</h1>
          {subtitle && <p style={{ color: C.muted, fontSize: 13, marginTop: 4 }}>{subtitle}</p>}
        </div>
        {right}
      </div>
      {children}
    </div>
  )
}

export function Card({ title, children, style, right }: {
  title?: ReactNode; children: ReactNode; style?: CSSProperties; right?: ReactNode
}) {
  return (
    <section style={{ background: C.panel, border: `1px solid ${C.border}`, borderRadius: 10, padding: 16, minWidth: 0, ...style }}>
      {(title || right) && (
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12, gap: 8 }}>
          <h3 style={{ fontSize: 12, fontWeight: 700, color: C.muted, textTransform: 'uppercase', letterSpacing: 1 }}>{title}</h3>
          {right}
        </div>
      )}
      {children}
    </section>
  )
}

export function Grid({ cols = 2, children, gap = 14 }: { cols?: number; children: ReactNode; gap?: number }) {
  return (
    <div style={{ display: 'grid', gridTemplateColumns: `repeat(auto-fit, minmax(${cols >= 4 ? 170 : cols === 3 ? 260 : 340}px, 1fr))`, gap, marginBottom: gap }}>
      {children}
    </div>
  )
}

export function Stat({ label, value, sub, color }: { label: string; value: ReactNode; sub?: ReactNode; color?: string }) {
  return (
    <div style={{ background: C.panel, border: `1px solid ${C.border}`, borderRadius: 10, padding: '12px 14px' }}>
      <div style={{ fontSize: 11, color: C.muted, textTransform: 'uppercase', letterSpacing: 0.8 }}>{label}</div>
      <div style={{ fontSize: 24, fontWeight: 700, color: color ?? '#f1f5f9', marginTop: 2, fontVariantNumeric: 'tabular-nums' }}>{value}</div>
      {sub && <div style={{ fontSize: 11, color: C.dim, marginTop: 2 }}>{sub}</div>}
    </div>
  )
}

export function Badge({ children, color = C.muted }: { children: ReactNode; color?: string }) {
  return (
    <span style={{ display: 'inline-block', padding: '2px 8px', borderRadius: 4, fontSize: 11, fontWeight: 700,
      color, background: `${color}1f`, border: `1px solid ${color}55`, whiteSpace: 'nowrap' }}>
      {children}
    </span>
  )
}

export function Banner({ kind = 'info', children }: { kind?: 'info' | 'warn' | 'error' | 'synthetic' | 'replay' | 'ok'; children: ReactNode }) {
  const col = { info: C.blue, warn: C.amber, error: C.red, synthetic: C.purple, replay: C.cyan, ok: C.green }[kind]
  return (
    <div style={{ border: `1px solid ${col}66`, background: `${col}14`, color: '#e2e8f0', borderRadius: 8,
      padding: '10px 14px', fontSize: 13, marginBottom: 12, borderLeft: `4px solid ${col}` }}>
      {children}
    </div>
  )
}

export function Table({ head, rows, empty = 'No records.' }: { head: ReactNode[]; rows: ReactNode[][]; empty?: string }) {
  if (!rows.length) return <p style={{ color: C.dim, fontStyle: 'italic', fontSize: 13 }}>{empty}</p>
  return (
    <div style={{ overflowX: 'auto' }}>
      <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12.5 }}>
        <thead>
          <tr>{head.map((h, i) => (
            <th key={i} style={{ textAlign: 'left', padding: '6px 8px', fontSize: 10.5, color: C.muted, textTransform: 'uppercase',
              letterSpacing: 0.6, borderBottom: `1px solid ${C.border}`, whiteSpace: 'nowrap' }}>{h}</th>
          ))}</tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i}>{r.map((c, j) => (
              <td key={j} style={{ padding: '6px 8px', borderBottom: `1px solid ${C.panel2}`, color: '#cbd5e1', verticalAlign: 'top' }}>{c}</td>
            ))}</tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

export function Btn({ children, onClick, disabled, kind = 'primary', small }: {
  children: ReactNode; onClick?: () => void; disabled?: boolean; kind?: 'primary' | 'ghost' | 'danger' | 'ok'; small?: boolean
}) {
  const col = { primary: C.blue, ghost: '#334155', danger: C.red, ok: C.green }[kind]
  return (
    <button onClick={onClick} disabled={disabled} style={{
      background: kind === 'ghost' ? 'transparent' : col, color: '#fff', border: `1px solid ${col}`,
      borderRadius: 6, padding: small ? '4px 10px' : '8px 14px', fontSize: small ? 12 : 13, fontWeight: 600,
      cursor: disabled ? 'not-allowed' : 'pointer', opacity: disabled ? 0.5 : 1, whiteSpace: 'nowrap',
    }}>{children}</button>
  )
}

export const inputStyle: CSSProperties = {
  background: C.panel2, border: `1px solid ${C.border}`, color: C.text, borderRadius: 6, padding: '7px 10px', fontSize: 13, width: '100%',
}

export function Field({ label, children, hint }: { label: string; children: ReactNode; hint?: string }) {
  return (
    <label style={{ display: 'block', marginBottom: 10 }}>
      <div style={{ fontSize: 11, color: C.muted, marginBottom: 4, textTransform: 'uppercase', letterSpacing: 0.6 }}>{label}</div>
      {children}
      {hint && <div style={{ fontSize: 11, color: C.dim, marginTop: 3 }}>{hint}</div>}
    </label>
  )
}

export function KV({ items }: { items: [string, ReactNode][] }) {
  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'max-content 1fr', gap: '4px 14px', fontSize: 12.5 }}>
      {items.map(([k, v]) => (
        <div key={k} style={{ display: 'contents' }}>
          <span style={{ color: C.muted }}>{k}</span>
          <span style={{ color: '#cbd5e1', wordBreak: 'break-all' }}>{v ?? '—'}</span>
        </div>
      ))}
    </div>
  )
}

export function Progress({ value, color = C.blue }: { value: number; color?: string }) {
  return (
    <div style={{ height: 6, background: C.panel2, borderRadius: 3, overflow: 'hidden' }}>
      <div style={{ width: `${Math.max(0, Math.min(100, value * 100))}%`, height: '100%', background: color, transition: 'width .4s' }} />
    </div>
  )
}

export function Empty({ children }: { children: ReactNode }) {
  return <p style={{ color: C.dim, fontStyle: 'italic', fontSize: 13 }}>{children}</p>
}

export function ErrorText({ error }: { error: string | null }) {
  if (!error) return null
  return <Banner kind="error">{error}</Banner>
}

/** Extract a readable message from an axios error. */
export function errMsg(e: unknown): string {
  const anyE = e as { response?: { data?: { detail?: unknown } }; message?: string }
  const d = anyE?.response?.data?.detail
  if (typeof d === 'string') return d
  if (d && typeof d === 'object') {
    const o = d as { message?: string; problems?: string[]; errors?: string[] }
    return [o.message, ...(o.problems ?? []), ...(o.errors ?? [])].filter(Boolean).join(' — ') || JSON.stringify(d)
  }
  return anyE?.message ?? 'Request failed'
}

export function downloadJson(name: string, data: unknown) {
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' })
  const a = document.createElement('a')
  a.href = URL.createObjectURL(blob)
  a.download = name
  a.click()
  URL.revokeObjectURL(a.href)
}

export function downloadCsv(name: string, rows: Record<string, unknown>[]) {
  if (!rows.length) return
  const cols = Object.keys(rows[0])
  const esc = (v: unknown) => `"${String(v ?? '').replace(/"/g, '""')}"`
  const csv = [cols.join(','), ...rows.map(r => cols.map(c => esc(r[c])).join(','))].join('\n')
  const blob = new Blob([csv], { type: 'text/csv' })
  const a = document.createElement('a')
  a.href = URL.createObjectURL(blob)
  a.download = name
  a.click()
  URL.revokeObjectURL(a.href)
}

export const PRIORITY_COL: Record<string, string> = { HIGH: C.red, MEDIUM: C.amber, LOW: C.green }
export const PROV_COL: Record<string, string> = {
  real_inference: C.blue, real: C.blue, recorded: C.cyan, recorded_replay: C.cyan, synthetic: C.purple,
}
export function ProvBadge({ p }: { p: string | null | undefined }) {
  const label = p === 'synthetic' ? 'SYNTHETIC' : p?.startsWith('recorded') ? 'RECORDED' : p === 'real_inference' || p === 'real' ? 'REAL INFERENCE' : (p ?? '—')
  return <Badge color={PROV_COL[p ?? ''] ?? C.muted}>{label}</Badge>
}
