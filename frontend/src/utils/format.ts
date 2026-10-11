/** Format a UTC ISO string for display. */
export function formatTs(ts: string | null | undefined): string {
  if (!ts) return '—'
  try {
    return new Date(ts).toLocaleString('en-GB', {
      year: 'numeric', month: 'short', day: '2-digit',
      hour: '2-digit', minute: '2-digit', second: '2-digit',
    })
  } catch {
    return ts
  }
}

/** Format confidence 0-1 as percentage string. */
export function fmtConf(c: number | null | undefined): string {
  if (c == null) return '—'
  return `${(c * 100).toFixed(1)}%`
}

/** Format a float to N decimal places, or —. */
export function fmtNum(n: number | null | undefined, dp = 2): string {
  if (n == null) return '—'
  return n.toFixed(dp)
}

/** Colour class for review priority. */
export function priorityColour(p: string): string {
  if (p === 'HIGH') return '#ef4444'
  if (p === 'MEDIUM') return '#f59e0b'
  return '#22c55e'
}

/** Colour class for job/model status. */
export function statusColour(s: string): string {
  if (s === 'COMPLETED' || s === 'TRAINED' || s === 'EVALUATED') return '#22c55e'
  if (s === 'RUNNING' || s === 'TRAINING' || s === 'EVALUATING') return '#3b82f6'
  if (s === 'FAILED') return '#ef4444'
  if (s === 'CANCELLED') return '#6b7280'
  if (s === 'PENDING') return '#f59e0b'
  return '#94a3b8'
}

/** Colour for correlation status. */
export function corrColour(s: string): string {
  if (s === 'CORRELATED') return '#22c55e'
  if (s === 'NOT_CORRELATED') return '#ef4444'
  return '#f59e0b' // INSUFFICIENT_METADATA
}

/** Sensor icon character. */
export function sensorIcon(sensor: string): string {
  if (sensor === 'satellite') return '🛰️'
  if (sensor === 'thermal') return '🌡️'
  if (sensor === 'radar') return '📡'
  return '📦'
}

/** Truncate a long string. */
export function trunc(s: string | null | undefined, n = 60): string {
  if (!s) return '—'
  return s.length > n ? s.slice(0, n) + '…' : s
}
