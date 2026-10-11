import { NavLink } from 'react-router-dom'
import { useHealth } from '../hooks/useHealth'

const NAV = [
  { to: '/',           icon: '🖥️',  label: 'Overview' },
  { to: '/satellite',  icon: '🛰️',  label: 'Satellite' },
  { to: '/thermal',    icon: '🌡️',  label: 'Thermal' },
  { to: '/radar',      icon: '📡',  label: 'Radar' },
  { to: '/fusion',     icon: '🔗',  label: 'Sensor Fusion' },
  { to: '/models',     icon: '🧠',  label: 'Models' },
  { to: '/training',   icon: '🎓',  label: 'Training' },
  { to: '/evaluation', icon: '🔬',  label: 'Evaluation' },
  { to: '/datasets',   icon: '🗄️',  label: 'Datasets' },
]

export default function Sidebar() {
  const { health, error } = useHealth(6000)

  return (
    <aside style={s.sidebar}>
      {/* Logo */}
      <div style={s.logo}>
        <span style={s.logoIcon}>🔭</span>
        <div>
          <div style={s.logoTitle}>SurveillanceAI</div>
          <div style={s.logoSub}>Research Prototype</div>
        </div>
      </div>

      {/* System status pill */}
      <div style={{
        ...s.statusPill,
        borderColor: error ? '#7f1d1d' : health ? '#166534' : '#334155',
        background: error ? '#1c080a' : health ? '#0f2a1a' : '#0f172a',
      }}>
        <span style={{
          ...s.dot,
          background: error ? '#ef4444' : health ? '#22c55e' : '#f59e0b',
        }} />
        <div style={s.statusText}>
          <span style={{ color: error ? '#f87171' : health ? '#86efac' : '#94a3b8', fontSize: 11, fontWeight: 600 }}>
            {error ? 'OFFLINE' : health ? 'OPERATIONAL' : 'CONNECTING…'}
          </span>
          {health && !error && (
            <span style={{ color: '#475569', fontSize: 10 }}>
              {health.device.toUpperCase()} · {health.gpu_available ? 'GPU' : 'CPU-only'}
            </span>
          )}
        </div>
      </div>

      {/* Nav links */}
      <nav style={s.nav}>
        {NAV.map(item => (
          <NavLink
            key={item.to}
            to={item.to}
            end={item.to === '/'}
            style={({ isActive }) => ({
              ...s.link,
              background: isActive ? '#1e3a5f' : 'transparent',
              color: isActive ? '#e2e8f0' : '#64748b',
              borderLeft: isActive ? '3px solid #3b82f6' : '3px solid transparent',
            })}
          >
            <span style={s.linkIcon}>{item.icon}</span>
            <span>{item.label}</span>
          </NavLink>
        ))}
      </nav>

      {/* Footer disclaimer */}
      <div style={s.disclaimer}>
        <p style={s.disclaimerText}>
          OFFLINE RESEARCH PROTOTYPE
        </p>
        <p style={s.disclaimerText}>
          Not connected to operational infrastructure.
        </p>
        {health && (
          <p style={{ ...s.disclaimerText, marginTop: 4 }}>
            v{health.version}
          </p>
        )}
      </div>
    </aside>
  )
}

const s: Record<string, React.CSSProperties> = {
  sidebar: {
    width: 220,
    minWidth: 220,
    background: '#0f172a',
    borderRight: '1px solid #1e293b',
    display: 'flex',
    flexDirection: 'column',
    height: '100vh',
    position: 'sticky',
    top: 0,
    overflowY: 'auto',
  },
  logo: {
    display: 'flex',
    alignItems: 'center',
    gap: 10,
    padding: '20px 16px 14px',
    borderBottom: '1px solid #1e293b',
  },
  logoIcon: { fontSize: 24 },
  logoTitle: { fontSize: 14, fontWeight: 700, color: '#f1f5f9', letterSpacing: 0.5 },
  logoSub: { fontSize: 10, color: '#475569', marginTop: 1 },
  statusPill: {
    margin: '10px 12px',
    border: '1px solid',
    borderRadius: 8,
    padding: '7px 10px',
    display: 'flex',
    alignItems: 'center',
    gap: 8,
  },
  dot: { width: 8, height: 8, borderRadius: '50%', flexShrink: 0 },
  statusText: { display: 'flex', flexDirection: 'column', gap: 1 },
  nav: { flex: 1, padding: '8px 0' },
  link: {
    display: 'flex',
    alignItems: 'center',
    gap: 10,
    padding: '9px 16px',
    textDecoration: 'none',
    fontSize: 13,
    fontWeight: 500,
    transition: 'background 0.15s, color 0.15s',
  },
  linkIcon: { fontSize: 16, width: 20, textAlign: 'center' },
  disclaimer: {
    padding: '12px 14px',
    borderTop: '1px solid #1e293b',
  },
  disclaimerText: {
    margin: '2px 0',
    fontSize: 9,
    color: '#334155',
    lineHeight: 1.5,
    textTransform: 'uppercase',
    letterSpacing: 0.5,
  },
}
