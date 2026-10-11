import { useLocation } from 'react-router-dom'

const PAGE_TITLES: Record<string, string> = {
  '/':           'System Overview',
  '/satellite':  'Satellite / Aerial Analysis',
  '/thermal':    'Thermal / Infrared Analysis',
  '/radar':      'Radar Track Analysis',
  '/fusion':     'Sensor Fusion & Event Prioritization',
  '/models':     'Model Management',
  '/training':   'Model Training',
  '/evaluation': 'Test & Evaluation',
  '/datasets':   'Dataset Management',
}

export default function TopBar() {
  const { pathname } = useLocation()
  const title = PAGE_TITLES[pathname] ?? 'AI Surveillance System'
  const now = new Date().toLocaleString('en-GB', {
    weekday: 'short', year: 'numeric', month: 'short',
    day: '2-digit', hour: '2-digit', minute: '2-digit',
  })

  return (
    <header style={s.bar}>
      <h2 style={s.title}>{title}</h2>
      <div style={s.right}>
        <span style={s.time}>{now}</span>
        <span style={s.tag}>OFFLINE RESEARCH PROTOTYPE</span>
      </div>
    </header>
  )
}

const s: Record<string, React.CSSProperties> = {
  bar: {
    height: 52,
    background: '#0f172a',
    borderBottom: '1px solid #1e293b',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: '0 24px',
    flexShrink: 0,
    position: 'sticky',
    top: 0,
    zIndex: 10,
  },
  title: {
    margin: 0,
    fontSize: 15,
    fontWeight: 600,
    color: '#94a3b8',
    letterSpacing: 0.3,
  },
  right: {
    display: 'flex',
    alignItems: 'center',
    gap: 16,
  },
  time: {
    fontSize: 12,
    color: '#334155',
    fontFamily: 'monospace',
  },
  tag: {
    fontSize: 10,
    fontWeight: 700,
    color: '#1d4ed8',
    background: '#1e3a8a22',
    border: '1px solid #1e3a8a',
    borderRadius: 4,
    padding: '3px 8px',
    letterSpacing: 0.8,
  },
}
