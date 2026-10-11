import { useState } from 'react'
import { getHealth, getStats } from '../utils/api'
import { usePolling } from './usePolling'
import type { SystemHealth, SystemStats } from '../types'

export function useHealth(intervalMs = 5000) {
  const [health, setHealth] = useState<SystemHealth | null>(null)
  const [stats, setStats] = useState<SystemStats | null>(null)
  const [error, setError] = useState<string | null>(null)

  usePolling(async () => {
    try {
      const [h, s] = await Promise.all([getHealth(), getStats()])
      setHealth(h)
      setStats(s)
      setError(null)
    } catch (e: unknown) {
      setError('Cannot reach backend — is it running on port 8000?')
    }
  }, intervalMs)

  return { health, stats, error }
}
