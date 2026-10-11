import { useEffect, useRef } from 'react'

/**
 * Calls `fn` immediately and then every `intervalMs` milliseconds.
 * Stops when the component unmounts or when `active` is false.
 */
export function usePolling(fn: () => void, intervalMs: number, active = true) {
  const savedFn = useRef(fn)
  useEffect(() => { savedFn.current = fn }, [fn])

  useEffect(() => {
    if (!active) return
    savedFn.current()
    const id = setInterval(() => savedFn.current(), intervalMs)
    return () => clearInterval(id)
  }, [intervalMs, active])
}
