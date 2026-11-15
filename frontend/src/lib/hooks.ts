import { useCallback, useEffect, useRef, useState } from "react";

/** Load data once (and again when ``deps`` change); ``interval`` keeps it fresh. */
export function useApi<T>(load: () => Promise<T>, deps: unknown[] = [], interval?: number) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<Error | null>(null);
  const [loading, setLoading] = useState(true);
  const loadRef = useRef(load);
  loadRef.current = load;

  const refresh = useCallback(async () => {
    try {
      const d = await loadRef.current();
      setData(d);
      setError(null);
    } catch (e) {
      setError(e as Error);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    setLoading(true);
    void refresh();
    if (!interval) return;
    const id = window.setInterval(() => {
      if (document.visibilityState === "visible") void refresh();
    }, interval);
    // a background tab does not poll; catch up at once when it is shown again (another tab may have acted)
    const shown = () => {
      if (document.visibilityState === "visible") void refresh();
    };
    document.addEventListener("visibilitychange", shown);
    window.addEventListener("focus", shown);
    return () => {
      window.clearInterval(id);
      document.removeEventListener("visibilitychange", shown);
      window.removeEventListener("focus", shown);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, interval]);

  return { data, error, loading, refresh, setData };
}

export function useDebounced<T>(value: T, ms = 250): T {
  const [v, setV] = useState(value);
  useEffect(() => {
    const id = window.setTimeout(() => setV(value), ms);
    return () => window.clearTimeout(id);
  }, [value, ms]);
  return v;
}

export function useHotkey(key: string, handler: (e: KeyboardEvent) => void) {
  const ref = useRef(handler);
  ref.current = handler;
  useEffect(() => {
    const on = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === key) {
        e.preventDefault();
        ref.current(e);
      }
    };
    window.addEventListener("keydown", on);
    return () => window.removeEventListener("keydown", on);
  }, [key]);
}
