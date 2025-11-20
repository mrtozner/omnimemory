import { useEffect, useState, useMemo } from 'react';
import type { Metrics, BackendMetrics } from '../services/api';
import { flattenMetrics } from '../services/api';

const API_BASE = 'http://localhost:8003';

export interface UseSSEOptions {
  sessionId?: string;
  tags?: Record<string, string>;
}

export function useSSE(toolId?: string, options?: UseSSEOptions) {
  const [data, setData] = useState<Metrics | null>(null);
  const [error, setError] = useState<Error | null>(null);
  const [isConnected, setIsConnected] = useState(false);

  // Serialize options to create stable dependency
  const sessionId = options?.sessionId;
  const tagsJson = useMemo(() => options?.tags || {}, [options?.tags]);

  const optionsKey = useMemo(() => {
    return JSON.stringify({ sessionId, tags: tagsJson });
  }, [sessionId, tagsJson]);

  useEffect(() => {
    // Build base URL with toolId
    let url = `${API_BASE}/stream/metrics`;
    const params = new URLSearchParams();

    if (toolId) {
      params.append('tool_id', toolId);
    }

    // Parse options from memoized key
    const opts = JSON.parse(optionsKey) as UseSSEOptions;

    // Add tags if sessionId or custom tags are provided
    if (opts?.sessionId) {
      const tags = JSON.stringify({ session_id: opts.sessionId });
      params.append('tags', tags);
    } else if (opts?.tags) {
      const tags = JSON.stringify(opts.tags);
      params.append('tags', tags);
    }

    // Construct final URL with query parameters
    if (params.toString()) {
      url = `${url}?${params.toString()}`;
    }

    let eventSource: EventSource | null = null;
    let reconnectTimeout: number | null = null;

    const connect = () => {
      try {
        eventSource = new EventSource(url);

        // Throttle updates to max 1 per 2 seconds for better performance
        // (reduced from 500ms to prevent excessive re-renders)
        let lastUpdate = 0;
        const THROTTLE_MS = 2000;

        eventSource.onopen = () => {
          setIsConnected(true);
          setError(null);
        };

        // Listen for the named 'metrics' event
        eventSource.addEventListener('metrics', (event: MessageEvent) => {
          try {
            // Throttle to prevent excessive updates
            const now = Date.now();
            if (now - lastUpdate < THROTTLE_MS) {
              return;
            }
            lastUpdate = now;

            const backendData = JSON.parse(event.data) as BackendMetrics;
            const flatData = flattenMetrics(backendData);
            setData(flatData);
          } catch (err) {
            if (import.meta.env.DEV) {
              console.error('[useSSE] Failed to parse SSE data:', err);
            }
          }
        });

        eventSource.onerror = () => {
          if (import.meta.env.DEV) {
            console.error('[useSSE] EventSource error, readyState:', eventSource?.readyState);
          }
          setIsConnected(false);
          setError(new Error('SSE connection failed'));
          eventSource?.close();

          // Auto-reconnect after 5 seconds
          reconnectTimeout = setTimeout(() => {
            connect();
          }, 5000);
        };
      } catch (err) {
        if (import.meta.env.DEV) {
          console.error('[useSSE] Failed to create EventSource:', err);
        }
        setError(err as Error);
      }
    };

    connect();

    return () => {
      if (reconnectTimeout) {
        clearTimeout(reconnectTimeout);
      }
      eventSource?.close();
      setIsConnected(false);
    };
  }, [toolId, optionsKey]);

  return { data, error, isConnected };
}
