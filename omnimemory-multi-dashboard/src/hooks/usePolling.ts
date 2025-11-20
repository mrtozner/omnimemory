import { useEffect, useState, useRef, useMemo } from 'react';
import type { Metrics } from '../services/api';

const API_BASE = 'http://localhost:8003';

export interface UsePollingOptions {
  sessionId?: string;
  tags?: Record<string, string>;
  interval?: number; // Polling interval in ms, default 5000 (5 seconds)
  enabled?: boolean; // Allow disabling polling
}

/**
 * Lightweight polling hook to replace heavy SSE connections
 * Polls the metrics API at configurable intervals
 */
export function usePolling(toolId?: string, options?: UsePollingOptions) {
  const [data, setData] = useState<Metrics | null>(null);
  const [error, setError] = useState<Error | null>(null);
  const [isConnected, setIsConnected] = useState(false);

  const optionsRef = useRef(options);
  optionsRef.current = options;

  useEffect(() => {
    const enabled = options?.enabled !== false;
    const interval = options?.interval || 5000; // Default 5 seconds

    if (!enabled) {
      setIsConnected(false);
      return;
    }

    let isMounted = true;
    let timeoutId: number | null = null;

    const fetchMetrics = async () => {
      try {
        // Build URL with parameters
        const url = `${API_BASE}/metrics/tool/${toolId || 'unknown'}`;

        const response = await fetch(url);

        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`);
        }

        const responseData = await response.json();

        if (isMounted) {
          // The /metrics/tool/{tool_id} endpoint returns { tool_id, hours, metrics: {...} }
          // Extract the metrics object and use it directly
          const metricsData = responseData.metrics || responseData;
          setData(metricsData as Metrics);
          setIsConnected(true);
          setError(null);
        }
      } catch (err) {
        if (isMounted) {
          setError(err as Error);
          setIsConnected(false);
        }
      } finally {
        // Schedule next poll
        if (isMounted) {
          timeoutId = window.setTimeout(fetchMetrics, interval);
        }
      }
    };

    // Start polling
    fetchMetrics();

    return () => {
      isMounted = false;
      if (timeoutId) {
        clearTimeout(timeoutId);
      }
    };
  }, [toolId, options?.interval, options?.enabled]);

  // Stabilize return object to prevent unnecessary rerenders
  return useMemo(
    () => ({ data, error, isConnected }),
    [data, error, isConnected]
  );
}
