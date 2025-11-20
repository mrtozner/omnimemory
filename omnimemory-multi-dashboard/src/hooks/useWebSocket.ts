import { useEffect, useState, useRef, useCallback } from 'react';
import type { Metrics, BackendMetrics } from '../services/api';

const WS_BASE = import.meta.env.VITE_WS_METRICS_URL || 'ws://localhost:8003';

export interface UseWebSocketOptions {
  sessionId?: string;
  tags?: Record<string, string>;
  enabled?: boolean;
  autoReconnect?: boolean;
  reconnectInterval?: number;
}

interface WebSocketMessage {
  type: 'connected' | 'metrics' | 'error';
  timestamp?: string;
  data?: unknown;
  error?: string;
  message?: string;
  interval?: number;
}

/**
 * WebSocket hook for real-time metrics streaming
 *
 * Provides bidirectional real-time communication with automatic reconnection.
 * More efficient than polling for high-frequency updates.
 *
 * @example
 * ```tsx
 * const { data, isConnected, error, reconnect } = useWebSocket('claude-code', {
 *   enabled: true,
 *   autoReconnect: true
 * });
 *
 * return (
 *   <div>
 *     {isConnected ? (
 *       <div>Tokens saved: {data?.tokens_saved || 0}</div>
 *     ) : (
 *       <div>Connecting... <button onClick={reconnect}>Retry</button></div>
 *     )}
 *   </div>
 * );
 * ```
 */
export function useWebSocket(toolId?: string, options?: UseWebSocketOptions) {
  const [data, setData] = useState<Metrics | null>(null);
  const [error, setError] = useState<Error | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [reconnectAttempt, setReconnectAttempt] = useState(0);

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<number | null>(null);
  const isMountedRef = useRef(true);

  const {
    enabled = true,
    sessionId,
    autoReconnect = true,
    reconnectInterval = 5000,
  } = options || {};

  const connect = useCallback(() => {
    if (!enabled || !isMountedRef.current) {
      return;
    }

    // Build WebSocket URL with query parameters
    const params = new URLSearchParams();
    if (toolId) {
      params.append('tool_id', toolId);
    }
    if (sessionId) {
      params.append('session_id', sessionId);
    }

    const url = `${WS_BASE}/ws/metrics${params.toString() ? '?' + params.toString() : ''}`;

    try {
      const ws = new WebSocket(url);
      wsRef.current = ws;

      ws.onopen = () => {
        if (!isMountedRef.current) return;
        console.log('WebSocket connected:', url);
        setIsConnected(true);
        setError(null);
        setReconnectAttempt(0);
      };

      ws.onmessage = (event) => {
        if (!isMountedRef.current) return;

        try {
          const message: WebSocketMessage = JSON.parse(event.data);

          switch (message.type) {
            case 'connected':
              console.log('WebSocket handshake complete:', message.message);
              break;

            case 'metrics': {
              // Extract metrics data from the message
              const metricsData = message.data as BackendMetrics | undefined;
              if (metricsData) {
                // Convert the metrics structure to match the expected Metrics type
                const metrics: Metrics = {
                  tokens_saved: metricsData.compression?.metrics?.total_tokens_saved || 0,
                  total_compressions: metricsData.compression?.metrics?.total_compressions || 0,
                  compression_ratio: metricsData.compression?.metrics?.overall_compression_ratio || 0,
                  avg_compression_ratio: metricsData.compression?.metrics?.avg_compression_ratio || 0,
                  total_embeddings: metricsData.embeddings?.mlx_metrics?.total_embeddings || 0,
                  cache_hit_rate: metricsData.embeddings?.mlx_metrics?.cache_hit_rate || 0,
                  cache_hits: metricsData.embeddings?.mlx_metrics?.cache_hits || 0,
                  cache_misses: metricsData.embeddings?.mlx_metrics?.cache_misses || 0,
                  tokens_processed: metricsData.embeddings?.mlx_metrics?.tokens_processed || 0,
                  quality_score: metricsData.compression?.metrics?.avg_quality_score || 0,
                  avg_quality_score: metricsData.compression?.metrics?.avg_quality_score || 0,
                  pattern_count: metricsData.procedural?.pattern_count || 0,
                  prediction_accuracy: 0,
                  total_original_tokens: metricsData.compression?.metrics?.total_original_tokens || 0,
                  total_compressed_tokens: metricsData.compression?.metrics?.total_compressed_tokens || 0,
                  graph_edges: metricsData.procedural?.graph_edge_count || 0,
                  total_successes: metricsData.procedural?.total_successes || 0,
                  total_failures: metricsData.procedural?.total_failures || 0,
                  timestamp: message.timestamp || new Date().toISOString(),
                };
                setData(metrics);
                setError(null);
              }
              break;
            }

            case 'error':
              console.error('WebSocket error message:', message.error);
              setError(new Error(message.error || 'Unknown error'));
              break;

            default:
              console.warn('Unknown WebSocket message type:', message);
          }
        } catch (err) {
          console.error('Failed to parse WebSocket message:', err);
          setError(err as Error);
        }
      };

      ws.onerror = (event) => {
        if (!isMountedRef.current) return;
        console.error('WebSocket error:', event);
        setError(new Error('WebSocket connection error'));
        setIsConnected(false);
      };

      ws.onclose = () => {
        if (!isMountedRef.current) return;
        console.log('WebSocket closed');
        setIsConnected(false);
        wsRef.current = null;

        // Auto-reconnect if enabled
        if (autoReconnect && enabled) {
          const attempt = reconnectAttempt + 1;
          setReconnectAttempt(attempt);

          // Exponential backoff: 5s, 10s, 20s, 40s, max 60s
          const delay = Math.min(reconnectInterval * Math.pow(2, Math.min(attempt - 1, 3)), 60000);

          console.log(`Reconnecting in ${delay}ms (attempt ${attempt})...`);
          reconnectTimeoutRef.current = window.setTimeout(() => {
            connect();
          }, delay);
        }
      };
    } catch (err) {
      console.error('Failed to create WebSocket:', err);
      setError(err as Error);
      setIsConnected(false);
    }
  }, [enabled, toolId, sessionId, autoReconnect, reconnectInterval, reconnectAttempt]);

  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }

    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }

    setIsConnected(false);
  }, []);

  const reconnect = useCallback(() => {
    disconnect();
    setReconnectAttempt(0);
    connect();
  }, [disconnect, connect]);

  // Connect on mount, disconnect on unmount
  useEffect(() => {
    isMountedRef.current = true;
    connect();

    return () => {
      isMountedRef.current = false;
      disconnect();
    };
  }, [connect, disconnect]);

  return {
    data,
    error,
    isConnected,
    reconnect,
    reconnectAttempt,
  };
}
