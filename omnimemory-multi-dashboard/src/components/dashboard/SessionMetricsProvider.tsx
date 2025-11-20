import type { ReactNode } from 'react';
import { usePolling } from '../../hooks/usePolling';
import type { Metrics } from '../../services/api';

export interface SessionMetricsProviderProps {
  sessionId: string;
  children: (metrics: Metrics | null, isConnected: boolean) => ReactNode;
}

/**
 * SessionMetricsProvider component
 *
 * Provides session-specific metrics via SSE using the session_id tag.
 * Uses render props pattern to pass metrics to children.
 *
 * @example
 * <SessionMetricsProvider sessionId="sess_123">
 *   {(metrics, isConnected) => (
 *     <div>
 *       {isConnected ? `Embeddings: ${metrics?.total_embeddings}` : 'Connecting...'}
 *     </div>
 *   )}
 * </SessionMetricsProvider>
 */
export function SessionMetricsProvider({ sessionId, children }: SessionMetricsProviderProps) {
  const { data: metrics, isConnected } = usePolling('claude-code', { sessionId, interval: 10000, enabled: true }); // Enable polling

  return <>{children(metrics, isConnected)}</>;
}
