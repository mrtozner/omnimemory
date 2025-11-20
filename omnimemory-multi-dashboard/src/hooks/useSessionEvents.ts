import { useState, useEffect, useRef } from 'react';

const API_BASE = 'http://localhost:8003';

export interface SessionEvent {
  event_type: string;
  tool_id?: string;
  tool_type?: string;
  timestamp: string;
  metadata?: Record<string, unknown>;
}

export interface UseSessionEventsReturn {
  events: SessionEvent[];
  loading: boolean;
  error: string | null;
  isStreaming: boolean;
}

/**
 * Hook for SSE (Server-Sent Events) streaming of session events
 *
 * Provides real-time updates about session activities via SSE.
 * Automatically handles initial fetch and streaming updates.
 *
 * @example
 * ```tsx
 * const { events, loading, error, isStreaming } = useSessionEvents('session-123');
 *
 * return (
 *   <div>
 *     <div className="status">
 *       {loading ? 'Loading...' : isStreaming ? '● Streaming' : '○ Not streaming'}
 *     </div>
 *
 *     {error && <div className="error">{error}</div>}
 *
 *     <div className="events-timeline">
 *       {events.map((event, idx) => (
 *         <TimelineEvent key={idx} event={event} />
 *       ))}
 *     </div>
 *   </div>
 * );
 * ```
 */
export function useSessionEvents(sessionId: string): UseSessionEventsReturn {
  const [events, setEvents] = useState<SessionEvent[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [isStreaming, setIsStreaming] = useState(false);

  const eventSourceRef = useRef<EventSource | null>(null);
  const isMountedRef = useRef(true);

  useEffect(() => {
    if (!sessionId) {
      setError('Session ID is required');
      setLoading(false);
      return;
    }

    isMountedRef.current = true;

    // Initial fetch of existing events
    const fetchInitialEvents = async () => {
      try {
        setLoading(true);
        setError(null);

        const response = await fetch(
          `${API_BASE}/sessions/${sessionId}/events`
        );

        if (!response.ok) {
          throw new Error(`Failed to fetch events: ${response.statusText}`);
        }

        const data = await response.json();

        if (isMountedRef.current) {
          setEvents(data.events || []);
        }
      } catch (err) {
        if (isMountedRef.current) {
          setError(err instanceof Error ? err.message : 'Failed to load events');
          console.error('[useSessionEvents] Error fetching initial events:', err);
        }
      } finally {
        if (isMountedRef.current) {
          setLoading(false);
        }
      }
    };

    fetchInitialEvents();

    // SSE for real-time event updates
    const eventSource = new EventSource(
      `${API_BASE}/sessions/${sessionId}/events/stream`
    );
    eventSourceRef.current = eventSource;

    eventSource.onopen = () => {
      if (isMountedRef.current) {
        console.log('[useSessionEvents] SSE connection opened');
        setIsStreaming(true);
        setError(null);
      }
    };

    eventSource.onmessage = (event) => {
      if (!isMountedRef.current) return;

      try {
        const newEvent = JSON.parse(event.data);

        if (import.meta.env.DEV) {
          console.log('[useSessionEvents] New event received:', newEvent);
        }

        setEvents(prev => [...prev, newEvent]);
      } catch (err) {
        console.error('[useSessionEvents] Failed to parse event:', err);
      }
    };

    eventSource.onerror = (err) => {
      if (!isMountedRef.current) return;

      console.error('[useSessionEvents] SSE error:', err);
      setIsStreaming(false);

      // Only set error if we're still mounted and haven't loaded initial data
      if (events.length === 0) {
        setError('Failed to connect to event stream');
      }

      eventSource.close();
    };

    return () => {
      isMountedRef.current = false;
      eventSource.close();
      setIsStreaming(false);
    };
    // events.length is intentionally not included to prevent re-subscribing
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sessionId]);

  return {
    events,
    loading,
    error,
    isStreaming
  };
}
