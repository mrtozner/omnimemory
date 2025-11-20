import { useState, useEffect, useCallback, useRef } from 'react';

const WS_BASE = import.meta.env.VITE_WS_GATEWAY_URL || 'ws://localhost:8009';

export interface ToolNotification {
  id: string;
  type: 'tool_joined' | 'tool_left' | 'context_merged';
  tool_type: string;
  tool_id: string;
  message: string;
  timestamp: string;
  metadata?: Record<string, unknown>;
}

export interface UseToolNotificationsOptions {
  sessionId?: string;
  enabled?: boolean;
  autoConnect?: boolean;
  reconnectInterval?: number;
}

/**
 * Hook for receiving real-time tool notifications via WebSocket
 *
 * Provides notifications when tools join/leave and contexts are merged.
 * Includes automatic reconnection and connection state management.
 *
 * @example
 * ```tsx
 * const {
 *   notifications,
 *   addNotification,
 *   removeNotification,
 *   clearAll,
 *   connected
 * } = useToolNotifications({
 *   sessionId: 'session-123',
 *   enabled: true
 * });
 *
 * return (
 *   <div>
 *     <div className="status">
 *       {connected ? '● Connected' : '○ Disconnected'}
 *     </div>
 *
 *     <div className="notifications">
 *       {notifications.map(notif => (
 *         <Notification
 *           key={notif.id}
 *           notification={notif}
 *           onDismiss={() => removeNotification(notif.id)}
 *         />
 *       ))}
 *     </div>
 *
 *     <button onClick={clearAll}>Clear All</button>
 *   </div>
 * );
 * ```
 */
export function useToolNotifications(options: UseToolNotificationsOptions = {}) {
  const {
    sessionId,
    enabled = true,
    autoConnect = true,
    reconnectInterval = 5000
  } = options;

  const [notifications, setNotifications] = useState<ToolNotification[]>([]);
  const [connected, setConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [reconnectAttempt, setReconnectAttempt] = useState(0);

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<number | null>(null);
  const isMountedRef = useRef(true);

  const addNotification = useCallback((notification: ToolNotification) => {
    setNotifications(prev => [...prev, notification]);
  }, []);

  const removeNotification = useCallback((id: string) => {
    setNotifications(prev => prev.filter(n => n.id !== id));
  }, []);

  const clearAll = useCallback(() => {
    setNotifications([]);
  }, []);

  const connect = useCallback(() => {
    if (!enabled || !isMountedRef.current) {
      return;
    }

    try {
      const url = `${WS_BASE}/ws/notifications`;
      const ws = new WebSocket(url);
      wsRef.current = ws;

      ws.onopen = () => {
        if (!isMountedRef.current) return;

        console.log('[useToolNotifications] WebSocket connected');
        setConnected(true);
        setError(null);
        setReconnectAttempt(0);

        // Subscribe to session if provided
        if (sessionId) {
          ws.send(JSON.stringify({
            type: 'subscribe',
            session_id: sessionId
          }));
          console.log('[useToolNotifications] Subscribed to session:', sessionId);
        }
      };

      ws.onmessage = (event) => {
        if (!isMountedRef.current) return;

        try {
          const notification = JSON.parse(event.data);

          // Add ID if not present
          const notificationWithId = {
            ...notification,
            id: notification.id || `notif-${Date.now()}-${Math.random()}`
          };

          addNotification(notificationWithId);

          if (import.meta.env.DEV) {
            console.log('[useToolNotifications] Received notification:', notificationWithId);
          }
        } catch (err) {
          console.error('[useToolNotifications] Failed to parse notification:', err);
        }
      };

      ws.onerror = (event) => {
        if (!isMountedRef.current) return;

        setError('WebSocket connection error');
        console.error('[useToolNotifications] WebSocket error:', event);
      };

      ws.onclose = () => {
        if (!isMountedRef.current) return;

        console.log('[useToolNotifications] WebSocket closed');
        setConnected(false);
        wsRef.current = null;

        // Auto-reconnect if enabled
        if (autoConnect && enabled) {
          const attempt = reconnectAttempt + 1;
          setReconnectAttempt(attempt);

          // Exponential backoff: 5s, 10s, 20s, 40s, max 60s
          const delay = Math.min(
            reconnectInterval * Math.pow(2, Math.min(attempt - 1, 3)),
            60000
          );

          console.log(`[useToolNotifications] Reconnecting in ${delay}ms (attempt ${attempt})...`);
          reconnectTimeoutRef.current = window.setTimeout(() => {
            connect();
          }, delay);
        }
      };
    } catch (err) {
      console.error('[useToolNotifications] Failed to create WebSocket:', err);
      setError(err instanceof Error ? err.message : 'Unknown error');
      setConnected(false);
    }
  }, [enabled, sessionId, autoConnect, reconnectInterval, reconnectAttempt, addNotification]);

  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }

    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }

    setConnected(false);
  }, []);

  // Connect on mount, disconnect on unmount
  useEffect(() => {
    isMountedRef.current = true;

    if (autoConnect) {
      connect();
    }

    return () => {
      isMountedRef.current = false;
      disconnect();
    };
  }, [autoConnect, connect, disconnect]);

  return {
    notifications,
    addNotification,
    removeNotification,
    clearAll,
    connected,
    error
  };
}
