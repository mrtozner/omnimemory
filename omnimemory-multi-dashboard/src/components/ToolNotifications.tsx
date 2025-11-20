import { useState, useEffect, useCallback } from 'react';
import { X } from 'lucide-react';
import { cn } from '../lib/utils';
import {
  getToolIcon,
  formatRelativeTime
} from '../utils/toolNotifications';
import type { ToolNotification } from '../utils/toolNotifications';

interface ToolNotificationsProps {
  sessionId?: string;
  maxNotifications?: number;
  autoDismissDelay?: number;
  enableSound?: boolean;
  useWebSocket?: boolean;
}

// Notification Toast Component
interface NotificationToastProps {
  notification: ToolNotification;
  onDismiss: () => void;
  isExiting?: boolean;
}

function NotificationToast({ notification, onDismiss, isExiting }: NotificationToastProps) {
  return (
    <div
      className={cn(
        'bg-white dark:bg-gray-800 rounded-lg shadow-lg border border-gray-200 dark:border-gray-700 p-4 min-w-80 max-w-md',
        'transition-all duration-300',
        isExiting ? 'animate-fade-out' : 'animate-slide-in-right'
      )}
      role="alert"
      aria-live="polite"
      aria-atomic="true"
    >
      <div className="flex items-start gap-3">
        {/* Tool Icon */}
        <div className="flex-shrink-0 mt-0.5">
          {getToolIcon(notification.tool_type)}
        </div>

        {/* Message */}
        <div className="flex-1 min-w-0">
          <p className="text-sm font-medium text-gray-900 dark:text-gray-100">
            {notification.message}
          </p>
          <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
            {formatRelativeTime(notification.timestamp)}
          </p>
        </div>

        {/* Close Button */}
        <button
          onClick={onDismiss}
          className={cn(
            'flex-shrink-0 text-gray-400 hover:text-gray-600 dark:hover:text-gray-200',
            'transition-colors rounded p-1 hover:bg-gray-100 dark:hover:bg-gray-700',
            'focus:outline-none focus:ring-2 focus:ring-blue-500'
          )}
          aria-label="Dismiss notification"
        >
          <X className="w-4 h-4" />
        </button>
      </div>
    </div>
  );
}

// Main ToolNotifications Component
export function ToolNotifications({
  sessionId,
  maxNotifications = 5,
  autoDismissDelay = 5000,
  enableSound = false,
  useWebSocket = false,
}: ToolNotificationsProps) {
  const [notifications, setNotifications] = useState<ToolNotification[]>([]);
  const [exitingIds, setExitingIds] = useState<Set<string>>(new Set());
  const [lastCheckTime, setLastCheckTime] = useState<string>(new Date().toISOString());

  // Dismiss notification with fade-out animation (defined first)
  const dismissNotification = useCallback((id: string) => {
    setExitingIds((prev) => new Set(prev).add(id));

    // Wait for fade-out animation to complete
    setTimeout(() => {
      setNotifications((prev) => prev.filter((n) => n.id !== id));
      setExitingIds((prev) => {
        const next = new Set(prev);
        next.delete(id);
        return next;
      });
    }, 300); // Match fade-out animation duration
  }, []);

  // Add notification (uses dismissNotification, so defined after it)
  const addNotification = useCallback((notification: ToolNotification) => {
    setNotifications((prev) => {
      // Check if notification already exists
      if (prev.some((n) => n.id === notification.id)) {
        return prev;
      }

      // If at max capacity, remove oldest
      const newNotifications = prev.length >= maxNotifications
        ? prev.slice(1)
        : prev;

      return [...newNotifications, notification];
    });

    // Play sound if enabled
    if (enableSound) {
      try {
        const audio = new Audio('/notification.mp3');
        audio.volume = 0.3;
        audio.play().catch(() => {
          // Ignore audio play errors
        });
      } catch {
        // Ignore audio errors
      }
    }

    // Auto-dismiss
    setTimeout(() => {
      dismissNotification(notification.id);
    }, autoDismissDelay);
  }, [maxNotifications, autoDismissDelay, enableSound, dismissNotification]);

  // WebSocket connection
  useEffect(() => {
    if (!useWebSocket) return;

    const wsUrl = import.meta.env.VITE_WS_GATEWAY_URL || 'ws://localhost:8009';
    const ws = new WebSocket(`${wsUrl}/ws/notifications`);

    ws.onopen = () => {
      console.log('[ToolNotifications] WebSocket connected');
      if (sessionId) {
        ws.send(JSON.stringify({ type: 'subscribe', session_id: sessionId }));
      }
    };

    ws.onmessage = (event) => {
      try {
        const notification: ToolNotification = JSON.parse(event.data);
        addNotification(notification);
      } catch (error) {
        console.error('[ToolNotifications] Failed to parse WebSocket message:', error);
      }
    };

    ws.onerror = (error) => {
      console.error('[ToolNotifications] WebSocket error:', error);
    };

    ws.onclose = () => {
      console.log('[ToolNotifications] WebSocket disconnected');
    };

    return () => {
      ws.close();
    };
  }, [useWebSocket, sessionId, addNotification]);

  // Polling (default)
  useEffect(() => {
    if (useWebSocket) return;

    const interval = setInterval(async () => {
      try {
        const url = sessionId
          ? `http://localhost:8009/api/v1/tools/notifications?session_id=${sessionId}&since=${encodeURIComponent(lastCheckTime)}`
          : `http://localhost:8009/api/v1/tools/notifications?since=${encodeURIComponent(lastCheckTime)}`;

        const response = await fetch(url);

        if (!response.ok) {
          console.error('[ToolNotifications] Failed to fetch notifications:', response.status);
          return;
        }

        const data = await response.json();

        if (data.notifications && Array.isArray(data.notifications)) {
          data.notifications.forEach((notification: ToolNotification) => {
            addNotification(notification);
          });

          if (data.notifications.length > 0) {
            setLastCheckTime(new Date().toISOString());
          }
        }
      } catch (error) {
        console.error('[ToolNotifications] Polling error:', error);
      }
    }, 2000); // Poll every 2 seconds

    return () => clearInterval(interval);
  }, [useWebSocket, sessionId, lastCheckTime, addNotification]);

  // Keyboard support (Escape to dismiss all)
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape' && notifications.length > 0) {
        notifications.forEach((notification) => {
          dismissNotification(notification.id);
        });
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [notifications, dismissNotification]);

  if (notifications.length === 0) {
    return null;
  }

  return (
    <div
      className="fixed top-4 right-4 z-50 space-y-2 pointer-events-none"
      aria-label="Tool notifications"
      role="region"
    >
      <div className="space-y-2 pointer-events-auto">
        {notifications.map((notification) => (
          <NotificationToast
            key={notification.id}
            notification={notification}
            onDismiss={() => dismissNotification(notification.id)}
            isExiting={exitingIds.has(notification.id)}
          />
        ))}
      </div>
    </div>
  );
}
