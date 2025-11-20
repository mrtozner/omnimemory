import { useState, useEffect } from 'react';
import { useToolRegistry } from './useToolRegistry';
import { useToolNotifications } from './useToolNotifications';

export interface UseRealtimeToolsOptions {
  projectId: string;
  sessionId?: string;
  pollInterval?: number; // Fallback polling if WebSocket fails
  enableNotifications?: boolean;
}

/**
 * Combined hook for real-time tool registry with notifications
 *
 * Provides unified access to tool registry data with automatic updates
 * via WebSocket notifications. Falls back to polling if WebSocket unavailable.
 *
 * @example
 * ```tsx
 * const {
 *   ideTools,
 *   agents,
 *   loading,
 *   error,
 *   notifications,
 *   connected,
 *   lastUpdate
 * } = useRealtimeTools({
 *   projectId: 'my-project',
 *   sessionId: 'session-123',
 *   enableNotifications: true
 * });
 *
 * return (
 *   <div>
 *     <div className="header">
 *       <h2>Tools ({ideTools.length + agents.length})</h2>
 *       <div className="status">
 *         {connected ? (
 *           <span className="text-green-500">● Live</span>
 *         ) : (
 *           <span className="text-gray-500">○ Polling</span>
 *         )}
 *       </div>
 *       <div className="last-update">
 *         Updated: {lastUpdate.toLocaleTimeString()}
 *       </div>
 *     </div>
 *
 *     {notifications.length > 0 && (
 *       <div className="notifications">
 *         {notifications.map(notif => (
 *           <Notification key={notif.id} notification={notif} />
 *         ))}
 *       </div>
 *     )}
 *
 *     <div className="tools-grid">
 *       {ideTools.map(tool => <ToolCard key={tool.tool_id} tool={tool} />)}
 *       {agents.map(agent => <AgentCard key={agent.tool_id} agent={agent} />)}
 *     </div>
 *   </div>
 * );
 * ```
 */
export function useRealtimeTools(options: UseRealtimeToolsOptions) {
  const {
    projectId,
    sessionId,
    pollInterval = 5000,
    enableNotifications = true
  } = options;

  const toolRegistry = useToolRegistry(projectId);
  const notifications = useToolNotifications({
    sessionId,
    enabled: enableNotifications,
    autoConnect: enableNotifications
  });

  const [lastUpdate, setLastUpdate] = useState<Date>(new Date());

  // Refetch tools when notifications indicate changes
  useEffect(() => {
    const toolChangeEvents = ['tool_joined', 'tool_left'];
    const latestNotification = notifications.notifications[notifications.notifications.length - 1];

    if (latestNotification && toolChangeEvents.includes(latestNotification.type)) {
      if (import.meta.env.DEV) {
        console.log('[useRealtimeTools] Tool change detected, refetching...');
      }

      toolRegistry.refetch();
      setLastUpdate(new Date());
    }
    // toolRegistry is a stable object, no need to include in deps
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [notifications.notifications]);

  // Fallback polling if WebSocket not connected
  useEffect(() => {
    // If notifications are disabled or not connected, use polling
    if (!enableNotifications || (!notifications.connected && pollInterval > 0)) {
      const interval = setInterval(() => {
        if (import.meta.env.DEV) {
          console.log('[useRealtimeTools] Polling for tool updates...');
        }

        toolRegistry.refetch();
        setLastUpdate(new Date());
      }, pollInterval);

      return () => clearInterval(interval);
    }
    // toolRegistry is a stable object, no need to include in deps
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [notifications.connected, enableNotifications, pollInterval]);

  return {
    // Tool registry data
    ideTools: toolRegistry.ideTools,
    agents: toolRegistry.agents,
    loading: toolRegistry.loading,
    error: toolRegistry.error,
    refetch: toolRegistry.refetch,

    // Notifications
    notifications: notifications.notifications,
    addNotification: notifications.addNotification,
    removeNotification: notifications.removeNotification,
    clearAllNotifications: notifications.clearAll,

    // Connection state
    connected: notifications.connected,
    notificationError: notifications.error,

    // Metadata
    lastUpdate
  };
}
