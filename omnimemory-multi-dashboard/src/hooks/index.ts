// Real-time hooks for Phase 2 Multi-Tool Context Bridge
export { useToolRegistry } from './useToolRegistry';
export type { ToolInfo, UseToolRegistryReturn } from './useToolRegistry';

export { useToolNotifications } from './useToolNotifications';
export type { ToolNotification, UseToolNotificationsOptions } from './useToolNotifications';

export { useSessionEvents } from './useSessionEvents';
export type { SessionEvent, UseSessionEventsReturn } from './useSessionEvents';

export { useRealtimeTools } from './useRealtimeTools';
export type { UseRealtimeToolsOptions } from './useRealtimeTools';

// Existing hooks (for backwards compatibility)
export { useWebSocket } from './useWebSocket';
export { useSSE } from './useSSE';
export { usePolling } from './usePolling';
