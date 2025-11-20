# Real-time Hooks Usage Guide

Phase 2 Multi-Tool Context Bridge - WebSocket/SSE Real-time Updates

## Overview

This package provides 4 new React hooks for real-time updates in the Multi-Tool Context Bridge:

1. **useToolRegistry** - Fetch IDE tools and agents from gateway
2. **useToolNotifications** - WebSocket notifications for tool events
3. **useSessionEvents** - SSE streaming of session events
4. **useRealtimeTools** - Combined hook with automatic updates

## Installation

These hooks are already integrated into the project. Simply import from the hooks directory:

```typescript
import {
  useToolRegistry,
  useToolNotifications,
  useSessionEvents,
  useRealtimeTools
} from '@/hooks';
```

## Hook Details

### 1. useToolRegistry

Fetches tools from the multi-tool gateway at `http://localhost:8009`.

**Features:**
- Fetches IDE tools and agents
- Manual refetch support
- Error handling
- Loading states

**Example:**

```tsx
import { useToolRegistry } from '@/hooks';

function ToolsList({ projectId }: { projectId: string }) {
  const { ideTools, agents, loading, error, refetch } = useToolRegistry(projectId);

  if (loading) return <div>Loading tools...</div>;
  if (error) return <div>Error: {error}</div>;

  return (
    <div>
      <button onClick={refetch}>Refresh</button>

      <h3>IDE Tools ({ideTools.length})</h3>
      {ideTools.map(tool => (
        <div key={tool.tool_id}>
          {tool.tool_type} - {tool.tool_id}
        </div>
      ))}

      <h3>Agents ({agents.length})</h3>
      {agents.map(agent => (
        <div key={agent.tool_id}>
          {agent.tool_type} - {agent.tool_id}
        </div>
      ))}
    </div>
  );
}
```

### 2. useToolNotifications

WebSocket connection for real-time tool join/leave notifications.

**Features:**
- WebSocket connection to `ws://localhost:8009/ws/notifications`
- Auto-reconnection with exponential backoff
- Session subscription
- Notification management

**Example:**

```tsx
import { useToolNotifications } from '@/hooks';

function ToolNotifications({ sessionId }: { sessionId?: string }) {
  const {
    notifications,
    removeNotification,
    clearAll,
    connected,
    error
  } = useToolNotifications({
    sessionId,
    enabled: true,
    autoConnect: true
  });

  return (
    <div className="fixed top-4 right-4 space-y-2">
      <div className="flex items-center gap-2">
        <div className={connected ? 'text-green-500' : 'text-red-500'}>
          {connected ? '● Connected' : '○ Disconnected'}
        </div>
        {error && <div className="text-red-500">{error}</div>}
      </div>

      {notifications.map(notif => (
        <div
          key={notif.id}
          className="bg-white shadow-lg rounded-lg p-4 border-l-4 border-blue-500"
        >
          <div className="flex justify-between items-start">
            <div>
              <div className="font-semibold">{notif.type}</div>
              <div className="text-sm text-gray-600">{notif.message}</div>
              <div className="text-xs text-gray-400 mt-1">
                {new Date(notif.timestamp).toLocaleTimeString()}
              </div>
            </div>
            <button
              onClick={() => removeNotification(notif.id)}
              className="text-gray-400 hover:text-gray-600"
            >
              ×
            </button>
          </div>
        </div>
      ))}

      {notifications.length > 0 && (
        <button
          onClick={clearAll}
          className="w-full py-2 bg-gray-100 hover:bg-gray-200 rounded"
        >
          Clear All
        </button>
      )}
    </div>
  );
}
```

### 3. useSessionEvents

SSE streaming of session events from metrics service.

**Features:**
- Initial fetch of existing events
- SSE stream for real-time updates
- Connection status
- Error handling

**Example:**

```tsx
import { useSessionEvents } from '@/hooks';

function SessionTimeline({ sessionId }: { sessionId: string }) {
  const { events, loading, error, isStreaming } = useSessionEvents(sessionId);

  if (loading) return <div>Loading events...</div>;
  if (error) return <div>Error: {error}</div>;

  return (
    <div>
      <div className="flex items-center gap-2 mb-4">
        <h3>Session Events</h3>
        {isStreaming && (
          <span className="text-green-500 text-sm">● Streaming</span>
        )}
      </div>

      <div className="space-y-2">
        {events.map((event, idx) => (
          <div key={idx} className="border-l-2 border-blue-500 pl-4 py-2">
            <div className="font-semibold">{event.event_type}</div>
            {event.tool_id && (
              <div className="text-sm text-gray-600">Tool: {event.tool_id}</div>
            )}
            <div className="text-xs text-gray-400">
              {new Date(event.timestamp).toLocaleString()}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
```

### 4. useRealtimeTools (Combined)

Combines `useToolRegistry` and `useToolNotifications` for automatic updates.

**Features:**
- Auto-refresh when tools connect/disconnect
- Fallback polling if WebSocket unavailable
- Unified interface
- Connection status

**Example:**

```tsx
import { useRealtimeTools } from '@/hooks';

function RealtimeToolsView({ projectId, sessionId }: {
  projectId: string;
  sessionId?: string;
}) {
  const {
    ideTools,
    agents,
    loading,
    error,
    notifications,
    connected,
    lastUpdate,
    removeNotification,
    clearAllNotifications
  } = useRealtimeTools({
    projectId,
    sessionId,
    enableNotifications: true,
    pollInterval: 10000 // Fallback to polling every 10s if WS fails
  });

  if (loading) return <div>Loading...</div>;
  if (error) return <div>Error: {error}</div>;

  return (
    <div className="p-6">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-2xl font-bold">
          Connected Tools ({ideTools.length + agents.length})
        </h2>

        <div className="flex items-center gap-4">
          <div className={`flex items-center gap-2 ${connected ? 'text-green-500' : 'text-gray-400'}`}>
            {connected ? '●' : '○'}
            <span className="text-sm">{connected ? 'Live' : 'Polling'}</span>
          </div>

          <div className="text-sm text-gray-500">
            Updated: {lastUpdate.toLocaleTimeString()}
          </div>
        </div>
      </div>

      {/* Notifications */}
      {notifications.length > 0 && (
        <div className="mb-6 space-y-2">
          <div className="flex items-center justify-between">
            <h3 className="font-semibold">Recent Activity</h3>
            <button
              onClick={clearAllNotifications}
              className="text-sm text-blue-600 hover:underline"
            >
              Clear All
            </button>
          </div>

          {notifications.slice(-5).map(notif => (
            <div
              key={notif.id}
              className="bg-blue-50 border-l-4 border-blue-500 p-3 rounded flex justify-between items-start"
            >
              <div>
                <div className="font-medium">{notif.message}</div>
                <div className="text-xs text-gray-500">
                  {new Date(notif.timestamp).toLocaleTimeString()}
                </div>
              </div>
              <button
                onClick={() => removeNotification(notif.id)}
                className="text-gray-400 hover:text-gray-600"
              >
                ×
              </button>
            </div>
          ))}
        </div>
      )}

      {/* IDE Tools */}
      <div className="mb-6">
        <h3 className="font-semibold mb-3">IDE Tools ({ideTools.length})</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {ideTools.map(tool => (
            <div key={tool.tool_id} className="border rounded-lg p-4">
              <div className="font-medium">{tool.tool_type}</div>
              <div className="text-sm text-gray-600">{tool.tool_id}</div>
              <div className="text-xs text-gray-400 mt-2">
                Last active: {new Date(tool.last_activity).toLocaleString()}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Agents */}
      <div>
        <h3 className="font-semibold mb-3">Agents ({agents.length})</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {agents.map(agent => (
            <div key={agent.tool_id} className="border rounded-lg p-4">
              <div className="font-medium">{agent.tool_type}</div>
              <div className="text-sm text-gray-600">{agent.tool_id}</div>
              <div className="text-xs text-gray-400 mt-2">
                Last active: {new Date(agent.last_activity).toLocaleString()}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
```

## Error Handling

All hooks include proper error handling:

```tsx
const { data, error, loading } = useToolRegistry(projectId);

if (loading) {
  return <LoadingSpinner />;
}

if (error) {
  return (
    <ErrorDisplay
      message={error}
      retry={() => window.location.reload()}
    />
  );
}

// Render data
```

## Connection Management

### WebSocket Reconnection

The `useToolNotifications` hook includes automatic reconnection with exponential backoff:

- 1st attempt: 5 seconds
- 2nd attempt: 10 seconds
- 3rd attempt: 20 seconds
- 4th+ attempt: 40 seconds
- Maximum: 60 seconds

### Fallback Polling

The `useRealtimeTools` hook automatically falls back to polling if WebSocket is unavailable:

```tsx
const tools = useRealtimeTools({
  projectId,
  pollInterval: 5000 // Poll every 5 seconds if WS fails
});
```

## Testing

### Unit Tests

```tsx
import { renderHook } from '@testing-library/react-hooks';
import { useToolRegistry } from '@/hooks';

test('useToolRegistry fetches tools', async () => {
  const { result, waitForNextUpdate } = renderHook(() =>
    useToolRegistry('test-project')
  );

  expect(result.current.loading).toBe(true);

  await waitForNextUpdate();

  expect(result.current.loading).toBe(false);
  expect(result.current.ideTools).toBeDefined();
  expect(result.current.agents).toBeDefined();
});
```

### Integration Tests

```tsx
import { render, screen, waitFor } from '@testing-library/react';
import { RealtimeToolsView } from '@/components';

test('displays connected tools', async () => {
  render(<RealtimeToolsView projectId="test" />);

  await waitFor(() => {
    expect(screen.getByText(/Connected Tools/i)).toBeInTheDocument();
  });

  expect(screen.getByText(/Live/i)).toBeInTheDocument();
});
```

## Performance Considerations

### Throttling Updates

To prevent excessive re-renders with high-frequency updates:

```tsx
import { useMemo } from 'react';

function ToolsGrid({ projectId }: { projectId: string }) {
  const { ideTools, agents } = useRealtimeTools({ projectId });

  // Memoize expensive computations
  const sortedTools = useMemo(() => {
    return [...ideTools, ...agents].sort((a, b) =>
      a.last_activity.localeCompare(b.last_activity)
    );
  }, [ideTools, agents]);

  return (
    <div>
      {sortedTools.map(tool => <ToolCard key={tool.tool_id} tool={tool} />)}
    </div>
  );
}
```

### Cleanup

All hooks properly clean up resources on unmount:
- WebSocket connections closed
- SSE connections closed
- Intervals cleared
- Timeouts cleared

## API Endpoints

The hooks connect to these endpoints:

### Multi-Tool Gateway (Port 8009)
- **HTTP**: `http://localhost:8009/api/v1/projects/{projectId}/all-tools`
- **WebSocket**: `ws://localhost:8009/ws/notifications`

### Metrics Service (Port 8003)
- **HTTP**: `http://localhost:8003/sessions/{sessionId}/events`
- **SSE**: `http://localhost:8003/sessions/{sessionId}/events/stream`

## Success Criteria

✅ All hooks created with TypeScript types
✅ WebSocket connection management with auto-reconnection
✅ SSE streaming with error handling
✅ Fallback polling support
✅ Comprehensive examples and documentation
✅ Proper resource cleanup
✅ Type-safe interfaces
✅ Error handling and retry logic

## Next Steps

1. Integrate hooks into Phase 2 components:
   - ToolsList component
   - SessionSharing component
   - ToolNotifications component

2. Add unit tests for each hook

3. Add integration tests for real-time updates

4. Monitor WebSocket/SSE connection stability in production
