# ToolNotifications Component

Real-time toast notifications for Phase 2 Multi-Tool Context Bridge Dashboard UI (Week 4-5).

## Overview

The `ToolNotifications` component displays toast-style notifications in the top-right corner when:
- Tools join/leave sessions
- Context is merged between tools
- Files are shared across tools

## Features

- ✅ Toast-style notifications in top-right corner
- ✅ Slide-in animation from right
- ✅ Auto-dismiss after 5 seconds (configurable)
- ✅ Manual dismiss with close button (X)
- ✅ Stack multiple notifications (max 5 by default)
- ✅ Tool-specific icons and messages
- ✅ Dark mode support
- ✅ Full TypeScript support
- ✅ Keyboard accessibility (Escape to dismiss all)
- ✅ ARIA live regions for screen readers
- ✅ WebSocket or polling modes
- ✅ Optional sound notifications
- ✅ Responsive design

## Installation

The component is already included in the omnimemory-multi-dashboard project.

Required dependencies (already installed):
- `react` ^18.3.1
- `lucide-react` ^0.364.0
- `tailwindcss` ^3.4.1
- `clsx` ^2.1.1
- `tailwind-merge` ^2.6.0

## Basic Usage

```tsx
import { ToolNotifications } from './components/ToolNotifications';

function App() {
  return (
    <>
      {/* Your app content */}

      {/* Add notifications component */}
      <ToolNotifications sessionId="your-session-id" />
    </>
  );
}
```

## Props

| Prop | Type | Default | Description |
|------|------|---------|-------------|
| `sessionId` | `string?` | `undefined` | Filter notifications for specific session. If omitted, shows all notifications. |
| `maxNotifications` | `number` | `5` | Maximum number of visible notifications. Older ones auto-dismissed when limit reached. |
| `autoDismissDelay` | `number` | `5000` | Time in milliseconds before auto-dismissing (5 seconds default). |
| `enableSound` | `boolean` | `false` | Play sound on new notifications. Requires `/notification.mp3` in public folder. |
| `useWebSocket` | `boolean` | `false` | Use WebSocket for real-time updates instead of polling. |

## Advanced Usage

### With WebSocket (Real-time)

```tsx
<ToolNotifications
  sessionId="session-123"
  useWebSocket={true}
/>
```

**WebSocket Endpoint**: `ws://localhost:8009/ws/notifications`

### With Custom Settings

```tsx
<ToolNotifications
  sessionId="session-123"
  maxNotifications={3}
  autoDismissDelay={10000}  // 10 seconds
  enableSound={true}
/>
```

### Global Notifications (All Sessions)

```tsx
<ToolNotifications />
```

## Notification Types

The component supports four notification types:

### 1. Tool Joined

```typescript
{
  id: "notif-123",
  type: "tool_joined",
  tool_type: "vscode",
  tool_id: "tool-456",
  message: "VSCode just joined your session",
  timestamp: "2025-11-15T12:00:00Z"
}
```

**Display**: "VSCode just joined your session"

### 2. Tool Left

```typescript
{
  id: "notif-124",
  type: "tool_left",
  tool_type: "cursor",
  tool_id: "tool-789",
  message: "Cursor left your session",
  timestamp: "2025-11-15T12:05:00Z"
}
```

**Display**: "Cursor left your session"

### 3. Context Merged

```typescript
{
  id: "notif-125",
  type: "context_merged",
  tool_type: "system",
  tool_id: "system",
  message: "Context merged from 2 tools (5 files)",
  timestamp: "2025-11-15T12:10:00Z",
  metadata: {
    tools_count: 2,
    files_merged: 5
  }
}
```

**Display**: "Context merged from 2 tools (5 files)"

### 4. File Shared

```typescript
{
  id: "notif-126",
  type: "file_shared",
  tool_type: "vscode",
  tool_id: "tool-456",
  message: "VSCode shared auth.ts",
  timestamp: "2025-11-15T12:15:00Z",
  metadata: {
    file_name: "auth.ts"
  }
}
```

**Display**: "VSCode shared auth.ts"

## Supported Tools

The component includes icons and display names for:

| Tool Type | Display Name | Icon |
|-----------|--------------|------|
| `cursor` | Cursor | Terminal (blue) |
| `vscode` | VSCode | Code2 (dark blue) |
| `claude-code` | Claude Code | Sparkles (purple) |
| `continue` | Continue | Terminal (green) |
| `n8n-agent` | n8n Agent | Workflow (orange) |
| `custom-agent` | Custom Agent | Plug (gray) |
| `windsurf` | Windsurf | Code2 (cyan) |
| `cline` | Cline | Box (indigo) |

Additional tools will use a default plug icon.

## API Endpoints

### Polling Mode (Default)

**Endpoint**: `GET /api/v1/tools/notifications`

**Query Parameters**:
- `since` (optional): ISO timestamp to get notifications after this time
- `session_id` (optional): Filter notifications for specific session

**Response**:
```json
{
  "notifications": [
    {
      "id": "notif-123",
      "type": "tool_joined",
      "tool_type": "vscode",
      "tool_id": "tool-456",
      "message": "VSCode just joined your session",
      "timestamp": "2025-11-15T12:00:00Z",
      "metadata": {}
    }
  ]
}
```

**Polling Interval**: 2 seconds

### WebSocket Mode

**Endpoint**: `ws://localhost:8009/ws/notifications`

**Subscribe Message**:
```json
{
  "type": "subscribe",
  "session_id": "session-123"
}
```

**Notification Message**:
```json
{
  "id": "notif-123",
  "type": "tool_joined",
  "tool_type": "vscode",
  "tool_id": "tool-456",
  "message": "VSCode just joined your session",
  "timestamp": "2025-11-15T12:00:00Z",
  "metadata": {}
}
```

## Animations

The component uses custom Tailwind CSS animations defined in `tailwind.config.js`:

- **Slide-in**: Notifications slide in from the right with fade effect (300ms)
- **Fade-out**: Smooth fade-out when dismissed (300ms)
- **Bounce-in**: Optional bounce effect on appearance

## Keyboard Accessibility

- **Escape**: Dismiss all visible notifications
- **Focus Management**: Close button receives focus for keyboard navigation
- **ARIA**: Proper `role="alert"` and `aria-live="polite"` for screen readers

## Styling

The component uses:
- Tailwind CSS utility classes
- CSS custom properties for theming
- Dark mode support via `dark:` variants
- Responsive design (mobile-friendly)

## TypeScript Support

Full TypeScript definitions included:

```typescript
interface ToolNotification {
  id: string;
  type: 'tool_joined' | 'tool_left' | 'context_merged' | 'file_shared';
  tool_type: string;
  tool_id: string;
  message: string;
  timestamp: string;
  metadata?: {
    tools_count?: number;
    files_merged?: number;
    file_name?: string;
  };
}

interface ToolNotificationsProps {
  sessionId?: string;
  maxNotifications?: number;
  autoDismissDelay?: number;
  enableSound?: boolean;
  useWebSocket?: boolean;
}
```

## Exported Utilities

The component exports helper functions for external use:

```typescript
import {
  getToolDisplayName,
  getToolIcon,
  generateNotificationMessage,
  formatRelativeTime
} from './components/ToolNotifications';

// Get display name
const name = getToolDisplayName('vscode'); // "VSCode"

// Get icon component
const icon = getToolIcon('cursor'); // <Terminal className="..." />

// Generate message
const message = generateNotificationMessage('tool_joined', 'vscode');
// "VSCode just joined your session"

// Format timestamp
const timeAgo = formatRelativeTime('2025-11-15T12:00:00Z');
// "2 minutes ago"
```

## Performance

- Efficient re-renders with `useCallback` hooks
- Automatic cleanup of dismissed notifications
- Debounced WebSocket reconnection
- Lazy audio loading (sound only loads when enabled)
- Maximum notification limit prevents memory bloat

## Browser Support

- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

Requires support for:
- CSS animations
- WebSocket API (for WebSocket mode)
- Fetch API (for polling mode)

## Known Limitations

1. Sound notifications require `/notification.mp3` in public folder (user must provide)
2. WebSocket reconnection not implemented (will reconnect on component remount)
3. Notification persistence not implemented (disappear on page reload)
4. No notification history or "view all" feature

## Future Enhancements

Potential improvements for future versions:

- [ ] Notification history panel
- [ ] Persistent notifications (survive page reload)
- [ ] WebSocket auto-reconnect with exponential backoff
- [ ] Desktop notifications API integration
- [ ] Notification preferences (mute specific tools, types)
- [ ] Custom notification templates
- [ ] Action buttons on notifications (e.g., "View file")
- [ ] Notification grouping (collapse multiple similar notifications)
- [ ] Analytics tracking (notification interaction metrics)

## Troubleshooting

### Notifications not appearing

1. Check API endpoint is running: `curl http://localhost:8009/api/v1/tools/notifications`
2. Check WebSocket connection (if using): Browser dev tools > Network > WS
3. Check console for errors
4. Verify notification data matches interface

### Animations not working

1. Verify Tailwind animations are compiled: Check `tailwind.config.js`
2. Rebuild: `npm run build`
3. Clear browser cache

### Dark mode issues

1. Ensure dark mode is enabled in parent component
2. Check Tailwind dark mode config: `darkMode: 'class'`
3. Verify `<html class="dark">` is set

## Contributing

To modify the component:

1. Edit `/src/components/ToolNotifications.tsx`
2. Test changes: `npm run dev`
3. Run type check: `npm run build`
4. Update this README if adding features

## License

Part of the OmniMemory Multi-Tool Context Bridge project.

## Related Components

- `SessionManager` - Manages active sessions
- `SessionTimeline` - Shows session history
- `SessionDetails` - Session detail view
- `ToolsList` - List of connected tools

## Changelog

### v1.0.0 (2025-11-15)

- Initial release
- Support for 4 notification types
- Polling and WebSocket modes
- Dark mode support
- Full accessibility features
- TypeScript definitions
