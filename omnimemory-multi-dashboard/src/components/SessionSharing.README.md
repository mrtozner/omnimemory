# SessionSharing Component

**Phase 2 Multi-Tool Context Bridge Dashboard UI - Week 4-5**

## Overview

The `SessionSharing` component displays which tools are sharing the same session and shows a timeline of collaboration events between tools. This component is part of the Phase 2 Multi-Tool Context Bridge feature.

## Features

### 1. Active Tools Display
- Shows all currently active tools in the session
- Pulsing indicator for active status
- Tool icons and names using the existing tool mapping system
- Shows when each tool joined the session
- Lists previously active tools that have left

### 2. Collaboration Timeline
- Vertical timeline with connecting line
- Event types supported:
  - `tool_joined`: When a tool joins the session
  - `tool_left`: When a tool leaves the session
  - `context_merged`: When context is merged from multiple tools
  - `file_accessed`: When a tool accesses a file
  - `search_executed`: When a tool performs a search
  - `decision_made`: When a tool makes a decision
  - `memory_saved`: When a tool saves information to memory

### 3. Real-time Updates
- Polls for updates every 5 seconds
- Automatically refreshes session info and events
- Shows loading state on initial load
- Error handling with user-friendly messages

### 4. Styling
- Dark mode support (consistent with dashboard theme)
- Color-coded events (green for joins, red for leaves, blue for merges, etc.)
- Responsive layout
- Hover effects and transitions
- Tailwind CSS classes

## API Endpoints

### Session Info
```
GET http://localhost:8009/api/v1/sessions/{session_id}
```
**Response:**
```json
{
  "session_id": "sess_abc123",
  "project_id": "proj_xyz",
  "tools": [
    {
      "tool_id": "cursor-123",
      "tool_type": "cursor",
      "joined_at": "2025-11-15T10:00:00Z"
    }
  ],
  "created_at": "2025-11-15T10:00:00Z",
  "updated_at": "2025-11-15T10:15:00Z"
}
```

### Collaboration Events (Fallback)
```
GET http://localhost:8003/sessions/{session_id}/events
```
**Response:**
```json
{
  "events": [
    {
      "event_type": "tool_joined",
      "tool_id": "cursor-123",
      "tool_type": "cursor",
      "timestamp": "2025-11-15T10:00:00Z"
    }
  ]
}
```

## Usage

### Basic Usage
```tsx
import { SessionSharing } from './components/SessionSharing';

function MyPage() {
  return <SessionSharing sessionId="sess_abc123" />;
}
```

### With Session Selection
```tsx
import { SessionSharing } from './components/SessionSharing';
import { SessionTimeline } from './components/SessionTimeline';

function SessionDetailsPage() {
  const [selectedSessionId, setSelectedSessionId] = useState<string | null>(null);

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 p-6">
      <SessionTimeline
        onSessionClick={(sessionId) => setSelectedSessionId(sessionId)}
      />

      {selectedSessionId && (
        <SessionSharing sessionId={selectedSessionId} />
      )}
    </div>
  );
}
```

## Component Props

```typescript
interface SessionSharingProps {
  sessionId: string;  // Required: The session ID to display
}
```

## Component Structure

### Main Component: `SessionSharing`
- Fetches session data and collaboration events
- Manages loading and error states
- Renders two main sections: Active Tools and Timeline

### Sub-components

#### `ActiveToolBadge`
- Displays an active tool with pulsing indicator
- Shows tool icon, name, and join time
- Color-coded by tool type

#### `InactiveToolBadge`
- Displays a previously active tool (grayed out)
- Shows join and leave times

#### `TimelineEvent`
- Displays a single collaboration event
- Event icon and color-coded by type
- Shows timestamp, description, and metadata

#### `EventMetadata`
- Displays additional event details
- File paths, merge counts, search queries, etc.

## Event Types and Colors

| Event Type | Icon | Color | Description |
|-----------|------|-------|-------------|
| `tool_joined` | `UserPlus` | Green | Tool joined the session |
| `tool_left` | `UserMinus` | Red | Tool left the session |
| `context_merged` | `RefreshCw` | Blue | Context merged from multiple tools |
| `file_accessed` | `FileText` | Purple | Tool accessed a file |
| `search_executed` | `Search` | Yellow | Tool performed a search |
| `decision_made` | `CheckCircle` | Green | Tool made a decision |
| `memory_saved` | `Save` | Indigo | Tool saved information |

## Styling Classes

The component uses Tailwind CSS with dark mode support:
- **Cards**: `bg-gray-800`, `border-gray-700`
- **Text**: `text-gray-100` (primary), `text-gray-400` (secondary)
- **Active indicators**: `bg-green-500`, `animate-pulse`
- **Timeline line**: `bg-gray-700`

## Dependencies

- `react` - Core React library
- `lucide-react` - Icons
- `../shared/Card` - Card components
- `../shared/LoadingSpinner` - Loading indicator
- `../utils/toolMapping` - Tool display utilities
- `../lib/utils` - Utility functions (cn for classNames)

## Time Formatting

Relative time is displayed for all timestamps:
- `5s ago` - Less than 60 seconds
- `3m ago` - Less than 60 minutes
- `2h ago` - Less than 24 hours
- `5d ago` - Less than 7 days
- `Nov 15` - More than 7 days

## Error Handling

The component handles errors gracefully:
- Shows error message if session fetch fails
- Falls back to metrics service if context bridge is unavailable
- Displays user-friendly error messages
- Continues polling even after transient errors

## Loading States

- Initial load: Shows centered loading spinner
- Empty states: Shows helpful messages with icons
- No active tools: "No active tools in this session"
- No events: "No collaboration events yet"

## Performance

- Polls every 5 seconds (configurable)
- Cleanup on unmount (clears intervals)
- Efficient re-renders using React hooks
- Minimal API calls (only fetches when sessionId changes)

## Future Enhancements

Potential improvements for future iterations:
1. **WebSocket support**: Real-time updates instead of polling
2. **Event filtering**: Filter timeline by event type or tool
3. **Event search**: Search events by description or metadata
4. **Export timeline**: Download timeline as JSON or CSV
5. **Event details modal**: Click event for more details
6. **Tool avatars**: Custom avatars for each tool type
7. **Session analytics**: Statistics on collaboration patterns

## Testing

Example test scenarios:
1. Single tool session (1 tool active)
2. Multi-tool session (2+ tools sharing context)
3. Tool joins and leaves
4. Context merge events
5. File access tracking
6. Search history
7. Empty session (no tools)
8. API errors (graceful degradation)

## Accessibility

- Semantic HTML structure
- ARIA labels on interactive elements
- Keyboard navigation support
- Screen reader friendly descriptions
- Color contrast meets WCAG AA standards

## Browser Support

Tested on:
- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

## Contributing

When adding new event types:
1. Add event type to `CollaborationEvent` interface
2. Add icon to `EVENT_ICONS` mapping
3. Add color to `EVENT_COLORS` mapping
4. Add title to `getEventTitle` function
5. Add description logic to `getEventDescription` function
6. Update this README with the new event type

## License

Part of the OmniMemory project.
