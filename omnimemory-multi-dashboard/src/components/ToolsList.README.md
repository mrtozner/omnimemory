# ToolsList Component

**Phase 2 Multi-Tool Context Bridge Dashboard - Week 4**

## Overview

The `ToolsList` component displays connected IDE tools and autonomous agents for a project, showing their status, capabilities, and last activity.

## Features

- **Separated Sections**: IDE tools and autonomous agents are displayed in separate sections
- **Real-time Status**: Activity status indicators (active/idle/inactive) based on last activity timestamp
- **Capabilities Display**: Badges showing tool capabilities (MCP, LSP, REST API, Execute Code, Edit Files)
- **Context Window Info**: Shows maximum context token size for each tool
- **Auto-refresh**: Polls for updates every 30 seconds
- **Responsive Grid**: 1/2/3 columns based on screen size
- **Dark Mode Support**: Full support for dark/light themes
- **Error Handling**: Graceful error states with retry functionality
- **Loading States**: Proper loading spinners while fetching data
- **Empty States**: User-friendly message when no tools are connected

## Props

```typescript
interface ToolsListProps {
  projectId: string;  // Required: The project ID to fetch tools for
}
```

## Usage

```tsx
import { ToolsList } from './components/ToolsList';

function MyPage() {
  return <ToolsList projectId="my-project-123" />;
}
```

## API Endpoint

Fetches from: `http://localhost:8009/api/v1/projects/{projectId}/all-tools`

**Expected Response:**
```json
{
  "ide_tools": [
    {
      "tool_id": "cursor-123",
      "tool_type": "cursor",
      "last_activity": "2025-11-15T10:30:00Z",
      "capabilities": {
        "supports_mcp": true,
        "max_context_tokens": 20000,
        "can_execute_code": true,
        "can_edit_files": true
      }
    }
  ],
  "agents": [
    {
      "tool_id": "n8n-agent-456",
      "tool_type": "n8n-agent",
      "last_activity": "2025-11-15T10:25:00Z",
      "capabilities": {
        "supports_rest": true,
        "max_context_tokens": 10000
      }
    }
  ]
}
```

## Supported Tool Types

### IDE Tools
- **Cursor** - Cursor AI Code Editor
- **VSCode** - Visual Studio Code
- **Claude Code** - Anthropic Claude Code Assistant
- **Continue** - Continue Dev VSCode Extension

### Autonomous Agents
- **n8n Agent** - n8n workflow automation agent
- **LangChain Agent** - LangChain-based agent
- **AutoGen Agent** - Microsoft AutoGen agent
- **Custom Agent** - Generic custom agent

## Activity Status

- **Active** (green): Last activity < 5 minutes ago
- **Idle** (yellow): Last activity 5-30 minutes ago
- **Inactive** (gray): Last activity > 30 minutes ago

## Capabilities Badges

- **MCP**: Supports Model Context Protocol
- **LSP**: Supports Language Server Protocol
- **REST API**: Supports REST API integration
- **Execute**: Can execute code
- **Edit**: Can edit files

## Styling

Uses Tailwind CSS with the project's design system:
- Card components from `shared/Card`
- Design tokens from `index.css` (--foreground, --background, etc.)
- Dark mode support via `dark:` classes
- Responsive grid layout

## Dependencies

- React 18+ (hooks: useState, useEffect, useCallback)
- lucide-react (icons)
- Tailwind CSS
- Shared components: Card, LoadingSpinner, ErrorState
- Utility: cn (className merging)

## Accessibility

- Semantic HTML structure
- ARIA labels for status indicators
- Keyboard navigation support
- Screen reader friendly text
- Sufficient color contrast in light/dark modes

## Performance

- Auto-refresh every 30 seconds (configurable)
- useCallback for optimized re-renders
- Cleanup on unmount (clears intervals)
- Conditional rendering for loading/error/empty states

## Error Handling

- Network errors: Shows ErrorState with retry button
- Loading states: Shows spinner while fetching
- Empty states: Shows friendly message when no tools
- Console error logging for debugging

## Future Enhancements

- [ ] Filter tools by type
- [ ] Sort tools by activity/name/type
- [ ] Click to view tool details
- [ ] Manual refresh button
- [ ] Configurable poll interval
- [ ] WebSocket support for real-time updates
- [ ] Tool connection/disconnection animations
