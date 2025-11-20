# OmniMemory Multi-Tool Dashboard

A modern, real-time dashboard for monitoring OmniMemory across multiple AI coding tools (Claude Code, Cursor, Codex, VSCode).

## Status: Phase 3 Complete ✅

**Current Version**: Phase 3 - Full Claude Code Dashboard with Real-Time Metrics
**Frontend**: http://localhost:8004
**Backend**: http://localhost:8003

### Completed Features

#### Phase 1: Foundation ✅
- React 18.3 + TypeScript + Vite
- React Router for navigation
- Zustand for state management
- Tailwind CSS for styling
- Radix UI components
- Dark mode by default

#### Phase 2: Backend Integration ✅
- FastAPI backend with SQLite
- SSE streaming for real-time metrics
- Session management API
- Configuration API
- 100% test coverage

#### Phase 3: Claude Code Dashboard ✅
- **Real-Time Metrics**: 4 metric cards updating every second via SSE
- **Historical Analytics**: 4 charts showing 24-hour trends
- **Configuration Panel**: Live settings with save functionality
- **Session Management**: Auto-start, tracking, and restart
- **Professional UI**: Responsive, accessible, dark-themed

## Features

### Real-Time Metrics (Updates every second)
- **Tokens Saved**: Total tokens saved via compression with ratio
- **Embeddings Generated**: Total embeddings with cache hit rate
- **Patterns Learned**: Workflow patterns with prediction accuracy
- **Active Session**: Current session status and duration

### Historical Analytics (24-hour charts)
- **Tokens Saved Over Time**: Line chart showing savings trend
- **Cache Hit Rate**: Area chart of embedding cache performance
- **Compression Ratio**: Bar chart of compression effectiveness
- **Service Usage**: Combined chart of embeddings vs compressions

### Configuration Management
- Enable/disable compression
- Enable/disable embeddings
- Enable/disable workflows
- Configure max token limit (1,000 - 1,000,000)

### Session Management
- Auto-start sessions on page load
- Real-time session metrics tracking
- Manual session restart capability
- Session persistence across navigation

## Getting Started

### Prerequisites

- Node.js 18+ or pnpm 8+
- Existing OmniMemory services running (ports 8000-8003)

### Installation

```bash
# Install dependencies
pnpm install

# Start development server (port 8004)
pnpm dev

# Build for production
pnpm build

# Preview production build
pnpm preview
```

### Development

The dashboard runs on `http://localhost:8004` by default.

## Project Structure

```
src/
├── components/
│   ├── layout/
│   │   ├── Layout.tsx       # Main layout wrapper
│   │   ├── Sidebar.tsx      # Left navigation
│   │   └── Header.tsx       # Top header with theme toggle
│   └── shared/
│       ├── MetricCard.tsx   # Reusable metric display
│       └── LoadingSpinner.tsx
├── pages/
│   ├── ClaudeCodePage.tsx   # Claude Code dashboard
│   ├── GlobalPage.tsx       # Global overview (placeholder)
│   └── SettingsPage.tsx     # Settings (placeholder)
├── hooks/
│   └── useSSE.ts           # Server-Sent Events hook
├── stores/
│   └── configStore.ts      # Zustand global state
├── types/
│   └── metrics.ts          # TypeScript types
├── lib/
│   └── utils.ts            # Utility functions
├── App.tsx                 # Router setup
└── main.tsx                # Entry point
```

## Roadmap

### Phase 1: Foundation ✅ COMPLETE
- ✅ Project setup with Vite + React + TypeScript
- ✅ Tailwind CSS with dark mode
- ✅ React Router navigation
- ✅ Basic layout and components
- ✅ Placeholder pages

### Phase 2: Backend Integration ✅ COMPLETE
- ✅ FastAPI backend with SQLite
- ✅ SSE streaming endpoint
- ✅ Session management API
- ✅ Configuration API
- ✅ Tool-specific data models
- ✅ 100% test coverage

### Phase 3: Real-time Metrics ✅ COMPLETE
- ✅ SSE connection with auto-reconnect
- ✅ Live metrics display (4 cards)
- ✅ Historical data fetching
- ✅ Comprehensive error handling
- ✅ Loading states

### Phase 4: Visualizations ✅ COMPLETE (Current)
- ✅ Recharts integration
- ✅ 4 historical charts (Line, Area, Bar, Composed)
- ✅ Real-time metric updates
- ✅ Dark theme chart styling
- ✅ Configuration panel
- ✅ Session statistics

### Phase 5: Multi-Tool Support (Next)
- [ ] Cursor integration
- [ ] Codex integration
- [ ] VSCode integration
- [ ] Cross-tool analytics
- [ ] Tool comparison view
- [ ] Combined metrics dashboard

## Tech Stack

- **Frontend**: React 18.3, TypeScript 5.6
- **Routing**: React Router 6
- **State**: Zustand 4.5
- **Styling**: Tailwind CSS 3.4
- **UI Components**: Radix UI
- **Charts**: Recharts 2.12 (Phase 4)
- **Build**: Vite 6.0
- **Package Manager**: pnpm

## API Endpoints (Backend: Port 8003)

### Session Management
- `GET /sessions/active` - Get all active sessions
- `POST /sessions/start` - Start new session
- `GET /sessions/{id}` - Get session details
- `POST /sessions/{id}/end` - End session

### Metrics
- `GET /metrics/current` - Current metrics snapshot
- `GET /metrics/tool/{tool_id}` - Tool-specific metrics
- `GET /metrics/tool/{tool_id}/history?hours=24` - Historical data
- `GET /metrics/compare?tool_ids=...` - Compare multiple tools
- `GET /stream/metrics?tool_id={tool_id}` - SSE stream (real-time)

### Configuration
- `GET /config/tool/{tool_id}` - Get tool configuration
- `PUT /config/tool/{tool_id}` - Update tool configuration

## Testing

Run the integration test suite:

```bash
./test-dashboard-integration.sh
```

**Results**: 12/12 tests passing
- Backend API endpoints (6 tests)
- Frontend server (2 tests)
- Data structure validation (2 tests)
- Integration flows (2 tests)

## Contributing

This is an internal OmniMemory project. Follow the existing code style and TypeScript strict mode.

## License

MIT - Part of the OmniMemory ecosystem
