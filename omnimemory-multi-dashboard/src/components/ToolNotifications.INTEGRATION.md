# ToolNotifications Integration Guide

Quick guide to integrate ToolNotifications into the Multi-Tool Context Bridge Dashboard.

## Step 1: Add to Main Layout

Edit `/src/App.tsx` or your main layout component:

```tsx
import { ToolNotifications } from './components/ToolNotifications';

function App() {
  return (
    <div className="app">
      {/* Your existing app content */}
      <Router>
        <Layout>
          <Routes>
            {/* Your routes */}
          </Routes>
        </Layout>
      </Router>

      {/* Add notifications at the end - it will position itself */}
      <ToolNotifications />
    </div>
  );
}
```

## Step 2: Session-Specific Notifications

If you want notifications for a specific session:

```tsx
import { ToolNotifications } from './components/ToolNotifications';
import { useSessionStore } from './stores/sessionStore'; // Your session store

function SessionPage() {
  const activeSessionId = useSessionStore((state) => state.activeSessionId);

  return (
    <div>
      {/* Your session content */}

      {/* Session-specific notifications */}
      <ToolNotifications sessionId={activeSessionId} />
    </div>
  );
}
```

## Step 3: Backend API Setup

The component expects notifications from the backend. Add these endpoints to your multi-tool service:

### Polling Endpoint (Required for default mode)

```python
# In your FastAPI/Flask app (port 8009)

@app.get("/api/v1/tools/notifications")
async def get_tool_notifications(
    since: Optional[str] = None,
    session_id: Optional[str] = None
):
    """
    Get tool notifications since a given timestamp.

    Args:
        since: ISO timestamp (optional) - only return notifications after this time
        session_id: Session ID (optional) - filter notifications for this session

    Returns:
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
    """
    # Your implementation here
    notifications = await get_notifications_from_db(since, session_id)
    return {"notifications": notifications}
```

### WebSocket Endpoint (Optional, for real-time)

```python
from fastapi import WebSocket

@app.websocket("/ws/notifications")
async def websocket_notifications(websocket: WebSocket):
    await websocket.accept()

    try:
        # Receive subscription message
        data = await websocket.receive_json()
        session_id = data.get("session_id")

        # Listen for tool events and send notifications
        while True:
            notification = await wait_for_tool_event(session_id)
            await websocket.send_json(notification)

    except WebSocketDisconnect:
        print("Client disconnected")
```

## Step 4: Generate Notifications

When tools join/leave or events occur, create notifications:

```python
# Example: Tool joined event
async def on_tool_joined(session_id: str, tool_type: str, tool_id: str):
    notification = {
        "id": generate_notification_id(),
        "type": "tool_joined",
        "tool_type": tool_type,
        "tool_id": tool_id,
        "message": f"{get_tool_display_name(tool_type)} just joined your session",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "metadata": {}
    }

    # Save to database or message queue
    await save_notification(notification)

    # If using WebSocket, broadcast to connected clients
    await broadcast_notification(session_id, notification)
```

## Step 5: Test Notifications

Create test notifications to verify integration:

```tsx
// Add test button to your dev tools
<Button onClick={async () => {
  await fetch('http://localhost:8009/api/v1/tools/test-notification', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      type: 'tool_joined',
      tool_type: 'vscode',
      session_id: 'test-session'
    })
  });
}}>
  Test Notification
</Button>
```

## Database Schema

Suggested database schema for notifications:

```sql
CREATE TABLE tool_notifications (
    id VARCHAR(255) PRIMARY KEY,
    type VARCHAR(50) NOT NULL,  -- tool_joined, tool_left, context_merged, file_shared
    tool_type VARCHAR(100) NOT NULL,
    tool_id VARCHAR(255) NOT NULL,
    session_id VARCHAR(255),
    message TEXT NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_timestamp (timestamp),
    INDEX idx_session_id (session_id),
    INDEX idx_type (type)
);
```

## Notification Events to Track

1. **Tool Joined**
   - When: Tool connects to session
   - Type: `tool_joined`
   - Metadata: None

2. **Tool Left**
   - When: Tool disconnects from session
   - Type: `tool_left`
   - Metadata: None

3. **Context Merged**
   - When: Context from multiple tools is merged
   - Type: `context_merged`
   - Metadata: `{ tools_count: number, files_merged: number }`

4. **File Shared**
   - When: Tool shares a file with session
   - Type: `file_shared`
   - Metadata: `{ file_name: string }`

## Environment Variables

Add to your `.env`:

```bash
# Notification settings
NOTIFICATION_POLLING_INTERVAL=2000  # ms
NOTIFICATION_AUTO_DISMISS_DELAY=5000  # ms
NOTIFICATION_MAX_STACK=5
NOTIFICATION_ENABLE_SOUND=false
NOTIFICATION_USE_WEBSOCKET=false
```

## Monitoring

Monitor notification delivery:

```python
# Track notification metrics
notification_metrics = {
    "total_sent": 0,
    "total_delivered": 0,
    "avg_delivery_time_ms": 0,
    "active_subscribers": 0
}

# Log notification events
logger.info(f"Notification sent: {notification['id']} to session {session_id}")
```

## Production Checklist

- [ ] API endpoints implemented and tested
- [ ] Database schema created
- [ ] Notification generation integrated with tool events
- [ ] Error handling for failed deliveries
- [ ] Rate limiting on notification API
- [ ] WebSocket connection management
- [ ] Notification retention policy (auto-delete old notifications)
- [ ] Metrics and monitoring
- [ ] Load testing for high notification volume

## Troubleshooting

### No notifications appearing

1. Check API is running: `curl http://localhost:8009/api/v1/tools/notifications`
2. Check browser console for errors
3. Verify CORS settings allow localhost:8004
4. Check network tab for failed requests

### Duplicate notifications

1. Ensure notification IDs are unique
2. Check component isn't mounted multiple times
3. Verify polling interval isn't too aggressive

### Performance issues

1. Reduce polling interval (increase from 2s to 5s)
2. Implement notification batching
3. Add database indexes
4. Consider WebSocket mode

## Next Steps

After integration:

1. Test all notification types
2. Verify dark mode styling
3. Test keyboard accessibility
4. Add analytics tracking
5. Monitor performance metrics
6. Gather user feedback

## Support

For issues or questions:
- Check the main README: `ToolNotifications.README.md`
- Review example: `ToolNotificationsExample.tsx`
- Check PHASE_2_MULTI_TOOL_CONTEXT_BRIDGE_SPEC.md
