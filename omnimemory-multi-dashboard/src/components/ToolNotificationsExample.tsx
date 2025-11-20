/**
 * ToolNotifications Usage Example
 *
 * This demonstrates how to integrate the ToolNotifications component
 * into your application for Phase 2 Multi-Tool Context Bridge.
 */

import { useState } from 'react';
import type { ToolNotification } from '../utils/toolNotifications';
import { ToolNotifications } from './ToolNotifications';
import { Button } from './shared/Button';

export function ToolNotificationsExample() {
  const [sessionId] = useState<string>('demo-session-123');

  // Simulate adding notifications for demo purposes
  const simulateNotification = (type: ToolNotification['type']) => {
    // In production, notifications come from the backend API
    // This is just for demonstration
    console.log(`Simulating ${type} notification`);
  };

  return (
    <div className="p-8 space-y-6">
      <div className="max-w-2xl">
        <h2 className="text-2xl font-bold mb-4">Tool Notifications Demo</h2>
        <p className="text-gray-600 dark:text-gray-400 mb-6">
          This component shows real-time notifications when tools join/leave sessions,
          context is merged, or files are shared.
        </p>

        <div className="space-y-4">
          <div>
            <h3 className="text-lg font-semibold mb-2">Features</h3>
            <ul className="list-disc list-inside space-y-1 text-sm text-gray-600 dark:text-gray-400">
              <li>Toast-style notifications in top-right corner</li>
              <li>Slide-in animation from right</li>
              <li>Auto-dismiss after 5 seconds (configurable)</li>
              <li>Manual dismiss with close button (X)</li>
              <li>Stacks up to 5 notifications (configurable)</li>
              <li>Tool-specific icons and messages</li>
              <li>Dark mode support</li>
              <li>Keyboard support (Escape to dismiss all)</li>
              <li>WebSocket or polling modes</li>
            </ul>
          </div>

          <div>
            <h3 className="text-lg font-semibold mb-2">Demo Controls</h3>
            <div className="flex flex-wrap gap-2">
              <Button onClick={() => simulateNotification('tool_joined')}>
                Simulate Tool Joined
              </Button>
              <Button onClick={() => simulateNotification('tool_left')}>
                Simulate Tool Left
              </Button>
              <Button onClick={() => simulateNotification('context_merged')}>
                Simulate Context Merged
              </Button>
              <Button onClick={() => simulateNotification('file_shared')}>
                Simulate File Shared
              </Button>
            </div>
          </div>

          <div>
            <h3 className="text-lg font-semibold mb-2">Integration</h3>
            <pre className="bg-gray-100 dark:bg-gray-800 p-4 rounded-lg text-xs overflow-x-auto">
{`// Basic usage with polling (default)
<ToolNotifications sessionId="your-session-id" />

// With WebSocket for real-time updates
<ToolNotifications
  sessionId="your-session-id"
  useWebSocket={true}
/>

// With custom settings
<ToolNotifications
  sessionId="your-session-id"
  maxNotifications={3}
  autoDismissDelay={10000}
  enableSound={true}
/>

// Global usage (all sessions)
<ToolNotifications />`}
            </pre>
          </div>

          <div>
            <h3 className="text-lg font-semibold mb-2">API Endpoint Requirements</h3>
            <div className="text-sm text-gray-600 dark:text-gray-400 space-y-2">
              <p>The component expects the following endpoints:</p>
              <div className="bg-gray-100 dark:bg-gray-800 p-3 rounded">
                <p className="font-mono text-xs">
                  <strong>Polling:</strong> GET /api/v1/tools/notifications?since=&lt;timestamp&gt;&session_id=&lt;id&gt;
                </p>
                <p className="font-mono text-xs mt-2">
                  <strong>WebSocket:</strong> ws://localhost:8009/ws/notifications
                </p>
              </div>
              <div className="bg-gray-100 dark:bg-gray-800 p-3 rounded mt-2">
                <p className="font-semibold mb-1">Response format:</p>
                <pre className="font-mono text-xs">
{`{
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
}`}
                </pre>
              </div>
            </div>
          </div>

          <div>
            <h3 className="text-lg font-semibold mb-2">Supported Tools</h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-sm">
              <div className="bg-gray-100 dark:bg-gray-800 p-2 rounded">VSCode</div>
              <div className="bg-gray-100 dark:bg-gray-800 p-2 rounded">Cursor</div>
              <div className="bg-gray-100 dark:bg-gray-800 p-2 rounded">Claude Code</div>
              <div className="bg-gray-100 dark:bg-gray-800 p-2 rounded">Continue</div>
              <div className="bg-gray-100 dark:bg-gray-800 p-2 rounded">Windsurf</div>
              <div className="bg-gray-100 dark:bg-gray-800 p-2 rounded">Cline</div>
              <div className="bg-gray-100 dark:bg-gray-800 p-2 rounded">n8n Agent</div>
              <div className="bg-gray-100 dark:bg-gray-800 p-2 rounded">Custom Agent</div>
            </div>
          </div>
        </div>
      </div>

      {/* Render the actual notification component */}
      <ToolNotifications
        sessionId={sessionId}
        maxNotifications={5}
        autoDismissDelay={5000}
        enableSound={false}
        useWebSocket={false}
      />
    </div>
  );
}

export default ToolNotificationsExample;
