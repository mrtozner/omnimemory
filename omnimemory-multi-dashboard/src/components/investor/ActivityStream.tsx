import { useEffect, useState, useRef } from 'react';
import { Card, CardHeader, CardTitle, CardContent, CardDescription } from '../shared/Card';
import { Activity, Zap, Archive, Search, Database, GitBranch } from 'lucide-react';
import type { Session, Metrics } from '../../services/api';
import { useSSE } from '../../hooks/useSSE';

// Activity event interface
interface ActivityEvent {
  timestamp: Date;
  tool: string;
  action: string;
  customer: string;
  description: string;
  sessionId?: string;
}

interface ActivityStreamProps {
  tools: string[];
  maxItems?: number;
  sessions?: Session[];
}

const ACTION_ICONS: Record<string, React.ReactNode> = {
  embed: <Zap className="h-4 w-4" />,
  compress: <Archive className="h-4 w-4" />,
  search: <Search className="h-4 w-4" />,
  store: <Database className="h-4 w-4" />,
  retrieve: <Database className="h-4 w-4" />,
  workflow: <GitBranch className="h-4 w-4" />,
  session_start: <Activity className="h-4 w-4" />,
  active: <Activity className="h-4 w-4" />,
};

const ACTION_COLORS: Record<string, string> = {
  embed: 'text-blue-400 bg-blue-500/10',
  compress: 'text-purple-400 bg-purple-500/10',
  search: 'text-green-400 bg-green-500/10',
  store: 'text-orange-400 bg-orange-500/10',
  retrieve: 'text-yellow-400 bg-yellow-500/10',
  workflow: 'text-pink-400 bg-pink-500/10',
  session_start: 'text-gray-400 bg-gray-500/10',
  active: 'text-green-400 bg-green-500/10',
};

const TOOL_COLORS: Record<string, string> = {
  'claude-code': 'bg-blue-500',
  'codex': 'bg-green-500',
  'gpt-4': 'bg-orange-500',
  'gemini': 'bg-purple-500',
};

/**
 * Generate activity events from real sessions
 */
function generateActivityFromSessions(sessions: Session[]): ActivityEvent[] {
  const events: ActivityEvent[] = [];
  const now = new Date();

  sessions.forEach((session) => {
    const startedAt = new Date(session.started_at);
    const durationMs = now.getTime() - startedAt.getTime();
    const durationMinutes = Math.floor(durationMs / 60000);

    // Add session start event
    events.push({
      timestamp: startedAt,
      tool: session.tool_id,
      action: 'session_start',
      customer: session.tool_id, // Use tool_id as customer since we don't have real customer data
      description: `Session started`,
      sessionId: session.session_id,
    });

    // If session has embeddings, add embedding activity
    if (session.total_embeddings > 0) {
      events.push({
        timestamp: new Date(startedAt.getTime() + 30000), // 30s after start
        tool: session.tool_id,
        action: 'embed',
        customer: session.tool_id,
        description: `Generated ${session.total_embeddings} embeddings`,
        sessionId: session.session_id,
      });
    }

    // If session has compressions, add compression activity
    if (session.total_compressions > 0) {
      events.push({
        timestamp: new Date(startedAt.getTime() + 60000), // 1 min after start
        tool: session.tool_id,
        action: 'compress',
        customer: session.tool_id,
        description: `Compressed ${session.tokens_saved} tokens saved`,
        sessionId: session.session_id,
      });
    }

    // If session is active, add current activity indicator
    if (!session.ended_at) {
      events.push({
        timestamp: new Date(now.getTime() - 10000), // 10 seconds ago
        tool: session.tool_id,
        action: 'active',
        customer: session.tool_id,
        description: `Active for ${durationMinutes} minutes`,
        sessionId: session.session_id,
      });
    }
  });

  // Sort by timestamp (most recent first) and limit
  return events.sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime());
}

/**
 * Generate activity events from SSE metrics changes
 * DISABLED: This was generating misleading activity from cumulative global counters
 * that included embeddings/compressions from other sessions/time periods.
 *
 * Now we only show activity from actual session data via generateActivityFromSessions()
 */
function generateEventsFromMetricsChange(
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  _prevMetrics: Metrics | null,
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  _newMetrics: Metrics,
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  _toolId: string
): ActivityEvent[] {
  // DISABLED: Don't show cumulative delta events - they're misleading
  // The total_embeddings counter never resets, so deltas include activity
  // from other sessions, not just the current active session
  return [];
}

export function ActivityStream({ tools, maxItems = 20, sessions = [] }: ActivityStreamProps) {
  const [activities, setActivities] = useState<ActivityEvent[]>([]);
  const [latestEventTime, setLatestEventTime] = useState<number>(0);
  const prevMetricsRef = useRef<Metrics | null>(null);

  // Connect to SSE metrics stream
  const toolId = tools.length > 0 ? tools[0] : 'claude-code';
  const { data: metricsData, isConnected } = useSSE(toolId, { sessionId: undefined });

  // Handle SSE metrics updates
  useEffect(() => {
    if (!metricsData) {
      return;
    }

    // Generate events from metrics changes
    const newEvents = generateEventsFromMetricsChange(
      prevMetricsRef.current,
      metricsData,
      toolId
    );

    // Update previous metrics reference
    prevMetricsRef.current = metricsData;

    // Add new events to the activity stream
    if (newEvents.length > 0) {
      // Track the latest event time for animation
      const latestTime = Math.max(...newEvents.map(e => e.timestamp.getTime()));
      setLatestEventTime(latestTime);

      setActivities((prev) => {
        const updated = [...newEvents, ...prev];
        return updated.slice(0, maxItems);
      });
    }
  }, [metricsData, toolId, maxItems]);

  // Initialize activities from sessions on mount
  useEffect(() => {
    if (sessions.length > 0 && activities.length === 0) {
      const initialActivities = generateActivityFromSessions(sessions);
      setActivities(initialActivities.slice(0, maxItems));
    }
  }, [sessions, maxItems, activities.length]);

  const formatTimeAgo = (timestamp: Date) => {
    const seconds = Math.floor((Date.now() - timestamp.getTime()) / 1000);
    if (seconds < 60) return `${seconds}s ago`;
    const minutes = Math.floor(seconds / 60);
    if (minutes < 60) return `${minutes}m ago`;
    const hours = Math.floor(minutes / 60);
    return `${hours}h ago`;
  };

  const hasActivity = activities.length > 0;

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="flex items-center gap-2">
              <Activity className="h-5 w-5 text-green-500" />
              Real-Time Activity
            </CardTitle>
            <CardDescription>
              {isConnected
                ? `Connected to live stream (${toolId})`
                : `Connecting to live stream...`}
            </CardDescription>
          </div>
          <div className="flex items-center gap-2 text-sm">
            {isConnected && (
              <div className="flex items-center gap-2 text-green-500">
                <span className="relative flex h-3 w-3">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75"></span>
                  <span className="relative inline-flex rounded-full h-3 w-3 bg-green-500"></span>
                </span>
                Live
              </div>
            )}
            {!isConnected && (
              <div className="flex items-center gap-2 text-yellow-500">
                <span className="relative flex h-3 w-3">
                  <span className="animate-pulse absolute inline-flex h-full w-full rounded-full bg-yellow-400 opacity-75"></span>
                  <span className="relative inline-flex rounded-full h-3 w-3 bg-yellow-500"></span>
                </span>
                Connecting
              </div>
            )}
          </div>
        </div>
      </CardHeader>
      <CardContent>
        {!hasActivity ? (
          <div className="text-center py-12">
            <Activity className="h-12 w-12 text-gray-600 mx-auto mb-4" />
            {isConnected ? (
              <>
                <p className="text-gray-400 mb-2">Waiting for activity</p>
                <p className="text-sm text-gray-500">
                  Connected - operations will appear here as they happen
                </p>
              </>
            ) : (
              <>
                <p className="text-gray-400 mb-2">Connecting to live stream</p>
                <p className="text-sm text-gray-500">
                  Please wait while we establish connection...
                </p>
              </>
            )}
          </div>
        ) : (
          <div className="space-y-2 max-h-[500px] overflow-y-auto pr-2">
            {activities.map((activity, index) => (
              <ActivityItem
                key={`${activity.timestamp.getTime()}-${index}`}
                activity={activity}
                formatTimeAgo={formatTimeAgo}
                isNew={activity.timestamp.getTime() === latestEventTime}
              />
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
}

interface ActivityItemProps {
  activity: ActivityEvent;
  formatTimeAgo: (timestamp: Date) => string;
  isNew: boolean;
}

function ActivityItem({ activity, formatTimeAgo, isNew }: ActivityItemProps) {
  const actionColor = ACTION_COLORS[activity.action] || 'text-gray-400 bg-gray-500/10';
  const toolColor = TOOL_COLORS[activity.tool] || 'bg-gray-500';

  return (
    <div
      className={`flex items-start gap-3 p-3 rounded-lg border border-gray-700 hover:border-gray-600 transition-all ${
        isNew ? 'animate-slideIn bg-gray-800/80' : 'bg-gray-800/30'
      }`}
    >
      {/* Action Icon */}
      <div className={`p-2 rounded-lg ${actionColor}`}>
        {ACTION_ICONS[activity.action] || <Activity className="h-4 w-4" />}
      </div>

      {/* Activity Details */}
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2 mb-1">
          {/* Tool Badge */}
          <span
            className={`px-2 py-0.5 rounded text-xs font-medium text-white ${toolColor}`}
          >
            {activity.tool}
          </span>

          {/* Customer Badge */}
          <span className="px-2 py-0.5 rounded text-xs font-medium bg-gray-700 text-gray-300">
            {activity.customer}
          </span>

          {/* Session ID Badge */}
          {activity.sessionId && (
            <span className="px-2 py-0.5 rounded text-xs font-mono bg-gray-800 text-gray-400 border border-gray-700">
              {activity.sessionId.substring(0, 8)}...
            </span>
          )}

          {/* Timestamp */}
          <span className="text-xs text-muted-foreground ml-auto">
            {formatTimeAgo(activity.timestamp)}
          </span>
        </div>

        {/* Description */}
        <p className="text-sm text-gray-300">{activity.description}</p>
      </div>
    </div>
  );
}
