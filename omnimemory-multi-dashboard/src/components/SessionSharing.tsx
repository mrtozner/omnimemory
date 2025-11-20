import { useState, useEffect } from 'react';
import {
  Users,
  UserPlus,
  UserMinus,
  RefreshCw,
  FileText,
  Search,
  CheckCircle,
  Save,
  AlertCircle,
  Clock,
} from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from './shared/Card';
import { LoadingSpinner } from './shared/LoadingSpinner';
import { getToolDisplay, getToolBgColor, getToolBorderColor } from '../utils/toolMapping';
import { cn } from '../lib/utils';

// Type Definitions
interface ToolInfo {
  tool_id: string;
  tool_type: string;
  joined_at: string;
  left_at?: string;
}

interface SessionInfo {
  session_id: string;
  project_id?: string;
  tools: ToolInfo[];
  created_at: string;
  updated_at: string;
}

interface CollaborationEvent {
  event_type: 'tool_joined' | 'tool_left' | 'context_merged' | 'file_accessed' | 'search_executed' | 'decision_made' | 'memory_saved';
  tool_id?: string;
  tool_type?: string;
  timestamp: string;
  metadata?: {
    file_path?: string;
    files_merged?: number;
    from_tools?: string[];
    query?: string;
    files_count?: number;
    searches_count?: number;
  };
}

interface SessionSharingProps {
  sessionId: string;
}

// Event Icons and Colors
const EVENT_ICONS: Record<string, React.ReactNode> = {
  tool_joined: <UserPlus className="h-4 w-4" />,
  tool_left: <UserMinus className="h-4 w-4" />,
  context_merged: <RefreshCw className="h-4 w-4" />,
  file_accessed: <FileText className="h-4 w-4" />,
  search_executed: <Search className="h-4 w-4" />,
  decision_made: <CheckCircle className="h-4 w-4" />,
  memory_saved: <Save className="h-4 w-4" />,
};

const EVENT_COLORS: Record<string, string> = {
  tool_joined: 'text-green-400 bg-green-500/10 border-green-500/30',
  tool_left: 'text-red-400 bg-red-500/10 border-red-500/30',
  context_merged: 'text-blue-400 bg-blue-500/10 border-blue-500/30',
  file_accessed: 'text-purple-400 bg-purple-500/10 border-purple-500/30',
  search_executed: 'text-yellow-400 bg-yellow-500/10 border-yellow-500/30',
  decision_made: 'text-green-400 bg-green-500/10 border-green-500/30',
  memory_saved: 'text-indigo-400 bg-indigo-500/10 border-indigo-500/30',
};

export function SessionSharing({ sessionId }: SessionSharingProps) {
  const [sessionInfo, setSessionInfo] = useState<SessionInfo | null>(null);
  const [events, setEvents] = useState<CollaborationEvent[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function fetchSessionData() {
      try {
        setError(null);

        // Fetch session info from context bridge service
        const sessionResponse = await fetch(
          `http://localhost:8009/api/v1/sessions/${sessionId}`
        );

        if (!sessionResponse.ok) {
          // If session not found in context bridge, try metrics service
          const metricsResponse = await fetch(
            `http://localhost:8003/sessions/${sessionId}`
          );
          if (!metricsResponse.ok) {
            throw new Error('Session not found');
          }
          const metricsData = await metricsResponse.json();
          // Transform metrics service data to SessionInfo format
          setSessionInfo({
            session_id: metricsData.session_id || sessionId,
            project_id: metricsData.project_id,
            tools: metricsData.tools || [],
            created_at: metricsData.started_at || new Date().toISOString(),
            updated_at: metricsData.last_activity || new Date().toISOString(),
          });
        } else {
          const sessionData = await sessionResponse.json();
          setSessionInfo(sessionData);
        }

        // Fetch collaboration events from metrics service
        const eventsResponse = await fetch(
          `http://localhost:8003/sessions/${sessionId}/events`
        );

        if (eventsResponse.ok) {
          const eventsData = await eventsResponse.json();
          setEvents(eventsData.events || []);
        } else {
          // If events endpoint doesn't exist, generate synthetic events from session info
          setEvents([]);
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Unknown error');
        console.error('Failed to fetch session data:', err);
      } finally {
        setLoading(false);
      }
    }

    fetchSessionData();

    // Poll for updates every 5 seconds
    const interval = setInterval(fetchSessionData, 5000);
    return () => clearInterval(interval);
  }, [sessionId]);

  if (loading) {
    return (
      <div className="flex items-center justify-center p-12">
        <LoadingSpinner />
      </div>
    );
  }

  if (error || !sessionInfo) {
    return (
      <Card className="p-6">
        <div className="flex items-center gap-3 text-red-400">
          <AlertCircle className="h-5 w-5" />
          <span>Error loading session: {error || 'Session not found'}</span>
        </div>
      </Card>
    );
  }

  const activeTools = sessionInfo.tools.filter(t => !t.left_at);
  const inactiveTools = sessionInfo.tools.filter(t => t.left_at);

  return (
    <div className="space-y-6">
      {/* Active Tools Section */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Users className="h-5 w-5 text-blue-500" />
            Active Tools in This Session
          </CardTitle>
          <CardDescription>
            {activeTools.length === 0
              ? 'No tools currently active'
              : `${activeTools.length} tool${activeTools.length === 1 ? '' : 's'} sharing context`}
          </CardDescription>
        </CardHeader>
        <CardContent>
          {activeTools.length === 0 ? (
            <div className="text-center py-8 text-gray-400">
              <Users className="h-12 w-12 mx-auto mb-3 opacity-50" />
              <p>No active tools in this session</p>
            </div>
          ) : (
            <div className="flex flex-wrap gap-4">
              {activeTools.map(tool => (
                <ActiveToolBadge key={tool.tool_id} tool={tool} />
              ))}
            </div>
          )}

          {inactiveTools.length > 0 && (
            <div className="mt-6 pt-6 border-t border-gray-700">
              <h4 className="text-sm font-semibold text-gray-400 mb-3">
                Previously Active ({inactiveTools.length})
              </h4>
              <div className="flex flex-wrap gap-3">
                {inactiveTools.map(tool => (
                  <InactiveToolBadge key={tool.tool_id} tool={tool} />
                ))}
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Collaboration Timeline */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Clock className="h-5 w-5 text-purple-500" />
            Collaboration Timeline
          </CardTitle>
          <CardDescription>
            Timeline of tool interactions and context sharing
          </CardDescription>
        </CardHeader>
        <CardContent>
          {events.length === 0 ? (
            <div className="text-center py-8 text-gray-400">
              <Clock className="h-12 w-12 mx-auto mb-3 opacity-50" />
              <p>No collaboration events yet</p>
              <p className="text-sm text-gray-500 mt-1">
                Events will appear as tools interact with the session
              </p>
            </div>
          ) : (
            <div className="relative">
              {/* Timeline Line */}
              <div className="absolute left-6 top-0 bottom-0 w-0.5 bg-gray-700" />

              {/* Timeline Events */}
              <div className="space-y-4">
                {events.map((event, index) => (
                  <TimelineEvent key={index} event={event} />
                ))}
              </div>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}

// Active Tool Badge Component
interface ActiveToolBadgeProps {
  tool: ToolInfo;
}

function ActiveToolBadge({ tool }: ActiveToolBadgeProps) {
  const toolDisplay = getToolDisplay(tool.tool_type);
  const ToolIcon = toolDisplay.icon;
  const bgColor = getToolBgColor(tool.tool_type);
  const borderColor = getToolBorderColor(tool.tool_type);

  const joinedTime = formatRelativeTime(tool.joined_at);

  return (
    <div
      className={cn(
        'flex items-center gap-3 px-4 py-3 rounded-lg border-2 transition-all',
        bgColor,
        borderColor,
        'hover:shadow-lg'
      )}
    >
      {/* Pulsing Active Indicator */}
      <div className="relative flex items-center justify-center">
        <span className="absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75 animate-ping" />
        <span className="relative inline-flex rounded-full h-3 w-3 bg-green-500" />
      </div>

      {/* Tool Info */}
      <div className="flex items-center gap-2">
        <ToolIcon className="h-6 w-6" />
        <div>
          <div className="font-semibold text-gray-100">{toolDisplay.name}</div>
          <div className="text-xs text-gray-400">Joined {joinedTime}</div>
        </div>
      </div>
    </div>
  );
}

// Inactive Tool Badge Component
interface InactiveToolBadgeProps {
  tool: ToolInfo;
}

function InactiveToolBadge({ tool }: InactiveToolBadgeProps) {
  const toolDisplay = getToolDisplay(tool.tool_type);
  const ToolIcon = toolDisplay.icon;

  const joinedTime = formatRelativeTime(tool.joined_at);
  const leftTime = tool.left_at ? formatRelativeTime(tool.left_at) : 'unknown';

  return (
    <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-gray-800/50 border border-gray-700">
      <ToolIcon className="h-4 w-4 text-gray-500" />
      <div className="text-sm">
        <span className="text-gray-400">{toolDisplay.name}</span>
        <span className="text-gray-600 ml-2">
          ({joinedTime} → {leftTime})
        </span>
      </div>
    </div>
  );
}

// Timeline Event Component
interface TimelineEventProps {
  event: CollaborationEvent;
}

function TimelineEvent({ event }: TimelineEventProps) {
  const eventColor = EVENT_COLORS[event.event_type] || 'text-gray-400 bg-gray-500/10 border-gray-500/30';
  const eventIcon = EVENT_ICONS[event.event_type] || <Clock className="h-4 w-4" />;
  const description = getEventDescription(event);
  const timestamp = formatRelativeTime(event.timestamp);

  const toolDisplay = event.tool_type ? getToolDisplay(event.tool_type) : null;
  const ToolIcon = toolDisplay?.icon;

  return (
    <div className="relative flex gap-4 group">
      {/* Event Icon */}
      <div
        className={cn(
          'relative z-10 flex items-center justify-center w-12 h-12 rounded-full border-2',
          eventColor,
          'group-hover:scale-110 transition-transform'
        )}
      >
        {eventIcon}
      </div>

      {/* Event Content */}
      <div className="flex-1 pb-8">
        <div className="bg-gray-800/50 rounded-lg p-4 border border-gray-700 hover:border-gray-600 transition-colors">
          {/* Event Header */}
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-2">
              {ToolIcon && <ToolIcon className="h-4 w-4 text-gray-400" />}
              <span className="font-semibold text-gray-100">
                {getEventTitle(event.event_type)}
              </span>
            </div>
            <span className="text-xs text-gray-500">{timestamp}</span>
          </div>

          {/* Event Description */}
          <p className="text-sm text-gray-300">{description}</p>

          {/* Event Metadata */}
          {event.metadata && Object.keys(event.metadata).length > 0 && (
            <div className="mt-3 pt-3 border-t border-gray-700">
              <EventMetadata metadata={event.metadata} />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// Event Metadata Component
interface EventMetadataProps {
  metadata: CollaborationEvent['metadata'];
}

function EventMetadata({ metadata }: EventMetadataProps) {
  if (!metadata) return null;

  return (
    <div className="flex flex-wrap gap-2 text-xs">
      {metadata.file_path && (
        <span className="px-2 py-1 rounded bg-purple-500/20 text-purple-300 font-mono">
          {metadata.file_path}
        </span>
      )}
      {metadata.files_merged !== undefined && (
        <span className="px-2 py-1 rounded bg-blue-500/20 text-blue-300">
          {metadata.files_merged} files merged
        </span>
      )}
      {metadata.from_tools && metadata.from_tools.length > 0 && (
        <span className="px-2 py-1 rounded bg-green-500/20 text-green-300">
          From {metadata.from_tools.length} tools
        </span>
      )}
      {metadata.query && (
        <span className="px-2 py-1 rounded bg-yellow-500/20 text-yellow-300 font-mono">
          "{metadata.query}"
        </span>
      )}
      {metadata.files_count !== undefined && (
        <span className="px-2 py-1 rounded bg-gray-500/20 text-gray-300">
          {metadata.files_count} files
        </span>
      )}
      {metadata.searches_count !== undefined && (
        <span className="px-2 py-1 rounded bg-gray-500/20 text-gray-300">
          {metadata.searches_count} searches
        </span>
      )}
    </div>
  );
}

// Helper Functions

function getEventTitle(eventType: string): string {
  const titles: Record<string, string> = {
    tool_joined: 'Tool Joined',
    tool_left: 'Tool Left',
    context_merged: 'Context Merged',
    file_accessed: 'File Accessed',
    search_executed: 'Search Executed',
    decision_made: 'Decision Made',
    memory_saved: 'Memory Saved',
  };
  return titles[eventType] || eventType;
}

function getEventDescription(event: CollaborationEvent): string {
  const toolDisplay = event.tool_type ? getToolDisplay(event.tool_type) : null;
  const toolName = toolDisplay?.name || event.tool_id || 'A tool';

  switch (event.event_type) {
    case 'tool_joined':
      return `${toolName} joined the session and began sharing context`;

    case 'tool_left':
      return `${toolName} left the session`;

    case 'context_merged':
      if (event.metadata?.from_tools && event.metadata.from_tools.length > 0) {
        const tools = event.metadata.from_tools.map(t => getToolDisplay(t).name).join(', ');
        const fileCount = event.metadata.files_merged || 0;
        return `Context merged from ${tools} (${fileCount} files combined)`;
      }
      return 'Context was merged between tools';

    case 'file_accessed': {
      const filePath = event.metadata?.file_path || 'a file';
      return `${toolName} accessed ${filePath}`;
    }

    case 'search_executed': {
      const query = event.metadata?.query || 'a query';
      return `${toolName} searched for "${query}"`;
    }

    case 'decision_made':
      return `${toolName} made a decision that was saved to shared context`;

    case 'memory_saved':
      return `${toolName} saved important information to shared memory`;

    default:
      return `${toolName} performed an action`;
  }
}

function formatRelativeTime(timestamp: string): string {
  const date = new Date(timestamp);
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  const diffSecs = Math.floor(diffMs / 1000);
  const diffMins = Math.floor(diffSecs / 60);
  const diffHours = Math.floor(diffMins / 60);
  const diffDays = Math.floor(diffHours / 24);

  if (diffSecs < 60) {
    return `${diffSecs}s ago`;
  } else if (diffMins < 60) {
    return `${diffMins}m ago`;
  } else if (diffHours < 24) {
    return `${diffHours}h ago`;
  } else if (diffDays < 7) {
    return `${diffDays}d ago`;
  } else {
    return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
  }
}
