import { useState, useEffect } from 'react';
import { Clock, FileText, Search, AlertCircle } from 'lucide-react';
import { Card, CardContent } from './shared/Card';
import { LoadingSpinner } from './shared/LoadingSpinner';
import { getToolDisplay } from '../utils/toolMapping';

interface SessionContext {
  files_accessed?: Array<{
    file_path: string;
    access_count?: number;
    last_accessed?: string;
  }>;
  recent_searches?: Array<{
    query: string;
    timestamp: string;
    results_count?: number;
  }>;
  saved_memories?: Array<unknown>;
  decisions?: Array<unknown>;
  file_importance_scores?: Record<string, number>;
  tool_specific?: Record<string, unknown>;
}

interface APISession {
  session_id: string;
  tool_id: string;
  project_id?: string;
  started_at: string;
  last_activity: string;
  ended_at?: string | null;
  pinned: number | boolean;
  archived: number | boolean;
  total_compressions?: number;
  total_embeddings?: number;
  tokens_saved?: number;
  context_json?: string;
  compressed_context?: string | null;
}

interface Session {
  session_id: string;
  tool_id: string;
  project_id?: string;
  started_at: string;
  last_activity: string;
  ended_at?: string;
  pinned: boolean;
  archived: boolean;
  total_compressions?: number;
  total_embeddings?: number;
  tokens_saved?: number;
  context?: SessionContext;
}

interface SessionTimelineProps {
  toolId?: string;
  limit?: number;
  onSessionClick?: (sessionId: string) => void;
}

export function SessionTimeline({
  toolId,
  limit = 50,
  onSessionClick
}: SessionTimelineProps) {
  const [sessions, setSessions] = useState<Session[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [filter, setFilter] = useState<'all' | 'active' | 'ended'>('all');

  useEffect(() => {
    async function fetchSessions() {
      try {
        setError(null);

        // Build query params
        const params = new URLSearchParams();
        if (toolId) params.append('tool_id', toolId);
        if (filter === 'ended') {
          params.append('include_archived', 'true');
        }
        params.append('limit', limit.toString());

        const response = await fetch(`http://localhost:8003/sessions?${params}`);
        if (!response.ok) {
          throw new Error(`Failed to fetch sessions: ${response.statusText}`);
        }

        const data = await response.json();
        const apiSessions: APISession[] = data.sessions || [];

        // Transform API sessions to our Session interface
        let sessionList: Session[] = apiSessions.map(s => {
          let context: SessionContext | undefined;

          // Parse context_json if it exists
          if (s.context_json) {
            try {
              context = JSON.parse(s.context_json);
            } catch (err) {
              console.error('Failed to parse context_json for session', s.session_id, err);
              context = undefined;
            }
          }

          return {
            session_id: s.session_id,
            tool_id: s.tool_id,
            project_id: s.project_id,
            started_at: s.started_at,
            last_activity: s.last_activity,
            ended_at: s.ended_at || undefined,
            pinned: Boolean(s.pinned),
            archived: Boolean(s.archived),
            total_compressions: s.total_compressions,
            total_embeddings: s.total_embeddings,
            tokens_saved: s.tokens_saved,
            context,
          };
        });

        // Client-side filtering based on status
        if (filter === 'active') {
          sessionList = sessionList.filter(s => !s.ended_at);
        } else if (filter === 'ended') {
          sessionList = sessionList.filter(s => s.ended_at);
        }

        setSessions(sessionList);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Unknown error');
        console.error('Failed to fetch sessions:', err);
      } finally {
        setLoading(false);
      }
    }

    fetchSessions();
    const interval = setInterval(fetchSessions, 10000); // Poll every 10s
    return () => clearInterval(interval);
  }, [toolId, filter, limit]);

  // Group sessions by date
  const groupedSessions = groupSessionsByDate(sessions);

  if (loading && sessions.length === 0) {
    return (
      <div className="flex items-center justify-center p-12">
        <LoadingSpinner />
      </div>
    );
  }

  if (error) {
    return (
      <Card className="p-6">
        <div className="flex items-center gap-3 text-red-400">
          <AlertCircle className="h-5 w-5" />
          <span>Error loading sessions: {error}</span>
        </div>
      </Card>
    );
  }

  return (
    <div className="space-y-4">
      {/* Filter Bar */}
      <div className="flex gap-2">
        {(['all', 'active', 'ended'] as const).map(status => (
          <button
            key={status}
            onClick={() => setFilter(status)}
            className={`px-4 py-2 rounded-lg font-medium transition-colors ${
              filter === status
                ? 'bg-blue-600 text-white'
                : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
            }`}
          >
            {status.charAt(0).toUpperCase() + status.slice(1)}
          </button>
        ))}
      </div>

      {/* Timeline */}
      {sessions.length === 0 ? (
        <Card className="p-12">
          <div className="text-center text-gray-400">
            <Clock className="h-12 w-12 mx-auto mb-4 opacity-50" />
            <p>No sessions found</p>
          </div>
        </Card>
      ) : (
        <div className="space-y-6">
          {groupedSessions.map(({ date, sessions: daySessions }) => (
            <div key={date}>
              <h3 className="text-sm font-semibold text-gray-400 mb-3 sticky top-0 bg-gray-900 py-2">
                {date}
              </h3>
              <div className="space-y-2">
                {daySessions.map(session => (
                  <SessionCard
                    key={session.session_id}
                    session={session}
                    onClick={() => onSessionClick?.(session.session_id)}
                  />
                ))}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

interface SessionCardProps {
  session: Session;
  onClick?: () => void;
}

function SessionCard({ session, onClick }: SessionCardProps) {
  const isActive = !session.ended_at;
  // Use actual API fields from tool_sessions table
  const filesCount = session.total_compressions || 0;
  const searchesCount = session.total_embeddings || 0;
  const toolInfo = getToolDisplay(session.tool_id);
  const ToolIcon = toolInfo.icon;

  return (
    <div
      onClick={onClick}
      className="cursor-pointer"
    >
      <Card
        className={`hover:border-blue-500/50 transition-all hover:shadow-lg ${
          isActive ? 'border-green-500/30 bg-green-500/5' : 'border-gray-700'
        }`}
      >
        <CardContent className="p-4">
          <div className="flex items-center justify-between">
            {/* Left: Status + Info */}
            <div className="flex items-center gap-3">
              {isActive && (
                <div className="w-2 h-2 rounded-full bg-green-400 animate-pulse" />
              )}
              <div>
                <div className="font-semibold text-gray-100 flex items-center gap-2">
                  <ToolIcon className="w-5 h-5" />
                  <span>{toolInfo.name}</span>
                  {session.pinned && (
                    <span className="text-xs bg-yellow-500/20 text-yellow-400 px-2 py-0.5 rounded">
                      Pinned
                    </span>
                  )}
                </div>
                <div className="text-xs text-gray-400">
                  {formatTime(session.started_at)}
                  {session.ended_at && ` → ${formatTime(session.ended_at)}`}
                </div>
              </div>
            </div>

            {/* Right: Metrics */}
            <div className="flex gap-4 text-sm">
              <div className="flex items-center gap-1 text-blue-400">
                <FileText className="h-4 w-4" />
                <span>{filesCount}</span>
              </div>
              <div className="flex items-center gap-1 text-green-400">
                <Search className="h-4 w-4" />
                <span>{searchesCount}</span>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

// Helper Functions

function groupSessionsByDate(sessions: Session[]): Array<{ date: string; sessions: Session[] }> {
  const groups = new Map<string, Session[]>();

  sessions.forEach(session => {
    const date = formatDateGroup(session.started_at);
    if (!groups.has(date)) {
      groups.set(date, []);
    }
    groups.get(date)!.push(session);
  });

  return Array.from(groups.entries()).map(([date, sessions]) => ({
    date,
    sessions: sessions.sort((a, b) =>
      new Date(b.started_at).getTime() - new Date(a.started_at).getTime()
    ),
  }));
}

function formatDateGroup(dateString: string): string {
  const date = new Date(dateString);
  const today = new Date();
  const yesterday = new Date(today);
  yesterday.setDate(yesterday.getDate() - 1);

  if (date.toDateString() === today.toDateString()) {
    return 'Today';
  } else if (date.toDateString() === yesterday.toDateString()) {
    return 'Yesterday';
  } else {
    return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
  }
}

function formatTime(dateString: string): string {
  const date = new Date(dateString);
  return date.toLocaleString('en-US', {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
    hour12: false,
  });
}
