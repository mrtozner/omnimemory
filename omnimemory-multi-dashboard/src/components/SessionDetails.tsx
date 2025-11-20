import { useState, useEffect } from 'react';
import {
  Clock, FileText, Search, Brain, MessageSquare,
  AlertCircle, Folder, Calendar
} from 'lucide-react';
import { Card, CardHeader, CardTitle, CardContent } from './shared/Card';
import { LoadingSpinner } from './shared/LoadingSpinner';
import { SessionManager } from './SessionManager';
import { getToolDisplay, getToolColor } from '../utils/toolMapping';

interface SessionContext {
  files_accessed?: Array<{
    path: string;
    accessed_at: string;
    order: number;
  }>;
  file_importance_scores?: Record<string, number>;
  searches?: Array<{
    query: string;
    timestamp: string;
  }>;
  decisions?: Array<{
    decision: string;
    timestamp: string;
  }>;
  saved_memories?: Array<{
    key: string;
    value: string;
  }>;
}

interface ToolOperation {
  id: string;
  session_id: string;
  tool_name: 'read' | 'search';
  operation_mode: string;
  parameters: Record<string, any>;
  file_path: string | null;
  tokens_original: number;
  tokens_actual: number;
  tokens_prevented: number;
  response_time_ms: number;
  tool_id: string;
  created_at: string;
}

interface Session {
  session_id: string;
  tool_id: string;
  tool_version?: string;
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

interface SessionDetailsProps {
  sessionId: string;
  onClose?: () => void;
}

type TabType = 'overview' | 'files' | 'searches' | 'memories';

export function SessionDetails({ sessionId, onClose }: SessionDetailsProps) {
  const [session, setSession] = useState<Session | null>(null);
  const [operations, setOperations] = useState<ToolOperation[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<TabType>('overview');

  useEffect(() => {
    async function fetchSession() {
      try {
        setError(null);
        setLoading(true);

        // Fetch session details
        const response = await fetch(`http://localhost:8003/sessions/${sessionId}`);
        if (!response.ok) {
          if (response.status === 404) {
            throw new Error('Session not found');
          }
          throw new Error('Failed to fetch session');
        }

        const data = await response.json();

        // Parse context if it's a string
        let parsedContext = data.context;
        if (typeof data.context === 'string') {
          try {
            parsedContext = JSON.parse(data.context);
          } catch {
            parsedContext = {};
          }
        }

        setSession({
          ...data,
          context: parsedContext,
          pinned: Boolean(data.pinned),
          archived: Boolean(data.archived),
        });
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Unknown error');
        console.error('Failed to fetch session:', err);
      } finally {
        setLoading(false);
      }
    }

    fetchSession();
    const interval = setInterval(fetchSession, 30000); // Refresh every 30s
    return () => clearInterval(interval);
  }, [sessionId]);

  // Fetch tool operations for this session
  useEffect(() => {
    async function fetchOperations() {
      try {
        const response = await fetch(
          `http://localhost:8003/metrics/tool-operations?session_id=${sessionId}&limit=100`
        );
        if (!response.ok) {
          console.error('Failed to fetch operations');
          return;
        }

        const data = await response.json();
        setOperations(data.operations || []);
      } catch (err) {
        console.error('Failed to fetch operations:', err);
      }
    }

    fetchOperations();
    const interval = setInterval(fetchOperations, 30000); // Refresh every 30s
    return () => clearInterval(interval);
  }, [sessionId]);

  if (loading) {
    return (
      <div className="flex items-center justify-center p-12">
        <LoadingSpinner />
      </div>
    );
  }

  if (error || !session) {
    return (
      <Card className="p-6">
        <div className="flex items-center gap-3 text-red-400">
          <AlertCircle className="h-5 w-5" />
          <span>{error || 'Session not found'}</span>
        </div>
      </Card>
    );
  }

  const isActive = !session.ended_at;
  // Use actual API fields from tool_sessions table
  const filesCount = session.total_compressions || 0;
  const searchesCount = session.total_embeddings || 0;
  const memoriesCount = session.context?.saved_memories?.length || 0;
  const decisionsCount = session.context?.decisions?.length || 0;
  const toolInfo = getToolDisplay(session.tool_id);
  const ToolIcon = toolInfo.icon;

  return (
    <div className="space-y-4">
      {/* Header */}
      <Card>
        <CardHeader>
          <div className="flex items-start justify-between">
            <div>
              <CardTitle className="flex items-center gap-3">
                <ToolIcon className="w-6 h-6" />
                <span className={getToolColor(session.tool_id)}>{toolInfo.name}</span>
                {isActive && (
                  <span className="flex items-center gap-1 text-sm font-normal text-green-400">
                    <div className="w-2 h-2 rounded-full bg-green-400 animate-pulse" />
                    Active
                  </span>
                )}
                {session.pinned && (
                  <span className="text-xs bg-yellow-500/20 text-yellow-400 px-2 py-1 rounded">
                    Pinned
                  </span>
                )}
                {session.archived && (
                  <span className="text-xs bg-gray-500/20 text-gray-400 px-2 py-1 rounded">
                    Archived
                  </span>
                )}
              </CardTitle>
              {session.tool_version && (
                <p className="text-sm text-gray-400 mt-1">Version {session.tool_version}</p>
              )}
            </div>
            {onClose && (
              <button
                onClick={onClose}
                className="text-gray-400 hover:text-gray-300"
              >
                ✕
              </button>
            )}
          </div>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div className="flex items-center gap-2">
              <Folder className="h-4 w-4 text-gray-400" />
              <span className="text-gray-400">Project:</span>
              <span className="text-gray-200 font-mono text-sm" title={session.project_id}>
                {session.project_id || 'N/A'}
              </span>
            </div>
            <div className="flex items-center gap-2">
              <Calendar className="h-4 w-4 text-gray-400" />
              <span className="text-gray-400">Started:</span>
              <span className="text-gray-200">
                {new Date(session.started_at).toLocaleString('en-US', {
                  month: 'short',
                  day: 'numeric',
                  year: 'numeric',
                  hour: '2-digit',
                  minute: '2-digit',
                  hour12: false,
                })}
              </span>
            </div>
            {session.ended_at && (
              <div className="flex items-center gap-2 col-span-2">
                <Clock className="h-4 w-4 text-gray-400" />
                <span className="text-gray-400">Ended:</span>
                <span className="text-gray-200">
                  {new Date(session.ended_at).toLocaleString('en-US', {
                    month: 'short',
                    day: 'numeric',
                    year: 'numeric',
                    hour: '2-digit',
                    minute: '2-digit',
                    hour12: false,
                  })}
                </span>
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Metrics Overview */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <MetricCard
          icon={<FileText className="h-5 w-5 text-blue-400" />}
          label="Files Accessed"
          value={filesCount}
        />
        <MetricCard
          icon={<Search className="h-5 w-5 text-green-400" />}
          label="Searches"
          value={searchesCount}
        />
        <MetricCard
          icon={<Brain className="h-5 w-5 text-purple-400" />}
          label="Memories"
          value={memoriesCount}
        />
        <MetricCard
          icon={<MessageSquare className="h-5 w-5 text-yellow-400" />}
          label="Decisions"
          value={decisionsCount}
        />
      </div>

      {/* Tabs */}
      <div className="border-b border-gray-700">
        <div className="flex gap-4">
          {([
            { id: 'overview', label: 'Overview' },
            { id: 'files', label: `Files (${filesCount})` },
            { id: 'searches', label: `Searches (${searchesCount})` },
            { id: 'memories', label: `Memories (${memoriesCount})` },
          ] as const).map(tab => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`px-4 py-2 border-b-2 transition-colors ${
                activeTab === tab.id
                  ? 'border-blue-500 text-blue-400'
                  : 'border-transparent text-gray-400 hover:text-gray-300'
              }`}
            >
              {tab.label}
            </button>
          ))}
        </div>
      </div>

      {/* Tab Content */}
      <div className="min-h-[300px]">
        {activeTab === 'overview' && <OverviewTab session={session} />}
        {activeTab === 'files' && (
          <FilesTab operations={operations.filter(op => op.tool_name === 'read')} />
        )}
        {activeTab === 'searches' && (
          <SearchesTab operations={operations.filter(op => op.tool_name === 'search')} />
        )}
        {activeTab === 'memories' && <MemoriesTab memories={session.context?.saved_memories || []} />}
      </div>

      {/* Actions */}
      <Card>
        <CardContent className="p-4">
          <SessionManager session={session} />
        </CardContent>
      </Card>
    </div>
  );
}

// Sub-components

function MetricCard({ icon, label, value }: { icon: React.ReactNode; label: string; value: number }) {
  return (
    <Card>
      <CardContent className="p-4">
        <div className="flex items-center gap-3">
          {icon}
          <div>
            <p className="text-2xl font-bold">{value}</p>
            <p className="text-xs text-gray-400">{label}</p>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

function OverviewTab({ session }: { session: Session }) {
  const duration = session.ended_at
    ? new Date(session.ended_at).getTime() - new Date(session.started_at).getTime()
    : Date.now() - new Date(session.started_at).getTime();

  const hours = Math.floor(duration / 3600000);
  const minutes = Math.floor((duration % 3600000) / 60000);

  return (
    <Card>
      <CardContent className="p-6 space-y-4">
        <div>
          <h3 className="font-semibold mb-2">Session Summary</h3>
          <div className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span className="text-gray-400">Duration:</span>
              <span>{hours}h {minutes}m</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Project ID:</span>
              <span className="font-mono text-xs">{session.project_id || 'N/A'}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Status:</span>
              <span className={session.ended_at ? 'text-gray-400' : 'text-green-400'}>
                {session.ended_at ? 'Ended' : 'Active'}
              </span>
            </div>
          </div>
        </div>

        {session.context?.decisions && session.context.decisions.length > 0 && (
          <div>
            <h3 className="font-semibold mb-2 flex items-center gap-2">
              <MessageSquare className="h-4 w-4" />
              Recent Decisions
            </h3>
            <div className="space-y-2">
              {session.context.decisions.slice(0, 3).map((decision, idx) => (
                <div key={idx} className="text-sm p-3 bg-gray-800 rounded">
                  <p>{decision.decision}</p>
                  <p className="text-xs text-gray-400 mt-1">
                    {new Date(decision.timestamp).toLocaleString('en-US', {
                      month: 'short',
                      day: 'numeric',
                      hour: '2-digit',
                      minute: '2-digit',
                      hour12: false,
                    })}
                  </p>
                </div>
              ))}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

function FilesTab({ operations }: { operations: ToolOperation[] }) {
  if (operations.length === 0) {
    return (
      <Card className="p-12">
        <div className="text-center text-gray-400">
          <FileText className="h-12 w-12 mx-auto mb-4 opacity-50" />
          <p>No files accessed in this session</p>
        </div>
      </Card>
    );
  }

  return (
    <Card>
      <CardContent className="p-4">
        <div className="space-y-2">
          {operations.map((op) => (
            <div
              key={op.id}
              className="flex items-center justify-between p-3 bg-gray-800 rounded hover:bg-gray-750"
            >
              <div className="flex items-center gap-3 flex-1 min-w-0">
                <FileText className="h-4 w-4 text-blue-400 flex-shrink-0" />
                <div className="flex-1 min-w-0">
                  <p className="text-sm truncate" title={op.file_path || 'Unknown'}>
                    {op.file_path || 'Unknown file'}
                  </p>
                  <div className="flex items-center gap-2 mt-1">
                    <span className="text-xs text-gray-500">Mode: {op.operation_mode}</span>
                    <span className="text-xs text-green-400">
                      Saved {op.tokens_prevented.toLocaleString()} tokens
                    </span>
                  </div>
                </div>
              </div>
              <span className="text-xs text-gray-400 ml-2 whitespace-nowrap">
                {new Date(op.created_at).toLocaleString('en-US', {
                  month: 'short',
                  day: 'numeric',
                  hour: '2-digit',
                  minute: '2-digit',
                  hour12: false,
                })}
              </span>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}

function SearchesTab({ operations }: { operations: ToolOperation[] }) {
  if (operations.length === 0) {
    return (
      <Card className="p-12">
        <div className="text-center text-gray-400">
          <Search className="h-12 w-12 mx-auto mb-4 opacity-50" />
          <p>No searches performed in this session</p>
        </div>
      </Card>
    );
  }

  return (
    <Card>
      <CardContent className="p-4">
        <div className="space-y-2">
          {operations.map((op) => (
            <div key={op.id} className="p-3 bg-gray-800 rounded">
              <div className="flex items-start gap-3">
                <Search className="h-4 w-4 text-green-400 mt-0.5 flex-shrink-0" />
                <div className="flex-1">
                  <p className="text-sm font-medium">
                    {op.parameters?.query || 'Unknown query'}
                  </p>
                  <div className="flex items-center gap-3 mt-1">
                    <span className="text-xs text-gray-500">
                      Mode: {op.operation_mode}
                    </span>
                    <span className="text-xs text-green-400">
                      Saved {op.tokens_prevented.toLocaleString()} tokens
                    </span>
                    <span className="text-xs text-gray-400">
                      {new Date(op.created_at).toLocaleString('en-US', {
                        month: 'short',
                        day: 'numeric',
                        hour: '2-digit',
                        minute: '2-digit',
                        hour12: false,
                      })}
                    </span>
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}

function MemoriesTab({ memories }: { memories: Array<{ key: string; value: string }> }) {
  if (memories.length === 0) {
    return (
      <Card className="p-12">
        <div className="text-center text-gray-400">
          <Brain className="h-12 w-12 mx-auto mb-4 opacity-50" />
          <p>No memories saved in this session</p>
        </div>
      </Card>
    );
  }

  return (
    <Card>
      <CardContent className="p-4">
        <div className="space-y-3">
          {memories.map((memory, idx) => (
            <div key={idx} className="p-4 bg-gray-800 rounded">
              <div className="flex items-start gap-3">
                <Brain className="h-4 w-4 text-purple-400 mt-0.5 flex-shrink-0" />
                <div className="flex-1">
                  <p className="text-sm font-semibold text-purple-400">{memory.key}</p>
                  <p className="text-sm text-gray-300 mt-1">{memory.value}</p>
                </div>
              </div>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}
