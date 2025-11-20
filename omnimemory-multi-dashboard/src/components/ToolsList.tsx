import { useState, useEffect, useCallback } from 'react';
import {
  Zap,
  Bot,
  Settings,
  Clock,
  Code2,
  FileCode,
  Cpu,
  Network
} from 'lucide-react';
import type { LucideIcon } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from './shared/Card';
import { LoadingSpinner } from './shared/LoadingSpinner';
import { ErrorState } from './shared/ErrorState';
import { cn } from '../lib/utils';

// Types
interface ToolCapabilities {
  supports_mcp?: boolean;
  supports_lsp?: boolean;
  supports_rest?: boolean;
  max_context_tokens: number;
  can_execute_code?: boolean;
  can_edit_files?: boolean;
}

interface ToolInfo {
  tool_id: string;
  tool_type: string;
  last_activity: string;
  capabilities: ToolCapabilities;
}

interface ToolsListProps {
  projectId: string;
}

interface ToolsResponse {
  ide_tools: ToolInfo[];
  agents: ToolInfo[];
}

// Tool type display configuration
interface ToolTypeConfig {
  name: string;
  icon: LucideIcon;
  color: string;
  bgColor: string;
  borderColor: string;
}

const IDE_TOOL_CONFIGS: Record<string, ToolTypeConfig> = {
  'cursor': {
    name: 'Cursor',
    icon: Zap,
    color: 'text-blue-400',
    bgColor: 'bg-blue-500/10',
    borderColor: 'border-blue-500/30'
  },
  'vscode': {
    name: 'VSCode',
    icon: Code2,
    color: 'text-blue-500',
    bgColor: 'bg-blue-600/10',
    borderColor: 'border-blue-600/30'
  },
  'claude-code': {
    name: 'Claude Code',
    icon: Bot,
    color: 'text-purple-400',
    bgColor: 'bg-purple-500/10',
    borderColor: 'border-purple-500/30'
  },
  'continue': {
    name: 'Continue',
    icon: FileCode,
    color: 'text-green-400',
    bgColor: 'bg-green-500/10',
    borderColor: 'border-green-500/30'
  }
};

const AGENT_CONFIGS: Record<string, ToolTypeConfig> = {
  'n8n-agent': {
    name: 'n8n Agent',
    icon: Network,
    color: 'text-orange-400',
    bgColor: 'bg-orange-500/10',
    borderColor: 'border-orange-500/30'
  },
  'custom-agent': {
    name: 'Custom Agent',
    icon: Settings,
    color: 'text-gray-400',
    bgColor: 'bg-gray-500/10',
    borderColor: 'border-gray-500/30'
  },
  'langchain-agent': {
    name: 'LangChain Agent',
    icon: Bot,
    color: 'text-green-400',
    bgColor: 'bg-green-500/10',
    borderColor: 'border-green-500/30'
  },
  'autogen-agent': {
    name: 'AutoGen Agent',
    icon: Cpu,
    color: 'text-cyan-400',
    bgColor: 'bg-cyan-500/10',
    borderColor: 'border-cyan-500/30'
  }
};

const DEFAULT_CONFIG: ToolTypeConfig = {
  name: 'Unknown Tool',
  icon: Settings,
  color: 'text-gray-400',
  bgColor: 'bg-gray-500/10',
  borderColor: 'border-gray-500/30'
};

// Helper functions
function getToolConfig(toolType: string, isAgent: boolean): ToolTypeConfig {
  const configs = isAgent ? AGENT_CONFIGS : IDE_TOOL_CONFIGS;
  return configs[toolType] || DEFAULT_CONFIG;
}

function getActivityStatus(lastActivity: string): 'active' | 'idle' | 'inactive' {
  const now = new Date();
  const activity = new Date(lastActivity);
  const diffMinutes = (now.getTime() - activity.getTime()) / (1000 * 60);

  if (diffMinutes < 5) return 'active';
  if (diffMinutes < 30) return 'idle';
  return 'inactive';
}

function formatRelativeTime(timestamp: string): string {
  const now = new Date();
  const time = new Date(timestamp);
  const diffSeconds = Math.floor((now.getTime() - time.getTime()) / 1000);

  if (diffSeconds < 60) return `${diffSeconds} sec ago`;
  const diffMinutes = Math.floor(diffSeconds / 60);
  if (diffMinutes < 60) return `${diffMinutes} min ago`;
  const diffHours = Math.floor(diffMinutes / 60);
  if (diffHours < 24) return `${diffHours} hour${diffHours > 1 ? 's' : ''} ago`;
  const diffDays = Math.floor(diffHours / 24);
  return `${diffDays} day${diffDays > 1 ? 's' : ''} ago`;
}

function formatTokenCount(tokens: number): string {
  if (tokens >= 1000) {
    return `${(tokens / 1000).toFixed(0)}K`;
  }
  return tokens.toString();
}

// Main Component
export function ToolsList({ projectId }: ToolsListProps) {
  const [ideTools, setIdeTools] = useState<ToolInfo[]>([]);
  const [agents, setAgents] = useState<ToolInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchTools = useCallback(async () => {
    try {
      setError(null);
      const response = await fetch(
        `http://localhost:8009/api/v1/projects/${projectId}/all-tools`
      );

      if (!response.ok) {
        throw new Error(`Failed to fetch tools: ${response.statusText}`);
      }

      const data: ToolsResponse = await response.json();
      setIdeTools(data.ide_tools || []);
      setAgents(data.agents || []);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load tools');
      console.error('Failed to fetch tools:', err);
    } finally {
      setLoading(false);
    }
  }, [projectId]);

  useEffect(() => {
    fetchTools();
    // Poll every 30 seconds for updates
    const interval = setInterval(fetchTools, 30000);
    return () => clearInterval(interval);
  }, [fetchTools]);

  if (loading && ideTools.length === 0 && agents.length === 0) {
    return (
      <div className="flex items-center justify-center p-12">
        <LoadingSpinner />
      </div>
    );
  }

  if (error) {
    return (
      <ErrorState
        error={error}
        context="Failed to load tools"
        onRetry={fetchTools}
      />
    );
  }

  const totalTools = ideTools.length + agents.length;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-foreground">Connected Tools</h2>
          <p className="text-sm text-muted-foreground mt-1">
            {totalTools} tool{totalTools !== 1 ? 's' : ''} connected to this project
          </p>
        </div>
      </div>

      {/* IDE Tools Section */}
      {ideTools.length > 0 && (
        <div>
          <h3 className="text-lg font-semibold text-foreground mb-4 flex items-center gap-2">
            <Code2 className="h-5 w-5" />
            IDE Tools
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {ideTools.map(tool => (
              <ToolCard key={tool.tool_id} tool={tool} type="ide" />
            ))}
          </div>
        </div>
      )}

      {/* Autonomous Agents Section */}
      {agents.length > 0 && (
        <div>
          <h3 className="text-lg font-semibold text-foreground mb-4 flex items-center gap-2">
            <Bot className="h-5 w-5" />
            Autonomous Agents
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {agents.map(agent => (
              <ToolCard key={agent.tool_id} tool={agent} type="agent" />
            ))}
          </div>
        </div>
      )}

      {/* Empty State */}
      {totalTools === 0 && (
        <Card className="p-12">
          <div className="text-center text-muted-foreground">
            <Settings className="h-12 w-12 mx-auto mb-4 opacity-50" />
            <p className="text-lg font-medium mb-2">No tools connected</p>
            <p className="text-sm">
              Connect an IDE or agent to start using OmniMemory
            </p>
          </div>
        </Card>
      )}
    </div>
  );
}

// Tool Card Component
interface ToolCardProps {
  tool: ToolInfo;
  type: 'ide' | 'agent';
}

function ToolCard({ tool, type }: ToolCardProps) {
  const config = getToolConfig(tool.tool_type, type === 'agent');
  const ToolIcon = config.icon;
  const status = getActivityStatus(tool.last_activity);
  const relativeTime = formatRelativeTime(tool.last_activity);

  const statusConfig = {
    active: {
      color: 'bg-green-400',
      label: 'Connected',
      textColor: 'text-green-400'
    },
    idle: {
      color: 'bg-yellow-400',
      label: 'Idle',
      textColor: 'text-yellow-400'
    },
    inactive: {
      color: 'bg-gray-400',
      label: 'Inactive',
      textColor: 'text-gray-400'
    }
  };

  const currentStatus = statusConfig[status];

  return (
    <Card
      className={cn(
        'transition-all hover:shadow-lg hover:border-primary/50',
        config.borderColor
      )}
    >
      <CardHeader className="pb-3">
        <div className="flex items-start justify-between">
          <div className="flex items-center gap-3">
            <div className={cn('p-2 rounded-lg', config.bgColor)}>
              <ToolIcon className={cn('h-5 w-5', config.color)} />
            </div>
            <div>
              <CardTitle className="text-base">{config.name}</CardTitle>
              <p className="text-xs text-muted-foreground mt-0.5">
                {tool.tool_id.split('-').slice(0, 2).join('-')}
              </p>
            </div>
          </div>
          <div
            className={cn('w-2 h-2 rounded-full', currentStatus.color)}
            title={currentStatus.label}
          />
        </div>
      </CardHeader>

      <CardContent className="pt-0 space-y-3">
        {/* Status */}
        <div className="flex items-center gap-2 text-sm">
          <span className={currentStatus.textColor}>{currentStatus.label}</span>
          <span className="text-muted-foreground">·</span>
          <div className="flex items-center gap-1 text-muted-foreground">
            <Clock className="h-3 w-3" />
            <span className="text-xs">{relativeTime}</span>
          </div>
        </div>

        {/* Capabilities */}
        <div>
          <p className="text-xs font-medium text-muted-foreground mb-2">Capabilities</p>
          <div className="flex flex-wrap gap-1.5">
            {tool.capabilities.supports_mcp && (
              <CapabilityBadge label="MCP" />
            )}
            {tool.capabilities.supports_lsp && (
              <CapabilityBadge label="LSP" />
            )}
            {tool.capabilities.supports_rest && (
              <CapabilityBadge label="REST API" />
            )}
            {tool.capabilities.can_execute_code && (
              <CapabilityBadge label="Execute" />
            )}
            {tool.capabilities.can_edit_files && (
              <CapabilityBadge label="Edit" />
            )}
          </div>
        </div>

        {/* Context Window */}
        <div className="pt-2 border-t border-border/50">
          <div className="flex items-center justify-between text-xs">
            <span className="text-muted-foreground">Context Window</span>
            <span className="font-medium text-foreground">
              {formatTokenCount(tool.capabilities.max_context_tokens)} tokens
            </span>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

// Capability Badge Component
interface CapabilityBadgeProps {
  label: string;
}

function CapabilityBadge({ label }: CapabilityBadgeProps) {
  return (
    <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-primary/10 text-primary border border-primary/20">
      {label}
    </span>
  );
}
