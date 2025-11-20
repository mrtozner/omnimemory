import { useState, useEffect, useCallback } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '../shared/Card';
import { api } from '../../services/api';
import { Brain, Zap, Code, Play, Github, Sparkles, Wind, Terminal, Eye, EyeOff } from 'lucide-react';
import { formatNumber } from '../../lib/utils';

const ICON_MAP = {
  brain: Brain,
  zap: Zap,
  code: Code,
  play: Play,
  github: Github,
  sparkles: Sparkles,
  wind: Wind,
  terminal: Terminal,
};

interface ToolMetrics {
  tool_id: string;
  active_sessions: number;
  total_tokens_saved: number;
  estimated_cost_saved: number;
}

interface Tool {
  id: string;
  name: string;
  icon: string;
  color: string;
  configured: boolean;
  metrics?: ToolMetrics;
  visible: boolean;
}

export function ToolsOverviewCard() {
  const [tools, setTools] = useState<Tool[]>([]);
  const [loading, setLoading] = useState(true);
  const [hiddenTools, setHiddenTools] = useState<Set<string>>(new Set());

  const loadTools = useCallback(async () => {
    try {
      const { tools: initializedTools } = await api.getInitializedTools();

      // Load metrics for each tool
      const toolsWithMetrics = await Promise.all(
        initializedTools.map(async (tool) => {
          try {
            const metrics = await api.getToolMetricsDetailed(tool.id, 24);
            return {
              ...tool,
              metrics,
              visible: !hiddenTools.has(tool.id),
            };
          } catch (err) {
            console.error(`Failed to load metrics for ${tool.id}:`, err);
            return {
              ...tool,
              metrics: undefined,
              visible: !hiddenTools.has(tool.id),
            };
          }
        })
      );

      setTools(toolsWithMetrics);
      setLoading(false);
    } catch (err) {
      console.error('Failed to load tools:', err);
      setLoading(false);
    }
  }, [hiddenTools]);

  useEffect(() => {
    loadTools();
    const interval = setInterval(loadTools, 10000); // Refresh every 10s
    return () => clearInterval(interval);
  }, [loadTools]);

  const toggleToolVisibility = (toolId: string) => {
    setHiddenTools(prev => {
      const next = new Set(prev);
      if (next.has(toolId)) {
        next.delete(toolId);
      } else {
        next.add(toolId);
      }
      return next;
    });

    setTools(prev =>
      prev.map(tool =>
        tool.id === toolId ? { ...tool, visible: !tool.visible } : tool
      )
    );
  };

  const visibleTools = tools.filter(t => t.visible);
  const totalSessions = visibleTools.reduce(
    (sum, t) => sum + (t.metrics?.active_sessions || 0),
    0
  );
  const totalTokensSaved = visibleTools.reduce(
    (sum, t) => sum + (t.metrics?.total_tokens_saved || 0),
    0
  );
  const totalCostSaved = visibleTools.reduce(
    (sum, t) => sum + (t.metrics?.estimated_cost_saved || 0),
    0
  );

  if (loading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Initialized Tools</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-center py-8">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          <span>Initialized Tools ({tools.length})</span>
          <span className="text-sm font-normal text-muted-foreground">
            {visibleTools.length} visible
          </span>
        </CardTitle>
      </CardHeader>
      <CardContent>
        {/* Overview Stats */}
        <div className="grid grid-cols-3 gap-4 mb-6 p-4 bg-muted/50 rounded-lg">
          <div className="text-center">
            <div className="text-2xl font-bold">{totalSessions}</div>
            <div className="text-xs text-muted-foreground">Active Sessions</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold">{formatNumber(totalTokensSaved)}</div>
            <div className="text-xs text-muted-foreground">Tokens Saved</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold">${totalCostSaved.toFixed(2)}</div>
            <div className="text-xs text-muted-foreground">Cost Saved</div>
          </div>
        </div>

        {/* Tool List */}
        <div className="space-y-2">
          {tools.map((tool) => {
            const Icon = ICON_MAP[tool.icon as keyof typeof ICON_MAP] || Brain;
            const isVisible = tool.visible;

            return (
              <div
                key={tool.id}
                className={`flex items-center justify-between p-3 rounded-lg border transition-all ${
                  isVisible
                    ? 'bg-card border-border'
                    : 'bg-muted/30 border-muted opacity-50'
                }`}
              >
                <div className="flex items-center gap-3">
                  <div
                    className="p-2 rounded-lg"
                    style={{ backgroundColor: `${tool.color}20` }}
                  >
                    <Icon className="h-4 w-4" style={{ color: tool.color }} />
                  </div>
                  <div>
                    <div className="font-medium text-sm">{tool.name}</div>
                    {tool.metrics && isVisible && (
                      <div className="text-xs text-muted-foreground">
                        {tool.metrics.active_sessions} sessions •{' '}
                        {formatNumber(tool.metrics.total_tokens_saved)} tokens saved
                      </div>
                    )}
                  </div>
                </div>
                <button
                  onClick={() => toggleToolVisibility(tool.id)}
                  className="p-1.5 rounded-md hover:bg-muted transition-colors"
                  title={isVisible ? 'Hide from dashboard' : 'Show on dashboard'}
                >
                  {isVisible ? (
                    <Eye className="h-4 w-4 text-muted-foreground" />
                  ) : (
                    <EyeOff className="h-4 w-4 text-muted-foreground" />
                  )}
                </button>
              </div>
            );
          })}
        </div>

        {tools.length === 0 && (
          <div className="text-center py-8 text-muted-foreground">
            <p>No tools configured with OmniMemory</p>
            <p className="text-sm mt-2">Run `omni init` to get started</p>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
