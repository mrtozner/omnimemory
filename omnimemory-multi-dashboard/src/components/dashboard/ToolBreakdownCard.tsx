import { useState, useEffect } from 'react';
import { Card, CardHeader, CardTitle, CardContent, CardDescription } from '../shared/Card';
import { LoadingSpinner } from '../shared/LoadingSpinner';
import { api } from '../../services/api';
import { formatNumber } from '../../lib/utils';
import { Brain, Zap, Activity, Circle, TrendingUp } from 'lucide-react';

interface ToolBreakdownData {
  hours: number;
  tools: Array<{
    tool_id: string;
    tokens_saved: number;
    total_compressions: number;
    total_embeddings: number;
    avg_cache_hit_rate: number;
    avg_compression_ratio: number;
    active_sessions: number;
    sample_count: number;
  }>;
  total: {
    tokens_saved: number;
    total_compressions: number;
    total_embeddings: number;
    active_sessions: number;
  };
}

const TOOL_NAMES: Record<string, string> = {
  'claude-code': 'Claude Code',
  'codex': 'OpenAI Codex',
  'cursor': 'Cursor',
  'copilot': 'GitHub Copilot',
  'gemini': 'Google Gemini',
};

const TOOL_ICONS: Record<string, React.ReactNode> = {
  'claude-code': <Brain className="h-5 w-5 text-blue-500" />,
  'codex': <Zap className="h-5 w-5 text-green-500" />,
  'cursor': <Activity className="h-5 w-5 text-purple-500" />,
  'copilot': <Zap className="h-5 w-5 text-orange-500" />,
  'gemini': <Brain className="h-5 w-5 text-pink-500" />,
};

const TOOL_COLORS: Record<string, string> = {
  'claude-code': 'bg-blue-500/20 border-blue-500/50',
  'codex': 'bg-green-500/20 border-green-500/50',
  'cursor': 'bg-purple-500/20 border-purple-500/50',
  'copilot': 'bg-orange-500/20 border-orange-500/50',
  'gemini': 'bg-pink-500/20 border-pink-500/50',
};

export function ToolBreakdownCard() {
  const [loading, setLoading] = useState(true);
  const [data, setData] = useState<ToolBreakdownData | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        setError(null);
        const breakdown = await api.getToolBreakdown(24);
        setData(breakdown);
      } catch (err) {
        console.error('Failed to fetch tool breakdown:', err);
        setError('Failed to load tool breakdown');
      } finally {
        setLoading(false);
      }
    };

    fetchData();

    // Refresh every 10 seconds
    const interval = setInterval(fetchData, 10000);
    return () => clearInterval(interval);
  }, []);

  if (loading && !data) {
    return (
      <Card className="border-2 border-purple-500/30">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <TrendingUp className="h-5 w-5 text-purple-400" />
            Tool Breakdown
          </CardTitle>
          <CardDescription>Metrics by tool (last 24 hours)</CardDescription>
        </CardHeader>
        <CardContent className="flex items-center justify-center h-48">
          <LoadingSpinner />
        </CardContent>
      </Card>
    );
  }

  if (error || !data) {
    return (
      <Card className="border-2 border-purple-500/30">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <TrendingUp className="h-5 w-5 text-purple-400" />
            Tool Breakdown
          </CardTitle>
          <CardDescription>Metrics by tool (last 24 hours)</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="text-center text-red-400 py-8">
            {error || 'No data available'}
          </div>
        </CardContent>
      </Card>
    );
  }

  const calculatePercentage = (value: number, total: number): number => {
    if (total === 0) return 0;
    return Math.round((value / total) * 100);
  };

  return (
    <Card className="border-2 border-purple-500/30 bg-gradient-to-br from-purple-900/10 to-blue-900/10">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <TrendingUp className="h-5 w-5 text-purple-400" />
          Tool Breakdown
        </CardTitle>
        <CardDescription>
          Metrics by tool (last {data.hours} hours)
        </CardDescription>
      </CardHeader>
      <CardContent>
        {data.tools.length === 0 ? (
          <div className="text-center text-gray-400 py-8">
            No tool activity in the last {data.hours} hours
          </div>
        ) : (
          <div className="space-y-3">
            {data.tools.map((tool) => {
              const toolName = TOOL_NAMES[tool.tool_id] || tool.tool_id;
              const toolIcon = TOOL_ICONS[tool.tool_id] || <Activity className="h-5 w-5" />;
              const toolColor = TOOL_COLORS[tool.tool_id] || 'bg-gray-500/20 border-gray-500/50';
              const tokensPercent = calculatePercentage(tool.tokens_saved, data.total.tokens_saved);
              const compressionsPercent = calculatePercentage(tool.total_compressions, data.total.total_compressions);
              const isActive = tool.active_sessions > 0;

              return (
                <div
                  key={tool.tool_id}
                  className={`rounded-lg border p-4 ${toolColor} hover:border-opacity-100 transition-all`}
                >
                  {/* Tool Header */}
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center gap-3">
                      {toolIcon}
                      <div>
                        <h3 className="font-semibold text-white">{toolName}</h3>
                        <div className="flex items-center gap-2 text-xs text-gray-400">
                          <Circle
                            className={`h-2 w-2 ${
                              isActive ? 'fill-green-500 text-green-500' : 'fill-gray-500 text-gray-500'
                            }`}
                          />
                          <span>
                            {isActive
                              ? `${tool.active_sessions} active session${tool.active_sessions > 1 ? 's' : ''}`
                              : 'Inactive'}
                          </span>
                        </div>
                      </div>
                    </div>
                    <div className="text-right">
                      <div className="text-2xl font-bold text-purple-300">
                        {tokensPercent}%
                      </div>
                      <div className="text-xs text-gray-400">of tokens prevented</div>
                    </div>
                  </div>

                  {/* Metrics Grid */}
                  <div className="grid grid-cols-3 gap-4">
                    <div>
                      <div className="text-xs text-gray-400 mb-1">Tokens Prevented</div>
                      <div className="text-lg font-semibold text-white">
                        {formatNumber(tool.tokens_saved)}
                      </div>
                    </div>
                    <div>
                      <div className="text-xs text-gray-400 mb-1">Compressions</div>
                      <div className="text-lg font-semibold text-white">
                        {formatNumber(tool.total_compressions)}
                      </div>
                      <div className="text-xs text-gray-500">{compressionsPercent}%</div>
                    </div>
                    <div>
                      <div className="text-xs text-gray-400 mb-1">Embeddings</div>
                      <div className="text-lg font-semibold text-white">
                        {formatNumber(tool.total_embeddings)}
                      </div>
                    </div>
                  </div>

                  {/* Performance Indicators */}
                  {tool.sample_count > 0 && (
                    <div className="mt-3 pt-3 border-t border-gray-700/50 flex gap-4 text-xs">
                      {tool.avg_cache_hit_rate > 0 && (
                        <div>
                          <span className="text-gray-400">Cache Hit: </span>
                          <span className="text-green-400 font-semibold">
                            {(tool.avg_cache_hit_rate * 100).toFixed(1)}%
                          </span>
                        </div>
                      )}
                      {tool.avg_compression_ratio > 0 && (
                        <div>
                          <span className="text-gray-400">Compression: </span>
                          <span className="text-blue-400 font-semibold">
                            {tool.avg_compression_ratio.toFixed(1)}%
                          </span>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              );
            })}

            {/* Totals Summary */}
            <div className="mt-6 pt-4 border-t border-purple-500/30">
              <div className="grid grid-cols-4 gap-4">
                <div className="text-center">
                  <div className="text-xs text-gray-400 mb-1">Total Tokens Prevented</div>
                  <div className="text-xl font-bold text-purple-300">
                    {formatNumber(data.total.tokens_saved)}
                  </div>
                </div>
                <div className="text-center">
                  <div className="text-xs text-gray-400 mb-1">Total Compressions</div>
                  <div className="text-xl font-bold text-blue-300">
                    {formatNumber(data.total.total_compressions)}
                  </div>
                </div>
                <div className="text-center">
                  <div className="text-xs text-gray-400 mb-1">Total Embeddings</div>
                  <div className="text-xl font-bold text-green-300">
                    {formatNumber(data.total.total_embeddings)}
                  </div>
                </div>
                <div className="text-center">
                  <div className="text-xs text-gray-400 mb-1">Active Sessions</div>
                  <div className="text-xl font-bold text-orange-300">
                    {data.total.active_sessions}
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
