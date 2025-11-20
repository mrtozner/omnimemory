import { useState, useEffect, useCallback } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '../shared/Card';
import { LoadingSpinner } from '../shared/LoadingSpinner';
import { usePolling } from '../../hooks/usePolling';
import { api, type Session, type HistoryDataPoint } from '../../services/api';
import { formatNumber, formatDuration } from '../../lib/utils';
import { Brain, Zap, Target, Activity } from 'lucide-react';
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  AreaChart,
  Area,
  ComposedChart,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import {
  CHART_COLORS,
  CHART_TOOLTIP_STYLE,
  CHART_AXIS_STYLE,
  CHART_GRID_STYLE,
  formatTimestamp
} from '../../utils/chartTheme';

interface ToolDetailTabProps {
  toolId: string;
  toolName: string;
}

export function ToolDetailTab({ toolId, toolName }: ToolDetailTabProps) {
  const [sessions, setSessions] = useState<Session[]>([]);
  const [historyData, setHistoryData] = useState<HistoryDataPoint[]>([]);
  const [loading, setLoading] = useState(true);

  // Real-time metrics via polling
  const { data: metrics } = usePolling(toolId, { interval: 5000 });

  const loadData = useCallback(async () => {
    try {
      setLoading(true);

      // Fetch active sessions for this tool
      const response = await fetch(
        `http://localhost:8003/sessions/active?tool_id=${toolId}`
      );

      if (!response.ok) {
        throw new Error('Failed to fetch sessions');
      }

      const data = await response.json();
      const sessionArray = data.sessions || [];

      // Set real data
      setSessions(sessionArray);

      // Load historical data
      try {
        const history = await api.getToolHistory(toolId, 24);
        setHistoryData(history);
      } catch (err) {
        console.error('Failed to load history:', err);
        setHistoryData([]);
      }

      setLoading(false);
    } catch (err) {
      console.error(`Failed to load ${toolName} data:`, err);
      setSessions([]);
      setHistoryData([]);
      setLoading(false);
    }
  }, [toolId, toolName]);

  useEffect(() => {
    loadData();
    const interval = setInterval(loadData, 30000);
    return () => clearInterval(interval);
  }, [loadData]);

  // Calculate aggregated metrics
  const metricsData = metrics as Record<string, unknown> | null;
  const aggregatedMetrics = {
    totalSessions: sessions.length,
    totalEmbeddings: (metricsData && typeof metricsData.total_embeddings === 'number') ? metricsData.total_embeddings : 0,
    totalCompressions: (metricsData && typeof metricsData.total_compressions === 'number') ? metricsData.total_compressions : 0,
    tokensSaved: (metricsData && typeof metricsData.total_tokens_saved === 'number') ? metricsData.total_tokens_saved : 0,
    cacheHitRate: (metricsData && typeof metricsData.avg_cache_hit_rate === 'number') ? metricsData.avg_cache_hit_rate : 0,
    compressionRatio: (metricsData && typeof metricsData.avg_compression_ratio === 'number') ? metricsData.avg_compression_ratio : 0,
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-96">
        <LoadingSpinner />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Metrics Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <Card className="bg-white/5 border border-white/10">
          <CardHeader>
            <CardTitle className="text-sm text-gray-400 flex items-center gap-2">
              <Activity className="h-4 w-4" />
              Active Sessions
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-3xl font-bold text-white">{aggregatedMetrics.totalSessions}</p>
          </CardContent>
        </Card>

        <Card className="bg-white/5 border border-white/10">
          <CardHeader>
            <CardTitle className="text-sm text-gray-400 flex items-center gap-2">
              <Zap className="h-4 w-4" />
              Tokens Saved
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-3xl font-bold text-white">
              {formatNumber(aggregatedMetrics.tokensSaved)}
            </p>
          </CardContent>
        </Card>

        <Card className="bg-white/5 border border-white/10">
          <CardHeader>
            <CardTitle className="text-sm text-gray-400 flex items-center gap-2">
              <Brain className="h-4 w-4" />
              Embeddings
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-3xl font-bold text-white">
              {formatNumber(aggregatedMetrics.totalEmbeddings)}
            </p>
          </CardContent>
        </Card>

        <Card className="bg-white/5 border border-white/10">
          <CardHeader>
            <CardTitle className="text-sm text-gray-400 flex items-center gap-2">
              <Target className="h-4 w-4" />
              Cache Hit Rate
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-3xl font-bold text-white">
              {((aggregatedMetrics.cacheHitRate || 0) * 100).toFixed(1)}%
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Tokens Saved Chart */}
        <Card>
          <CardHeader>
            <CardTitle>Tokens Saved (24h)</CardTitle>
          </CardHeader>
          <CardContent>
            {historyData.length > 0 ? (
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={historyData}>
                  <CartesianGrid {...CHART_GRID_STYLE} />
                  <XAxis dataKey="timestamp" tickFormatter={formatTimestamp} {...CHART_AXIS_STYLE} />
                  <YAxis {...CHART_AXIS_STYLE} />
                  <Tooltip
                    contentStyle={CHART_TOOLTIP_STYLE}
                    labelStyle={{ color: '#F3F4F6', marginBottom: '4px' }}
                    itemStyle={{ color: '#E5E7EB' }}
                    labelFormatter={formatTimestamp}
                  />
                  <Line
                    type="monotone"
                    dataKey="tokens_saved"
                    stroke={CHART_COLORS.primary}
                    strokeWidth={2}
                    dot={{ r: 3 }}
                    activeDot={{ r: 5 }}
                  />
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <div className="h-[300px] flex items-center justify-center text-gray-400">
                No data available
              </div>
            )}
          </CardContent>
        </Card>

        {/* Cache Hit Rate */}
        <Card>
          <CardHeader>
            <CardTitle>Cache Hit Rate (24h)</CardTitle>
          </CardHeader>
          <CardContent>
            {historyData.length > 0 ? (
              <ResponsiveContainer width="100%" height={300}>
                <AreaChart data={historyData}>
                  <CartesianGrid {...CHART_GRID_STYLE} />
                  <XAxis dataKey="timestamp" tickFormatter={formatTimestamp} {...CHART_AXIS_STYLE} />
                  <YAxis {...CHART_AXIS_STYLE} />
                  <Tooltip
                    contentStyle={CHART_TOOLTIP_STYLE}
                    labelStyle={{ color: '#F3F4F6', marginBottom: '4px' }}
                    itemStyle={{ color: '#E5E7EB' }}
                    labelFormatter={formatTimestamp}
                  />
                  <Area
                    type="monotone"
                    dataKey="cache_hit_rate"
                    stroke={CHART_COLORS.success}
                    fill={CHART_COLORS.success}
                    fillOpacity={0.3}
                  />
                </AreaChart>
              </ResponsiveContainer>
            ) : (
              <div className="h-[300px] flex items-center justify-center text-gray-400">
                No data available
              </div>
            )}
          </CardContent>
        </Card>

        {/* Compression Ratio */}
        <Card>
          <CardHeader>
            <CardTitle>Compression Ratio (24h)</CardTitle>
          </CardHeader>
          <CardContent>
            {historyData.length > 0 ? (
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={historyData}>
                  <CartesianGrid {...CHART_GRID_STYLE} />
                  <XAxis dataKey="timestamp" tickFormatter={formatTimestamp} {...CHART_AXIS_STYLE} />
                  <YAxis {...CHART_AXIS_STYLE} />
                  <Tooltip
                    contentStyle={CHART_TOOLTIP_STYLE}
                    labelStyle={{ color: '#F3F4F6', marginBottom: '4px' }}
                    itemStyle={{ color: '#E5E7EB' }}
                    labelFormatter={formatTimestamp}
                  />
                  <Bar dataKey="compression_ratio" fill={CHART_COLORS.warning} radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <div className="h-[300px] flex items-center justify-center text-gray-400">
                No data available
              </div>
            )}
          </CardContent>
        </Card>

        {/* Service Usage */}
        <Card>
          <CardHeader>
            <CardTitle>Service Usage (24h)</CardTitle>
          </CardHeader>
          <CardContent>
            {historyData.length > 0 ? (
              <ResponsiveContainer width="100%" height={300}>
                <ComposedChart data={historyData}>
                  <CartesianGrid {...CHART_GRID_STYLE} />
                  <XAxis dataKey="timestamp" tickFormatter={formatTimestamp} {...CHART_AXIS_STYLE} />
                  <YAxis {...CHART_AXIS_STYLE} />
                  <Tooltip
                    contentStyle={CHART_TOOLTIP_STYLE}
                    labelStyle={{ color: '#F3F4F6', marginBottom: '4px' }}
                    itemStyle={{ color: '#E5E7EB' }}
                    labelFormatter={formatTimestamp}
                  />
                  <Legend wrapperStyle={{ paddingTop: '20px' }} />
                  <Bar dataKey="total_embeddings" fill={CHART_COLORS.secondary} name="Embeddings" radius={[4, 4, 0, 0]} />
                  <Line
                    type="monotone"
                    dataKey="total_compressions"
                    stroke={CHART_COLORS.danger}
                    strokeWidth={2}
                    name="Compressions"
                    dot={{ r: 3 }}
                    activeDot={{ r: 5 }}
                  />
                </ComposedChart>
              </ResponsiveContainer>
            ) : (
              <div className="h-[300px] flex items-center justify-center text-gray-400">
                No data available
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Active Sessions Table */}
      <Card>
        <CardHeader>
          <CardTitle>Active Sessions</CardTitle>
        </CardHeader>
        <CardContent>
          {sessions.length === 0 ? (
            <div className="text-center py-12">
              <p className="text-gray-400">No active {toolName} sessions</p>
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-white/10">
                    <th className="text-left py-3 px-4 text-gray-400 font-medium">Session ID</th>
                    <th className="text-left py-3 px-4 text-gray-400 font-medium">Started</th>
                    <th className="text-left py-3 px-4 text-gray-400 font-medium">Duration</th>
                    <th className="text-right py-3 px-4 text-gray-400 font-medium">Tokens Saved</th>
                  </tr>
                </thead>
                <tbody>
                  {sessions.map(session => {
                    const duration = formatDuration(
                      (Date.now() - new Date(session.started_at).getTime()) / 1000
                    );
                    return (
                      <tr key={session.session_id} className="border-b border-white/5">
                        <td className="py-3 px-4 text-white font-mono text-sm">
                          {session.session_id.slice(0, 12)}...
                        </td>
                        <td className="py-3 px-4 text-gray-300">
                          {new Date(session.started_at).toLocaleString()}
                        </td>
                        <td className="py-3 px-4 text-gray-300">{duration}</td>
                        <td className="py-3 px-4 text-right text-purple-400 font-bold">
                          {formatNumber(session.tokens_saved || 0)}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
