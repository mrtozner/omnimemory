import { Card, CardContent } from '../shared/Card';
import { Network, Activity, Target } from 'lucide-react';
import { useEffect, useState } from 'react';
import { api } from '../../services/api';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import type { OrchestrationResponse } from '../../types/unified';
import { ErrorState } from '../shared/ErrorState';
import { LoadingState } from '../shared/LoadingState';

export function OrchestrationMetricsCard() {
  const [data, setData] = useState<OrchestrationResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function fetchOrchestrationMetrics() {
      try {
        setError(null);

        const result = await api.getUnifiedOrchestration();
        setData(result as unknown as OrchestrationResponse);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load orchestration metrics');
        console.error('Failed to fetch orchestration metrics:', err);
      } finally {
        setLoading(false);
      }
    }

    fetchOrchestrationMetrics();
    const interval = setInterval(fetchOrchestrationMetrics, 15000); // Poll every 15s (slower for gateway calls)
    return () => clearInterval(interval);
  }, []);

  if (loading && !data) {
    return (
      <Card>
        <CardContent className="pt-6">
          <LoadingState message="Loading orchestration metrics..." compact />
        </CardContent>
      </Card>
    );
  }

  if (error && !data) {
    return (
      <Card>
        <CardContent className="pt-6">
          <ErrorState
            error={error}
            context="Orchestration Metrics"
            compact={true}
          />
        </CardContent>
      </Card>
    );
  }

  if (!data) {
    return (
      <Card>
        <CardContent className="pt-6">
          <div className="h-64 flex items-center justify-center text-muted-foreground">
            No orchestration data available
          </div>
        </CardContent>
      </Card>
    );
  }

  // Prepare bar chart data for query types with safe access
  const queryTypes = data?.query_types || {};
  const queryTypeData = Object.entries(queryTypes).map(([name, value]) => ({
    name: name.replace('_', ' '),
    count: value as number,
  }));

  // Prepare bar chart data for sources used with safe access
  const sourcesUsed = data?.sources_used || {};
  const sourcesData = [
    {
      name: 'File Context',
      count: sourcesUsed.file_context_only || 0,
      fill: '#3b82f6',
    },
    {
      name: 'Agent Memory',
      count: sourcesUsed.agent_memory_only || 0,
      fill: '#10b981',
    },
    {
      name: 'Cross-Memory',
      count: sourcesUsed.cross_memory || 0,
      fill: '#8b5cf6',
    },
  ];

  return (
    <Card className="border-purple-500/30 bg-gradient-to-br from-purple-900/20 to-purple-800/10">
      <CardContent className="pt-6">
        {/* Header */}
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-3">
            <Network className="h-6 w-6 text-purple-400" />
            <h3 className="text-xl font-bold text-purple-300">Orchestration Metrics</h3>
          </div>
        </div>

        {/* KPIs Grid */}
        <div className="grid grid-cols-3 gap-4 mb-6">
          <div className="text-center p-3 rounded-lg bg-black/20">
            <div className="text-2xl font-bold text-purple-400">
              {(data?.total_queries || 0).toLocaleString()}
            </div>
            <div className="text-xs text-gray-400 mt-1">Total Queries</div>
          </div>
          <div className="text-center p-3 rounded-lg bg-black/20">
            <div className="text-2xl font-bold text-purple-400">
              {(data?.avg_orchestration_overhead_ms || 0).toFixed(2)}ms
            </div>
            <div className="text-xs text-gray-400 mt-1">Avg Overhead</div>
          </div>
          <div className="text-center p-3 rounded-lg bg-black/20">
            <div className="text-2xl font-bold text-purple-400">
              {((data?.cache_hit_rate || 0) * 100).toFixed(1)}%
            </div>
            <div className="text-xs text-gray-400 mt-1">Cache Hit Rate</div>
          </div>
        </div>

        {/* Query Types Chart */}
        <div className="mb-6">
          <h4 className="text-sm font-semibold text-gray-300 mb-3 flex items-center gap-2">
            <Activity className="h-4 w-4 text-purple-400" />
            Query Type Distribution
          </h4>
          <ResponsiveContainer width="100%" height={180}>
            <BarChart data={queryTypeData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis
                dataKey="name"
                tick={{ fill: '#9ca3af', fontSize: 11 }}
                angle={-15}
                textAnchor="end"
                height={60}
              />
              <YAxis tick={{ fill: '#9ca3af', fontSize: 11 }} />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#1f2937',
                  border: '1px solid #374151',
                  borderRadius: '8px',
                }}
              />
              <Bar dataKey="count" fill="#a855f7" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Sources Used Chart */}
        <div>
          <h4 className="text-sm font-semibold text-gray-300 mb-3 flex items-center gap-2">
            <Target className="h-4 w-4 text-purple-400" />
            Memory Source Usage
          </h4>
          <ResponsiveContainer width="100%" height={180}>
            <BarChart data={sourcesData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis
                dataKey="name"
                tick={{ fill: '#9ca3af', fontSize: 11 }}
              />
              <YAxis tick={{ fill: '#9ca3af', fontSize: 11 }} />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#1f2937',
                  border: '1px solid #374151',
                  borderRadius: '8px',
                }}
              />
              <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                {sourcesData.map((entry, index) => (
                  <Bar key={`bar-${index}`} dataKey="count" fill={entry.fill} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </CardContent>
    </Card>
  );
}
