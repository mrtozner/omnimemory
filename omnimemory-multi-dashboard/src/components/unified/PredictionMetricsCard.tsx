import { Card, CardContent } from '../shared/Card';
import { Brain, TrendingUp, Zap } from 'lucide-react';
import { useEffect, useState } from 'react';
import { api } from '../../services/api';
import { PieChart, Pie, Cell, ResponsiveContainer, Legend, Tooltip } from 'recharts';
import type { PredictionsResponse } from '../../types/unified';
import { ErrorState } from '../shared/ErrorState';
import { LoadingState } from '../shared/LoadingState';

const COLORS = {
  file_context: '#3b82f6', // Blue
  agent_memory: '#10b981', // Green
  cross_memory: '#8b5cf6', // Purple
};

export function PredictionMetricsCard() {
  const [data, setData] = useState<PredictionsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function fetchPredictionMetrics() {
      try {
        setError(null);

        const result = await api.getUnifiedPredictions();
        setData(result as unknown as PredictionsResponse);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load prediction metrics');
        console.error('Failed to fetch prediction metrics:', err);
      } finally {
        setLoading(false);
      }
    }

    fetchPredictionMetrics();
    const interval = setInterval(fetchPredictionMetrics, 15000); // Poll every 15s (slower for gateway calls)
    return () => clearInterval(interval);
  }, []);

  if (loading && !data) {
    return (
      <Card>
        <CardContent className="pt-6">
          <LoadingState message="Loading prediction metrics..." compact />
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
            context="Prediction Metrics"
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
            No prediction data available
          </div>
        </CardContent>
      </Card>
    );
  }

  // Prepare pie chart data with safe access
  const sourceContributions = data?.source_contributions || {};
  const pieData = [
    {
      name: 'File Context',
      value: (sourceContributions.file_context || 0) * 100,
      color: COLORS.file_context,
    },
    {
      name: 'Agent Memory',
      value: (sourceContributions.agent_memory || 0) * 100,
      color: COLORS.agent_memory,
    },
    {
      name: 'Cross-Memory',
      value: (sourceContributions.cross_memory || 0) * 100,
      color: COLORS.cross_memory,
    },
  ];

  return (
    <Card className="border-orange-500/30 bg-gradient-to-br from-orange-900/20 to-orange-800/10">
      <CardContent className="pt-6">
        {/* Header */}
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-3">
            <Brain className="h-6 w-6 text-orange-400" />
            <h3 className="text-xl font-bold text-orange-300">Prediction Metrics</h3>
          </div>
        </div>

        {/* KPIs Grid */}
        <div className="grid grid-cols-3 gap-4 mb-6">
          <div className="text-center p-3 rounded-lg bg-black/20">
            <div className="text-2xl font-bold text-orange-400">
              {(data?.metrics?.total_predictions || 0).toLocaleString()}
            </div>
            <div className="text-xs text-gray-400 mt-1">Total Predictions</div>
          </div>
          <div className="text-center p-3 rounded-lg bg-black/20">
            <div className="text-2xl font-bold text-orange-400">
              {((data?.metrics?.avg_confidence || 0) * 100).toFixed(1)}%
            </div>
            <div className="text-xs text-gray-400 mt-1">Avg Confidence</div>
          </div>
          <div className="text-center p-3 rounded-lg bg-black/20">
            <div className="text-2xl font-bold text-orange-400">
              {(data?.metrics?.avg_execution_time_ms || 0).toFixed(2)}ms
            </div>
            <div className="text-xs text-gray-400 mt-1">Avg Time</div>
          </div>
        </div>

        {/* Source Contributions Pie Chart */}
        <div className="mb-4">
          <h4 className="text-sm font-semibold text-gray-300 mb-3 flex items-center gap-2">
            <TrendingUp className="h-4 w-4 text-orange-400" />
            Source Contributions
          </h4>
          <ResponsiveContainer width="100%" height={200}>
            <PieChart>
              <Pie
                data={pieData}
                dataKey="value"
                nameKey="name"
                cx="50%"
                cy="50%"
                outerRadius={70}
                label={(entry) => `${entry.value.toFixed(1)}%`}
                labelLine={false}
              >
                {pieData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip
                formatter={(value: number) => `${value.toFixed(1)}%`}
                contentStyle={{
                  backgroundColor: '#1f2937',
                  border: '1px solid #374151',
                  borderRadius: '8px',
                }}
              />
              <Legend
                verticalAlign="bottom"
                height={36}
                iconType="circle"
                formatter={(value) => (
                  <span className="text-sm text-gray-300">{value}</span>
                )}
              />
            </PieChart>
          </ResponsiveContainer>
        </div>

        {/* Recent Predictions */}
        <div>
          <h4 className="text-sm font-semibold text-gray-300 mb-3 flex items-center gap-2">
            <Zap className="h-4 w-4 text-orange-400" />
            Recent Predictions
          </h4>
          <div className="space-y-2">
            {(data?.predictions || []).map((pred, index) => (
              <div
                key={index}
                className="flex items-center justify-between p-2 rounded bg-black/20"
              >
                <div className="flex items-center gap-2">
                  <div
                    className="w-2 h-2 rounded-full"
                    style={{
                      backgroundColor: COLORS[pred.source],
                    }}
                  />
                  <span className="text-sm text-gray-300">{pred.predicted_item}</span>
                  <span className="text-xs text-gray-500">({pred.prediction_type})</span>
                </div>
                <div className="text-sm font-semibold text-orange-400">
                  {(pred.confidence * 100).toFixed(0)}%
                </div>
              </div>
            ))}
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
