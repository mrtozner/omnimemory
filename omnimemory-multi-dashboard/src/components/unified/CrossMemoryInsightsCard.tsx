import { Card, CardContent } from '../shared/Card';
import { GitBranch, TrendingUp, Database } from 'lucide-react';
import { useEffect, useState } from 'react';
import { api } from '../../services/api';
import type { InsightsResponse } from '../../types/unified';
import { ErrorState } from '../shared/ErrorState';
import { LoadingState } from '../shared/LoadingState';

export function CrossMemoryInsightsCard() {
  const [data, setData] = useState<InsightsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function fetchCrossMemoryInsights() {
      try {
        setError(null);

        const result = await api.getUnifiedInsights();
        setData(result as unknown as InsightsResponse);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load cross-memory insights');
        console.error('Failed to fetch cross-memory insights:', err);
      } finally {
        setLoading(false);
      }
    }

    fetchCrossMemoryInsights();
    const interval = setInterval(fetchCrossMemoryInsights, 15000); // Poll every 15s (slower for gateway calls)
    return () => clearInterval(interval);
  }, []);

  if (loading && !data) {
    return (
      <Card>
        <CardContent className="pt-6">
          <LoadingState message="Loading cross-memory insights..." compact />
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
            context="Cross-Memory Insights"
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
            No insights data available
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className="border-blue-500/30 bg-gradient-to-br from-blue-900/20 to-blue-800/10">
      <CardContent className="pt-6">
        {/* Header */}
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-3">
            <GitBranch className="h-6 w-6 text-blue-400" />
            <h3 className="text-xl font-bold text-blue-300">Cross-Memory Insights</h3>
          </div>
        </div>

        {/* KPIs Grid */}
        <div className="grid grid-cols-4 gap-4 mb-6">
          <div className="text-center p-3 rounded-lg bg-black/20">
            <div className="text-2xl font-bold text-blue-400">
              {(data?.pattern_library_size || 0).toLocaleString()}
            </div>
            <div className="text-xs text-gray-400 mt-1">Pattern Library</div>
          </div>
          <div className="text-center p-3 rounded-lg bg-black/20">
            <div className="text-2xl font-bold text-blue-400">
              {data?.patterns_detected_today || 0}
            </div>
            <div className="text-xs text-gray-400 mt-1">Detected Today</div>
          </div>
          <div className="text-center p-3 rounded-lg bg-black/20">
            <div className="text-2xl font-bold text-blue-400">
              {((data?.model_accuracy || 0) * 100).toFixed(1)}%
            </div>
            <div className="text-xs text-gray-400 mt-1">Model Accuracy</div>
          </div>
          <div className="text-center p-3 rounded-lg bg-black/20">
            <div className="text-2xl font-bold text-blue-400">
              {((data?.learning_rate || 0) * 100).toFixed(2)}%
            </div>
            <div className="text-xs text-gray-400 mt-1">Learning Rate</div>
          </div>
        </div>

        {/* Model Performance */}
        <div className="mb-6 p-4 rounded-lg bg-black/20 border border-blue-500/30">
          <div className="flex items-center gap-2 mb-3">
            <Database className="h-4 w-4 text-blue-400" />
            <span className="text-sm font-semibold text-gray-300">Model Performance</span>
          </div>
          <div className="space-y-3">
            <div>
              <div className="flex justify-between text-sm mb-1">
                <span className="text-gray-400">Accuracy</span>
                <span className="text-blue-400 font-semibold">
                  {((data?.model_accuracy || 0) * 100).toFixed(1)}%
                </span>
              </div>
              <div className="h-2 bg-gray-700 rounded-full overflow-hidden">
                <div
                  className="h-full bg-blue-400 transition-all"
                  style={{ width: `${(data?.model_accuracy || 0) * 100}%` }}
                />
              </div>
            </div>
          </div>
        </div>

        {/* Top Correlations */}
        <div>
          <h4 className="text-sm font-semibold text-gray-300 mb-3 flex items-center gap-2">
            <TrendingUp className="h-4 w-4 text-blue-400" />
            Top Pattern Correlations
          </h4>
          <div className="space-y-3">
            {(data?.top_correlations || []).map((correlation, index) => (
              <div
                key={index}
                className="p-3 rounded-lg bg-black/20 border border-gray-700/50"
              >
                <div className="flex items-start justify-between mb-2">
                  <div className="flex-1">
                    <div className="text-sm font-medium text-gray-200 mb-1">
                      {correlation.pattern}
                    </div>
                    <div className="text-xs text-gray-500">
                      {correlation.occurrences.toLocaleString()} occurrences
                    </div>
                  </div>
                  <div className="text-right">
                    <div
                      className={`text-lg font-bold ${
                        correlation.correlation_strength >= 0.9
                          ? 'text-green-400'
                          : correlation.correlation_strength >= 0.8
                          ? 'text-blue-400'
                          : 'text-yellow-400'
                      }`}
                    >
                      {(correlation.correlation_strength * 100).toFixed(0)}%
                    </div>
                    <div className="text-xs text-gray-500">strength</div>
                  </div>
                </div>
                {/* Strength Bar */}
                <div className="h-1.5 bg-gray-700 rounded-full overflow-hidden">
                  <div
                    className={`h-full transition-all ${
                      correlation.correlation_strength >= 0.9
                        ? 'bg-green-400'
                        : correlation.correlation_strength >= 0.8
                        ? 'bg-blue-400'
                        : 'bg-yellow-400'
                    }`}
                    style={{ width: `${correlation.correlation_strength * 100}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Info Footer */}
        <div className="mt-6 p-3 rounded-lg bg-blue-500/10 border border-blue-500/20">
          <p className="text-xs text-gray-400 leading-relaxed">
            <strong className="text-blue-300">Cross-Memory Patterns:</strong> These patterns
            emerge from analyzing both file access history (File Context Memory) and agent
            invocation sequences (Agent Memory). Strong correlations enable predictive
            prefetching and proactive suggestions.
          </p>
        </div>
      </CardContent>
    </Card>
  );
}
