import { Card, CardContent } from '../shared/Card';
import { Lightbulb, ThumbsUp, AlertCircle, TrendingUp } from 'lucide-react';
import { useEffect, useState } from 'react';
import { api } from '../../services/api';
import type { SuggestionsResponse } from '../../types/unified';
import { ErrorState } from '../shared/ErrorState';
import { LoadingState } from '../shared/LoadingState';

export function SuggestionMetricsCard() {
  const [data, setData] = useState<SuggestionsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function fetchSuggestionMetrics() {
      try {
        setError(null);

        const result = await api.getUnifiedSuggestions();
        setData(result as unknown as SuggestionsResponse);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load suggestion metrics');
        console.error('Failed to fetch suggestion metrics:', err);
      } finally {
        setLoading(false);
      }
    }

    fetchSuggestionMetrics();
    const interval = setInterval(fetchSuggestionMetrics, 15000); // Poll every 15s (slower for gateway calls)
    return () => clearInterval(interval);
  }, []);

  if (loading && !data) {
    return (
      <Card>
        <CardContent className="pt-6">
          <LoadingState message="Loading suggestion metrics..." compact />
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
            context="Suggestion Metrics"
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
            No suggestion data available
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className="border-green-500/30 bg-gradient-to-br from-green-900/20 to-green-800/10">
      <CardContent className="pt-6">
        {/* Header */}
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-3">
            <Lightbulb className="h-6 w-6 text-green-400" />
            <h3 className="text-xl font-bold text-green-300">Suggestion Metrics</h3>
          </div>
        </div>

        {/* KPIs Grid */}
        <div className="grid grid-cols-4 gap-4 mb-6">
          <div className="text-center p-3 rounded-lg bg-black/20">
            <div className="text-2xl font-bold text-green-400">
              {data?.total_suggestions_generated || 0}
            </div>
            <div className="text-xs text-gray-400 mt-1">Generated</div>
          </div>
          <div className="text-center p-3 rounded-lg bg-black/20">
            <div className="text-2xl font-bold text-green-400">
              {data?.suggestions_shown || 0}
            </div>
            <div className="text-xs text-gray-400 mt-1">Shown</div>
          </div>
          <div className="text-center p-3 rounded-lg bg-black/20">
            <div className="text-2xl font-bold text-green-400">
              {data?.suggestions_accepted || 0}
            </div>
            <div className="text-xs text-gray-400 mt-1">Accepted</div>
          </div>
          <div className="text-center p-3 rounded-lg bg-black/20">
            <div className="text-2xl font-bold text-green-400">
              {(data?.avg_generation_time_ms || 0).toFixed(2)}ms
            </div>
            <div className="text-xs text-gray-400 mt-1">Avg Time</div>
          </div>
        </div>

        {/* Acceptance & False Positive Rates */}
        <div className="grid grid-cols-2 gap-4 mb-6">
          <div className="p-4 rounded-lg bg-black/20 border border-green-500/30">
            <div className="flex items-center gap-2 mb-2">
              <ThumbsUp className="h-4 w-4 text-green-400" />
              <span className="text-sm font-semibold text-gray-300">Acceptance Rate</span>
            </div>
            <div className="text-3xl font-bold text-green-400">
              {((data?.acceptance_rate || 0) * 100).toFixed(1)}%
            </div>
            <div className="mt-2 h-2 bg-gray-700 rounded-full overflow-hidden">
              <div
                className="h-full bg-green-400 transition-all"
                style={{ width: `${(data?.acceptance_rate || 0) * 100}%` }}
              />
            </div>
          </div>

          <div className="p-4 rounded-lg bg-black/20 border border-yellow-500/30">
            <div className="flex items-center gap-2 mb-2">
              <AlertCircle className="h-4 w-4 text-yellow-400" />
              <span className="text-sm font-semibold text-gray-300">False Positive Rate</span>
            </div>
            <div className="text-3xl font-bold text-yellow-400">
              {((data?.false_positive_rate || 0) * 100).toFixed(1)}%
            </div>
            <div className="mt-2 h-2 bg-gray-700 rounded-full overflow-hidden">
              <div
                className="h-full bg-yellow-400 transition-all"
                style={{ width: `${(data?.false_positive_rate || 0) * 100}%` }}
              />
            </div>
          </div>
        </div>

        {/* Feedback by Type Table */}
        <div>
          <h4 className="text-sm font-semibold text-gray-300 mb-3 flex items-center gap-2">
            <TrendingUp className="h-4 w-4 text-green-400" />
            Feedback by Type
          </h4>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-gray-700">
                  <th className="text-left py-2 px-2 text-gray-400 font-medium">Type</th>
                  <th className="text-right py-2 px-2 text-gray-400 font-medium">Generated</th>
                  <th className="text-right py-2 px-2 text-gray-400 font-medium">Shown</th>
                  <th className="text-right py-2 px-2 text-gray-400 font-medium">Accepted</th>
                  <th className="text-right py-2 px-2 text-gray-400 font-medium">Rate</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(data?.feedback_by_type || {}).map(([type, metrics]) => (
                  <tr key={type} className="border-b border-gray-700/50">
                    <td className="py-2 px-2 text-gray-300 capitalize">
                      {type.replace('_', ' ')}
                    </td>
                    <td className="text-right py-2 px-2 text-gray-300">
                      {metrics.generated}
                    </td>
                    <td className="text-right py-2 px-2 text-gray-300">
                      {metrics.shown}
                    </td>
                    <td className="text-right py-2 px-2 text-gray-300">
                      {metrics.accepted}
                    </td>
                    <td className="text-right py-2 px-2">
                      <span
                        className={`font-semibold ${
                          metrics.acceptance_rate >= 0.7
                            ? 'text-green-400'
                            : metrics.acceptance_rate >= 0.5
                            ? 'text-yellow-400'
                            : 'text-red-400'
                        }`}
                      >
                        {(metrics.acceptance_rate * 100).toFixed(0)}%
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
