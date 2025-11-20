import { Card, CardContent } from '../shared/Card';
import { Target, Clock, HardDrive } from 'lucide-react';
import { useEffect, useState } from 'react';
import { api } from '../../services/api';
import type { CachePerformanceResponse } from '../../types/fileContext';
import { ErrorState } from '../shared/ErrorState';
import { LoadingState } from '../shared/LoadingState';

export function CachePerformanceCards() {
  const [data, setData] = useState<CachePerformanceResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function fetchCachePerformance() {
      try {
        setError(null);

        const result = await api.getCachePerformance() as unknown as CachePerformanceResponse;
        setData(result);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load cache performance');
        console.error('Failed to fetch cache performance:', err);
      } finally {
        setLoading(false);
      }
    }

    fetchCachePerformance();
    const interval = setInterval(fetchCachePerformance, 10000); // Poll every 10s
    return () => clearInterval(interval);
  }, []);

  if (loading && !data) {
    return (
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Card>
          <CardContent className="pt-6">
            <LoadingState message="Loading cache performance..." compact />
          </CardContent>
        </Card>
      </div>
    );
  }

  if (error && !data) {
    return (
      <ErrorState
        error={error}
        context="Cache Performance"
        compact={true}
      />
    );
  }

  if (!data) {
    return (
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Card>
          <CardContent className="pt-6">
            <div className="h-24 flex items-center justify-center text-muted-foreground">
              No data available
            </div>
          </CardContent>
        </Card>
      </div>
    );
  }

  // Extract values with safe defaults
  const cacheHitRate = data.cache_hit_rate ?? 0;
  const cacheHits = data.cache_hits ?? 0;
  const totalRequests = data.total_requests ?? 0;
  const avgLatencyMs = data.avg_latency_ms ?? 0;
  const storageUsedMb = data.storage_used_mb ?? 0;

  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
      {/* Cache Hit Rate */}
      <Card className="border-green-500/30 bg-gradient-to-br from-green-900/20 to-green-800/10">
        <CardContent className="pt-6">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-2">
              <Target className="h-5 w-5 text-green-400" />
              <span className="text-sm font-medium text-muted-foreground">Cache Hit Rate</span>
            </div>
          </div>
          <div className="space-y-2">
            <div className="text-4xl font-bold text-green-400">
              {cacheHitRate.toFixed(1)}%
            </div>
            <div className="text-xs text-muted-foreground">
              {cacheHits.toLocaleString()} hits / {totalRequests.toLocaleString()} requests
            </div>
          </div>
          {/* Progress Bar */}
          <div className="mt-4 h-2 bg-gray-700 rounded-full overflow-hidden">
            <div
              className="h-full bg-green-400 transition-all"
              style={{ width: `${Math.min(Math.max(cacheHitRate, 0), 100)}%` }}
            />
          </div>
        </CardContent>
      </Card>

      {/* Average Latency */}
      <Card className="border-blue-500/30 bg-gradient-to-br from-blue-900/20 to-blue-800/10">
        <CardContent className="pt-6">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-2">
              <Clock className="h-5 w-5 text-blue-400" />
              <span className="text-sm font-medium text-muted-foreground">Avg Latency</span>
            </div>
          </div>
          <div className="space-y-2">
            <div className="text-4xl font-bold text-blue-400">
              {avgLatencyMs.toFixed(1)}
              <span className="text-xl ml-1">ms</span>
            </div>
            <div className="text-xs text-muted-foreground">
              {avgLatencyMs < 50 ? 'Excellent' : avgLatencyMs < 100 ? 'Good' : 'Needs improvement'}
            </div>
          </div>
          {/* Latency Indicator */}
          <div className="mt-4 flex gap-1">
            {[1, 2, 3, 4, 5].map((bar) => (
              <div
                key={bar}
                className={`flex-1 h-2 rounded ${
                  bar <= Math.ceil(Math.max((100 - avgLatencyMs) / 20, 0))
                    ? 'bg-blue-400'
                    : 'bg-gray-700'
                }`}
              />
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Storage Used */}
      <Card className="border-purple-500/30 bg-gradient-to-br from-purple-900/20 to-purple-800/10">
        <CardContent className="pt-6">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-2">
              <HardDrive className="h-5 w-5 text-purple-400" />
              <span className="text-sm font-medium text-muted-foreground">Storage Used</span>
            </div>
          </div>
          <div className="space-y-2">
            <div className="text-4xl font-bold text-purple-400">
              {storageUsedMb.toFixed(1)}
              <span className="text-xl ml-1">MB</span>
            </div>
            <div className="text-xs text-muted-foreground">
              {storageUsedMb < 100
                ? 'Low usage'
                : storageUsedMb < 500
                ? 'Moderate usage'
                : 'High usage'}
            </div>
          </div>
          {/* Storage Bar */}
          <div className="mt-4 h-2 bg-gray-700 rounded-full overflow-hidden">
            <div
              className="h-full bg-purple-400 transition-all"
              style={{ width: `${Math.min((storageUsedMb / 1000) * 100, 100)}%` }}
            />
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
