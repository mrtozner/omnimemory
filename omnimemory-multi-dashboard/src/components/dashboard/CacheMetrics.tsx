import React, { useEffect, useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '../shared/Card';
import { CircularProgressbar, buildStyles } from 'react-circular-progressbar';
import { api } from '../../services/api';
import { LoadingSpinner } from '../shared/LoadingSpinner';
import { AlertCircle } from 'lucide-react';
import 'react-circular-progressbar/dist/styles.css';

interface CacheStatsData {
  hot_cache: {
    hit_rate: number;
    size_mb: number;
    entries: number;
    avg_latency_ms: number;
    hits: number;
    misses: number;
  };
  file_hash_cache: {
    hit_rate: number;
    size_mb: number;
    entries: number;
    avg_latency_ms: number;
    hits: number;
    misses: number;
    disk_size_mb: number;
  };
  overall: {
    total_hit_rate: number;
    total_hits: number;
    total_misses: number;
    memory_saved_mb: number;
    tokens_prevented_from_api: number;
  };
}

export const CacheMetrics = React.memo(() => {
  const [stats, setStats] = useState<CacheStatsData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchCacheStats = async () => {
      try {
        setLoading(true);
        const data = await api.getCacheStats();
        setStats(data as unknown as CacheStatsData);
        setError(null);
      } catch (err) {
        console.error('Failed to fetch cache stats:', err);
        setError(err instanceof Error ? err.message : 'Failed to load cache statistics');
      } finally {
        setLoading(false);
      }
    };

    fetchCacheStats();

    // Refresh every 30 seconds
    const interval = setInterval(fetchCacheStats, 30000);
    return () => clearInterval(interval);
  }, []);

  if (loading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Cache Performance</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-center h-64">
            <LoadingSpinner />
          </div>
        </CardContent>
      </Card>
    );
  }

  if (error || !stats) {
    return (
      <Card className="border-red-500/50 bg-red-500/10">
        <CardHeader>
          <CardTitle>Cache Performance</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center gap-2 text-red-500">
            <AlertCircle className="h-5 w-5" />
            <span className="font-medium">{error || 'No data available'}</span>
          </div>
        </CardContent>
      </Card>
    );
  }

  // Helper function to get color based on hit rate
  const getHitRateColor = (rate: number): string => {
    if (rate >= 0.9) return '#10b981'; // Green - excellent
    if (rate >= 0.75) return '#f59e0b'; // Yellow - good
    return '#ef4444'; // Red - needs improvement
  };

  // Helper function to get latency status
  const getLatencyStatus = (latency: number): { text: string; color: string } => {
    if (latency < 1) return { text: 'excellent', color: 'text-green-500' };
    if (latency < 5) return { text: 'good', color: 'text-yellow-500' };
    return { text: 'slow', color: 'text-red-500' };
  };

  // Format large numbers
  const formatNumber = (num: number | null | undefined): string => {
    if (num == null) return '0';
    if (num >= 1000000) return `${(num / 1000000).toFixed(1)}M`;
    if (num >= 1000) return `${(num / 1000).toFixed(1)}K`;
    return num.toString();
  };

  const hotCacheHitRate = Math.round(stats.hot_cache.hit_rate * 100);
  const fileHashHitRate = Math.round(stats.file_hash_cache.hit_rate * 100);
  const overallHitRate = Math.round(stats.overall.total_hit_rate * 100);

  const hotLatencyStatus = getLatencyStatus(stats.hot_cache.avg_latency_ms);
  const fileHashLatencyStatus = getLatencyStatus(stats.file_hash_cache.avg_latency_ms);

  return (
    <Card>
      <CardHeader>
        <CardTitle>Cache Performance</CardTitle>
      </CardHeader>
      <CardContent>
        {/* Hit Rate Gauges */}
        <div className="grid grid-cols-2 gap-8 mb-8">
          {/* Hot Cache Gauge */}
          <div className="text-center">
            <div className="w-48 h-48 mx-auto mb-4">
              <CircularProgressbar
                value={hotCacheHitRate}
                text={`${hotCacheHitRate}%`}
                styles={buildStyles({
                  textSize: '16px',
                  pathColor: getHitRateColor(stats.hot_cache.hit_rate),
                  textColor: getHitRateColor(stats.hot_cache.hit_rate),
                  trailColor: 'hsl(var(--muted))',
                  backgroundColor: 'hsl(var(--background))',
                })}
              />
            </div>
            <p className="text-sm font-medium text-foreground">
              Hot Cache Hit Rate
            </p>
            <p className="text-xs text-muted-foreground mt-1">
              Target: &gt;90%
            </p>
          </div>

          {/* File Hash Cache Gauge */}
          <div className="text-center">
            <div className="w-48 h-48 mx-auto mb-4">
              <CircularProgressbar
                value={fileHashHitRate}
                text={`${fileHashHitRate}%`}
                styles={buildStyles({
                  textSize: '16px',
                  pathColor: getHitRateColor(stats.file_hash_cache.hit_rate),
                  textColor: getHitRateColor(stats.file_hash_cache.hit_rate),
                  trailColor: 'hsl(var(--muted))',
                  backgroundColor: 'hsl(var(--background))',
                })}
              />
            </div>
            <p className="text-sm font-medium text-foreground">
              File Hash Cache Hit Rate
            </p>
            <p className="text-xs text-muted-foreground mt-1">
              Target: &gt;75%
            </p>
          </div>
        </div>

        {/* Stats Grid */}
        <div className="grid grid-cols-2 gap-6">
          {/* Hot Cache Stats */}
          <div className="space-y-4">
            <h3 className="text-sm font-semibold text-foreground border-b border-gray-700 pb-2">
              Hot Cache Stats
            </h3>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-muted-foreground">Entries:</span>
                <span className="font-medium">{formatNumber(stats.hot_cache.entries ?? 0)}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Size:</span>
                <span className="font-medium">{stats.hot_cache.size_mb.toFixed(1)} MB</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Latency:</span>
                <span className={`font-medium ${hotLatencyStatus.color}`}>
                  {stats.hot_cache.avg_latency_ms.toFixed(3)}ms ({hotLatencyStatus.text})
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Hits / Misses:</span>
                <span className="font-medium">
                  {formatNumber(stats.hot_cache.hits ?? 0)} / {formatNumber(stats.hot_cache.misses ?? 0)}
                </span>
              </div>
            </div>
          </div>

          {/* File Hash Cache Stats */}
          <div className="space-y-4">
            <h3 className="text-sm font-semibold text-foreground border-b border-gray-700 pb-2">
              File Hash Cache Stats
            </h3>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-muted-foreground">Entries:</span>
                <span className="font-medium">{formatNumber(stats.file_hash_cache.entries ?? 0)}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Memory:</span>
                <span className="font-medium">{stats.file_hash_cache.size_mb.toFixed(1)} MB</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Disk:</span>
                <span className="font-medium">{stats.file_hash_cache.disk_size_mb.toFixed(1)} MB</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Latency:</span>
                <span className={`font-medium ${fileHashLatencyStatus.color}`}>
                  {stats.file_hash_cache.avg_latency_ms.toFixed(2)}ms ({fileHashLatencyStatus.text})
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Hits / Misses:</span>
                <span className="font-medium">
                  {formatNumber(stats.file_hash_cache.hits ?? 0)} / {formatNumber(stats.file_hash_cache.misses ?? 0)}
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* Overall Impact */}
        <div className="mt-6 pt-6 border-t border-gray-700">
          <h3 className="text-sm font-semibold text-foreground mb-4">Overall Impact</h3>
          <div className="grid grid-cols-3 gap-4 text-sm">
            <div className="text-center">
              <div className="text-2xl font-bold text-primary mb-1">{overallHitRate}%</div>
              <div className="text-xs text-muted-foreground">Combined Hit Rate</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-green-500 mb-1">
                {stats.overall.memory_saved_mb.toFixed(1)} MB
              </div>
              <div className="text-xs text-muted-foreground">Memory Saved</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-blue-500 mb-1">
                {formatNumber(stats.overall.tokens_prevented_from_api ?? 0)}
              </div>
              <div className="text-xs text-muted-foreground">Tokens Prevented</div>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
});

CacheMetrics.displayName = 'CacheMetrics';
