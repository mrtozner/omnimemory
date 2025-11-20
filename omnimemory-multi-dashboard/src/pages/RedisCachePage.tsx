import { useState, useEffect } from 'react';
import { Layers, Zap, HardDrive, User, Users, Clock, AlertCircle } from 'lucide-react';
import { Card, CardHeader, CardTitle, CardContent, CardDescription } from '../components/shared/Card';
import { MetricCard } from '../components/shared/MetricCard';

interface CacheTier {
  name: string;
  description: string;
  keys: number;
  ttl: string;
  scope: string;
}

interface CacheStats {
  status: string;
  l1_tier: CacheTier;
  l2_tier: CacheTier;
  l3_tier: CacheTier;
  team_tier: CacheTier;
  overall: {
    total_keys: number;
    cache_hits: number;
    cache_misses: number;
    hit_rate: number;
    memory_used_mb: number;
    memory_peak_mb: number;
  };
  health: {
    healthy: boolean;
    latency_ms: number;
    compression_enabled: boolean;
  };
  error?: string;
  message?: string;
}

export function RedisCachePage() {
  const [stats, setStats] = useState<CacheStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchStats = async () => {
    try {
      const response = await fetch('http://localhost:8003/api/cache/stats');
      if (!response.ok) {
        throw new Error('Failed to fetch cache stats');
      }
      const data = await response.json();
      setStats(data);
      setError(data.status === 'error' ? data.message : null);
    } catch (err) {
      console.error('Failed to fetch cache stats:', err);
      setError('Unable to connect to cache metrics service');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchStats();

    // Update every 5 seconds
    const interval = setInterval(fetchStats, 5000);

    return () => clearInterval(interval);
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-lg text-gray-600 dark:text-gray-400">Loading cache stats...</div>
      </div>
    );
  }

  const isHealthy = stats?.status === 'healthy';

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-gray-900 dark:text-white flex items-center gap-3">
          <Layers className="w-8 h-8 text-blue-600" />
          Cache Performance
        </h1>
        <p className="mt-2 text-gray-600 dark:text-gray-400">
          Multi-tier caching with L1 (user), L2 (team-shared), L3 (workflow)
        </p>
      </div>

      {/* Error Alert */}
      {error && (
        <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-4 flex items-start gap-3">
          <AlertCircle className="w-5 h-5 text-red-600 flex-shrink-0 mt-0.5" />
          <div>
            <h3 className="font-semibold text-red-900 dark:text-red-100">Cache Unavailable</h3>
            <p className="text-sm text-red-700 dark:text-red-300 mt-1">{error}</p>
          </div>
        </div>
      )}

      {/* Key Metrics */}
      {isHealthy && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          <MetricCard
            title="Overall Hit Rate"
            value={`${stats.overall?.hit_rate?.toFixed(1) || 0}%`}
            description={`${stats.overall?.cache_hits?.toLocaleString() || 0} hits / ${stats.overall?.cache_misses?.toLocaleString() || 0} misses`}
            icon={<Zap className="w-6 h-6 text-green-600" />}
            trend={stats.overall?.hit_rate && stats.overall.hit_rate > 80 ? 'up' : stats.overall?.hit_rate && stats.overall.hit_rate < 50 ? 'down' : 'neutral'}
          />

          <MetricCard
            title="L1: User Cache"
            value={stats.l1_tier?.keys?.toLocaleString() || '0'}
            description={`Personal • ${stats.l1_tier?.ttl || '1hr'} TTL`}
            icon={<User className="w-6 h-6 text-blue-600" />}
          />

          <MetricCard
            title="L2: Repository Cache"
            value={stats.l2_tier?.keys?.toLocaleString() || '0'}
            description={`Team-Shared • ${stats.l2_tier?.ttl || '7 days'} TTL`}
            icon={<Users className="w-6 h-6 text-purple-600" />}
          />

          <MetricCard
            title="L3: Workflow Cache"
            value={stats.l3_tier?.keys?.toLocaleString() || '0'}
            description={`Sessions • ${stats.l3_tier?.ttl || '30 days'} TTL`}
            icon={<Clock className="w-6 h-6 text-orange-600" />}
          />
        </div>
      )}

      {/* Detailed Stats */}
      {isHealthy && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Cache Tier Breakdown */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Layers className="w-5 h-5" />
                Cache Tiers
              </CardTitle>
              <CardDescription>Multi-tier caching architecture</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {/* L1 */}
              <div className="flex items-center justify-between py-3 border-b border-gray-200 dark:border-gray-700">
                <div className="flex items-center gap-3">
                  <div className="w-2 h-2 bg-blue-500 rounded-full" />
                  <div>
                    <div className="text-sm font-medium text-gray-900 dark:text-gray-100">
                      L1: User Cache
                    </div>
                    <p className="text-xs text-gray-600 dark:text-gray-400">
                      Personal • Fast • {stats.l1_tier?.ttl || '1hr'} TTL
                    </p>
                  </div>
                </div>
                <span className="font-mono text-sm font-semibold">
                  {stats.l1_tier?.keys?.toLocaleString() || 0} keys
                </span>
              </div>

              {/* L2 */}
              <div className="flex items-center justify-between py-3 border-b border-gray-200 dark:border-gray-700">
                <div className="flex items-center gap-3">
                  <div className="w-2 h-2 bg-purple-500 rounded-full" />
                  <div>
                    <div className="text-sm font-medium text-gray-900 dark:text-gray-100">
                      L2: Repository Cache
                    </div>
                    <p className="text-xs text-gray-600 dark:text-gray-400">
                      Team-Shared • {stats.l2_tier?.ttl || '7 day'} TTL
                    </p>
                  </div>
                </div>
                <span className="font-mono text-sm font-semibold">
                  {stats.l2_tier?.keys?.toLocaleString() || 0} keys
                </span>
              </div>

              {/* L3 */}
              <div className="flex items-center justify-between py-3 border-b border-gray-200 dark:border-gray-700">
                <div className="flex items-center gap-3">
                  <div className="w-2 h-2 bg-orange-500 rounded-full" />
                  <div>
                    <div className="text-sm font-medium text-gray-900 dark:text-gray-100">
                      L3: Workflow Cache
                    </div>
                    <p className="text-xs text-gray-600 dark:text-gray-400">
                      Long-term • {stats.l3_tier?.ttl || '30 day'} TTL
                    </p>
                  </div>
                </div>
                <span className="font-mono text-sm font-semibold">
                  {stats.l3_tier?.keys?.toLocaleString() || 0} keys
                </span>
              </div>

              {/* Total */}
              <div className="flex items-center justify-between py-3 pt-4 border-t-2 border-gray-300 dark:border-gray-600">
                <span className="font-medium text-gray-900 dark:text-gray-100">Total Keys</span>
                <span className="font-bold text-xl text-blue-600 dark:text-blue-400">
                  {stats.overall?.total_keys?.toLocaleString() || 0}
                </span>
              </div>
            </CardContent>
          </Card>

          {/* Memory & Performance */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <HardDrive className="w-5 h-5" />
                Memory & Performance
              </CardTitle>
              <CardDescription>Cache health and resource usage</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex justify-between items-center py-2 border-b border-gray-200 dark:border-gray-700">
                <span className="text-gray-600 dark:text-gray-400">Memory Used</span>
                <span className="font-semibold">{stats.overall?.memory_used_mb?.toFixed(2) || 0} MB</span>
              </div>
              <div className="flex justify-between items-center py-2 border-b border-gray-200 dark:border-gray-700">
                <span className="text-gray-600 dark:text-gray-400">Memory Peak</span>
                <span className="font-semibold">{stats.overall?.memory_peak_mb?.toFixed(2) || 0} MB</span>
              </div>
              <div className="flex justify-between items-center py-2 border-b border-gray-200 dark:border-gray-700">
                <span className="text-gray-600 dark:text-gray-400">Cache Hits</span>
                <span className="font-semibold text-green-600 dark:text-green-400">
                  {stats.overall?.cache_hits?.toLocaleString() || 0}
                </span>
              </div>
              <div className="flex justify-between items-center py-2 border-b border-gray-200 dark:border-gray-700">
                <span className="text-gray-600 dark:text-gray-400">Cache Misses</span>
                <span className="font-semibold text-red-600 dark:text-red-400">
                  {stats.overall?.cache_misses?.toLocaleString() || 0}
                </span>
              </div>
              <div className="flex justify-between items-center py-2 border-b border-gray-200 dark:border-gray-700">
                <span className="text-gray-600 dark:text-gray-400">Latency</span>
                <span className="font-semibold">{stats.health?.latency_ms?.toFixed(2) || 0} ms</span>
              </div>
              <div className="flex justify-between items-center py-2">
                <span className="text-gray-600 dark:text-gray-400">Compression</span>
                <span className={`font-semibold ${stats.health?.compression_enabled ? 'text-green-600 dark:text-green-400' : 'text-gray-500'}`}>
                  {stats.health?.compression_enabled ? 'Enabled' : 'Disabled'}
                </span>
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Performance Impact Note */}
      {isHealthy && (
        <Card className="bg-gradient-to-br from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 border-blue-200 dark:border-blue-800">
          <CardContent className="p-6">
            <div className="flex items-start gap-3">
              <Zap className="w-6 h-6 text-blue-600 flex-shrink-0 mt-1" />
              <div>
                <h3 className="font-semibold text-blue-900 dark:text-blue-100 mb-2">
                  Multi-Tier Performance
                </h3>
                <p className="text-sm text-blue-700 dark:text-blue-300">
                  <strong>L1 (User)</strong>: Personal cache for instant access ({stats.l1_tier?.ttl || '1hr'})
                  <br />
                  <strong>L2 (Repository)</strong>: Team-shared cache saves 80-90% tokens ({stats.l2_tier?.ttl || '7 days'})
                  <br />
                  <strong>L3 (Workflow)</strong>: Long-term session continuity ({stats.l3_tier?.ttl || '30 days'})
                  <br /><br />
                  Current: <strong>{stats.overall?.hit_rate?.toFixed(1) || 0}%</strong> hit rate with{' '}
                  <strong>{stats.overall?.total_keys?.toLocaleString() || 0}</strong> cached items using{' '}
                  <strong>{stats.overall?.memory_used_mb?.toFixed(2) || 0} MB</strong> of memory.
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
