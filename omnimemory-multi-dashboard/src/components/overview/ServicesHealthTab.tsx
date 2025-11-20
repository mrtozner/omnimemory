import { useState, useEffect, useCallback } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '../shared/Card';
import { LoadingSpinner } from '../shared/LoadingSpinner';
import { api, type ServiceHealth, type SessionStats, type AggregateMetrics } from '../../services/api';
import { CheckCircle, XCircle, AlertCircle, Server, Activity } from 'lucide-react';

interface ServiceHealthState {
  [port: number]: ServiceHealth;
}

// Extended SessionStats interface for this component
interface ExtendedSessionStats extends SessionStats {
  total_tools?: number;
  avg_session_duration_seconds?: number;
}

// Extended AggregateMetrics interface for this component
interface ExtendedAggregateMetrics extends Omit<AggregateMetrics, 'total_sessions'> {
  total_sessions?: number;
}

export function ServicesHealthTab() {
  const [serviceHealth, setServiceHealth] = useState<ServiceHealthState>({});
  const [sessionStats, setSessionStats] = useState<ExtendedSessionStats | null>(null);
  const [aggregates, setAggregates] = useState<ExtendedAggregateMetrics | null>(null);
  const [loading, setLoading] = useState(true);

  const fetchServiceHealth = useCallback(async (port: number): Promise<ServiceHealth> => {
    const startTime = Date.now();
    try {
      const data = await api.getServiceHealth(port);
      const responseTime = Date.now() - startTime;
      return {
        status: 'healthy',
        responseTime,
        uptime: (data.uptime as string) || 'N/A',
        data,
      };
    } catch {
      return {
        status: 'error',
        responseTime: Date.now() - startTime,
        data: null,
      };
    }
  }, []);

  const fetchAllServices = useCallback(async () => {
    setLoading(true);
    const ports = [8000, 8001, 8002, 8003];
    const results = await Promise.allSettled([
      fetchServiceHealth(8000),
      fetchServiceHealth(8001),
      fetchServiceHealth(8002),
      fetchServiceHealth(8003),
    ]);

    const healthMap: ServiceHealthState = {};
    ports.forEach((port, index) => {
      const result = results[index];
      if (result.status === 'fulfilled') {
        healthMap[port] = result.value;
      } else {
        healthMap[port] = {
          status: 'error',
          data: null,
        };
      }
    });

    setServiceHealth(healthMap);

    // Fetch session stats and aggregates
    try {
      const [stats, agg] = await Promise.all([
        api.getSessionStats(),
        api.getMetricsAggregates('24h')
      ]);
      setSessionStats(stats);
      setAggregates(agg);
    } catch (err) {
      console.error('Failed to fetch stats:', err);
    }

    setLoading(false);
  }, [fetchServiceHealth]);

  useEffect(() => {
    fetchAllServices();
    const interval = setInterval(fetchAllServices, 30000);
    return () => clearInterval(interval);
  }, [fetchAllServices]);

  const services = [
    { name: 'Embeddings Service', port: 8000, description: 'Semantic search and vector operations' },
    { name: 'Compression Service', port: 8001, description: 'Context compression and token optimization' },
    { name: 'Procedural Memory', port: 8002, description: 'Workflow learning and pattern recognition' },
    { name: 'Metrics API', port: 8003, description: 'Session tracking and analytics' },
  ];

  if (loading) {
    return (
      <div className="flex items-center justify-center h-96">
        <LoadingSpinner />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Service Health Overview */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {services.map(service => {
          const health = serviceHealth[service.port];
          const StatusIcon =
            health.status === 'healthy' ? CheckCircle :
            health.status === 'error' ? XCircle :
            AlertCircle;

          const statusColor =
            health.status === 'healthy' ? 'text-green-400' :
            health.status === 'error' ? 'text-red-400' :
            'text-yellow-400';

          const borderColor =
            health.status === 'healthy' ? 'border-green-500/30' :
            health.status === 'error' ? 'border-red-500/30' :
            'border-yellow-500/30';

          return (
            <Card key={service.port} className={`bg-white/5 border ${borderColor}`}>
              <CardHeader>
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <Server className={`h-6 w-6 ${statusColor}`} />
                    <div>
                      <CardTitle className="text-white">{service.name}</CardTitle>
                      <p className="text-sm text-gray-400 mt-1">{service.description}</p>
                    </div>
                  </div>
                  <StatusIcon className={`w-8 h-8 ${statusColor}`} />
                </div>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-400">Port</span>
                    <span className="text-white font-mono">{service.port}</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-400">Status</span>
                    <span className={`${statusColor} font-semibold capitalize`}>
                      {health.status}
                    </span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-400">Response Time</span>
                    <span className="text-white">
                      {health.responseTime ? `${health.responseTime}ms` : 'N/A'}
                    </span>
                  </div>
                  {health.uptime && (
                    <div className="flex justify-between items-center">
                      <span className="text-sm text-gray-400">Uptime</span>
                      <span className="text-white">{health.uptime}</span>
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>
          );
        })}
      </div>

      {/* System Status Summary */}
      <Card className="bg-gradient-to-br from-blue-900/20 to-purple-900/20 border-blue-500/30">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Server className="h-5 w-5 text-blue-400" />
            System Status Summary
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div>
              <p className="text-sm text-gray-400 mb-2">Healthy Services</p>
              <p className="text-3xl font-bold text-green-400">
                {Object.values(serviceHealth).filter(h => h.status === 'healthy').length} / {services.length}
              </p>
            </div>
            <div>
              <p className="text-sm text-gray-400 mb-2">Average Response Time</p>
              <p className="text-3xl font-bold text-white">
                {Math.round(
                  Object.values(serviceHealth)
                    .filter(h => h.responseTime)
                    .reduce((sum, h) => sum + (h.responseTime || 0), 0) /
                    Object.values(serviceHealth).filter(h => h.responseTime).length || 1
                )}ms
              </p>
            </div>
            <div>
              <p className="text-sm text-gray-400 mb-2">System Health</p>
              <p className="text-3xl font-bold text-blue-400">
                {Object.values(serviceHealth).filter(h => h.status === 'healthy').length === services.length
                  ? 'Excellent'
                  : Object.values(serviceHealth).filter(h => h.status === 'healthy').length >= services.length / 2
                  ? 'Good'
                  : 'Degraded'}
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* System Metrics */}
      {sessionStats && (
        <Card className="bg-gradient-to-br from-purple-900/20 to-blue-900/20 border-purple-500/30">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Activity className="h-5 w-5 text-purple-400" />
              System Statistics
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
              <div>
                <p className="text-sm text-gray-400 mb-1">Total Sessions</p>
                <p className="text-2xl font-bold text-white">
                  {sessionStats.total_sessions_lifetime.toLocaleString()}
                </p>
              </div>
              <div>
                <p className="text-sm text-gray-400 mb-1">Active Sessions</p>
                <p className="text-2xl font-bold text-green-400">
                  {sessionStats.active_sessions}
                </p>
              </div>
              <div>
                <p className="text-sm text-gray-400 mb-1">Tools Used</p>
                <p className="text-2xl font-bold text-blue-400">
                  {sessionStats.total_tools}
                </p>
              </div>
              <div>
                <p className="text-sm text-gray-400 mb-1">Avg Duration</p>
                <p className="text-2xl font-bold text-purple-400">
                  {Math.floor((sessionStats.avg_session_duration_seconds || 0) / 60)}m
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* 24h Aggregates */}
      {aggregates && (
        <Card className="bg-gradient-to-br from-green-900/20 to-blue-900/20 border-green-500/30">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Activity className="h-5 w-5 text-green-400" />
              Last 24 Hours
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
              <div>
                <p className="text-sm text-gray-400 mb-1">Sessions</p>
                <p className="text-2xl font-bold text-white">
                  {aggregates.total_sessions || 0}
                </p>
              </div>
              <div>
                <p className="text-sm text-gray-400 mb-1">Embeddings</p>
                <p className="text-2xl font-bold text-white">
                  {(aggregates.total_embeddings || 0).toLocaleString()}
                </p>
              </div>
              <div>
                <p className="text-sm text-gray-400 mb-1">Tokens Saved</p>
                <p className="text-2xl font-bold text-green-400">
                  {(aggregates.total_tokens_saved || 0).toLocaleString()}
                </p>
              </div>
              <div>
                <p className="text-sm text-gray-400 mb-1">Compression Ratio</p>
                <p className="text-2xl font-bold text-blue-400">
                  {((aggregates.avg_compression_ratio || 0) * 100).toFixed(0)}%
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Service Architecture Diagram */}
      <Card>
        <CardHeader>
          <CardTitle>Service Architecture</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div className="bg-gray-800/50 rounded-lg p-6">
              <h4 className="text-white font-semibold mb-4">Data Flow</h4>
              <div className="space-y-3 text-sm text-gray-300">
                <div className="flex items-center gap-3">
                  <div className="w-32 text-right text-gray-400">AI Tool</div>
                  <div className="text-gray-500">→</div>
                  <div className="flex-1 bg-blue-500/20 border border-blue-500/30 rounded px-3 py-2">
                    Metrics API (8003)
                  </div>
                </div>
                <div className="flex items-center gap-3">
                  <div className="w-32 text-right text-gray-400">Context Request</div>
                  <div className="text-gray-500">→</div>
                  <div className="flex-1 bg-purple-500/20 border border-purple-500/30 rounded px-3 py-2">
                    Embeddings Service (8000)
                  </div>
                </div>
                <div className="flex items-center gap-3">
                  <div className="w-32 text-right text-gray-400">Compression</div>
                  <div className="text-gray-500">→</div>
                  <div className="flex-1 bg-green-500/20 border border-green-500/30 rounded px-3 py-2">
                    Compression Service (8001)
                  </div>
                </div>
                <div className="flex items-center gap-3">
                  <div className="w-32 text-right text-gray-400">Workflow</div>
                  <div className="text-gray-500">→</div>
                  <div className="flex-1 bg-orange-500/20 border border-orange-500/30 rounded px-3 py-2">
                    Procedural Memory (8002)
                  </div>
                </div>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
