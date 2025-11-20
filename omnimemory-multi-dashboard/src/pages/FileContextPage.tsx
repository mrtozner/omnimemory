import { useState, useEffect, useMemo } from 'react';
import { FileText, Activity, ChevronDown, ChevronUp, Zap, Package } from 'lucide-react';
import { TierDistributionChart } from '../components/file-context/TierDistributionChart';
import { TokenSavingsChart } from '../components/file-context/TokenSavingsChart';
import { CachePerformanceCards } from '../components/file-context/CachePerformanceCards';
import { CrossToolUsageMatrix } from '../components/file-context/CrossToolUsageMatrix';
import { FileAccessHeatmap } from '../components/file-context/FileAccessHeatmap';
import { Card, CardContent } from '../components/shared/Card';
import { LoadingState } from '../components/shared/LoadingState';
import { ErrorState } from '../components/shared/ErrorState';
import { api } from '../services/api';
import type { ToolOperation } from '../services/api';
import { formatNumber } from '../lib/utils';

export function FileContextPage() {
  // Last refresh timestamp for header display
  const [lastRefresh, setLastRefresh] = useState(new Date());

  // Operations state
  const [operations, setOperations] = useState<ToolOperation[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [filters, setFilters] = useState({
    toolName: '',
    operationMode: '',
    searchQuery: '',
  });
  const [sortField, setSortField] = useState<'created_at' | 'tokens_prevented' | 'response_time_ms'>('created_at');
  const [sortDirection, setSortDirection] = useState<'asc' | 'desc'>('desc');
  const [currentPage, setCurrentPage] = useState(1);
  const [totalOperations, setTotalOperations] = useState(0);
  const itemsPerPage = 50;

  // Fetch operations
  const fetchOperations = async () => {
    try {
      setLoading(true);
      setError(null);

      const offset = (currentPage - 1) * itemsPerPage;

      const response = await api.getToolOperations(
        undefined, // session_id
        filters.toolName || undefined,
        filters.operationMode || undefined,
        undefined, // tool_id
        undefined, // start_date
        undefined, // end_date
        itemsPerPage,
        offset
      );

      setOperations(response.operations);
      setTotalOperations(response.total);
    } catch (err) {
      console.error('Failed to fetch operations:', err);
      setError(err instanceof Error ? err.message : 'Failed to load operations');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchOperations();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [filters.toolName, filters.operationMode, currentPage]);

  // Auto-refresh operations table every 30 seconds
  useEffect(() => {
    const interval = setInterval(() => {
      setLastRefresh(new Date());
      fetchOperations();
    }, 30000);

    return () => clearInterval(interval);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Client-side filtering and sorting
  const filteredAndSortedOps = useMemo(() => {
    let filtered = [...operations];

    // Client-side search filter (for file path)
    if (filters.searchQuery) {
      filtered = filtered.filter(op =>
        op.file_path?.toLowerCase().includes(filters.searchQuery.toLowerCase())
      );
    }

    // Sort
    filtered.sort((a, b) => {
      let aVal, bVal;
      if (sortField === 'created_at') {
        aVal = new Date(a.created_at).getTime();
        bVal = new Date(b.created_at).getTime();
      } else if (sortField === 'tokens_prevented') {
        aVal = a.tokens_prevented;
        bVal = b.tokens_prevented;
      } else {
        aVal = a.response_time_ms;
        bVal = b.response_time_ms;
      }

      return sortDirection === 'asc' ? aVal - bVal : bVal - aVal;
    });

    return filtered;
  }, [operations, filters.searchQuery, sortField, sortDirection]);

  // Format helpers
  const formatTime = (date: Date) => {
    return date.toLocaleTimeString('en-US', {
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit'
    });
  };

  const formatTimeAgo = (timestamp: string): string => {
    const now = new Date();
    const then = new Date(timestamp);
    const diffMs = now.getTime() - then.getTime();
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMins / 60);
    const diffDays = Math.floor(diffHours / 24);

    if (diffDays > 0) return `${diffDays}d ago`;
    if (diffHours > 0) return `${diffHours}h ago`;
    if (diffMins > 0) return `${diffMins}m ago`;
    return 'just now';
  };

  const formatCost = (tokens: number): string => {
    const costPerMillion = 3.0;
    const cost = (tokens / 1_000_000) * costPerMillion;
    return `$${cost.toFixed(4)}`;
  };

  return (
    <div className="space-y-4 md:space-y-6">
      {/* Header */}
      <div className="relative overflow-hidden rounded-2xl bg-gradient-to-br from-purple-600/20 via-blue-600/20 to-green-600/20 p-4 md:p-8 border border-purple-500/30">
        <div className="relative z-10">
          <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
            <div>
              <div className="flex items-center gap-3 mb-3">
                <FileText className="h-6 w-6 md:h-8 md:w-8 text-purple-400" />
                <h1 className="text-2xl md:text-4xl font-bold bg-gradient-to-r from-purple-400 via-blue-400 to-green-400 bg-clip-text text-transparent">
                  File Context Metrics
                </h1>
              </div>
              <p className="text-base md:text-xl text-gray-300">
                Real-time tier-based file context management and token savings
              </p>
            </div>
            <div className="flex items-center gap-2 text-sm text-gray-400">
              <Activity className="h-4 w-4 animate-pulse text-green-400" />
              <span>Last updated: {formatTime(lastRefresh)}</span>
            </div>
          </div>
        </div>

        {/* Background Decoration */}
        <div className="absolute top-0 right-0 w-64 h-64 bg-purple-500/10 rounded-full blur-3xl"></div>
        <div className="absolute bottom-0 left-0 w-64 h-64 bg-blue-500/10 rounded-full blur-3xl"></div>
      </div>

      {/* Filters */}
      <Card>
        <CardContent className="pt-4 md:pt-6">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3 md:gap-4">
            <div>
              <label className="text-xs md:text-sm text-muted-foreground mb-2 block">Tool Name</label>
              <select
                value={filters.toolName}
                onChange={(e) => setFilters(prev => ({ ...prev, toolName: e.target.value }))}
                className="w-full px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-blue-500 min-h-[44px]"
              >
                <option value="">All Tools</option>
                <option value="read">Read</option>
                <option value="search">Search</option>
              </select>
            </div>

            <div>
              <label className="text-xs md:text-sm text-muted-foreground mb-2 block">Operation Mode</label>
              <select
                value={filters.operationMode}
                onChange={(e) => setFilters(prev => ({ ...prev, operationMode: e.target.value }))}
                className="w-full px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-blue-500 min-h-[44px]"
              >
                <option value="">All Modes</option>
                <option value="full">Full</option>
                <option value="overview">Overview</option>
                <option value="symbol">Symbol</option>
                <option value="semantic">Semantic</option>
                <option value="tri_index">Tri-Index</option>
                <option value="references">References</option>
              </select>
            </div>

            <div>
              <label className="text-xs md:text-sm text-muted-foreground mb-2 block">Search File Path</label>
              <input
                type="text"
                placeholder="Search by file path..."
                value={filters.searchQuery}
                onChange={(e) => setFilters(prev => ({ ...prev, searchQuery: e.target.value }))}
                className="w-full px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-blue-500 min-h-[44px]"
              />
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Operations Table */}
      <div>
        <h2 className="text-xl md:text-2xl font-bold mb-3 md:mb-4 flex items-center gap-2">
          <Activity className="h-5 w-5 md:h-6 md:w-6 text-purple-400" />
          Recent Operations
        </h2>

        {loading ? (
          <LoadingState message="Loading operations..." />
        ) : error ? (
          <ErrorState
            error={error}
            context="Failed to Load Operations"
            onRetry={fetchOperations}
            compact={false}
          />
        ) : filteredAndSortedOps.length === 0 ? (
          <Card>
            <CardContent className="p-12 text-center">
              <p className="text-muted-foreground">No operations found</p>
            </CardContent>
          </Card>
        ) : (
          <>
            {/* Desktop Table */}
            <Card className="hidden md:block">
              <CardContent className="p-0">
                <div className="overflow-x-auto">
                  <table className="w-full">
                    <thead className="bg-gray-800/50 border-b border-gray-700">
                      <tr>
                        <th className="px-4 py-3 text-left text-sm font-medium text-muted-foreground">File Path</th>
                        <th className="px-4 py-3 text-left text-sm font-medium text-muted-foreground">Tool/Mode</th>
                        <th
                          className="px-4 py-3 text-left text-sm font-medium text-muted-foreground cursor-pointer hover:text-white"
                          onClick={() => {
                            if (sortField === 'created_at') {
                              setSortDirection(prev => prev === 'asc' ? 'desc' : 'asc');
                            } else {
                              setSortField('created_at');
                              setSortDirection('desc');
                            }
                          }}
                        >
                          <div className="flex items-center gap-1">
                            Timestamp
                            {sortField === 'created_at' && (
                              sortDirection === 'asc' ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />
                            )}
                          </div>
                        </th>
                        <th
                          className="px-4 py-3 text-right text-sm font-medium text-muted-foreground cursor-pointer hover:text-white"
                          onClick={() => {
                            if (sortField === 'tokens_prevented') {
                              setSortDirection(prev => prev === 'asc' ? 'desc' : 'asc');
                            } else {
                              setSortField('tokens_prevented');
                              setSortDirection('desc');
                            }
                          }}
                        >
                          <div className="flex items-center justify-end gap-1">
                            Tokens Prevented
                            {sortField === 'tokens_prevented' && (
                              sortDirection === 'asc' ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />
                            )}
                          </div>
                        </th>
                        <th
                          className="px-4 py-3 text-right text-sm font-medium text-muted-foreground cursor-pointer hover:text-white"
                          onClick={() => {
                            if (sortField === 'response_time_ms') {
                              setSortDirection(prev => prev === 'asc' ? 'desc' : 'asc');
                            } else {
                              setSortField('response_time_ms');
                              setSortDirection('desc');
                            }
                          }}
                        >
                          <div className="flex items-center justify-end gap-1">
                            Response Time
                            {sortField === 'response_time_ms' && (
                              sortDirection === 'asc' ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />
                            )}
                          </div>
                        </th>
                        <th className="px-4 py-3 text-right text-sm font-medium text-muted-foreground">Cost Saved</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-gray-800">
                      {filteredAndSortedOps.map((op) => (
                        <tr key={op.id} className="hover:bg-gray-800/30 transition-colors">
                          <td className="px-4 py-3">
                            <div className="flex items-center gap-2">
                              <FileText className="h-4 w-4 text-blue-400 flex-shrink-0" />
                              <span className="text-sm truncate max-w-md" title={op.file_path}>
                                {op.file_path || 'N/A'}
                              </span>
                            </div>
                          </td>
                          <td className="px-4 py-3">
                            <div>
                              <div className="text-sm font-medium capitalize">{op.tool_name}</div>
                              <div className="text-xs text-muted-foreground capitalize">{op.operation_mode}</div>
                            </div>
                          </td>
                          <td className="px-4 py-3 text-sm text-muted-foreground">
                            {formatTimeAgo(op.created_at)}
                          </td>
                          <td className="px-4 py-3 text-sm text-right font-medium text-green-400">
                            {formatNumber(op.tokens_prevented)}
                          </td>
                          <td className="px-4 py-3 text-sm text-right">
                            {op.response_time_ms.toFixed(1)}ms
                          </td>
                          <td className="px-4 py-3 text-sm text-right text-green-400">
                            {formatCost(op.tokens_prevented)}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </CardContent>
            </Card>

            {/* Mobile Cards */}
            <div className="md:hidden space-y-4">
              {filteredAndSortedOps.map((op) => (
                <Card key={op.id}>
                  <CardContent className="p-4">
                    <div className="space-y-3">
                      <div className="flex items-start gap-2">
                        <FileText className="h-4 w-4 text-blue-400 flex-shrink-0 mt-0.5" />
                        <div className="flex-1 min-w-0">
                          <div className="text-sm font-medium truncate" title={op.file_path}>
                            {op.file_path || 'N/A'}
                          </div>
                          <div className="text-xs text-muted-foreground mt-1">
                            {formatTimeAgo(op.created_at)}
                          </div>
                        </div>
                      </div>

                      <div className="flex items-center gap-4 text-xs">
                        <div>
                          <span className="text-muted-foreground">Tool: </span>
                          <span className="capitalize">{op.tool_name}</span>
                        </div>
                        <div>
                          <span className="text-muted-foreground">Mode: </span>
                          <span className="capitalize">{op.operation_mode}</span>
                        </div>
                      </div>

                      <div className="grid grid-cols-3 gap-2 pt-2 border-t border-gray-800">
                        <div>
                          <div className="text-xs text-muted-foreground">Tokens</div>
                          <div className="text-sm font-medium text-green-400">
                            {formatNumber(op.tokens_prevented)}
                          </div>
                        </div>
                        <div>
                          <div className="text-xs text-muted-foreground">Time</div>
                          <div className="text-sm">{op.response_time_ms.toFixed(1)}ms</div>
                        </div>
                        <div>
                          <div className="text-xs text-muted-foreground">Saved</div>
                          <div className="text-sm text-green-400">
                            {formatCost(op.tokens_prevented)}
                          </div>
                        </div>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>

            {/* Pagination */}
            {totalOperations > itemsPerPage && (
              <div className="flex flex-col sm:flex-row items-center justify-between gap-3">
                <div className="text-xs sm:text-sm text-muted-foreground text-center sm:text-left">
                  Showing {((currentPage - 1) * itemsPerPage) + 1} to {Math.min(currentPage * itemsPerPage, totalOperations)} of {totalOperations} operations
                </div>
                <div className="flex gap-2">
                  <button
                    onClick={() => setCurrentPage(prev => Math.max(1, prev - 1))}
                    disabled={currentPage === 1}
                    className="px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white text-sm disabled:opacity-50 disabled:cursor-not-allowed hover:bg-gray-700 transition-colors min-h-[44px]"
                  >
                    Previous
                  </button>
                  <button
                    onClick={() => setCurrentPage(prev => prev + 1)}
                    disabled={currentPage * itemsPerPage >= totalOperations}
                    className="px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white text-sm disabled:opacity-50 disabled:cursor-not-allowed hover:bg-gray-700 transition-colors min-h-[44px]"
                  >
                    Next
                  </button>
                </div>
              </div>
            )}
          </>
        )}
      </div>

      {/* Cache Performance Cards */}
      <div>
        <h2 className="text-xl md:text-2xl font-bold mb-3 md:mb-4 flex items-center gap-2">
          <Activity className="h-5 w-5 md:h-6 md:w-6 text-green-400" />
          Cache Performance
        </h2>
        <CachePerformanceCards />
      </div>

      {/* Token Savings */}
      <div>
        <TokenSavingsChart />
      </div>

      {/* Cross-Tool Usage Matrix */}
      <div>
        <CrossToolUsageMatrix />
      </div>

      {/* Tier Distribution */}
      <div>
        <TierDistributionChart />
      </div>

      {/* File Access Heatmap */}
      <div>
        <FileAccessHeatmap />
      </div>

      {/* Info Footer */}
      <div className="grid md:grid-cols-3 gap-4 md:gap-6">
        <div className="p-3 md:p-4 rounded-lg bg-green-500/10 border border-green-500/30">
          <div className="text-2xl mb-2">🌱</div>
          <h3 className="text-base md:text-lg font-semibold mb-2 text-green-300">FRESH Tier</h3>
          <p className="text-xs md:text-sm text-gray-400">
            Recently accessed files with high context value. Kept uncompressed for instant access.
          </p>
        </div>

        <div className="p-3 md:p-4 rounded-lg bg-blue-500/10 border border-blue-500/30">
          <Zap className="w-6 h-6 mb-2 text-blue-400" />
          <h3 className="text-base md:text-lg font-semibold mb-2 text-blue-300">RECENT Tier</h3>
          <p className="text-xs md:text-sm text-gray-400">
            Moderately used files. Lightly compressed to balance access speed and storage.
          </p>
        </div>

        <div className="p-3 md:p-4 rounded-lg bg-yellow-500/10 border border-yellow-500/30">
          <Package className="w-6 h-6 mb-2 text-yellow-400" />
          <h3 className="text-base md:text-lg font-semibold mb-2 text-yellow-300">AGING Tier</h3>
          <p className="text-xs md:text-sm text-gray-400">
            Infrequently accessed files. Heavily compressed to save storage with acceptable retrieval time.
          </p>
        </div>
      </div>
    </div>
  );
}
