import { useState, useEffect } from 'react';
import { DollarSign, Activity, AlertCircle, BarChart3, Zap } from 'lucide-react';
import { CostSavingsCard } from '../components/api-savings/CostSavingsCard';
import { TokenPreventionChart } from '../components/api-savings/TokenPreventionChart';
import { ToolBreakdownCard } from '../components/api-savings/ToolBreakdownCard';
import { ModeUsageCard } from '../components/api-savings/ModeUsageCard';
import { ROICalculator } from '../components/api-savings/ROICalculator';
import { LoadingState } from '../components/shared/LoadingState';
import { ErrorState } from '../components/shared/ErrorState';
import { api } from '../services/api';
import type { APISavingsData, ToolBreakdownStats } from '../services/api';

type TimeRange = '1h' | '24h' | '7d' | '30d' | 'all';

export function APISavingsPage() {
  const [timeRange, setTimeRange] = useState<TimeRange>('24h');
  const [selectedTool, setSelectedTool] = useState<string>('');
  const [savingsData, setSavingsData] = useState<APISavingsData | null>(null);
  const [breakdownData, setBreakdownData] = useState<ToolBreakdownStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lastRefresh, setLastRefresh] = useState(new Date());
  const [availableTools, setAvailableTools] = useState<string[]>([]);

  // Fetch available tools on mount
  useEffect(() => {
    const fetchTools = async () => {
      try {
        const toolsData = await api.getInitializedTools();
        const toolIds = toolsData.tools.map((tool) => tool.id);
        setAvailableTools(toolIds);
      } catch (err) {
        console.error('Failed to fetch tools:', err);
        // Continue with empty tools list
      }
    };

    fetchTools();
  }, []);

  // Fetch data when time range or tool filter changes
  useEffect(() => {
    const fetchData = async () => {
      setLoading(true);
      setError(null);

      try {
        // Fetch API savings data
        const savings = await api.getAPISavings(
          timeRange,
          selectedTool || undefined
        );

        // Fetch tool breakdown data
        // Note: getToolBreakdownByMode doesn't support 'all', so use '30d' as fallback
        const breakdownTimeRange = timeRange === 'all' ? '30d' : timeRange;
        const breakdown = await api.getToolBreakdownByMode(
          breakdownTimeRange,
          selectedTool || undefined
        );

        // Use real data
        setSavingsData(savings);
        setBreakdownData(breakdown);

        setLastRefresh(new Date());
      } catch (err) {
        console.error('Failed to fetch API savings data:', err);
        setError(err instanceof Error ? err.message : 'Failed to load API savings data')
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, [timeRange, selectedTool]);

  const formatTime = (date: Date) => {
    return date.toLocaleTimeString('en-US', {
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
    });
  };

  const handleRetry = () => {
    setError(null);
    window.location.reload();
  };

  // Loading state
  if (loading && !savingsData) {
    return (
      <div className="space-y-6">
        {/* Header */}
        <div className="relative overflow-hidden rounded-2xl bg-gradient-to-br from-green-600/20 via-blue-600/20 to-purple-600/20 p-8 border border-green-500/30">
          <div className="relative z-10">
            <div className="flex items-center gap-3 mb-3">
              <DollarSign className="h-8 w-8 text-green-400" />
              <h1 className="text-4xl font-bold bg-gradient-to-r from-green-400 via-blue-400 to-purple-400 bg-clip-text text-transparent">
                API Savings & Memory
              </h1>
            </div>
            <p className="text-xl text-gray-300">
              Real-time cost savings and token prevention metrics
            </p>
          </div>
        </div>

        {/* Loading State */}
        <LoadingState message="Loading API savings data..." />
      </div>
    );
  }

  // Error state
  if (error) {
    return (
      <div className="space-y-6">
        {/* Header */}
        <div className="relative overflow-hidden rounded-2xl bg-gradient-to-br from-green-600/20 via-blue-600/20 to-purple-600/20 p-8 border border-green-500/30">
          <div className="relative z-10">
            <div className="flex items-center gap-3 mb-3">
              <DollarSign className="h-8 w-8 text-green-400" />
              <h1 className="text-4xl font-bold bg-gradient-to-r from-green-400 via-blue-400 to-purple-400 bg-clip-text text-transparent">
                API Savings & Memory
              </h1>
            </div>
            <p className="text-xl text-gray-300">
              Real-time cost savings and token prevention metrics
            </p>
          </div>
        </div>

        {/* Error State */}
        <ErrorState
          error={error}
          context="Failed to Load API Savings Data"
          onRetry={handleRetry}
        />
      </div>
    );
  }

  // No data state
  if (!savingsData || !breakdownData) {
    return (
      <div className="space-y-6">
        {/* Header */}
        <div className="relative overflow-hidden rounded-2xl bg-gradient-to-br from-green-600/20 via-blue-600/20 to-purple-600/20 p-8 border border-green-500/30">
          <div className="relative z-10">
            <div className="flex items-center gap-3 mb-3">
              <DollarSign className="h-8 w-8 text-green-400" />
              <h1 className="text-4xl font-bold bg-gradient-to-r from-green-400 via-blue-400 to-purple-400 bg-clip-text text-transparent">
                API Savings & Memory
              </h1>
            </div>
            <p className="text-xl text-gray-300">
              Real-time cost savings and token prevention metrics
            </p>
          </div>
        </div>

        <div className="flex items-center justify-center h-64">
          <div className="text-center">
            <AlertCircle className="h-12 w-12 text-yellow-400 mx-auto mb-4" />
            <p className="text-xl text-muted-foreground">
              No API savings data available for the selected time range
            </p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="relative overflow-hidden rounded-2xl bg-gradient-to-br from-green-600/20 via-blue-600/20 to-purple-600/20 p-8 border border-green-500/30">
        <div className="relative z-10">
          <div className="flex items-center justify-between flex-wrap gap-4">
            <div>
              <div className="flex items-center gap-3 mb-3">
                <DollarSign className="h-8 w-8 text-green-400" />
                <h1 className="text-4xl font-bold bg-gradient-to-r from-green-400 via-blue-400 to-purple-400 bg-clip-text text-transparent">
                  API Savings & Memory
                </h1>
              </div>
              <p className="text-xl text-gray-300">
                Real-time cost savings and token prevention metrics
              </p>
            </div>

            <div className="flex items-center gap-4">
              {/* Time Range Selector */}
              <div className="flex items-center gap-2">
                <label className="text-sm text-muted-foreground">
                  Time Range:
                </label>
                <select
                  value={timeRange}
                  onChange={(e) => setTimeRange(e.target.value as TimeRange)}
                  className="px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  <option value="1h">Last Hour</option>
                  <option value="24h">Last 24 Hours</option>
                  <option value="7d">Last 7 Days</option>
                  <option value="30d">Last 30 Days</option>
                  <option value="all">All Time</option>
                </select>
              </div>

              {/* Tool Filter */}
              <div className="flex items-center gap-2">
                <label className="text-sm text-muted-foreground">Tool:</label>
                <select
                  value={selectedTool}
                  onChange={(e) => setSelectedTool(e.target.value)}
                  className="px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  <option value="">All Tools</option>
                  {availableTools.map((toolId) => (
                    <option key={toolId} value={toolId}>
                      {toolId}
                    </option>
                  ))}
                </select>
              </div>

              {/* Last Updated */}
              <div className="flex items-center gap-2 text-sm text-gray-400">
                <Activity className="h-4 w-4 animate-pulse text-green-400" />
                <span>Last updated: {formatTime(lastRefresh)}</span>
              </div>
            </div>
          </div>
        </div>

        {/* Background Decoration */}
        <div className="absolute top-0 right-0 w-64 h-64 bg-green-500/10 rounded-full blur-3xl"></div>
        <div className="absolute bottom-0 left-0 w-64 h-64 bg-blue-500/10 rounded-full blur-3xl"></div>
      </div>

      {/* Cost Savings Overview */}
      <CostSavingsCard
        baselineCost={savingsData.api_cost_baseline}
        actualCost={savingsData.api_cost_actual}
        totalSaved={savingsData.total_cost_saved}
        savingsPercentage={savingsData.savings_percentage}
      />

      {/* Token Prevention Trends */}
      <TokenPreventionChart trends={savingsData.trends} />

      {/* Tool Breakdown and Mode Usage Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Tool Breakdown */}
        <ToolBreakdownCard breakdown={breakdownData} />

        {/* Mode Usage */}
        <ModeUsageCard breakdownByMode={savingsData.breakdown_by_mode} />
      </div>

      {/* ROI Calculator */}
      <ROICalculator data={savingsData} />

      {/* Info Footer */}
      <div className="grid md:grid-cols-3 gap-6">
        <div className="p-4 rounded-lg bg-green-500/10 border border-green-500/30">
          <DollarSign className="w-6 h-6 mb-2 text-green-400" />
          <h3 className="text-lg font-semibold mb-2 text-green-300">
            Immediate ROI
          </h3>
          <p className="text-sm text-gray-400">
            OmniMemory provides immediate cost savings from day 1. No upfront
            investment required - savings start with the first operation.
          </p>
        </div>

        <div className="p-4 rounded-lg bg-blue-500/10 border border-blue-500/30">
          <BarChart3 className="w-6 h-6 mb-2 text-blue-400" />
          <h3 className="text-lg font-semibold mb-2 text-blue-300">
            Token Prevention
          </h3>
          <p className="text-sm text-gray-400">
            By intelligently caching and compressing, OmniMemory prevents
            unnecessary tokens from being sent to expensive LLM APIs.
          </p>
        </div>

        <div className="p-4 rounded-lg bg-purple-500/10 border border-purple-500/30">
          <Zap className="w-6 h-6 mb-2 text-purple-400" />
          <h3 className="text-lg font-semibold mb-2 text-purple-300">
            Smart Operations
          </h3>
          <p className="text-sm text-gray-400">
            Different operation modes (full, overview, symbol, semantic)
            optimize for specific use cases, maximizing efficiency.
          </p>
        </div>
      </div>
    </div>
  );
}
