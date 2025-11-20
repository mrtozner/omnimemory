import { useEffect, useState, useMemo, useCallback } from 'react';
import { useSearchParams } from 'react-router-dom';
import { Card, CardHeader, CardTitle, CardContent, CardDescription } from '../components/shared/Card';
import { LoadingSpinner } from '../components/shared/LoadingSpinner';
import { LoadingState } from '../components/shared/LoadingState';
import { ErrorState } from '../components/shared/ErrorState';
import { ToolCard } from '../components/investor/ToolCard';
import { ToolBreakdownCard } from '../components/dashboard/ToolBreakdownCard';
import { ToolsOverviewCard } from '../components/dashboard/ToolsOverviewCard';
import { CostBreakdownChart } from '../components/investor/CostBreakdownChart';
import { ActivityStream } from '../components/investor/ActivityStream';
import { api, type Session, type ToolBreakdownStats, type APISavingsData, type SessionStats } from '../services/api';
import { formatNumber } from '../lib/utils';
import { Brain, Zap, DollarSign, Activity, TrendingUp, Sparkles, Clock, FileText, Search, Globe, Bot, Settings as SettingsIcon, Link, Flame } from 'lucide-react';
import { getToolDisplay, getToolColor } from '../utils/toolMapping';
import { Tabs, TabsList, TabsTrigger, TabsContent } from '../components/shared/Tabs';
import { ToolDetailTab } from '../components/overview/ToolDetailTab';
import { ServicesHealthTab } from '../components/overview/ServicesHealthTab';

// Metrics interface for tool cards
interface ToolMetrics {
  total_embeddings: number;
  total_compressions: number;
  tokens_saved: number;
  cache_hit_rate: number;
  compression_ratio: number;
  avg_compression_ratio: number;
  quality_score: number;
  avg_quality_score: number;
  pattern_count: number;
  graph_edges: number;
  total_successes: number;
  total_failures: number;
  total_original_tokens: number;
  total_compressed_tokens: number;
  cache_hits: number;
  tokens_processed: number;
  prediction_accuracy: number;
}

// Format timestamp in user's local time
function formatLocalTime(timestamp: string): string {
  const date = new Date(timestamp);
  return date.toLocaleString('en-US', {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
    hour12: false,
  });
}

const TOOLS = ['claude-code', 'codex'];

export function OverviewPage() {
  const [searchParams, setSearchParams] = useSearchParams();
  const initialTab = searchParams.get('tab') || 'all';
  const [selectedTab, setSelectedTab] = useState(initialTab);

  const [loading, setLoading] = useState(true);
  const [allActiveSessions, setAllActiveSessions] = useState<Session[]>([]);
  const [claudeSessions, setClaudeSessions] = useState<Session[]>([]);
  const [codexSessions, setCodexSessions] = useState<Session[]>([]);
  const [historicalMetrics, setHistoricalMetrics] = useState<{
    claude: {
      embeddings: number;
      compressions: number;
      tokensSaved: number;
      cacheHitRate: number;
      compressionRatio: number;
      qualityScore: number;
      patternCount: number;
    };
    codex: {
      embeddings: number;
      compressions: number;
      tokensSaved: number;
      cacheHitRate: number;
      compressionRatio: number;
      qualityScore: number;
      patternCount: number;
    };
  } | null>(null);
  const [sessionStats, setSessionStats] = useState<SessionStats | null>(null);

  // Tool Usage & Operations state
  const [toolBreakdownData, setToolBreakdownData] = useState<ToolBreakdownStats | null>(null);
  const [apiSavingsData, setApiSavingsData] = useState<APISavingsData | null>(null);
  const [toolDataLoading, setToolDataLoading] = useState(true);
  const [toolDataError, setToolDataError] = useState<string | null>(null);
  const [toolTimeRange, setToolTimeRange] = useState<'1h' | '24h' | '7d' | '30d'>('24h');

  // Memoize initialize function to prevent unnecessary re-renders
  const initialize = useCallback(async () => {
    try {
      setLoading(true);

      // Load real data from API
      const activeSessions = await api.getActiveSessions();
      const sessionArray = Array.isArray(activeSessions)
        ? activeSessions
        : (activeSessions as { sessions?: Session[] })?.sessions || [];

      // Fetch aggregate metrics (last 24 hours)
      let aggregates = null;
      try {
        aggregates = await api.getAggregates(24);
      } catch (err) {
        console.error('Failed to load aggregate metrics:', err);
      }

      // Fetch session statistics
      try {
        const stats = await api.getSessionStats();
        setSessionStats(stats);
      } catch (err) {
        console.error('Failed to load session stats:', err);
      }

      // Store ALL active sessions (including cursor, etc.)
      setAllActiveSessions(sessionArray);

      // Also filter for tool-specific cards
      const claudeData = sessionArray.filter((s: Session) => s.tool_id === 'claude-code');
      const codexData = sessionArray.filter((s: Session) => s.tool_id === 'codex');

      setClaudeSessions(claudeData);
      setCodexSessions(codexData);

      // Use aggregate metrics as fallback when no active sessions
      // These represent the last 24 hours of actual usage
      setHistoricalMetrics({
        claude: {
          embeddings: aggregates?.total_embeddings || 0,
          compressions: aggregates?.total_compressions || 0,
          tokensSaved: aggregates?.total_tokens_saved || 0,
          cacheHitRate: aggregates?.avg_cache_hit_rate || 0,
          compressionRatio: aggregates?.avg_compression_ratio || 0,
          qualityScore: 0, // Not in aggregates API
          patternCount: 0, // Not in aggregates API
        },
        codex: {
          embeddings: 0, // Aggregates API is global, not per-tool
          compressions: 0,
          tokensSaved: 0,
          cacheHitRate: 0,
          compressionRatio: 0,
          qualityScore: 0,
          patternCount: 0,
        },
      });

      setLoading(false);
    } catch (err) {
      console.error('Failed to initialize:', err);
      setLoading(false);
    }
  }, []);

  // Fetch tool usage data
  const fetchToolUsageData = useCallback(async () => {
    try {
      setToolDataLoading(true);
      const [breakdown, savings] = await Promise.all([
        api.getToolBreakdownByMode(toolTimeRange),
        api.getAPISavings(toolTimeRange)
      ]);
      setToolBreakdownData(breakdown);
      setApiSavingsData(savings);
      setToolDataError(null);
    } catch (err) {
      console.error('Failed to fetch tool usage data:', err);
      setToolDataError(err instanceof Error ? err.message : 'Failed to load tool usage data');
    } finally {
      setToolDataLoading(false);
    }
  }, [toolTimeRange]);

  useEffect(() => {
    initialize();

    // Poll for updates every 10 seconds
    const interval = setInterval(initialize, 10000);
    return () => clearInterval(interval);
  }, [initialize]);

  useEffect(() => {
    fetchToolUsageData();
  }, [fetchToolUsageData]);

  // Update URL when tab changes
  const handleTabChange = (value: string) => {
    setSelectedTab(value);
    setSearchParams(value === 'all' ? {} : { tab: value });
  };

  // Aggregate metrics from all sessions
  const allSessions = useMemo(() => {
    return [...claudeSessions, ...codexSessions];
  }, [claudeSessions, codexSessions]);

  const aggregatedMetrics = useMemo(() => {
    // If we have active sessions, use session data
    if (allSessions && allSessions.length > 0) {
      const totals = allSessions.reduce(
        (acc, session) => ({
          totalCompressions: acc.totalCompressions + (session.total_compressions || 0),
          totalEmbeddings: acc.totalEmbeddings + (session.total_embeddings || 0),
          tokensSaved: acc.tokensSaved + (session.tokens_saved || 0),
        }),
        { totalCompressions: 0, totalEmbeddings: 0, tokensSaved: 0 }
      );

      const COST_PER_MILLION_TOKENS = 3.0;
      const estimatedSavings = (totals.tokensSaved / 1_000_000) * COST_PER_MILLION_TOKENS;

      return { ...totals, estimatedSavings };
    }

    // Otherwise, use historical data (last 24 hours)
    if (historicalMetrics) {
      const totals = {
        totalCompressions: historicalMetrics.claude.compressions + historicalMetrics.codex.compressions,
        totalEmbeddings: historicalMetrics.claude.embeddings + historicalMetrics.codex.embeddings,
        tokensSaved: historicalMetrics.claude.tokensSaved + historicalMetrics.codex.tokensSaved,
      };

      const COST_PER_MILLION_TOKENS = 3.0;
      const estimatedSavings = (totals.tokensSaved / 1_000_000) * COST_PER_MILLION_TOKENS;

      return { ...totals, estimatedSavings };
    }

    return { totalCompressions: 0, totalEmbeddings: 0, tokensSaved: 0, estimatedSavings: 0 };
  }, [allSessions, historicalMetrics]);

  // Filter sessions for Claude Code and calculate metrics
  const claudeCodeSessions = useMemo(() => {
    return allSessions.filter(s => s.tool_id === 'claude-code');
  }, [allSessions]);

  const claudeCodeMetrics = useMemo(() => {
    // Use session data if available
    if (claudeCodeSessions.length > 0) {
      const totals = claudeCodeSessions.reduce(
        (acc, session) => ({
          totalCompressions: acc.totalCompressions + (session.total_compressions || 0),
          totalEmbeddings: acc.totalEmbeddings + (session.total_embeddings || 0),
          tokensSaved: acc.tokensSaved + (session.tokens_saved || 0),
        }),
        { totalCompressions: 0, totalEmbeddings: 0, tokensSaved: 0 }
      );

      const COST_PER_MILLION_TOKENS = 3.0;
      const estimatedSavings = (totals.tokensSaved / 1_000_000) * COST_PER_MILLION_TOKENS;

      return {
        ...totals,
        estimatedSavings,
        cacheHitRate: 0,
        compressionRatio: 0,
        qualityScore: 0,
        patternCount: 0,
      };
    }

    // Fall back to historical data
    if (historicalMetrics) {
      const totals = {
        totalCompressions: historicalMetrics.claude.compressions,
        totalEmbeddings: historicalMetrics.claude.embeddings,
        tokensSaved: historicalMetrics.claude.tokensSaved,
        cacheHitRate: historicalMetrics.claude.cacheHitRate,
        compressionRatio: historicalMetrics.claude.compressionRatio,
        qualityScore: historicalMetrics.claude.qualityScore,
        patternCount: historicalMetrics.claude.patternCount,
      };

      const COST_PER_MILLION_TOKENS = 3.0;
      const estimatedSavings = (totals.tokensSaved / 1_000_000) * COST_PER_MILLION_TOKENS;

      return { ...totals, estimatedSavings };
    }

    return {
      totalCompressions: 0,
      totalEmbeddings: 0,
      tokensSaved: 0,
      estimatedSavings: 0,
      cacheHitRate: 0,
      compressionRatio: 0,
      qualityScore: 0,
      patternCount: 0,
    };
  }, [claudeCodeSessions, historicalMetrics]);

  // Filter sessions for Codex and calculate metrics
  const codexSessionsFiltered = useMemo(() => {
    return allSessions.filter(s => s.tool_id === 'codex');
  }, [allSessions]);

  const codexMetricsAggregated = useMemo(() => {
    // Use session data if available
    if (codexSessionsFiltered.length > 0) {
      const totals = codexSessionsFiltered.reduce(
        (acc, session) => ({
          totalCompressions: acc.totalCompressions + (session.total_compressions || 0),
          totalEmbeddings: acc.totalEmbeddings + (session.total_embeddings || 0),
          tokensSaved: acc.tokensSaved + (session.tokens_saved || 0),
        }),
        { totalCompressions: 0, totalEmbeddings: 0, tokensSaved: 0 }
      );

      const COST_PER_MILLION_TOKENS = 3.0;
      const estimatedSavings = (totals.tokensSaved / 1_000_000) * COST_PER_MILLION_TOKENS;

      return {
        ...totals,
        estimatedSavings,
        cacheHitRate: 0,
        compressionRatio: 0,
        qualityScore: 0,
        patternCount: 0,
      };
    }

    // Fall back to historical data
    if (historicalMetrics) {
      const totals = {
        totalCompressions: historicalMetrics.codex.compressions,
        totalEmbeddings: historicalMetrics.codex.embeddings,
        tokensSaved: historicalMetrics.codex.tokensSaved,
        cacheHitRate: historicalMetrics.codex.cacheHitRate,
        compressionRatio: historicalMetrics.codex.compressionRatio,
        qualityScore: historicalMetrics.codex.qualityScore,
        patternCount: historicalMetrics.codex.patternCount,
      };

      const COST_PER_MILLION_TOKENS = 3.0;
      const estimatedSavings = (totals.tokensSaved / 1_000_000) * COST_PER_MILLION_TOKENS;

      return { ...totals, estimatedSavings };
    }

    return {
      totalCompressions: 0,
      totalEmbeddings: 0,
      tokensSaved: 0,
      estimatedSavings: 0,
      cacheHitRate: 0,
      compressionRatio: 0,
      qualityScore: 0,
      patternCount: 0,
    };
  }, [codexSessionsFiltered, historicalMetrics]);

  // Convert aggregated metrics to ToolMetrics format
  const claudeCodeMetricsFormatted: ToolMetrics = useMemo(() => ({
    total_embeddings: claudeCodeMetrics.totalEmbeddings,
    total_compressions: claudeCodeMetrics.totalCompressions,
    tokens_saved: claudeCodeMetrics.tokensSaved,
    cache_hit_rate: claudeCodeMetrics.cacheHitRate,
    compression_ratio: claudeCodeMetrics.compressionRatio,
    avg_compression_ratio: claudeCodeMetrics.compressionRatio,
    quality_score: claudeCodeMetrics.qualityScore,
    avg_quality_score: claudeCodeMetrics.qualityScore,
    pattern_count: claudeCodeMetrics.patternCount,
    graph_edges: 0, // Not available in API
    total_successes: 0, // Not available in API
    total_failures: 0, // Not available in API
    total_original_tokens: 0, // Not available in API
    total_compressed_tokens: 0, // Not available in API
    cache_hits: 0, // Not available in API
    tokens_processed: 0, // Not available in API
    prediction_accuracy: 0, // Not available in API
  }), [claudeCodeMetrics]);

  const codexMetricsFormatted: ToolMetrics = useMemo(() => ({
    total_embeddings: codexMetricsAggregated.totalEmbeddings,
    total_compressions: codexMetricsAggregated.totalCompressions,
    tokens_saved: codexMetricsAggregated.tokensSaved,
    cache_hit_rate: codexMetricsAggregated.cacheHitRate,
    compression_ratio: codexMetricsAggregated.compressionRatio,
    avg_compression_ratio: codexMetricsAggregated.compressionRatio,
    quality_score: codexMetricsAggregated.qualityScore,
    avg_quality_score: codexMetricsAggregated.qualityScore,
    pattern_count: codexMetricsAggregated.patternCount,
    graph_edges: 0, // Not available in API
    total_successes: 0, // Not available in API
    total_failures: 0, // Not available in API
    total_original_tokens: 0, // Not available in API
    total_compressed_tokens: 0, // Not available in API
    cache_hits: 0, // Not available in API
    tokens_processed: 0, // Not available in API
    prediction_accuracy: 0, // Not available in API
  }), [codexMetricsAggregated]);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-96">
        <LoadingSpinner />
      </div>
    );
  }

  // Use all active sessions (includes cursor, codex, claude-code, etc.)
  const activeSessions = allActiveSessions.filter(s => !s.ended_at);

  return (
    <div className="space-y-4 md:space-y-6">
      {/* Page Header */}
      <div className="bg-gradient-to-r from-purple-900/50 to-pink-900/50 rounded-xl p-4 md:p-8 border border-purple-500/20">
        <h1 className="text-2xl md:text-3xl font-bold text-white mb-2">Dashboard Overview</h1>
        <p className="text-sm md:text-base text-gray-300">Multi-tool metrics and performance monitoring</p>
      </div>

      {/* Tabs */}
      <Tabs value={selectedTab} onValueChange={handleTabChange}>
        <TabsList>
          <TabsTrigger value="all">
            <Globe className="w-4 h-4 mr-2" />
            All Tools
          </TabsTrigger>
          <TabsTrigger value="claude-code">
            <Bot className="w-4 h-4 mr-2" />
            Claude Code
          </TabsTrigger>
          <TabsTrigger value="codex">
            <Brain className="w-4 h-4 mr-2" />
            Codex
          </TabsTrigger>
          <TabsTrigger value="services">
            <SettingsIcon className="w-4 h-4 mr-2" />
            Services
          </TabsTrigger>
        </TabsList>

        {/* All Tools Tab - Current overview content */}
        <TabsContent value="all">
          {/* Hero Section */}
      <div className="relative overflow-hidden rounded-2xl bg-gradient-to-br from-purple-600/20 via-blue-600/20 to-green-600/20 p-4 md:p-8 border border-purple-500/30">
        <div className="relative z-10">
          <div className="flex items-center gap-3 mb-3">
            <Sparkles className="h-6 w-6 md:h-8 md:w-8 text-purple-400" />
            <h1 className="text-2xl md:text-4xl font-bold bg-gradient-to-r from-purple-400 via-blue-400 to-green-400 bg-clip-text text-transparent">
              OmniMemory
            </h1>
          </div>
          <p className="text-base md:text-xl text-gray-300 mb-4 md:mb-6">
            Universal Context Memory - Intelligent memory across all your AI tools
          </p>

          {/* Stats Bar */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 md:gap-4">
            <StatCard
              icon={<Activity className="h-5 w-5" />}
              label="Total Sessions"
              value={sessionStats?.total_sessions_lifetime || 0}
              color="text-blue-400"
            />
            <StatCard
              icon={<Zap className="h-5 w-5" />}
              label="Active Sessions"
              value={activeSessions.length}
              color="text-green-400"
            />
            <StatCard
              icon={<Brain className="h-5 w-5" />}
              label="Tokens Saved"
              value={formatNumber(aggregatedMetrics.tokensSaved)}
              color="text-purple-400"
            />
            <StatCard
              icon={<DollarSign className="h-5 w-5" />}
              label="Est. Savings"
              value={`$${aggregatedMetrics.estimatedSavings.toFixed(2)}`}
              color="text-green-400"
            />
          </div>

          {/* Active Sessions Details */}
          <ActiveSessionsList sessions={activeSessions} />
        </div>

        {/* Background Decoration */}
        <div className="absolute top-0 right-0 w-64 h-64 bg-purple-500/10 rounded-full blur-3xl"></div>
        <div className="absolute bottom-0 left-0 w-64 h-64 bg-blue-500/10 rounded-full blur-3xl"></div>
      </div>

      {/* Tool Breakdown by Tool */}
      <ToolBreakdownCard />

      {/* Initialized Tools Overview */}
      <ToolsOverviewCard />

      {/* Tool Cards Grid */}
      <div>
        <h2 className="text-xl md:text-2xl font-bold mb-3 md:mb-4 flex items-center gap-2">
          <TrendingUp className="h-5 w-5 md:h-6 md:w-6 text-purple-400" />
          Connected Tools
        </h2>
        <div className="grid gap-4 md:gap-6 lg:grid-cols-2">
          <ToolCard
            name="Claude Code"
            toolId="claude-code"
            icon={<Brain className="h-6 w-6 text-blue-500" />}
            color="blue"
            metrics={claudeCodeMetricsFormatted}
            sessions={claudeCodeSessions}
          />
          <ToolCard
            name="OpenAI Codex"
            toolId="codex"
            icon={<Zap className="h-6 w-6 text-green-500" />}
            color="green"
            metrics={codexMetricsFormatted}
            sessions={codexSessionsFiltered}
          />
        </div>
      </div>

      {/* Cost Analytics */}
      <div>
        <h2 className="text-xl md:text-2xl font-bold mb-3 md:mb-4 flex items-center gap-2">
          <DollarSign className="h-5 w-5 md:h-6 md:w-6 text-green-400" />
          Cost Analytics
        </h2>
        <CostBreakdownChart tools={TOOLS} />
      </div>

      {/* Activity Feed */}
      <div>
        <h2 className="text-xl md:text-2xl font-bold mb-3 md:mb-4 flex items-center gap-2">
          <Activity className="h-5 w-5 md:h-6 md:w-6 text-green-400" />
          Real-Time Activity
        </h2>
        <ActivityStream tools={TOOLS} maxItems={15} sessions={activeSessions} />
      </div>

      {/* Tool Usage & Operations */}
      <div>
          <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 mb-3 md:mb-4">
            <h2 className="text-xl md:text-2xl font-bold flex items-center gap-2">
              <Activity className="h-5 w-5 md:h-6 md:w-6 text-blue-400" />
              Tool Usage & Operations
            </h2>
            <select
              value={toolTimeRange}
              onChange={(e) => setToolTimeRange(e.target.value as typeof toolTimeRange)}
              className="w-full sm:w-auto px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 min-h-[44px]"
            >
              <option value="1h">Last Hour</option>
              <option value="24h">Last 24 Hours</option>
              <option value="7d">Last 7 Days</option>
              <option value="30d">Last 30 Days</option>
            </select>
          </div>

          {toolDataLoading ? (
            <LoadingState message="Loading tool usage data..." />
          ) : toolDataError ? (
            <ErrorState
              error={toolDataError}
              context="Failed to Load Tool Usage Data"
              onRetry={fetchToolUsageData}
            />
          ) : toolBreakdownData && apiSavingsData ? (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 md:gap-6">
              {/* Read Operations Card */}
              <Card className="border-blue-500/30">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <FileText className="h-5 w-5 text-blue-400" />
                    Read Operations
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3">
                    <div className="flex justify-between items-center">
                      <span className="text-sm text-muted-foreground">Total Operations</span>
                      <span className="text-lg font-bold text-blue-400">
                        {formatNumber(toolBreakdownData.read.total_operations)}
                      </span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-sm text-muted-foreground">Tokens Prevented</span>
                      <span className="text-lg font-bold text-green-400">
                        {formatNumber(toolBreakdownData.read.total_tokens_prevented)}
                      </span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-sm text-muted-foreground">Avg Response Time</span>
                      <span className="text-sm font-medium">
                        {toolBreakdownData.read.avg_response_time_ms.toFixed(1)}ms
                      </span>
                    </div>
                    <div className="pt-2 border-t border-gray-700">
                      <div className="text-xs text-muted-foreground mb-2">By Mode:</div>
                      <div className="space-y-1">
                        {Object.entries(toolBreakdownData.read.by_mode).map(([mode, stats]) => (
                          <div key={mode} className="flex justify-between items-center text-xs">
                            <span className="capitalize">{mode}</span>
                            <span className="text-muted-foreground">
                              {stats.count} ops · {formatNumber(stats.tokens_prevented)} tokens
                            </span>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>

              {/* Search Operations Card */}
              <Card className="border-green-500/30">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Search className="h-5 w-5 text-green-400" />
                    Search Operations
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3">
                    <div className="flex justify-between items-center">
                      <span className="text-sm text-muted-foreground">Total Operations</span>
                      <span className="text-lg font-bold text-green-400">
                        {formatNumber(toolBreakdownData.search.total_operations)}
                      </span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-sm text-muted-foreground">Tokens Prevented</span>
                      <span className="text-lg font-bold text-green-400">
                        {formatNumber(toolBreakdownData.search.total_tokens_prevented)}
                      </span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-sm text-muted-foreground">Avg Response Time</span>
                      <span className="text-sm font-medium">
                        {toolBreakdownData.search.avg_response_time_ms.toFixed(1)}ms
                      </span>
                    </div>
                    <div className="pt-2 border-t border-gray-700">
                      <div className="text-xs text-muted-foreground mb-2">By Mode:</div>
                      <div className="space-y-1">
                        {Object.entries(toolBreakdownData.search.by_mode).map(([mode, stats]) => (
                          <div key={mode} className="flex justify-between items-center text-xs">
                            <span className="capitalize">{mode}</span>
                            <span className="text-muted-foreground">
                              {stats.count} ops · {formatNumber(stats.tokens_prevented)} tokens
                            </span>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>
          ) : null}
      </div>

      {/* Key Metrics Summary */}
          <Card className="border-2 border-purple-500/30 bg-gradient-to-br from-purple-900/20 to-blue-900/20">
            <CardHeader>
              <CardTitle className="text-2xl">Value Proposition</CardTitle>
              <CardDescription>Why OmniMemory is transforming AI tooling</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid md:grid-cols-3 gap-4 md:gap-6">
                <ValueProp
                  title="Universal Memory"
                  description="Works with ANY AI tool - Claude, GPT, Codex, Gemini. Single source of truth for context."
                  icon={<Link className="w-8 h-8 text-purple-400" />}
                />
                <ValueProp
                  title="Cost Control"
                  description="Tag-based allocation enables precise multi-tenant billing. 90%+ cost reduction using free providers."
                  icon={<DollarSign className="w-8 h-8 text-green-400" />}
                />
                <ValueProp
                  title="Real-Time Tracking"
                  description="Session-specific metrics in real-time, not batch. Production-ready with SOC2 audit trails."
                  icon={<Flame className="w-8 h-8 text-orange-400" />}
                />
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Claude Code Tab */}
        <TabsContent value="claude-code">
          <ToolDetailTab toolId="claude-code" toolName="Claude Code" />
        </TabsContent>

        {/* Codex Tab */}
        <TabsContent value="codex">
          <ToolDetailTab toolId="codex" toolName="OpenAI Codex" />
        </TabsContent>

        {/* Services Tab */}
        <TabsContent value="services">
          <ServicesHealthTab />
        </TabsContent>
      </Tabs>
    </div>
  );
}

interface StatCardProps {
  icon: React.ReactNode;
  label: string;
  value: string | number;
  color: string;
}

function StatCard({ icon, label, value, color }: StatCardProps) {
  return (
    <div className="bg-gray-800/50 backdrop-blur rounded-lg p-3 md:p-4 border border-gray-700">
      <div className="flex items-center gap-2 mb-2">
        <span className={color}>{icon}</span>
        <span className="text-xs md:text-sm text-muted-foreground">{label}</span>
      </div>
      <div className={`text-xl md:text-2xl font-bold ${color}`}>{value}</div>
    </div>
  );
}

interface ValuePropProps {
  title: string;
  description: string;
  icon: React.ReactNode;
}

function ValueProp({ title, description, icon }: ValuePropProps) {
  return (
    <div className="p-3 md:p-4 rounded-lg bg-gray-800/50 border border-gray-700">
      <div className="mb-3">{icon}</div>
      <h3 className="text-base md:text-lg font-semibold mb-2 text-purple-300">{title}</h3>
      <p className="text-xs md:text-sm text-gray-400">{description}</p>
    </div>
  );
}

interface ActiveSessionsListProps {
  sessions: Session[];
}

function ActiveSessionsList({ sessions }: ActiveSessionsListProps) {
  if (sessions.length === 0) {
    return (
      <div className="mt-4 p-4 rounded-lg bg-gray-800/30 border border-gray-700/50">
        <p className="text-sm text-gray-400 text-center">No active sessions</p>
      </div>
    );
  }

  return (
    <div className="mt-4 p-4 rounded-lg bg-gray-800/30 border border-gray-700/50">
      <div className="flex items-center gap-2 mb-3">
        <Clock className="h-4 w-4 text-green-400" />
        <h4 className="text-sm font-semibold text-gray-300">Active Sessions Details</h4>
      </div>
      <div className="space-y-2">
        {sessions.map((session) => {
          const toolInfo = getToolDisplay(session.tool_id);
          const ToolIcon = toolInfo.icon;
          return (
            <div
              key={session.session_id}
              className="flex items-center justify-between p-2 rounded bg-gray-900/50 border border-gray-700/30"
            >
              <div className="flex items-center gap-3">
                <div className="w-2 h-2 rounded-full bg-green-400 animate-pulse"></div>
                <div className="flex items-center gap-2">
                  <ToolIcon className={`w-5 h-5 ${getToolColor(session.tool_id)}`} />
                  <div>
                    <span className={`text-sm font-medium ${getToolColor(session.tool_id)}`}>
                      {toolInfo.name}
                    </span>
                    {session.tool_version && (
                      <span className="text-xs text-gray-500 ml-2">
                        v{session.tool_version}
                      </span>
                    )}
                  </div>
                </div>
              </div>
              <div className="flex items-center gap-4 text-xs text-gray-400">
                <span>Started {formatLocalTime(session.started_at)}</span>
                <div className="flex gap-2">
                  <span title="Compressions">{formatNumber(session.total_compressions)} comp</span>
                  <span title="Embeddings">{formatNumber(session.total_embeddings)} emb</span>
                  <span title="Tokens Saved" className="text-green-400">
                    {formatNumber(session.tokens_saved)} tokens
                  </span>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
