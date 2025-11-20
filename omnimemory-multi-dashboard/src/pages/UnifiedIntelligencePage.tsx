import { useState, useEffect } from 'react';
import { Brain, Activity, HardDrive, Bot, Lightbulb, Link, Target, RefreshCw, Zap } from 'lucide-react';
import { PredictionMetricsCard } from '../components/unified/PredictionMetricsCard';
import { OrchestrationMetricsCard } from '../components/unified/OrchestrationMetricsCard';
import { SuggestionMetricsCard } from '../components/unified/SuggestionMetricsCard';
import { CrossMemoryInsightsCard } from '../components/unified/CrossMemoryInsightsCard';

export function UnifiedIntelligencePage() {
  const [lastRefresh, setLastRefresh] = useState(new Date());

  // Auto-refresh timestamp every 30 seconds
  useEffect(() => {
    const interval = setInterval(() => {
      setLastRefresh(new Date());
    }, 30000);

    return () => clearInterval(interval);
  }, []);

  const formatTime = (date: Date) => {
    return date.toLocaleTimeString('en-US', {
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit'
    });
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="relative overflow-hidden rounded-2xl bg-gradient-to-br from-orange-600/20 via-purple-600/20 to-blue-600/20 p-8 border border-orange-500/30">
        <div className="relative z-10">
          <div className="flex items-center justify-between">
            <div>
              <div className="flex items-center gap-3 mb-3">
                <Brain className="h-8 w-8 text-orange-400" />
                <h1 className="text-4xl font-bold bg-gradient-to-r from-orange-400 via-purple-400 to-blue-400 bg-clip-text text-transparent">
                  Unified Intelligence System
                </h1>
              </div>
              <p className="text-xl text-gray-300">
                Dual Memory System: File Context + Agent Memory
              </p>
              <p className="text-sm text-gray-400 mt-2">
                Predictive Engine • Memory Orchestrator • Proactive Suggestions • Cross-Memory Patterns
              </p>
            </div>
            <div className="flex items-center gap-2 text-sm text-gray-400">
              <Activity className="h-4 w-4 animate-pulse text-green-400" />
              <span>Last updated: {formatTime(lastRefresh)}</span>
            </div>
          </div>
        </div>

        {/* Background Decoration */}
        <div className="absolute top-0 right-0 w-64 h-64 bg-orange-500/10 rounded-full blur-3xl"></div>
        <div className="absolute bottom-0 left-0 w-64 h-64 bg-purple-500/10 rounded-full blur-3xl"></div>
        <div className="absolute bottom-0 right-0 w-48 h-48 bg-blue-500/10 rounded-full blur-3xl"></div>
      </div>

      {/* System Overview */}
      <div className="grid md:grid-cols-2 gap-6">
        <div className="p-6 rounded-xl bg-gradient-to-br from-blue-900/30 to-blue-800/20 border border-blue-500/30">
          <HardDrive className="w-8 h-8 mb-3 text-blue-400" />
          <h3 className="text-xl font-bold mb-2 text-blue-300">File Context Memory</h3>
          <p className="text-sm text-gray-400">
            Tracks file access patterns, maintains tiered cache (FRESH/RECENT/AGING), and predicts next files to access based on historical patterns.
          </p>
        </div>

        <div className="p-6 rounded-xl bg-gradient-to-br from-green-900/30 to-green-800/20 border border-green-500/30">
          <Bot className="w-8 h-8 mb-3 text-green-400" />
          <h3 className="text-xl font-bold mb-2 text-green-300">Agent Memory</h3>
          <p className="text-sm text-gray-400">
            Records agent invocations, tool usage sequences, and learns workflow patterns to predict which agents to invoke next.
          </p>
        </div>
      </div>

      {/* Prediction Metrics */}
      <div>
        <h2 className="text-2xl font-bold mb-4 flex items-center gap-2">
          <Brain className="h-6 w-6 text-orange-400" />
          Predictive Engine
        </h2>
        <PredictionMetricsCard />
      </div>

      {/* Orchestration Metrics */}
      <div>
        <h2 className="text-2xl font-bold mb-4 flex items-center gap-2">
          <Activity className="h-6 w-6 text-purple-400" />
          Memory Orchestration
        </h2>
        <OrchestrationMetricsCard />
      </div>

      {/* Suggestions and Insights Grid */}
      <div className="grid lg:grid-cols-2 gap-6">
        {/* Suggestion Metrics */}
        <div>
          <h2 className="text-2xl font-bold mb-4 flex items-center gap-2">
            <Lightbulb className="h-6 w-6 text-yellow-400" />
            Proactive Suggestions
          </h2>
          <SuggestionMetricsCard />
        </div>

        {/* Cross-Memory Insights */}
        <div>
          <h2 className="text-2xl font-bold mb-4 flex items-center gap-2">
            <Link className="h-6 w-6 text-purple-400" />
            Cross-Memory Patterns
          </h2>
          <CrossMemoryInsightsCard />
        </div>
      </div>

      {/* Info Footer */}
      <div className="grid md:grid-cols-3 gap-6">
        <div className="p-4 rounded-lg bg-orange-500/10 border border-orange-500/30">
          <Target className="w-6 h-6 mb-2 text-orange-400" />
          <h3 className="text-lg font-semibold mb-2 text-orange-300">Predictive Engine</h3>
          <p className="text-sm text-gray-400">
            Combines file and agent memory to predict next actions with high confidence. Cross-memory predictions achieve 95%+ accuracy.
          </p>
        </div>

        <div className="p-4 rounded-lg bg-purple-500/10 border border-purple-500/30">
          <RefreshCw className="w-6 h-6 mb-2 text-purple-400" />
          <h3 className="text-lg font-semibold mb-2 text-purple-300">Memory Orchestrator</h3>
          <p className="text-sm text-gray-400">
            Intelligently routes queries to the right memory system with minimal overhead. Achieves 67%+ cache hit rate.
          </p>
        </div>

        <div className="p-4 rounded-lg bg-green-500/10 border border-green-500/30">
          <Zap className="w-6 h-6 mb-2 text-green-400" />
          <h3 className="text-lg font-semibold mb-2 text-green-300">Proactive Suggestions</h3>
          <p className="text-sm text-gray-400">
            Delivers timely suggestions based on learned patterns. Achieves 75%+ acceptance rate with continuous learning.
          </p>
        </div>
      </div>
    </div>
  );
}
