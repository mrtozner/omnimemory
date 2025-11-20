import { useState, useEffect } from 'react';
import { SWEBenchCard } from '../components/benchmark/SWEBenchCard';
import { CompetitiveComparison } from '../components/benchmark/CompetitiveComparison';
import { CostSavingsCalculator } from '../components/benchmark/CostSavingsCalculator';
import { api, type SWEBenchResults, type TokenSavingsResults } from '../services/api';
import { Award, TrendingUp } from 'lucide-react';

export function BenchmarkPage() {
  const [sweBenchData, setSweBenchData] = useState<SWEBenchResults | null>(null);
  const [tokenData, setTokenData] = useState<TokenSavingsResults | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchData();
  }, []);

  const fetchData = async () => {
    try {
      const [sweBench, tokens] = await Promise.all([
        api.getSWEBenchResults(),
        api.getTokenSavings()
      ]);
      setSweBenchData(sweBench);
      setTokenData(tokens);
    } catch (error) {
      console.error('Failed to fetch benchmark data:', error);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-lg text-gray-600 dark:text-gray-400">Loading benchmarks...</div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
          Benchmarks & Competitive Analysis
        </h1>
        <p className="mt-2 text-gray-600 dark:text-gray-400">
          Transparent validation, competitive positioning, and cost analysis
        </p>
      </div>

      {/* SWE-Bench Validation */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2">
          {sweBenchData?.data?.omnimemory && (
            <SWEBenchCard
              passRate={sweBenchData.data.omnimemory.pass_at_1}
              totalTests={sweBenchData.data.omnimemory.total_tests}
              testsPassed={sweBenchData.data.omnimemory.tests_passed}
              contextRetention={sweBenchData.data.omnimemory.avg_context_retention}
            />
          )}
        </div>

        <div className="space-y-4">
          <div className="bg-gradient-to-br from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 border border-blue-200 dark:border-blue-800 p-6 rounded-lg">
            <Award className="w-8 h-8 text-blue-600 mb-3" />
            <div className="text-3xl font-bold text-blue-600">
              {tokenData?.data?.summary?.semantic_cache_savings || '30-60%'}
            </div>
            <div className="text-sm text-gray-600 dark:text-gray-400 mt-1">
              Token Savings (Cache)
            </div>
          </div>

          <div className="bg-gradient-to-br from-purple-50 to-pink-50 dark:from-purple-900/20 dark:to-pink-900/20 border border-purple-200 dark:border-purple-800 p-6 rounded-lg">
            <TrendingUp className="w-8 h-8 text-purple-600 mb-3" />
            <div className="text-3xl font-bold text-purple-600">
              {tokenData?.data?.summary?.compression_savings || '85-95%'}
            </div>
            <div className="text-sm text-gray-600 dark:text-gray-400 mt-1">
              Compression Ratio
            </div>
          </div>
        </div>
      </div>

      {/* Competitive Comparison */}
      <CompetitiveComparison />

      {/* Cost Savings Calculator */}
      <CostSavingsCalculator />
    </div>
  );
}
