import { Card, CardHeader, CardTitle, CardContent } from '../shared/Card';
import { TrendingUp, Clock, Zap, DollarSign } from 'lucide-react';
import type { APISavingsData } from '../../services/api';

interface ROICalculatorProps {
  data: APISavingsData;
}

export function ROICalculator({ data }: ROICalculatorProps) {
  const formatCurrency = (value: number): string => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    }).format(value);
  };

  const formatNumber = (num: number): string => {
    return num.toLocaleString();
  };

  // Calculate metrics
  const avgSavingsPerOperation =
    data.total_operations > 0
      ? data.total_cost_saved / data.total_operations
      : 0;

  // Find most expensive and fastest operation modes
  const modeEntries = Object.entries(data.breakdown_by_mode);

  const mostExpensiveMode = modeEntries.reduce(
    (max, [mode, stats]) =>
      stats.tokens_prevented > max.tokens
        ? { mode, tokens: stats.tokens_prevented, cost: stats.cost_saved }
        : max,
    { mode: 'N/A', tokens: 0, cost: 0 }
  );

  // Calculate projected savings based on current usage
  // Assume this time range continues at same rate
  const timeRangeHours =
    data.time_range === '1h' ? 1 :
    data.time_range === '24h' ? 24 :
    data.time_range === '7d' ? 168 :
    data.time_range === '30d' ? 720 :
    24; // default to 24h

  const projectedMonthlySavings = (data.total_cost_saved / timeRangeHours) * 720; // 30 days
  const projectedAnnualSavings = projectedMonthlySavings * 12;

  // Estimate developer hours saved (rough estimate: 1 hour per 10,000 tokens saved)
  const estimatedHoursSaved = data.total_tokens_prevented / 10000;

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <TrendingUp className="h-5 w-5 text-green-500" />
          ROI & Metrics
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {/* Payback Period */}
          <div className="p-4 rounded-lg bg-green-500/10 border border-green-500/30">
            <div className="flex items-center gap-2 mb-2">
              <Zap className="h-5 w-5 text-green-400" />
              <span className="text-sm text-muted-foreground">
                Payback Period
              </span>
            </div>
            <div className="text-2xl font-bold text-green-400">Immediate</div>
            <p className="text-xs text-muted-foreground mt-1">
              Savings from day 1
            </p>
          </div>

          {/* Avg Savings Per Operation */}
          <div className="p-4 rounded-lg bg-blue-500/10 border border-blue-500/30">
            <div className="flex items-center gap-2 mb-2">
              <DollarSign className="h-5 w-5 text-blue-400" />
              <span className="text-sm text-muted-foreground">
                Avg Savings/Op
              </span>
            </div>
            <div className="text-2xl font-bold text-blue-400">
              {formatCurrency(avgSavingsPerOperation)}
            </div>
            <p className="text-xs text-muted-foreground mt-1">
              Per operation
            </p>
          </div>

          {/* Most Expensive Mode */}
          <div className="p-4 rounded-lg bg-purple-500/10 border border-purple-500/30">
            <div className="flex items-center gap-2 mb-2">
              <TrendingUp className="h-5 w-5 text-purple-400" />
              <span className="text-sm text-muted-foreground">
                Top Mode
              </span>
            </div>
            <div className="text-lg font-bold text-purple-400 truncate">
              {mostExpensiveMode.mode}
            </div>
            <p className="text-xs text-muted-foreground mt-1">
              {formatCurrency(mostExpensiveMode.cost)} saved
            </p>
          </div>

          {/* Developer Hours Saved */}
          <div className="p-4 rounded-lg bg-yellow-500/10 border border-yellow-500/30">
            <div className="flex items-center gap-2 mb-2">
              <Clock className="h-5 w-5 text-yellow-400" />
              <span className="text-sm text-muted-foreground">
                Hours Saved
              </span>
            </div>
            <div className="text-2xl font-bold text-yellow-400">
              {estimatedHoursSaved.toFixed(1)}h
            </div>
            <p className="text-xs text-muted-foreground mt-1">
              Estimated time saved
            </p>
          </div>
        </div>

        {/* Projections */}
        {data.time_range !== 'all' && (
          <div className="mt-6 grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="p-4 rounded-lg bg-gradient-to-br from-green-900/20 to-green-800/10 border border-green-500/30">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm text-muted-foreground">
                  Projected Monthly Savings
                </span>
                <TrendingUp className="h-4 w-4 text-green-400" />
              </div>
              <div className="text-3xl font-bold text-green-400">
                {formatCurrency(projectedMonthlySavings)}
              </div>
              <p className="text-xs text-muted-foreground mt-1">
                Based on current {data.time_range} usage pattern
              </p>
            </div>

            <div className="p-4 rounded-lg bg-gradient-to-br from-blue-900/20 to-blue-800/10 border border-blue-500/30">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm text-muted-foreground">
                  Projected Annual Savings
                </span>
                <TrendingUp className="h-4 w-4 text-blue-400" />
              </div>
              <div className="text-3xl font-bold text-blue-400">
                {formatCurrency(projectedAnnualSavings)}
              </div>
              <p className="text-xs text-muted-foreground mt-1">
                Extrapolated from {data.time_range} data
              </p>
            </div>
          </div>
        )}

        {/* Summary Stats */}
        <div className="mt-6 p-4 rounded-lg bg-gray-800/50 border border-gray-700">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-center">
            <div>
              <div className="text-sm text-muted-foreground mb-1">
                Total Operations
              </div>
              <div className="text-xl font-bold">
                {formatNumber(data.total_operations)}
              </div>
            </div>
            <div>
              <div className="text-sm text-muted-foreground mb-1">
                Tokens Processed
              </div>
              <div className="text-xl font-bold text-blue-400">
                {formatNumber(data.total_tokens_processed)}
              </div>
            </div>
            <div>
              <div className="text-sm text-muted-foreground mb-1">
                Tokens Prevented
              </div>
              <div className="text-xl font-bold text-green-400">
                {formatNumber(data.total_tokens_prevented)}
              </div>
            </div>
            <div>
              <div className="text-sm text-muted-foreground mb-1">
                Time Range
              </div>
              <div className="text-xl font-bold text-purple-400">
                {data.time_range}
              </div>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
