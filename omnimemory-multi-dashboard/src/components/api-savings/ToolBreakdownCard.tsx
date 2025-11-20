import { Card, CardHeader, CardTitle, CardContent } from '../shared/Card';
import { FileText, Search, Clock, Zap } from 'lucide-react';
import type { ToolBreakdownStats } from '../../services/api';

interface ToolBreakdownCardProps {
  breakdown: ToolBreakdownStats;
}

export function ToolBreakdownCard({ breakdown }: ToolBreakdownCardProps) {
  const formatNumber = (num: number): string => {
    if (num >= 1000000) {
      return `${(num / 1000000).toFixed(2)}M`;
    }
    if (num >= 1000) {
      return `${(num / 1000).toFixed(1)}K`;
    }
    return num.toString();
  };

  const totalOperations =
    breakdown.read.total_operations + breakdown.search.total_operations;

  const readPercentage =
    totalOperations > 0
      ? (breakdown.read.total_operations / totalOperations) * 100
      : 0;
  const searchPercentage =
    totalOperations > 0
      ? (breakdown.search.total_operations / totalOperations) * 100
      : 0;

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Zap className="h-5 w-5 text-yellow-500" />
          Tool Usage Breakdown
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-6">
          {/* Read Operations */}
          <div className="p-4 rounded-lg bg-blue-500/10 border border-blue-500/30">
            <div className="flex items-center gap-2 mb-4">
              <FileText className="h-5 w-5 text-blue-400" />
              <h3 className="text-lg font-semibold text-blue-400">
                Read Operations
              </h3>
              <span className="ml-auto text-sm text-muted-foreground">
                {readPercentage.toFixed(1)}% of total
              </span>
            </div>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div>
                <div className="text-sm text-muted-foreground mb-1">Count</div>
                <div className="text-2xl font-bold text-blue-400">
                  {breakdown.read.total_operations.toLocaleString()}
                </div>
              </div>
              <div>
                <div className="text-sm text-muted-foreground mb-1">
                  Tokens Prevented
                </div>
                <div className="text-2xl font-bold text-blue-400">
                  {formatNumber(breakdown.read.total_tokens_prevented)}
                </div>
              </div>
              <div>
                <div className="text-sm text-muted-foreground mb-1">
                  Avg Response Time
                </div>
                <div className="text-2xl font-bold text-blue-400 flex items-baseline">
                  {breakdown.read.avg_response_time_ms.toFixed(0)}
                  <span className="text-sm ml-1">ms</span>
                </div>
              </div>
              <div>
                <div className="text-sm text-muted-foreground mb-1">
                  Tokens Actual
                </div>
                <div className="text-2xl font-bold text-blue-400">
                  {formatNumber(breakdown.read.total_tokens_actual)}
                </div>
              </div>
            </div>
          </div>

          {/* Search Operations */}
          <div className="p-4 rounded-lg bg-purple-500/10 border border-purple-500/30">
            <div className="flex items-center gap-2 mb-4">
              <Search className="h-5 w-5 text-purple-400" />
              <h3 className="text-lg font-semibold text-purple-400">
                Search Operations
              </h3>
              <span className="ml-auto text-sm text-muted-foreground">
                {searchPercentage.toFixed(1)}% of total
              </span>
            </div>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div>
                <div className="text-sm text-muted-foreground mb-1">Count</div>
                <div className="text-2xl font-bold text-purple-400">
                  {breakdown.search.total_operations.toLocaleString()}
                </div>
              </div>
              <div>
                <div className="text-sm text-muted-foreground mb-1">
                  Tokens Prevented
                </div>
                <div className="text-2xl font-bold text-purple-400">
                  {formatNumber(breakdown.search.total_tokens_prevented)}
                </div>
              </div>
              <div>
                <div className="text-sm text-muted-foreground mb-1">
                  Avg Response Time
                </div>
                <div className="text-2xl font-bold text-purple-400 flex items-baseline">
                  {breakdown.search.avg_response_time_ms.toFixed(0)}
                  <span className="text-sm ml-1">ms</span>
                </div>
              </div>
              <div>
                <div className="text-sm text-muted-foreground mb-1">
                  Tokens Actual
                </div>
                <div className="text-2xl font-bold text-purple-400">
                  {formatNumber(breakdown.search.total_tokens_actual)}
                </div>
              </div>
            </div>
          </div>

          {/* Overall Stats */}
          <div className="flex items-center justify-between p-4 rounded-lg bg-green-500/10 border border-green-500/30">
            <div className="flex items-center gap-2">
              <Clock className="h-5 w-5 text-green-400" />
              <span className="text-sm text-muted-foreground">
                Total Cost Saved
              </span>
            </div>
            <div className="text-2xl font-bold text-green-400">
              ${breakdown.total_cost_saved.toFixed(2)}
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
