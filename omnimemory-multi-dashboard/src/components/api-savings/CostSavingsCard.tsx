import { Card, CardContent, CardHeader, CardTitle } from '../shared/Card';
import { DollarSign, TrendingDown, TrendingUp } from 'lucide-react';

interface CostSavingsCardProps {
  baselineCost: number;
  actualCost: number;
  totalSaved: number;
  savingsPercentage: number;
}

export function CostSavingsCard({
  baselineCost,
  actualCost,
  totalSaved,
  savingsPercentage,
}: CostSavingsCardProps) {
  const formatCurrency = (value: number): string => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    }).format(value);
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <DollarSign className="h-5 w-5 text-green-500" />
          Cost Savings Overview
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {/* Baseline Cost */}
          <div className="p-4 rounded-lg bg-red-500/10 border border-red-500/30">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm text-muted-foreground">Baseline Cost</span>
              <TrendingUp className="h-4 w-4 text-red-400" />
            </div>
            <div className="text-3xl font-bold text-red-400">
              {formatCurrency(baselineCost)}
            </div>
            <p className="text-xs text-muted-foreground mt-1">
              Without OmniMemory
            </p>
          </div>

          {/* Actual Cost */}
          <div className="p-4 rounded-lg bg-blue-500/10 border border-blue-500/30">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm text-muted-foreground">Actual Cost</span>
              <DollarSign className="h-4 w-4 text-blue-400" />
            </div>
            <div className="text-3xl font-bold text-blue-400">
              {formatCurrency(actualCost)}
            </div>
            <p className="text-xs text-muted-foreground mt-1">
              With OmniMemory
            </p>
          </div>

          {/* Total Saved */}
          <div className="p-4 rounded-lg bg-green-500/10 border border-green-500/30">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm text-muted-foreground">Total Saved</span>
              <TrendingDown className="h-4 w-4 text-green-400" />
            </div>
            <div className="text-3xl font-bold text-green-400">
              {formatCurrency(totalSaved)}
            </div>
            <p className="text-xs text-muted-foreground mt-1">
              {savingsPercentage.toFixed(1)}% reduction
            </p>
          </div>
        </div>

        {/* Savings Percentage Bar */}
        <div className="mt-6">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm font-medium">Savings Rate</span>
            <span className="text-sm font-bold text-green-400">
              {savingsPercentage.toFixed(1)}%
            </span>
          </div>
          <div className="h-3 bg-gray-700 rounded-full overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-green-500 to-green-400 transition-all"
              style={{ width: `${Math.min(savingsPercentage, 100)}%` }}
            />
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
