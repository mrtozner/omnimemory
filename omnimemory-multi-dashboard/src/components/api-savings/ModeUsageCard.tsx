import { Card, CardHeader, CardTitle, CardContent } from '../shared/Card';
import { Activity, ArrowUpDown } from 'lucide-react';
import { useState } from 'react';
import type { BreakdownItem } from '../../services/api';

interface ModeUsageCardProps {
  breakdownByMode: Record<string, BreakdownItem>;
}

type SortColumn = 'mode' | 'operations' | 'tokens' | 'cost' | 'avg_cost';
type SortDirection = 'asc' | 'desc';

export function ModeUsageCard({ breakdownByMode }: ModeUsageCardProps) {
  const [sortColumn, setSortColumn] = useState<SortColumn>('operations');
  const [sortDirection, setSortDirection] = useState<SortDirection>('desc');

  const formatNumber = (num: number): string => {
    if (num >= 1000000) {
      return `${(num / 1000000).toFixed(2)}M`;
    }
    if (num >= 1000) {
      return `${(num / 1000).toFixed(1)}K`;
    }
    return num.toString();
  };

  const formatCurrency = (value: number): string => {
    return `$${value.toFixed(2)}`;
  };

  const handleSort = (column: SortColumn) => {
    if (sortColumn === column) {
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc');
    } else {
      setSortColumn(column);
      setSortDirection('desc');
    }
  };

  // Convert breakdown to array and add calculated fields
  const modes = Object.entries(breakdownByMode).map(([mode, data]) => ({
    mode,
    operations: data.operations,
    tokens_prevented: data.tokens_prevented,
    cost_saved: data.cost_saved,
    avg_cost_per_operation: data.operations > 0 ? data.cost_saved / data.operations : 0,
  }));

  // Sort modes
  const sortedModes = [...modes].sort((a, b) => {
    let comparison = 0;
    switch (sortColumn) {
      case 'mode':
        comparison = a.mode.localeCompare(b.mode);
        break;
      case 'operations':
        comparison = a.operations - b.operations;
        break;
      case 'tokens':
        comparison = a.tokens_prevented - b.tokens_prevented;
        break;
      case 'cost':
        comparison = a.cost_saved - b.cost_saved;
        break;
      case 'avg_cost':
        comparison = a.avg_cost_per_operation - b.avg_cost_per_operation;
        break;
    }
    return sortDirection === 'asc' ? comparison : -comparison;
  });

  const totalOperations = modes.reduce((sum, m) => sum + m.operations, 0);

  if (modes.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Activity className="h-5 w-5 text-blue-500" />
            Usage by Operation Mode
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="h-[200px] flex items-center justify-center text-muted-foreground">
            No mode data available
          </div>
        </CardContent>
      </Card>
    );
  }

  const SortIcon = ({ column }: { column: SortColumn }) => {
    if (sortColumn !== column) {
      return <ArrowUpDown className="h-4 w-4 text-gray-500" />;
    }
    return (
      <ArrowUpDown
        className={`h-4 w-4 ${
          sortDirection === 'asc' ? 'text-blue-400' : 'text-green-400'
        }`}
      />
    );
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Activity className="h-5 w-5 text-blue-500" />
          Usage by Operation Mode
        </CardTitle>
      </CardHeader>
      <CardContent>
        {/* Desktop Table View */}
        <div className="hidden md:block overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-gray-700">
                <th
                  className="text-left py-3 px-4 cursor-pointer hover:bg-gray-800/50"
                  onClick={() => handleSort('mode')}
                >
                  <div className="flex items-center gap-2">
                    Mode
                    <SortIcon column="mode" />
                  </div>
                </th>
                <th
                  className="text-right py-3 px-4 cursor-pointer hover:bg-gray-800/50"
                  onClick={() => handleSort('operations')}
                >
                  <div className="flex items-center justify-end gap-2">
                    Operations
                    <SortIcon column="operations" />
                  </div>
                </th>
                <th
                  className="text-right py-3 px-4 cursor-pointer hover:bg-gray-800/50"
                  onClick={() => handleSort('tokens')}
                >
                  <div className="flex items-center justify-end gap-2">
                    Tokens Prevented
                    <SortIcon column="tokens" />
                  </div>
                </th>
                <th
                  className="text-right py-3 px-4 cursor-pointer hover:bg-gray-800/50"
                  onClick={() => handleSort('cost')}
                >
                  <div className="flex items-center justify-end gap-2">
                    Cost Saved
                    <SortIcon column="cost" />
                  </div>
                </th>
                <th
                  className="text-right py-3 px-4 cursor-pointer hover:bg-gray-800/50"
                  onClick={() => handleSort('avg_cost')}
                >
                  <div className="flex items-center justify-end gap-2">
                    Avg/Operation
                    <SortIcon column="avg_cost" />
                  </div>
                </th>
                <th className="text-right py-3 px-4">% of Total</th>
              </tr>
            </thead>
            <tbody>
              {sortedModes.map((mode) => {
                const percentage =
                  totalOperations > 0
                    ? (mode.operations / totalOperations) * 100
                    : 0;
                return (
                  <tr
                    key={mode.mode}
                    className="border-b border-gray-800 hover:bg-gray-800/30"
                  >
                    <td className="py-3 px-4 font-mono text-sm">
                      {mode.mode}
                    </td>
                    <td className="py-3 px-4 text-right">
                      {mode.operations.toLocaleString()}
                    </td>
                    <td className="py-3 px-4 text-right text-green-400">
                      {formatNumber(mode.tokens_prevented)}
                    </td>
                    <td className="py-3 px-4 text-right text-blue-400">
                      {formatCurrency(mode.cost_saved)}
                    </td>
                    <td className="py-3 px-4 text-right text-purple-400">
                      {formatCurrency(mode.avg_cost_per_operation)}
                    </td>
                    <td className="py-3 px-4 text-right text-muted-foreground">
                      {percentage.toFixed(1)}%
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>

        {/* Mobile Card View */}
        <div className="md:hidden space-y-3">
          {sortedModes.map((mode) => {
            const percentage =
              totalOperations > 0
                ? (mode.operations / totalOperations) * 100
                : 0;
            return (
              <div
                key={mode.mode}
                className="p-4 rounded-lg bg-gray-800/50 border border-gray-700"
              >
                <div className="flex items-center justify-between mb-3">
                  <span className="font-mono text-sm font-semibold">
                    {mode.mode}
                  </span>
                  <span className="text-sm text-muted-foreground">
                    {percentage.toFixed(1)}%
                  </span>
                </div>
                <div className="grid grid-cols-2 gap-3 text-sm">
                  <div>
                    <div className="text-muted-foreground mb-1">Operations</div>
                    <div className="font-semibold">
                      {mode.operations.toLocaleString()}
                    </div>
                  </div>
                  <div>
                    <div className="text-muted-foreground mb-1">
                      Tokens Prevented
                    </div>
                    <div className="font-semibold text-green-400">
                      {formatNumber(mode.tokens_prevented)}
                    </div>
                  </div>
                  <div>
                    <div className="text-muted-foreground mb-1">Cost Saved</div>
                    <div className="font-semibold text-blue-400">
                      {formatCurrency(mode.cost_saved)}
                    </div>
                  </div>
                  <div>
                    <div className="text-muted-foreground mb-1">
                      Avg/Operation
                    </div>
                    <div className="font-semibold text-purple-400">
                      {formatCurrency(mode.avg_cost_per_operation)}
                    </div>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </CardContent>
    </Card>
  );
}
