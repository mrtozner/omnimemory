import { Card, CardHeader, CardTitle, CardContent, CardDescription } from '../shared/Card';
import { PieChart, Pie, Cell, ResponsiveContainer, Legend, Tooltip, BarChart, Bar, XAxis, YAxis, CartesianGrid } from 'recharts';
import { DollarSign, TrendingDown } from 'lucide-react';
import { useEffect, useState } from 'react';
import { CHART_COLORS, CHART_TOOLTIP_STYLE, CHART_GRID_STYLE, CHART_AXIS_STYLE } from '../../utils/chartTheme';

interface CostBreakdownChartProps {
  tools: string[];
}

interface CostData {
  name: string;
  value: number;
  color: string;
  savings: number;
}

const TOOL_COLORS: Record<string, string> = {
  'claude-code': CHART_COLORS.secondary,
  'codex': CHART_COLORS.success,
  'gpt-4': CHART_COLORS.warning,
  'gemini': CHART_COLORS.primary,
};

export function CostBreakdownChart({ tools }: CostBreakdownChartProps) {
  const [pieData, setPieData] = useState<CostData[]>([]);
  const [savingsData, setSavingsData] = useState<Array<{ name: string; before: number; after: number; savings: number }>>([]);
  const [totalSavings, setTotalSavings] = useState(0);

  useEffect(() => {
    // Generate demo cost data
    const costData = tools.map((tool) => {
      const baseCost = Math.random() * 100 + 50;
      const savings = baseCost * (0.5 + Math.random() * 0.4); // 50-90% savings

      return {
        name: tool,
        value: baseCost - savings,
        color: TOOL_COLORS[tool] || '#6b7280',
        savings,
      };
    });

    setPieData(costData);

    // Generate savings comparison data
    const savingsComparison = tools.map((tool) => {
      const beforeCost = Math.random() * 150 + 100;
      const afterCost = beforeCost * (0.1 + Math.random() * 0.2); // 80-90% reduction
      return {
        name: tool,
        before: beforeCost,
        after: afterCost,
        savings: beforeCost - afterCost,
      };
    });

    setSavingsData(savingsComparison);
    setTotalSavings(savingsComparison.reduce((sum, item) => sum + item.savings, 0));
  }, [tools]);

  const formatCurrency = (value: number) => `$${value.toFixed(2)}`;

  return (
    <div className="grid gap-6 lg:grid-cols-2">
      {/* Pie Chart - Current Cost Distribution */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <DollarSign className="h-5 w-5 text-green-500" />
            Cost Distribution
          </CardTitle>
          <CardDescription>Current spending across tools</CardDescription>
        </CardHeader>
        <CardContent>
          {pieData.length > 0 ? (
            <div>
              <ResponsiveContainer width="100%" height={300}>
                <PieChart>
                  <Pie
                    data={pieData}
                    dataKey="value"
                    nameKey="name"
                    cx="50%"
                    cy="50%"
                    outerRadius={100}
                    label={({ name, percent }) => `${name} (${(percent * 100).toFixed(0)}%)`}
                    labelLine={true}
                  >
                    {pieData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip
                    formatter={(value: number) => formatCurrency(value)}
                    contentStyle={CHART_TOOLTIP_STYLE}
                    labelStyle={{ color: '#F3F4F6' }}
                    itemStyle={{ color: '#E5E7EB' }}
                  />
                </PieChart>
              </ResponsiveContainer>

              {/* Cost Summary */}
              <div className="mt-4 space-y-2">
                {pieData.map((item) => (
                  <div key={item.name} className="flex items-center justify-between p-2 rounded bg-gray-800/50">
                    <div className="flex items-center gap-2">
                      <div
                        className="w-3 h-3 rounded-full"
                        style={{ backgroundColor: item.color }}
                      />
                      <span className="text-sm font-medium">{item.name}</span>
                    </div>
                    <span className="text-sm text-green-400 font-medium">
                      {formatCurrency(item.value)}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <div className="h-[300px] flex items-center justify-center text-muted-foreground">
              Loading cost data...
            </div>
          )}
        </CardContent>
      </Card>

      {/* Bar Chart - Cost Savings Comparison */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <TrendingDown className="h-5 w-5 text-green-500" />
            Cost Savings
          </CardTitle>
          <CardDescription>
            Before vs After OmniMemory (Total saved: {formatCurrency(totalSavings)})
          </CardDescription>
        </CardHeader>
        <CardContent>
          {savingsData.length > 0 ? (
            <div>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={savingsData}>
                  <CartesianGrid {...CHART_GRID_STYLE} />
                  <XAxis dataKey="name" {...CHART_AXIS_STYLE} />
                  <YAxis {...CHART_AXIS_STYLE} tickFormatter={(value) => `$${value}`} />
                  <Tooltip
                    formatter={(value: number) => formatCurrency(value)}
                    contentStyle={CHART_TOOLTIP_STYLE}
                    labelStyle={{ color: '#F3F4F6', marginBottom: '4px' }}
                    itemStyle={{ color: '#E5E7EB' }}
                  />
                  <Legend wrapperStyle={{ paddingTop: '20px' }} iconType="rect" />
                  <Bar dataKey="before" fill={CHART_COLORS.danger} name="Before" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="after" fill={CHART_COLORS.success} name="After" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>

              {/* Savings Details */}
              <div className="mt-4 p-4 rounded-lg bg-green-500/10 border border-green-500/20">
                <h4 className="text-sm font-medium text-green-400 mb-2">Savings Breakdown</h4>
                <div className="space-y-2">
                  {savingsData.map((item) => (
                    <div key={item.name} className="flex items-center justify-between text-sm">
                      <span className="text-muted-foreground">{item.name}</span>
                      <span className="text-green-400 font-medium">
                        {formatCurrency(item.savings)} ({((item.savings / item.before) * 100).toFixed(1)}%)
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          ) : (
            <div className="h-[300px] flex items-center justify-center text-muted-foreground">
              Loading savings data...
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
