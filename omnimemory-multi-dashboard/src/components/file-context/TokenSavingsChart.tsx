import { Card, CardHeader, CardTitle, CardContent, CardDescription } from '../shared/Card';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';
import { TrendingUp, DollarSign, Zap } from 'lucide-react';
import { useEffect, useState } from 'react';
import { api } from '../../services/api';
import type { TokenSavingsResponse } from '../../types/fileContext';
import { ErrorState } from '../shared/ErrorState';
import { LoadingState } from '../shared/LoadingState';
import { CHART_COLORS, CHART_TOOLTIP_STYLE, CHART_GRID_STYLE, CHART_AXIS_STYLE, formatTooltipValue } from '../../utils/chartTheme';

export function TokenSavingsChart() {
  const [data, setData] = useState<TokenSavingsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function fetchTokenSavings() {
      try {
        setError(null);

        const result = await api.getFileContextTokenSavings() as unknown as TokenSavingsResponse;
        setData(result);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load token savings');
        console.error('Failed to fetch token savings:', err);
      } finally {
        setLoading(false);
      }
    }

    fetchTokenSavings();
    const interval = setInterval(fetchTokenSavings, 10000); // Poll every 10s
    return () => clearInterval(interval);
  }, []);

  const formatNumber = (num: number | undefined | null): string => {
    // Handle undefined, null, or NaN values
    if (num === undefined || num === null || isNaN(num)) {
      return '0';
    }

    if (num >= 1000000) {
      return `${(num / 1000000).toFixed(2)}M`;
    }
    if (num >= 1000) {
      return `${(num / 1000).toFixed(1)}K`;
    }
    return num.toString();
  };

  const formatCurrency = (value: number | undefined | null): string => {
    // Handle undefined, null, or NaN values
    if (value === undefined || value === null || isNaN(value)) {
      return '$0.00';
    }
    return `$${value.toFixed(2)}`;
  };

  if (loading && !data) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <TrendingUp className="h-5 w-5 text-green-500" />
            Token Savings
          </CardTitle>
        </CardHeader>
        <CardContent>
          <LoadingState message="Loading token savings data..." compact />
        </CardContent>
      </Card>
    );
  }

  if (error && !data) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <TrendingUp className="h-5 w-5 text-green-500" />
            Token Savings
          </CardTitle>
        </CardHeader>
        <CardContent>
          <ErrorState
            error={error}
            context="Token Savings"
            compact={true}
          />
        </CardContent>
      </Card>
    );
  }

  if (!data) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <TrendingUp className="h-5 w-5 text-green-500" />
            Token Savings
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="h-[400px] flex items-center justify-center text-muted-foreground">
            No token savings data available
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      {/* KPI Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {/* Total Tokens Saved */}
        <Card className="border-green-500/30 bg-gradient-to-br from-green-900/20 to-green-800/10">
          <CardContent className="pt-6">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-sm text-muted-foreground mb-1">Total Tokens Saved</div>
                <div className="text-3xl font-bold text-green-400">
                  {formatNumber(data.total_tokens_saved)}
                </div>
              </div>
              <Zap className="h-10 w-10 text-green-400 opacity-50" />
            </div>
          </CardContent>
        </Card>

        {/* Cost Saved */}
        <Card className="border-blue-500/30 bg-gradient-to-br from-blue-900/20 to-blue-800/10">
          <CardContent className="pt-6">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-sm text-muted-foreground mb-1">Cost Saved</div>
                <div className="text-3xl font-bold text-blue-400">
                  {formatCurrency(data.cost_saved_usd)}
                </div>
              </div>
              <DollarSign className="h-10 w-10 text-blue-400 opacity-50" />
            </div>
          </CardContent>
        </Card>

        {/* Projected Monthly Savings */}
        <Card className="border-purple-500/30 bg-gradient-to-br from-purple-900/20 to-purple-800/10">
          <CardContent className="pt-6">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-sm text-muted-foreground mb-1">Projected Monthly</div>
                <div className="text-3xl font-bold text-purple-400">
                  {formatCurrency(data.projected_monthly_savings)}
                </div>
              </div>
              <TrendingUp className="h-10 w-10 text-purple-400 opacity-50" />
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Line Chart */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <TrendingUp className="h-5 w-5 text-green-500" />
            Token Savings Over Time
          </CardTitle>
          <CardDescription>
            Tokens saved and cumulative savings trend
          </CardDescription>
        </CardHeader>
        <CardContent>
          {data.history && data.history.length > 0 ? (
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={data.history}>
                <CartesianGrid {...CHART_GRID_STYLE} />
                <XAxis
                  dataKey="timestamp"
                  {...CHART_AXIS_STYLE}
                  tickFormatter={(value) => {
                    const date = new Date(value);
                    return `${date.getHours()}:${date.getMinutes().toString().padStart(2, '0')}`;
                  }}
                />
                <YAxis
                  yAxisId="left"
                  {...CHART_AXIS_STYLE}
                  stroke={CHART_COLORS.success}
                  tickFormatter={(value) => formatNumber(value)}
                />
                <YAxis
                  yAxisId="right"
                  orientation="right"
                  {...CHART_AXIS_STYLE}
                  stroke={CHART_COLORS.secondary}
                  tickFormatter={(value) => formatNumber(value)}
                />
                <Tooltip
                  formatter={(value: number, name: string) => [formatTooltipValue(value, name), name]}
                  labelFormatter={(label) => {
                    const date = new Date(label);
                    return date.toLocaleString();
                  }}
                  contentStyle={CHART_TOOLTIP_STYLE}
                  labelStyle={{ color: '#F3F4F6', marginBottom: '4px' }}
                  itemStyle={{ color: '#E5E7EB' }}
                />
                <Legend wrapperStyle={{ paddingTop: '20px' }} iconType="line" />
                <Line
                  yAxisId="left"
                  type="monotone"
                  dataKey="tokens_saved"
                  stroke={CHART_COLORS.success}
                  strokeWidth={2}
                  dot={{ fill: CHART_COLORS.success, r: 3 }}
                  activeDot={{ r: 5 }}
                  name="Tokens Saved"
                />
                <Line
                  yAxisId="right"
                  type="monotone"
                  dataKey="cumulative"
                  stroke={CHART_COLORS.secondary}
                  strokeWidth={2}
                  dot={{ fill: CHART_COLORS.secondary, r: 3 }}
                  activeDot={{ r: 5 }}
                  name="Cumulative"
                />
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <div className="h-[300px] flex items-center justify-center text-muted-foreground">
              No history data available
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
