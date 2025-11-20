import { Card, CardHeader, CardTitle, CardContent, CardDescription } from '../shared/Card';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from 'recharts';
import { TrendingUp } from 'lucide-react';
import type { TrendDataPoint } from '../../services/api';
import {
  CHART_COLORS,
  CHART_TOOLTIP_STYLE,
  CHART_GRID_STYLE,
  CHART_AXIS_STYLE
} from '../../utils/chartTheme';

interface TokenPreventionChartProps {
  trends: TrendDataPoint[];
}

export function TokenPreventionChart({ trends }: TokenPreventionChartProps) {
  const formatNumber = (num: number): string => {
    if (num >= 1000000) {
      return `${(num / 1000000).toFixed(2)}M`;
    }
    if (num >= 1000) {
      return `${(num / 1000).toFixed(1)}K`;
    }
    return num.toString();
  };

  const formatTimestamp = (timestamp: string): string => {
    const date = new Date(timestamp);
    return `${date.getHours()}:${date.getMinutes().toString().padStart(2, '0')}`;
  };

  if (!trends || trends.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <TrendingUp className="h-5 w-5 text-green-500" />
            Token Prevention Trends
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="h-[300px] flex items-center justify-center text-muted-foreground">
            No trend data available
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <TrendingUp className="h-5 w-5 text-green-500" />
          Token Prevention Trends
        </CardTitle>
        <CardDescription>
          Tokens prevented and operations count over time
        </CardDescription>
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={trends}>
            <CartesianGrid {...CHART_GRID_STYLE} />
            <XAxis
              dataKey="timestamp"
              {...CHART_AXIS_STYLE}
              tickFormatter={formatTimestamp}
            />
            <YAxis
              yAxisId="left"
              {...CHART_AXIS_STYLE}
              stroke={CHART_COLORS.success}
              tickFormatter={formatNumber}
            />
            <YAxis
              yAxisId="right"
              orientation="right"
              {...CHART_AXIS_STYLE}
              stroke={CHART_COLORS.secondary}
            />
            <Tooltip
              formatter={(value: number, name: string) => {
                if (name === 'Tokens Prevented') {
                  return [formatNumber(value), name];
                }
                return [value, name];
              }}
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
              dataKey="tokens_prevented"
              stroke={CHART_COLORS.success}
              strokeWidth={2}
              dot={{ fill: CHART_COLORS.success, r: 3 }}
              activeDot={{ r: 5 }}
              name="Tokens Prevented"
            />
            <Line
              yAxisId="right"
              type="monotone"
              dataKey="operations"
              stroke={CHART_COLORS.secondary}
              strokeWidth={2}
              dot={{ fill: CHART_COLORS.secondary, r: 3 }}
              activeDot={{ r: 5 }}
              name="Operations Count"
            />
          </LineChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
}
