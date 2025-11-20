import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import {
  CHART_COLORS,
  CHART_TOOLTIP_STYLE,
  CHART_AXIS_STYLE,
  CHART_GRID_STYLE,
  CHART_HEIGHTS,
  formatTooltipValue,
  formatAxisLabel
} from '../../utils/chartTheme';

interface DataKey {
  key: string;
  name: string;
  color?: string;
}

interface StyledBarChartProps {
  data: Record<string, unknown>[];
  dataKeys: DataKey[];
  xAxisKey: string;
  height?: number;
  showLegend?: boolean;
  showGrid?: boolean;
  stacked?: boolean;
  xAxisFormatter?: (value: unknown) => string;
  yAxisFormatter?: (value: unknown) => string;
}

export function StyledBarChart({
  data,
  dataKeys,
  xAxisKey,
  height = CHART_HEIGHTS.medium,
  showLegend = true,
  showGrid = true,
  stacked = false,
  xAxisFormatter = (value) => formatAxisLabel(value as string | number),
  yAxisFormatter = (value) => (value as number).toLocaleString(),
}: StyledBarChartProps) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={data}>
        {showGrid && <CartesianGrid {...CHART_GRID_STYLE} />}

        <XAxis
          dataKey={xAxisKey}
          {...CHART_AXIS_STYLE}
          tickFormatter={xAxisFormatter}
        />

        <YAxis
          {...CHART_AXIS_STYLE}
          tickFormatter={yAxisFormatter}
        />

        <Tooltip
          contentStyle={CHART_TOOLTIP_STYLE}
          labelStyle={{ color: '#F3F4F6', marginBottom: '4px' }}
          itemStyle={{ color: '#E5E7EB' }}
          formatter={(value: number, name: string) => [formatTooltipValue(value, name), name]}
        />

        {showLegend && (
          <Legend
            wrapperStyle={{ paddingTop: '20px' }}
            iconType="rect"
          />
        )}

        {dataKeys.map((item) => (
          <Bar
            key={item.key}
            dataKey={item.key}
            name={item.name}
            fill={item.color || CHART_COLORS.primary}
            radius={[4, 4, 0, 0]}
            stackId={stacked ? 'stack' : undefined}
          />
        ))}
      </BarChart>
    </ResponsiveContainer>
  );
}
