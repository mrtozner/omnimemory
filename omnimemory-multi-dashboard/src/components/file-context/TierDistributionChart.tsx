import { Card, CardHeader, CardTitle, CardContent, CardDescription } from '../shared/Card';
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip } from 'recharts';
import { Layers } from 'lucide-react';
import { useEffect, useState } from 'react';
import { api } from '../../services/api';
import type { TierDistributionResponse } from '../../types/fileContext';
import { ErrorState } from '../shared/ErrorState';
import { LoadingState } from '../shared/LoadingState';
import { CHART_COLORS, CHART_TOOLTIP_STYLE } from '../../utils/chartTheme';

const TIER_COLORS = {
  FRESH: CHART_COLORS.success,    // Green
  RECENT: CHART_COLORS.secondary, // Blue
  AGING: CHART_COLORS.warning,    // Amber
  ARCHIVE: CHART_COLORS.gray      // Gray
};

interface ChartDataPoint {
  name: string;
  value: number;
  count: number;
  percentage: number;
  color: string;
}

export function TierDistributionChart() {
  const [chartData, setChartData] = useState<ChartDataPoint[]>([]);
  const [totalFiles, setTotalFiles] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function fetchTierDistribution() {
      try {
        setError(null);

        const data = await api.getTierDistribution() as unknown as TierDistributionResponse;

        // Transform API data to chart format
        const transformed: ChartDataPoint[] = [
          {
            name: 'FRESH',
            value: data.tiers.FRESH.count,
            count: data.tiers.FRESH.count,
            percentage: data.tiers.FRESH.percentage,
            color: TIER_COLORS.FRESH
          },
          {
            name: 'RECENT',
            value: data.tiers.RECENT.count,
            count: data.tiers.RECENT.count,
            percentage: data.tiers.RECENT.percentage,
            color: TIER_COLORS.RECENT
          },
          {
            name: 'AGING',
            value: data.tiers.AGING.count,
            count: data.tiers.AGING.count,
            percentage: data.tiers.AGING.percentage,
            color: TIER_COLORS.AGING
          },
          {
            name: 'ARCHIVE',
            value: data.tiers.ARCHIVE.count,
            count: data.tiers.ARCHIVE.count,
            percentage: data.tiers.ARCHIVE.percentage,
            color: TIER_COLORS.ARCHIVE
          }
        ];

        setChartData(transformed);
        setTotalFiles(data.total_files);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load tier distribution');
        console.error('Failed to fetch tier distribution:', err);
      } finally {
        setLoading(false);
      }
    }

    fetchTierDistribution();
    const interval = setInterval(fetchTierDistribution, 10000); // Poll every 10s
    return () => clearInterval(interval);
  }, []);

  if (loading && chartData.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Layers className="h-5 w-5 text-purple-500" />
            Tier Distribution
          </CardTitle>
        </CardHeader>
        <CardContent>
          <LoadingState message="Loading tier distribution..." compact />
        </CardContent>
      </Card>
    );
  }

  if (error && chartData.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Layers className="h-5 w-5 text-purple-500" />
            Tier Distribution
          </CardTitle>
        </CardHeader>
        <CardContent>
          <ErrorState
            error={error}
            context="Tier Distribution"
            compact={true}
          />
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Layers className="h-5 w-5 text-purple-500" />
          Tier Distribution
        </CardTitle>
        <CardDescription>
          File context tiers across the memory system
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="flex flex-col lg:flex-row gap-6">
          {/* Donut Chart */}
          <div className="flex-1 relative">
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie
                  data={chartData}
                  dataKey="value"
                  nameKey="name"
                  cx="50%"
                  cy="50%"
                  innerRadius={70}
                  outerRadius={110}
                  paddingAngle={2}
                >
                  {chartData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip
                  formatter={(value: number, name: string, entry: { payload?: { percentage?: number } }) => [
                    `${value} files (${entry.payload?.percentage?.toFixed(1) || '0'}%)`,
                    name
                  ]}
                  contentStyle={CHART_TOOLTIP_STYLE}
                  labelStyle={{ color: '#F3F4F6' }}
                  itemStyle={{ color: '#E5E7EB' }}
                />
              </PieChart>
            </ResponsiveContainer>

            {/* Center Text */}
            <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 text-center pointer-events-none">
              <div className="text-3xl font-bold text-foreground">{totalFiles}</div>
              <div className="text-sm text-muted-foreground">Total Files</div>
            </div>
          </div>

          {/* Legend with Stats */}
          <div className="flex-1 space-y-3">
            {chartData.map((item) => (
              <div
                key={item.name}
                className="flex items-center justify-between p-3 rounded-lg bg-gray-800/50 border border-gray-700/50"
              >
                <div className="flex items-center gap-3">
                  <div
                    className="w-4 h-4 rounded-full"
                    style={{ backgroundColor: item.color }}
                  />
                  <div>
                    <div className="text-sm font-medium text-foreground">{item.name}</div>
                    <div className="text-xs text-muted-foreground">
                      {item.percentage.toFixed(1)}% of total
                    </div>
                  </div>
                </div>
                <div className="text-right">
                  <div className="text-lg font-bold" style={{ color: item.color }}>
                    {item.count}
                  </div>
                  <div className="text-xs text-muted-foreground">files</div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
