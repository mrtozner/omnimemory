import React, { useMemo } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '../shared/Card';
import { StyledLineChart } from '../shared/StyledLineChart';
import { CHART_COLORS } from '../../utils/chartTheme';

interface TokenUsageChartProps {
  history: Array<{
    timestamp: string;
    tokens_saved: number;
  }>;
}

export const TokenUsageChart = React.memo<TokenUsageChartProps>(({ history }) => {
  // Format timestamps for display (show only seconds) - memoized for performance
  const formattedData = useMemo(() => {
    return history.map((point) => ({
      ...point,
      time: new Date(point.timestamp).toLocaleTimeString('en-US', {
        minute: '2-digit',
        second: '2-digit',
      }),
    }));
  }, [history]);

  return (
    <Card>
      <CardHeader>
        <CardTitle>Token Savings Over Time (Last 60s)</CardTitle>
      </CardHeader>
      <CardContent>
        {formattedData.length > 0 ? (
          <StyledLineChart
            data={formattedData}
            dataKeys={[
              { key: 'tokens_saved', name: 'Tokens Saved', color: CHART_COLORS.primary }
            ]}
            xAxisKey="time"
            height={300}
          />
        ) : (
          <div className="h-[300px] flex items-center justify-center text-muted-foreground">
            Collecting real-time data...
          </div>
        )}
      </CardContent>
    </Card>
  );
});

TokenUsageChart.displayName = 'TokenUsageChart';
