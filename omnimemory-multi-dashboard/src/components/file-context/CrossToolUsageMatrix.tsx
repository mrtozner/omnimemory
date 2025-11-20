import { Card, CardHeader, CardTitle, CardContent, CardDescription } from '../shared/Card';
import { Grid3x3 } from 'lucide-react';
import { useEffect, useState } from 'react';
import { api } from '../../services/api';
import type { CrossToolUsageResponse, CrossToolUsageData } from '../../types/fileContext';
import { ErrorState } from '../shared/ErrorState';
import { LoadingState } from '../shared/LoadingState';

const TOOLS = ['Claude Code', 'Cursor', 'VSCode', 'ChatGPT'];
const HOURS = Array.from({ length: 24 }, (_, i) => i);

// Color intensity based on hit count
const getHeatColor = (count: number, maxCount: number): string => {
  if (count === 0) return 'bg-gray-800';

  const intensity = maxCount > 0 ? count / maxCount : 0;

  if (intensity >= 0.8) return 'bg-green-500';
  if (intensity >= 0.6) return 'bg-green-600';
  if (intensity >= 0.4) return 'bg-green-700';
  if (intensity >= 0.2) return 'bg-green-800';
  return 'bg-green-900';
};

export function CrossToolUsageMatrix() {
  const [data, setData] = useState<CrossToolUsageResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [hoveredCell, setHoveredCell] = useState<{ tool: string; hour: number; count: number } | null>(null);

  useEffect(() => {
    async function fetchCrossToolUsage() {
      try {
        setError(null);

        const result = await api.getCrossToolUsage() as unknown as CrossToolUsageResponse;
        setData(result);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load cross-tool usage');
        console.error('Failed to fetch cross-tool usage:', err);
      } finally {
        setLoading(false);
      }
    }

    fetchCrossToolUsage();
    const interval = setInterval(fetchCrossToolUsage, 10000); // Poll every 10s
    return () => clearInterval(interval);
  }, []);

  if (loading && !data) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Grid3x3 className="h-5 w-5 text-blue-500" />
            Cross-Tool Usage Matrix
          </CardTitle>
        </CardHeader>
        <CardContent>
          <LoadingState message="Loading cross-tool usage data..." compact />
        </CardContent>
      </Card>
    );
  }

  if (error && !data) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Grid3x3 className="h-5 w-5 text-blue-500" />
            Cross-Tool Usage Matrix
          </CardTitle>
        </CardHeader>
        <CardContent>
          <ErrorState
            error={error}
            context="Cross-Tool Usage"
            compact={true}
          />
        </CardContent>
      </Card>
    );
  }

  if (!data || !data.tools || data.tools.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Grid3x3 className="h-5 w-5 text-blue-500" />
            Cross-Tool Usage Matrix
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="h-[400px] flex items-center justify-center text-muted-foreground">
            No cross-tool usage data available
          </div>
        </CardContent>
      </Card>
    );
  }

  // Create a map for quick lookups
  const toolDataMap = new Map<string, Record<number, number>>();
  data.tools.forEach((toolData: CrossToolUsageData) => {
    toolDataMap.set(toolData.tool_id, toolData.hourly_hits);
  });

  // Find max count for color scaling
  let maxCount = 0;
  data.tools.forEach((toolData: CrossToolUsageData) => {
    Object.values(toolData.hourly_hits).forEach((count) => {
      if (count > maxCount) maxCount = count;
    });
  });

  const getHitCount = (toolId: string, hour: number): number => {
    const toolData = toolDataMap.get(toolId);
    return toolData?.[hour] || 0;
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Grid3x3 className="h-5 w-5 text-blue-500" />
          Cross-Tool Usage Matrix
        </CardTitle>
        <CardDescription>
          Cache hits per tool by hour of day (24-hour format)
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="overflow-x-auto">
          <div className="inline-block min-w-full">
            {/* Heatmap Grid */}
            <div className="grid gap-1" style={{ gridTemplateColumns: `120px repeat(24, 1fr)` }}>
              {/* Header Row */}
              <div className="text-xs font-medium text-muted-foreground p-2"></div>
              {HOURS.map((hour) => (
                <div
                  key={`hour-${hour}`}
                  className="text-xs font-medium text-center text-muted-foreground p-1"
                >
                  {hour}
                </div>
              ))}

              {/* Tool Rows */}
              {TOOLS.map((tool) => (
                <div key={tool} className="contents">
                  {/* Tool Label */}
                  <div className="text-xs font-medium text-muted-foreground p-2 flex items-center">
                    {tool}
                  </div>

                  {/* Hour Cells */}
                  {HOURS.map((hour) => {
                    const count = getHitCount(tool, hour);
                    const colorClass = getHeatColor(count, maxCount);

                    return (
                      <div
                        key={`${tool}-${hour}`}
                        className={`${colorClass} rounded cursor-pointer transition-all duration-200 hover:ring-2 hover:ring-blue-400 hover:scale-105 relative group min-h-[32px]`}
                        onMouseEnter={() => setHoveredCell({ tool, hour, count })}
                        onMouseLeave={() => setHoveredCell(null)}
                      >
                        {/* Tooltip on hover */}
                        {hoveredCell?.tool === tool && hoveredCell?.hour === hour && (
                          <div className="absolute z-10 bottom-full left-1/2 transform -translate-x-1/2 mb-2 px-3 py-2 bg-gray-900 border border-gray-700 rounded-lg shadow-lg whitespace-nowrap">
                            <div className="text-xs font-semibold text-white">{tool}</div>
                            <div className="text-xs text-gray-300">
                              Hour: {hour}:00
                            </div>
                            <div className="text-xs text-green-400 font-bold">
                              {count} cache hits
                            </div>
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              ))}
            </div>

            {/* Legend */}
            <div className="mt-6 flex items-center justify-center gap-4">
              <span className="text-xs text-muted-foreground">Less</span>
              <div className="flex gap-1">
                <div className="w-6 h-6 bg-gray-800 rounded"></div>
                <div className="w-6 h-6 bg-green-900 rounded"></div>
                <div className="w-6 h-6 bg-green-800 rounded"></div>
                <div className="w-6 h-6 bg-green-700 rounded"></div>
                <div className="w-6 h-6 bg-green-600 rounded"></div>
                <div className="w-6 h-6 bg-green-500 rounded"></div>
              </div>
              <span className="text-xs text-muted-foreground">More</span>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
