import { Card, CardHeader, CardTitle, CardContent, CardDescription } from '../shared/Card';
import { FileText, TrendingUp } from 'lucide-react';
import { useEffect, useState } from 'react';
import { api } from '../../services/api';
import type { FileAccessHeatmapResponse, FileAccessRecord } from '../../types/fileContext';
import { ErrorState } from '../shared/ErrorState';
import { LoadingState } from '../shared/LoadingState';

type SortField = 'file_path' | 'access_count' | 'last_accessed';
type SortDirection = 'asc' | 'desc';

const TIER_COLORS = {
  FRESH: { bg: 'bg-green-500/20', text: 'text-green-400', border: 'border-green-500/30' },
  RECENT: { bg: 'bg-blue-500/20', text: 'text-blue-400', border: 'border-blue-500/30' },
  AGING: { bg: 'bg-yellow-500/20', text: 'text-yellow-400', border: 'border-yellow-500/30' },
  ARCHIVE: { bg: 'bg-gray-500/20', text: 'text-gray-400', border: 'border-gray-500/30' },
};

export function FileAccessHeatmap() {
  const [data, setData] = useState<FileAccessHeatmapResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [sortField, setSortField] = useState<SortField>('access_count');
  const [sortDirection, setSortDirection] = useState<SortDirection>('desc');

  useEffect(() => {
    async function fetchFileAccessHeatmap() {
      try {
        setError(null);

        const result = await api.getFileAccessHeatmap() as unknown as FileAccessHeatmapResponse;
        setData(result);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load file access heatmap');
        console.error('Failed to fetch file access heatmap:', err);
      } finally {
        setLoading(false);
      }
    }

    fetchFileAccessHeatmap();
    const interval = setInterval(fetchFileAccessHeatmap, 10000); // Poll every 10s
    return () => clearInterval(interval);
  }, []);

  const handleSort = (field: SortField) => {
    if (sortField === field) {
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc');
    } else {
      setSortField(field);
      setSortDirection('desc');
    }
  };

  const getSortedFiles = (): FileAccessRecord[] => {
    if (!data?.files) return [];

    const sorted = [...data.files].sort((a, b) => {
      let compareA: string | number = a[sortField];
      let compareB: string | number = b[sortField];

      if (sortField === 'last_accessed') {
        compareA = new Date(a.last_accessed).getTime();
        compareB = new Date(b.last_accessed).getTime();
      }

      if (sortDirection === 'asc') {
        return compareA > compareB ? 1 : -1;
      } else {
        return compareA < compareB ? 1 : -1;
      }
    });

    return sorted;
  };

  const formatDate = (dateString: string): string => {
    const date = new Date(dateString);
    return date.toLocaleString('en-US', {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  const formatFileName = (path: string): string => {
    const parts = path.split('/');
    const fileName = parts[parts.length - 1];
    return `${fileName}`;
  };

  const formatDirPath = (path: string): string => {
    const parts = path.split('/');
    return parts.slice(0, -1).join('/');
  };

  if (loading && !data) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <FileText className="h-5 w-5 text-purple-500" />
            File Access Heatmap
          </CardTitle>
        </CardHeader>
        <CardContent>
          <LoadingState message="Loading file access data..." compact />
        </CardContent>
      </Card>
    );
  }

  if (error && !data) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <FileText className="h-5 w-5 text-purple-500" />
            File Access Heatmap
          </CardTitle>
        </CardHeader>
        <CardContent>
          <ErrorState
            error={error}
            context="File Access Heatmap"
            compact={true}
          />
        </CardContent>
      </Card>
    );
  }

  if (!data || !data.files || data.files.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <FileText className="h-5 w-5 text-purple-500" />
            File Access Heatmap
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="h-[400px] flex items-center justify-center text-muted-foreground">
            No file access data available
          </div>
        </CardContent>
      </Card>
    );
  }

  const sortedFiles = getSortedFiles();

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <FileText className="h-5 w-5 text-purple-500" />
          File Access Heatmap
        </CardTitle>
        <CardDescription>
          Top {sortedFiles.length} most accessed files across all tools
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-gray-700">
                <th
                  className="text-left p-3 text-sm font-semibold text-gray-300 cursor-pointer hover:text-white transition-colors"
                  onClick={() => handleSort('file_path')}
                >
                  <div className="flex items-center gap-2">
                    File Path
                    {sortField === 'file_path' && (
                      <span className="text-xs">{sortDirection === 'asc' ? '↑' : '↓'}</span>
                    )}
                  </div>
                </th>
                <th
                  className="text-left p-3 text-sm font-semibold text-gray-300 cursor-pointer hover:text-white transition-colors"
                  onClick={() => handleSort('access_count')}
                >
                  <div className="flex items-center gap-2">
                    Access Count
                    {sortField === 'access_count' && (
                      <span className="text-xs">{sortDirection === 'asc' ? '↑' : '↓'}</span>
                    )}
                  </div>
                </th>
                <th className="text-left p-3 text-sm font-semibold text-gray-300">
                  Tools
                </th>
                <th className="text-left p-3 text-sm font-semibold text-gray-300">
                  Current Tier
                </th>
                <th
                  className="text-left p-3 text-sm font-semibold text-gray-300 cursor-pointer hover:text-white transition-colors"
                  onClick={() => handleSort('last_accessed')}
                >
                  <div className="flex items-center gap-2">
                    Last Accessed
                    {sortField === 'last_accessed' && (
                      <span className="text-xs">{sortDirection === 'asc' ? '↑' : '↓'}</span>
                    )}
                  </div>
                </th>
              </tr>
            </thead>
            <tbody>
              {sortedFiles.map((file, index) => {
                const tierColor = TIER_COLORS[file.current_tier];
                const isTopFile = index < 3;

                return (
                  <tr
                    key={file.file_path}
                    className={`border-b border-gray-800 hover:bg-gray-800/50 transition-colors ${
                      isTopFile ? 'bg-purple-500/5' : ''
                    }`}
                  >
                    <td className="p-3">
                      <div className="flex items-start gap-2">
                        {isTopFile && (
                          <TrendingUp className="h-4 w-4 text-purple-400 flex-shrink-0 mt-0.5" />
                        )}
                        <div className="min-w-0">
                          <div className="text-sm font-medium text-white truncate">
                            {formatFileName(file.file_path)}
                          </div>
                          <div className="text-xs text-gray-500 truncate">
                            {formatDirPath(file.file_path)}
                          </div>
                        </div>
                      </div>
                    </td>
                    <td className="p-3">
                      <div className="text-sm font-bold text-green-400">
                        {file.access_count.toLocaleString()}
                      </div>
                    </td>
                    <td className="p-3">
                      <div className="flex flex-wrap gap-1">
                        {file.tools.map((tool) => (
                          <span
                            key={tool}
                            className="px-2 py-1 text-xs rounded-full bg-blue-500/20 text-blue-400 border border-blue-500/30"
                          >
                            {tool}
                          </span>
                        ))}
                      </div>
                    </td>
                    <td className="p-3">
                      <span
                        className={`px-3 py-1 text-xs rounded-full ${tierColor.bg} ${tierColor.text} border ${tierColor.border} font-medium`}
                      >
                        {file.current_tier}
                      </span>
                    </td>
                    <td className="p-3">
                      <div className="text-xs text-gray-400">
                        {formatDate(file.last_accessed)}
                      </div>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>

        {/* Summary Footer */}
        <div className="mt-4 p-4 rounded-lg bg-gray-800/30 border border-gray-700/50">
          <div className="flex items-center justify-between text-sm">
            <div className="text-gray-400">
              Showing top {sortedFiles.length} files of {data.total_files} total
            </div>
            <div className="flex items-center gap-4">
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded-full bg-purple-500"></div>
                <span className="text-xs text-gray-400">Top 3 most accessed</span>
              </div>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
