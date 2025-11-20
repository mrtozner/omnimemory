// Standardized color palette for charts
export const CHART_COLORS = {
  primary: '#8B5CF6',      // Purple
  secondary: '#3B82F6',    // Blue
  success: '#10B981',      // Green
  warning: '#F59E0B',      // Amber
  danger: '#EF4444',       // Red
  info: '#06B6D4',         // Cyan
  purple: '#A855F7',       // Light purple
  pink: '#EC4899',         // Pink
  indigo: '#6366F1',       // Indigo
  gray: '#6B7280',         // Gray
};

// Multi-series color palette (for charts with multiple lines/bars)
export const CHART_COLOR_PALETTE = [
  CHART_COLORS.primary,
  CHART_COLORS.secondary,
  CHART_COLORS.success,
  CHART_COLORS.warning,
  CHART_COLORS.info,
  CHART_COLORS.purple,
  CHART_COLORS.pink,
];

// Standardized tooltip styling
export const CHART_TOOLTIP_STYLE = {
  backgroundColor: '#1F2937',
  border: '1px solid #374151',
  borderRadius: '8px',
  padding: '8px 12px',
};

// Standardized axis styling
export const CHART_AXIS_STYLE = {
  stroke: '#9CA3AF',
  fontSize: 12,
  fontFamily: 'Inter, system-ui, sans-serif',
};

// Standardized grid styling
export const CHART_GRID_STYLE = {
  stroke: '#ffffff20',
  strokeDasharray: '3 3',
};

// Standard chart margins
export const CHART_MARGINS = {
  top: 10,
  right: 30,
  bottom: 30,
  left: 40,
};

// Responsive chart height
export const CHART_HEIGHTS = {
  small: 200,
  medium: 300,
  large: 400,
};

// Custom tooltip formatter
export const formatTooltipValue = (value: number, name: string): string => {
  if (name.toLowerCase().includes('tokens') || name.toLowerCase().includes('cost')) {
    return value.toLocaleString();
  }
  if (name.toLowerCase().includes('rate') || name.toLowerCase().includes('ratio') || name.toLowerCase().includes('percentage')) {
    return `${(value * 100).toFixed(1)}%`;
  }
  return value.toLocaleString();
};

// Axis label formatter (prevent truncation)
export const formatAxisLabel = (value: string | number, maxLength: number = 15): string => {
  const str = String(value);
  if (str.length <= maxLength) return str;
  return str.substring(0, maxLength - 3) + '...';
};

// Date formatter for time series
export const formatChartDate = (dateString: string): string => {
  const date = new Date(dateString);
  return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
};

export const formatChartTime = (dateString: string): string => {
  const date = new Date(dateString);
  return date.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });
};

// Format timestamp for charts (used in multiple components)
export const formatTimestamp = (timestamp: string): string => {
  const date = new Date(timestamp);
  return date.toLocaleTimeString('en-US', {
    hour: '2-digit',
    minute: '2-digit',
    hour12: false
  });
};
