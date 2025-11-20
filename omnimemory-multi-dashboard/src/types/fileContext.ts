// File Context Tier Metrics Types

export interface TierData {
  count: number;
  percentage: number;
}

export interface TierDistributionResponse {
  tiers: {
    FRESH: TierData;
    RECENT: TierData;
    AGING: TierData;
    ARCHIVE: TierData;
  };
  total_files: number;
  timestamp: string;
}

export interface TokenSavingsHistoryPoint {
  timestamp: string;
  tokens_saved: number;
  cumulative: number;
}

export interface TokenSavingsResponse {
  total_tokens_saved: number;
  cost_saved_usd: number;
  projected_monthly_savings: number;
  history: TokenSavingsHistoryPoint[];
}

export interface CachePerformanceResponse {
  cache_hit_rate: number;
  avg_latency_ms: number;
  storage_used_mb: number;
  total_requests: number;
  cache_hits: number;
  cache_misses: number;
}

export interface CrossToolUsageData {
  tool_id: string;
  hourly_hits: Record<number, number>; // hour (0-23) -> hit count
}

export interface CrossToolUsageResponse {
  tools: CrossToolUsageData[];
  timestamp: string;
}

export interface FileAccessRecord {
  file_path: string;
  access_count: number;
  tools: string[];
  current_tier: 'FRESH' | 'RECENT' | 'AGING' | 'ARCHIVE';
  last_accessed: string;
}

export interface FileAccessHeatmapResponse {
  files: FileAccessRecord[];
  total_files: number;
  timestamp: string;
}
