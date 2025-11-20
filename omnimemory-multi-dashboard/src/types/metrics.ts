export interface MetricsData {
  timestamp: string;
  tool: 'claude-code' | 'cursor' | 'codex' | 'vscode';
  totalContextTokens: number;
  compressionRatio: number;
  retrievalLatency: number;
  memoriesStored: number;
  activeUsers: number;
}

export interface ToolConfig {
  id: string;
  name: string;
  enabled: boolean;
  color: string;
  icon: string;
}

export interface DashboardMetrics {
  currentMetrics: MetricsData;
  historicalData: MetricsData[];
  tools: ToolConfig[];
}

export interface ActiveSession {
  session_id: string;
  tool_id: string;
  tool_version: string | null;
  started_at: string;
  last_activity: string;
  total_compressions: number;
  total_embeddings: number;
  tokens_saved: number;
}

export interface ActiveSessionsResponse {
  active_sessions: ActiveSession[];
  count: number;
  tool_id: string | null;
}

export interface CacheStats {
  hot_cache: {
    hit_rate: number;
    size_mb: number;
    entries: number;
    avg_latency_ms: number;
    hits: number;
    misses: number;
  };
  file_hash_cache: {
    hit_rate: number;
    size_mb: number;
    entries: number;
    avg_latency_ms: number;
    hits: number;
    misses: number;
    disk_size_mb: number;
  };
  overall: {
    total_hit_rate: number;
    total_hits: number;
    total_misses: number;
    memory_saved_mb: number;
    tokens_prevented_from_api: number;
  };
}
