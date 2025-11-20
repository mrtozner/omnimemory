// Unified Intelligence System Types

export interface Prediction {
  prediction_type: string;
  predicted_item: string;
  confidence: number;
  source: 'file_context' | 'agent_memory' | 'cross_memory';
}

export interface PredictionMetrics {
  total_predictions: number;
  avg_confidence: number;
  avg_execution_time_ms: number;
}

export interface SourceContributions {
  file_context: number;
  agent_memory: number;
  cross_memory: number;
}

export interface PredictionsResponse {
  predictions: Prediction[];
  metrics: PredictionMetrics;
  source_contributions: SourceContributions;
  timestamp: string;
}

export interface QueryTypes {
  FILE_SEARCH: number;
  TASK_CONTEXT: number;
  PREDICTION: number;
  MIXED: number;
}

export interface SourcesUsed {
  file_context_only: number;
  agent_memory_only: number;
  cross_memory: number;
}

export interface OrchestrationResponse {
  total_queries: number;
  avg_orchestration_overhead_ms: number;
  cache_hit_rate: number;
  query_types: QueryTypes;
  sources_used: SourcesUsed;
  timestamp: string;
}

export interface FeedbackMetrics {
  generated: number;
  shown: number;
  accepted: number;
  acceptance_rate: number;
}

export interface FeedbackByType {
  next_action: FeedbackMetrics;
  tool_recommendation: FeedbackMetrics;
  file_prefetch: FeedbackMetrics;
  workflow_hint: FeedbackMetrics;
}

export interface SuggestionsResponse {
  total_suggestions_generated: number;
  suggestions_shown: number;
  suggestions_accepted: number;
  acceptance_rate: number;
  false_positive_rate: number;
  avg_generation_time_ms: number;
  feedback_by_type: FeedbackByType;
  timestamp: string;
}

export interface PatternCorrelation {
  pattern: string;
  correlation_strength: number;
  occurrences: number;
}

export interface InsightsResponse {
  pattern_library_size: number;
  patterns_detected_today: number;
  top_correlations: PatternCorrelation[];
  learning_rate: number;
  model_accuracy: number;
  timestamp: string;
}
