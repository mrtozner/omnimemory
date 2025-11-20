import { Link } from 'react-router-dom';
import { Card, CardHeader, CardTitle, CardContent } from '../shared/Card';
import { formatNumber } from '../../lib/utils';
import { Activity, TrendingUp, Zap, DollarSign } from 'lucide-react';
import type { Session } from '../../services/api';

// Tool metrics interface
interface ToolMetrics {
  total_embeddings: number;
  total_compressions: number;
  tokens_saved: number;
  cache_hit_rate: number;
  compression_ratio: number;
  avg_compression_ratio: number;
  quality_score: number;
  avg_quality_score: number;
  pattern_count: number;
  graph_edges: number;
  total_successes: number;
  total_failures: number;
  total_original_tokens: number;
  total_compressed_tokens: number;
  cache_hits: number;
  tokens_processed: number;
  prediction_accuracy: number;
}

interface ToolCardProps {
  name: string;
  toolId: string;
  icon: React.ReactNode;
  color: 'blue' | 'green' | 'purple' | 'orange';
  metrics: ToolMetrics;
  sessions: Session[];
}

const colorClasses = {
  blue: {
    border: 'border-blue-500',
    bg: 'bg-blue-500/10',
    text: 'text-blue-500',
    hover: 'hover:border-blue-400',
  },
  green: {
    border: 'border-green-500',
    bg: 'bg-green-500/10',
    text: 'text-green-500',
    hover: 'hover:border-green-400',
  },
  purple: {
    border: 'border-purple-500',
    bg: 'bg-purple-500/10',
    text: 'text-purple-500',
    hover: 'hover:border-purple-400',
  },
  orange: {
    border: 'border-orange-500',
    bg: 'bg-orange-500/10',
    text: 'text-orange-500',
    hover: 'hover:border-orange-400',
  },
};

export function ToolCard({ name, toolId, icon, color, metrics, sessions }: ToolCardProps) {
  const activeSessions = sessions.filter((s) => !s.ended_at);
  const colors = colorClasses[color];

  // Calculate estimated cost savings (assuming $0.001 per 1K tokens)
  const estimatedCost = (metrics.tokens_saved / 1000) * 0.001;

  return (
    <Card className={`border-t-4 ${colors.border} ${colors.hover} transition-all duration-200 hover:shadow-xl`}>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className={`p-2 rounded-lg ${colors.bg}`}>
              {icon}
            </div>
            <div>
              <CardTitle className="text-xl">{name}</CardTitle>
              <p className="text-sm text-muted-foreground mt-1">
                Real-time context memory
              </p>
            </div>
          </div>
          <div className={`px-3 py-1 rounded-full ${colors.bg} ${colors.text} text-sm font-medium flex items-center gap-1`}>
            <Activity className="h-3 w-3" />
            {activeSessions.length} active
          </div>
        </div>
      </CardHeader>

      <CardContent>
        <div className="space-y-4">
          {/* Metrics Grid */}
          <div className="grid grid-cols-2 gap-3">
            <MetricItem
              icon={<Zap className="h-4 w-4" />}
              label="Embeddings"
              value={formatNumber(metrics.total_embeddings)}
              color={colors.text}
            />
            <MetricItem
              icon={<TrendingUp className="h-4 w-4" />}
              label="Tokens Saved"
              value={formatNumber(metrics.tokens_saved)}
              color={colors.text}
              highlight
            />
            <MetricItem
              icon={<Activity className="h-4 w-4" />}
              label="Compressions"
              value={formatNumber(metrics.total_compressions)}
              color={colors.text}
            />
            <MetricItem
              icon={<DollarSign className="h-4 w-4" />}
              label="Est. Savings"
              value={`$${estimatedCost.toFixed(2)}`}
              color={colors.text}
            />
          </div>

          {/* Recent Sessions Preview */}
          <div className="pt-4 border-t border-gray-700">
            <h3 className="text-sm font-medium text-muted-foreground mb-3">
              Recent Sessions
            </h3>
            <div className="space-y-2">
              {sessions.slice(0, 3).map((session) => (
                <SessionPreview key={session.session_id} session={session} />
              ))}
            </div>
          </div>

          {/* View Details Link */}
          <Link
            to={`/${toolId}`}
            className={`block w-full text-center py-2 px-4 rounded-lg border ${colors.border} ${colors.text} hover:${colors.bg} transition-colors font-medium`}
          >
            View Details →
          </Link>
        </div>
      </CardContent>
    </Card>
  );
}

interface MetricItemProps {
  icon: React.ReactNode;
  label: string;
  value: string | number;
  color: string;
  highlight?: boolean;
}

function MetricItem({ icon, label, value, color, highlight }: MetricItemProps) {
  return (
    <div className={`p-3 rounded-lg ${highlight ? 'bg-gray-800/80 border border-gray-700' : 'bg-gray-800/50'}`}>
      <div className="flex items-center gap-2 mb-1">
        <span className={color}>{icon}</span>
        <span className="text-xs text-muted-foreground">{label}</span>
      </div>
      <div className={`text-lg font-bold ${highlight ? color : ''}`}>
        {value}
      </div>
    </div>
  );
}

function SessionPreview({ session }: { session: Session }) {
  const isActive = !session.ended_at;
  const duration = isActive
    ? Math.floor((Date.now() - new Date(session.started_at).getTime()) / 1000 / 60)
    : Math.floor(
        (new Date(session.ended_at!).getTime() - new Date(session.started_at).getTime()) / 1000 / 60
      );

  return (
    <div className="flex items-center justify-between p-2 rounded bg-gray-800/30 text-sm">
      <div className="flex items-center gap-2">
        <span
          className={`w-2 h-2 rounded-full ${
            isActive ? 'bg-green-500 animate-pulse' : 'bg-gray-500'
          }`}
        />
        <span className="text-muted-foreground">
          {new Date(session.started_at).toLocaleTimeString()}
        </span>
      </div>
      <span className="font-medium">
        {duration}m | {formatNumber(session.tokens_saved)} tokens
      </span>
    </div>
  );
}
