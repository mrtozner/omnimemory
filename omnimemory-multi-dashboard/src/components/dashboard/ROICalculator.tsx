import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '../shared/Card';
import { formatNumber, formatCurrency } from '../../lib/utils';
import { DollarSign, CheckCircle } from 'lucide-react';

interface ROICalculatorProps {
  tokensSaved: number;
  totalOriginal: number;
  compressionRatio: number;
}

export const ROICalculator = React.memo<ROICalculatorProps>(({
  tokensSaved,
  totalOriginal,
}) => {
  // Claude Sonnet 4.5 Pricing (≤ 200K tokens)
  // Input: $3 per MTok = $0.003 per 1K
  // Output: $15 per MTok = $0.015 per 1K
  // Average (assuming 50% input, 50% output): $0.009 per 1K
  const CLAUDE_SONNET_45_INPUT_COST_PER_1K = 0.003;
  const CLAUDE_SONNET_45_OUTPUT_COST_PER_1K = 0.015;
  const CLAUDE_SONNET_45_AVG_COST_PER_1K = (CLAUDE_SONNET_45_INPUT_COST_PER_1K + CLAUDE_SONNET_45_OUTPUT_COST_PER_1K) / 2;

  const costSaved = (tokensSaved / 1000) * CLAUDE_SONNET_45_AVG_COST_PER_1K;
  const compressionPct = totalOriginal > 0
    ? (tokensSaved / totalOriginal * 100)
    : 0;

  return (
    <Card className="roi-calculator bg-gradient-to-br from-card to-card/50 border-2 border-green-500/30">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <DollarSign className="w-5 h-5 text-green-400" />
          Active Token Savings (Mem0-Style Retrieval)
        </CardTitle>
        <p className="text-sm text-muted-foreground mt-1">
          OmniMemory uses selective context retrieval to reduce API costs by 90%+. Claude Code retrieves compressed content from cache instead of reading full files.
        </p>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <ROIMetric
            title="Original Tokens"
            value={formatNumber(totalOriginal)}
          />
          <ROIMetric
            title="Tokens Saved"
            value={formatNumber(tokensSaved)}
          />
          <ROIMetric
            title="Compression Rate"
            value={`${compressionPct.toFixed(1)}%`}
          />
          <ROIMetric
            title="Cost Savings (Sonnet 4.5)"
            value={formatCurrency(costSaved)}
            note={
              <span className="flex items-center gap-1">
                <CheckCircle className="w-3 h-3" />
                Active via MCP omnimemory_get_context
              </span>
            }
            highlight={true}
          />
        </div>
        <div className="mt-4 p-3 bg-green-500/10 border border-green-500/30 rounded-lg text-sm text-green-200 flex items-start gap-2">
          <CheckCircle className="w-4 h-4 flex-shrink-0 mt-0.5" />
          <div>
            <strong>Cost optimization active!</strong> Claude Code uses <code className="bg-black/30 px-1 py-0.5 rounded">omnimemory_get_context</code> to retrieve compressed file content from cache, avoiding full file reads and reducing API token usage by {compressionPct.toFixed(0)}%.
          </div>
        </div>
      </CardContent>
    </Card>
  );
});

ROICalculator.displayName = 'ROICalculator';

interface ROIMetricProps {
  title: string;
  value: string;
  note?: React.ReactNode;
  highlight?: boolean;
}

const ROIMetric = React.memo<ROIMetricProps>(({
  title,
  value,
  note,
  highlight = false,
}) => (
  <div
    className={`p-4 rounded-lg text-center transition-all ${
      highlight
        ? 'bg-gradient-to-br from-primary/10 to-primary/5 border-2 border-primary animate-pulse-glow'
        : 'bg-muted/30 border border-border'
    }`}
  >
    <p className="text-xs text-muted-foreground uppercase tracking-wide mb-2">
      {title}
    </p>
    <p
      className={`text-2xl font-bold ${
        highlight
          ? 'text-primary'
          : 'text-foreground'
      }`}
    >
      {value}
    </p>
    {note && (
      <p className="text-xs text-muted-foreground mt-1">{note}</p>
    )}
  </div>
));

ROIMetric.displayName = 'ROIMetric';
