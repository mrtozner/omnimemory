import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '../shared/Card';
import { CircularProgressbar, buildStyles } from 'react-circular-progressbar';
import 'react-circular-progressbar/dist/styles.css';

interface CompressionGaugesProps {
  compressionRatio: number;  // 0-100
  qualityScore: number;      // 0-100
}

export const CompressionGauges = React.memo<CompressionGaugesProps>(({
  compressionRatio,
  qualityScore,
}) => {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Compression Performance</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-2 gap-8">
          {/* Compression Ratio Gauge */}
          <div className="text-center">
            <div className="w-48 h-48 mx-auto mb-4">
              <CircularProgressbar
                value={compressionRatio}
                text={`${compressionRatio.toFixed(1)}%`}
                styles={buildStyles({
                  textSize: '16px',
                  pathColor: 'hsl(var(--primary))',
                  textColor: 'hsl(var(--primary))',
                  trailColor: 'hsl(var(--muted))',
                  backgroundColor: 'hsl(var(--background))',
                })}
              />
            </div>
            <p className="text-sm font-medium text-foreground">
              Compression Ratio
            </p>
            <p className="text-xs text-muted-foreground mt-1">
              Target: 94.4%
            </p>
          </div>

          {/* Quality Score Gauge */}
          <div className="text-center">
            <div className="w-48 h-48 mx-auto mb-4">
              <CircularProgressbar
                value={qualityScore}
                text={`${qualityScore.toFixed(1)}%`}
                styles={buildStyles({
                  textSize: '16px',
                  pathColor: '#10b981',
                  textColor: '#10b981',
                  trailColor: 'hsl(var(--muted))',
                  backgroundColor: 'hsl(var(--background))',
                })}
              />
            </div>
            <p className="text-sm font-medium text-foreground">
              Quality Score
            </p>
            <p className="text-xs text-muted-foreground mt-1">
              Semantic preservation
            </p>
          </div>
        </div>
      </CardContent>
    </Card>
  );
});

CompressionGauges.displayName = 'CompressionGauges';
