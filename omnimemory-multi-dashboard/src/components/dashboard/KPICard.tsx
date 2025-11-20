import React from 'react';
import { cn } from '../../lib/utils';

interface KPICardProps {
  title: string;
  value: string | number;
  subtitle?: string;
  delta?: string;
  icon?: React.ReactNode;
  trend?: 'up' | 'down' | 'neutral';
  className?: string;
}

export const KPICard = React.memo<KPICardProps>(({
  title,
  value,
  subtitle,
  delta,
  icon,
  trend = 'neutral',
  className,
}) => {
  const trendColors = {
    up: 'bg-green-500/20 text-green-400 border-green-500/50',
    down: 'bg-red-500/20 text-red-400 border-red-500/50',
    neutral: 'bg-gray-500/20 text-gray-400 border-gray-500/50',
  };

  return (
    <div
      className={cn(
        'rounded-lg border bg-card p-6 shadow-sm transition-all hover:border-primary/50 hover:-translate-y-1',
        className
      )}
    >
      <div className="flex items-center justify-between mb-4">
        <p className="text-xs text-muted-foreground uppercase tracking-wide font-medium">
          {title}
        </p>
        {icon && <div className="text-muted-foreground opacity-70">{icon}</div>}
      </div>

      <p className="text-3xl font-bold tracking-tight text-primary mb-2">
        {value}
      </p>

      {subtitle && (
        <p className="text-sm text-muted-foreground mb-2">{subtitle}</p>
      )}

      {delta && (
        <span
          className={cn(
            'inline-block px-2 py-1 rounded text-xs font-bold border',
            trendColors[trend]
          )}
        >
          {delta}
        </span>
      )}
    </div>
  );
});

KPICard.displayName = 'KPICard';
