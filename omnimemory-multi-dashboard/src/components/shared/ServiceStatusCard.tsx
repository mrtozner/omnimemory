import { Card, CardContent } from './Card';
import { Button } from './Button';
import { cn } from '../../lib/utils';
import { Server, Eye } from 'lucide-react';

interface ServiceStatusCardProps {
  name: string;
  port: number;
  status: 'healthy' | 'error' | 'unknown';
  responseTime?: number;
  uptime?: string;
  onViewDetails?: () => void;
  className?: string;
}

export function ServiceStatusCard({
  name,
  port,
  status,
  responseTime,
  uptime,
  onViewDetails,
  className,
}: ServiceStatusCardProps) {
  const statusColor = {
    healthy: 'bg-green-500',
    error: 'bg-red-500',
    unknown: 'bg-yellow-500',
  }[status];

  const statusText = {
    healthy: 'Healthy',
    error: 'Error',
    unknown: 'Unknown',
  }[status];

  const statusTextColor = {
    healthy: 'text-green-500',
    error: 'text-red-500',
    unknown: 'text-yellow-500',
  }[status];

  return (
    <Card className={cn('transition-all hover:shadow-lg', className)}>
      <CardContent className="pt-6">
        <div className="flex items-start justify-between mb-4">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-muted rounded-lg">
              <Server className="h-5 w-5 text-muted-foreground" />
            </div>
            <div>
              <h3 className="text-lg font-semibold">{name}</h3>
              <p className="text-sm text-muted-foreground">Port {port}</p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <span className={cn('text-xs font-medium', statusTextColor)}>
              {statusText}
            </span>
            <div className={cn('w-3 h-3 rounded-full', statusColor)} />
          </div>
        </div>

        <div className="space-y-2 text-sm">
          <div className="flex justify-between items-center">
            <span className="text-muted-foreground">Response Time</span>
            <span className="font-medium">
              {responseTime !== undefined ? `${responseTime}ms` : 'N/A'}
            </span>
          </div>
          <div className="flex justify-between items-center">
            <span className="text-muted-foreground">Uptime</span>
            <span className="font-medium">{uptime || 'N/A'}</span>
          </div>
        </div>

        {onViewDetails && (
          <Button
            variant="secondary"
            size="sm"
            className="w-full mt-4"
            onClick={onViewDetails}
          >
            <Eye className="h-4 w-4 mr-2" />
            View Details
          </Button>
        )}
      </CardContent>
    </Card>
  );
}
