import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function formatNumber(num: number | null | undefined): string {
  if (num == null) return '0';
  return new Intl.NumberFormat().format(num);
}

export function formatDuration(seconds: number): string {
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const secs = Math.floor(seconds % 60);

  if (hours > 0) {
    return `${hours}h ${minutes}m`;
  } else if (minutes > 0) {
    return `${minutes}m ${secs}s`;
  }
  return `${secs}s`;
}

export function formatTimestamp(timestamp: string): string {
  return new Date(timestamp).toLocaleTimeString();
}

export function formatDate(timestamp: string): string {
  return new Date(timestamp).toLocaleDateString();
}

export function formatPercent(value: number | null | undefined): string {
  if (value == null || isNaN(value)) return '0.0%';
  return `${value.toFixed(1)}%`;
}

export function formatCurrency(amount: number): string {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  }).format(amount);
}

interface MetricsWithAccuracy {
  total_successes?: number;
  total_failures?: number;
}

interface ServiceHealthStatus {
  status?: string;
}

export function calculateAccuracy(metrics: MetricsWithAccuracy | null | undefined): number {
  if (!metrics) return 0;
  const successes = metrics.total_successes || 0;
  const failures = metrics.total_failures || 0;
  const total = successes + failures;
  return total > 0 ? Math.round((successes / total) * 100) : 0;
}

export function calculateUptime(serviceHealth: Record<string, ServiceHealthStatus> | null | undefined): number {
  if (!serviceHealth) return 0;
  const services = Object.values(serviceHealth);
  const healthy = services.filter((s) => s?.status === 'healthy').length;
  return services.length > 0 ? Math.round((healthy / services.length) * 100) : 0;
}

export function calculateDelta(current: number = 0, previous: number = 0): string {
  const diff = current - previous;
  if (diff > 0) return `+${formatNumber(diff)}`;
  if (diff < 0) return `${formatNumber(diff)}`;
  return '0';
}
