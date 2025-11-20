import { Loader2 } from 'lucide-react';

interface LoadingStateProps {
  message?: string;
  compact?: boolean;
}

export function LoadingState({
  message = 'Loading...',
  compact = false
}: LoadingStateProps) {
  if (compact) {
    return (
      <div className="flex items-center justify-center gap-2 p-4">
        <Loader2 className="w-4 h-4 animate-spin text-purple-400" />
        <p className="text-gray-400 text-sm">{message}</p>
      </div>
    );
  }

  return (
    <div className="flex flex-col items-center justify-center gap-4 py-12">
      <Loader2 className="w-8 h-8 animate-spin text-purple-400" />
      <p className="text-gray-400">{message}</p>
    </div>
  );
}
