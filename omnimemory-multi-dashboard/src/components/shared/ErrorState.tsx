import { AlertCircle, RefreshCw } from 'lucide-react';

interface ErrorStateProps {
  error: string;
  onRetry?: () => void;
  context?: string;
  compact?: boolean;
  showDetails?: boolean;
}

export function ErrorState({
  error,
  onRetry,
  context = 'Error Loading Data',
  compact = false,
  showDetails = false
}: ErrorStateProps) {
  if (compact) {
    return (
      <div className="bg-red-900/20 border border-red-500/30 rounded-lg p-3">
        <div className="flex items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            <AlertCircle className="w-4 h-4 text-red-400 flex-shrink-0" />
            <p className="text-red-200 text-sm">{error}</p>
          </div>
          {onRetry && (
            <button
              onClick={onRetry}
              className="px-3 py-2 bg-red-500/20 hover:bg-red-500/30
                         border border-red-500/50 rounded text-red-200 text-sm
                         transition-colors flex items-center gap-1 min-h-[44px]"
            >
              <RefreshCw className="w-3 h-3" />
              Retry
            </button>
          )}
        </div>
      </div>
    );
  }

  return (
    <div className="bg-red-900/20 border border-red-500/30 rounded-lg p-6">
      <div className="flex items-start gap-4">
        <AlertCircle className="w-6 h-6 text-red-400 flex-shrink-0 mt-1" />
        <div className="flex-1">
          <h3 className="text-lg font-semibold text-red-200 mb-2">
            {context}
          </h3>
          <p className="text-red-300 mb-4">
            {showDetails ? error : 'Unable to load data. Please try again.'}
          </p>
          {showDetails && (
            <details className="mb-4">
              <summary className="text-red-400 text-sm cursor-pointer hover:text-red-300">
                Technical Details
              </summary>
              <pre className="mt-2 p-3 bg-black/30 rounded text-xs text-red-300 overflow-x-auto">
                {error}
              </pre>
            </details>
          )}
          {onRetry && (
            <button
              onClick={onRetry}
              className="px-4 py-3 bg-red-500/20 hover:bg-red-500/30
                         border border-red-500/50 rounded-lg text-red-200
                         transition-colors flex items-center gap-2 min-h-[44px]"
            >
              <RefreshCw className="w-4 h-4" />
              Try Again
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

// Convenience component for network errors
export function NetworkErrorState({ onRetry }: { onRetry?: () => void }) {
  return (
    <ErrorState
      error="Unable to connect to the metrics service. Please ensure the backend is running."
      context="Connection Error"
      onRetry={onRetry}
    />
  );
}

// Convenience component for data not found
export function NotFoundErrorState({
  resource = 'data',
  onRetry
}: {
  resource?: string;
  onRetry?: () => void;
}) {
  return (
    <ErrorState
      error={`The requested ${resource} was not found.`}
      context="Not Found"
      onRetry={onRetry}
    />
  );
}
