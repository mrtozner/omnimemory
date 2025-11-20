import { useState } from 'react';
import { Pin, Archive, Download, Loader2, Check, X } from 'lucide-react';

interface Session {
  session_id: string;
  tool_id: string;
  pinned: boolean;
  archived: boolean;
  ended_at?: string;
}

interface SessionManagerProps {
  session: Session;
  onUpdate?: () => void;
}

export function SessionManager({ session, onUpdate }: SessionManagerProps) {
  const [pinning, setPinning] = useState(false);
  const [archiving, setArchiving] = useState(false);
  const [exporting, setExporting] = useState(false);
  const [toast, setToast] = useState<{ message: string; type: 'success' | 'error' } | null>(null);

  const showToast = (message: string, type: 'success' | 'error') => {
    setToast({ message, type });
    setTimeout(() => setToast(null), 3000);
  };

  const handlePin = async () => {
    try {
      setPinning(true);
      const endpoint = session.pinned ? 'unpin' : 'pin';
      const response = await fetch(
        `http://localhost:8003/sessions/${session.session_id}/${endpoint}`,
        { method: 'POST' }
      );

      if (!response.ok) throw new Error('Failed to update pin status');

      showToast(
        session.pinned ? 'Session unpinned' : 'Session pinned successfully',
        'success'
      );
      onUpdate?.();
    } catch (err) {
      showToast(
        err instanceof Error ? err.message : 'Failed to update pin status',
        'error'
      );
    } finally {
      setPinning(false);
    }
  };

  const handleArchive = async () => {
    try {
      setArchiving(true);
      const endpoint = session.archived ? 'unarchive' : 'archive';
      const response = await fetch(
        `http://localhost:8003/sessions/${session.session_id}/${endpoint}`,
        { method: 'POST' }
      );

      if (!response.ok) throw new Error('Failed to update archive status');

      showToast(
        session.archived ? 'Session unarchived' : 'Session archived successfully',
        'success'
      );
      onUpdate?.();
    } catch (err) {
      showToast(
        err instanceof Error ? err.message : 'Failed to update archive status',
        'error'
      );
    } finally {
      setArchiving(false);
    }
  };

  const handleExport = async () => {
    try {
      setExporting(true);
      const response = await fetch(
        `http://localhost:8003/sessions/${session.session_id}/export`
      );

      if (!response.ok) throw new Error('Failed to export session');

      const data = await response.json();

      // Create download link
      const blob = new Blob([JSON.stringify(data, null, 2)], {
        type: 'application/json',
      });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `session-${session.session_id}.json`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);

      showToast('Session exported successfully', 'success');
    } catch (err) {
      showToast(
        err instanceof Error ? err.message : 'Failed to export session',
        'error'
      );
    } finally {
      setExporting(false);
    }
  };

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap gap-3">
        <ActionButton
          icon={pinning ? <Loader2 className="animate-spin" /> : <Pin />}
          label={session.pinned ? 'Unpin' : 'Pin Session'}
          onClick={handlePin}
          disabled={pinning}
          variant={session.pinned ? 'warning' : 'default'}
        />

        <ActionButton
          icon={archiving ? <Loader2 className="animate-spin" /> : <Archive />}
          label={session.archived ? 'Unarchive' : 'Archive Session'}
          onClick={handleArchive}
          disabled={archiving}
          variant={session.archived ? 'secondary' : 'default'}
        />

        <ActionButton
          icon={exporting ? <Loader2 className="animate-spin" /> : <Download />}
          label="Export Session"
          onClick={handleExport}
          disabled={exporting}
          variant="default"
        />
      </div>

      {/* Toast Notification */}
      {toast && (
        <Toast
          message={toast.message}
          type={toast.type}
          onClose={() => setToast(null)}
        />
      )}
    </div>
  );
}

// Action Button Component
interface ActionButtonProps {
  icon: React.ReactNode;
  label: string;
  onClick: () => void;
  disabled?: boolean;
  variant?: 'default' | 'warning' | 'secondary';
}

function ActionButton({ icon, label, onClick, disabled, variant = 'default' }: ActionButtonProps) {
  const baseClasses = 'flex items-center gap-2 px-4 py-3 rounded-lg font-medium transition-all min-h-[44px]';

  const variantClasses = {
    default: 'bg-blue-600 hover:bg-blue-700 text-white',
    warning: 'bg-yellow-600 hover:bg-yellow-700 text-white',
    secondary: 'bg-gray-700 hover:bg-gray-600 text-gray-200',
  };

  const disabledClasses = 'opacity-50 cursor-not-allowed';

  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className={`${baseClasses} ${variantClasses[variant]} ${disabled ? disabledClasses : ''}`}
    >
      <span className="h-4 w-4">{icon}</span>
      {label}
    </button>
  );
}

// Toast Component
interface ToastProps {
  message: string;
  type: 'success' | 'error';
  onClose: () => void;
}

function Toast({ message, type, onClose }: ToastProps) {
  return (
    <div className="fixed bottom-4 right-4 z-50 animate-slide-up">
      <div
        className={`flex items-center gap-3 px-4 py-3 rounded-lg shadow-lg ${
          type === 'success'
            ? 'bg-green-600 text-white'
            : 'bg-red-600 text-white'
        }`}
      >
        <span className="h-5 w-5">
          {type === 'success' ? <Check /> : <X />}
        </span>
        <span className="font-medium">{message}</span>
        <button
          onClick={onClose}
          className="ml-2 text-white/80 hover:text-white p-2 min-w-[44px] min-h-[44px] flex items-center justify-center"
        >
          <X className="h-4 w-4" />
        </button>
      </div>
    </div>
  );
}
