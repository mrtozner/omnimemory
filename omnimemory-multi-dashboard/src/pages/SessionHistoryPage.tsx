import { useState } from 'react';
import { SessionTimeline } from '../components/SessionTimeline';
import { SessionDetails } from '../components/SessionDetails';
import { X } from 'lucide-react';

export function SessionHistoryPage() {
  const [selectedSessionId, setSelectedSessionId] = useState<string | null>(null);
  const [toolFilter, setToolFilter] = useState<string | undefined>(undefined);

  return (
    <div className="p-6 space-y-6">
      {/* Page Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-100">Session History</h1>
          <p className="text-gray-400 mt-1">
            View and manage all your OmniMemory sessions
          </p>
        </div>

        {/* Tool Filter */}
        <select
          value={toolFilter || ''}
          onChange={(e) => setToolFilter(e.target.value || undefined)}
          className="px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg text-gray-200"
        >
          <option value="">All Tools</option>
          <option value="claude-code">Claude Code</option>
          <option value="cursor">Cursor</option>
          <option value="codex">Codex</option>
        </select>
      </div>

      {/* Main Content */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Timeline - Left Side (2/3 width on large screens) */}
        <div className={selectedSessionId ? 'lg:col-span-1' : 'lg:col-span-3'}>
          <SessionTimeline
            toolId={toolFilter}
            limit={50}
            onSessionClick={(sessionId) => setSelectedSessionId(sessionId)}
          />
        </div>

        {/* Details - Right Side (1/3 width on large screens) */}
        {selectedSessionId && (
          <div className="lg:col-span-2">
            <div className="sticky top-6">
              <div className="bg-gray-800 rounded-lg p-1 mb-4">
                <button
                  onClick={() => setSelectedSessionId(null)}
                  className="w-full flex items-center justify-center gap-2 px-4 py-2 text-gray-400 hover:text-gray-200 transition-colors"
                >
                  <X className="h-4 w-4" />
                  Close Details
                </button>
              </div>
              <SessionDetails
                sessionId={selectedSessionId}
                onClose={() => setSelectedSessionId(null)}
              />
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
