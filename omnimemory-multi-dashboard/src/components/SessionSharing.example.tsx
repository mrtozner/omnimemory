/**
 * Example usage of SessionSharing component
 *
 * This file demonstrates how to integrate the SessionSharing component
 * into a page or parent component.
 */

import { SessionSharing } from './SessionSharing';

export function SessionSharingExample() {
  // Example 1: Basic usage with a session ID
  const sessionId = 'sess_abc123';

  return (
    <div className="p-6">
      <h1 className="text-2xl font-bold mb-6">Session Collaboration</h1>

      <SessionSharing sessionId={sessionId} />
    </div>
  );
}

// Example 2: Integration with session selection
export function SessionSharingWithSelection() {
  const [selectedSessionId, setSelectedSessionId] = useState<string | null>(null);

  return (
    <div className="p-6 space-y-6">
      <div>
        <h2 className="text-xl font-semibold mb-4">Select a Session</h2>
        <select
          className="bg-gray-800 text-white px-4 py-2 rounded border border-gray-700"
          onChange={(e) => setSelectedSessionId(e.target.value)}
        >
          <option value="">Select a session...</option>
          <option value="sess_abc123">Session ABC123</option>
          <option value="sess_xyz789">Session XYZ789</option>
        </select>
      </div>

      {selectedSessionId && (
        <SessionSharing sessionId={selectedSessionId} />
      )}
    </div>
  );
}

// Example 3: Integration with SessionTimeline
import { useState } from 'react';
import { SessionTimeline } from './SessionTimeline';

export function SessionDetailsPage() {
  const [selectedSessionId, setSelectedSessionId] = useState<string | null>(null);

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 p-6">
      {/* Left: Session Timeline */}
      <div>
        <h2 className="text-xl font-semibold mb-4">Session History</h2>
        <SessionTimeline
          onSessionClick={(sessionId) => setSelectedSessionId(sessionId)}
        />
      </div>

      {/* Right: Session Sharing Details */}
      <div>
        {selectedSessionId ? (
          <>
            <h2 className="text-xl font-semibold mb-4">
              Session Collaboration Details
            </h2>
            <SessionSharing sessionId={selectedSessionId} />
          </>
        ) : (
          <div className="flex items-center justify-center h-full text-gray-400">
            <p>Select a session to view collaboration details</p>
          </div>
        )}
      </div>
    </div>
  );
}
