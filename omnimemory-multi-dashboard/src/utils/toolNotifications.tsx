import React from 'react';
import { Code2, Terminal, Sparkles, Box, Workflow, Plug } from 'lucide-react';

// Notification Types (re-exported for convenience)
export interface ToolNotification {
  id: string;
  type: 'tool_joined' | 'tool_left' | 'context_merged' | 'file_shared';
  tool_type: string;
  tool_id: string;
  message: string;
  timestamp: string;
  metadata?: {
    tools_count?: number;
    files_merged?: number;
    file_name?: string;
  };
}

// Tool Display Names
export function getToolDisplayName(toolType: string): string {
  const names: Record<string, string> = {
    'cursor': 'Cursor',
    'vscode': 'VSCode',
    'claude-code': 'Claude Code',
    'continue': 'Continue',
    'n8n-agent': 'n8n Agent',
    'custom-agent': 'Custom Agent',
    'windsurf': 'Windsurf',
    'cline': 'Cline',
  };
  return names[toolType.toLowerCase()] || toolType;
}

// Tool Icons
export function getToolIcon(toolType: string): React.ReactNode {
  const icons: Record<string, React.ReactNode> = {
    'cursor': <Terminal className="w-5 h-5 text-blue-500" />,
    'vscode': <Code2 className="w-5 h-5 text-blue-600" />,
    'claude-code': <Sparkles className="w-5 h-5 text-purple-500" />,
    'continue': <Terminal className="w-5 h-5 text-green-500" />,
    'n8n-agent': <Workflow className="w-5 h-5 text-orange-500" />,
    'custom-agent': <Plug className="w-5 h-5 text-gray-500" />,
    'windsurf': <Code2 className="w-5 h-5 text-cyan-500" />,
    'cline': <Box className="w-5 h-5 text-indigo-500" />,
  };
  return icons[toolType.toLowerCase()] || <Plug className="w-5 h-5 text-gray-500" />;
}

// Generate notification message
export function generateNotificationMessage(
  type: string,
  toolType: string,
  metadata?: ToolNotification['metadata']
): string {
  const toolName = getToolDisplayName(toolType);

  switch (type) {
    case 'tool_joined':
      return `${toolName} just joined your session`;
    case 'tool_left':
      return `${toolName} left your session`;
    case 'context_merged': {
      const count = metadata?.tools_count || 2;
      const files = metadata?.files_merged || 0;
      return `Context merged from ${count} tools (${files} files)`;
    }
    case 'file_shared': {
      const fileName = metadata?.file_name || 'a file';
      return `${toolName} shared ${fileName}`;
    }
    default:
      return 'Session update';
  }
}

// Format relative time
export function formatRelativeTime(timestamp: string): string {
  const now = new Date();
  const past = new Date(timestamp);
  const diffInSeconds = Math.floor((now.getTime() - past.getTime()) / 1000);

  if (diffInSeconds < 10) return 'just now';
  if (diffInSeconds < 60) return `${diffInSeconds} seconds ago`;

  const diffInMinutes = Math.floor(diffInSeconds / 60);
  if (diffInMinutes < 60) return diffInMinutes === 1 ? '1 minute ago' : `${diffInMinutes} minutes ago`;

  const diffInHours = Math.floor(diffInMinutes / 60);
  if (diffInHours < 24) return diffInHours === 1 ? '1 hour ago' : `${diffInHours} hours ago`;

  const diffInDays = Math.floor(diffInHours / 24);
  return diffInDays === 1 ? '1 day ago' : `${diffInDays} days ago`;
}
