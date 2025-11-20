import { Bot, Zap, Brain, Play, Github, Sparkles, HelpCircle } from 'lucide-react';
import type { LucideIcon } from 'lucide-react';

export interface ToolInfo {
  name: string;
  icon: LucideIcon;
  color: string;
  description?: string;
}

export const TOOL_DISPLAY_NAMES: Record<string, ToolInfo> = {
  'claude-code': {
    name: 'Claude Code',
    icon: Bot,
    color: 'purple',
    description: 'Anthropic Claude Code Assistant'
  },
  'cursor': {
    name: 'Cursor',
    icon: Zap,
    color: 'blue',
    description: 'Cursor AI Code Editor'
  },
  'codex': {
    name: 'OpenAI Codex',
    icon: Brain,
    color: 'green',
    description: 'OpenAI Codex API'
  },
  'continue': {
    name: 'Continue',
    icon: Play,
    color: 'orange',
    description: 'Continue Dev VSCode Extension'
  },
  'copilot': {
    name: 'GitHub Copilot',
    icon: Github,
    color: 'indigo',
    description: 'GitHub Copilot'
  },
  'tabnine': {
    name: 'Tabnine',
    icon: Sparkles,
    color: 'violet',
    description: 'Tabnine AI Assistant'
  },
};

export function getToolDisplay(toolId: string): ToolInfo {
  return TOOL_DISPLAY_NAMES[toolId] || {
    name: toolId || 'Unknown Tool',
    icon: HelpCircle,
    color: 'gray',
    description: 'Unknown tool'
  };
}

export function getToolColor(toolId: string): string {
  const tool = getToolDisplay(toolId);
  const colorMap: Record<string, string> = {
    purple: 'text-purple-400',
    blue: 'text-blue-400',
    green: 'text-green-400',
    orange: 'text-orange-400',
    indigo: 'text-indigo-400',
    violet: 'text-violet-400',
    gray: 'text-gray-400',
  };
  return colorMap[tool.color] || 'text-gray-400';
}

export function getToolBgColor(toolId: string): string {
  const tool = getToolDisplay(toolId);
  const colorMap: Record<string, string> = {
    purple: 'bg-purple-500/10',
    blue: 'bg-blue-500/10',
    green: 'bg-green-500/10',
    orange: 'bg-orange-500/10',
    indigo: 'bg-indigo-500/10',
    violet: 'bg-violet-500/10',
    gray: 'bg-gray-500/10',
  };
  return colorMap[tool.color] || 'bg-gray-500/10';
}

export function getToolBorderColor(toolId: string): string {
  const tool = getToolDisplay(toolId);
  const colorMap: Record<string, string> = {
    purple: 'border-purple-500/30',
    blue: 'border-blue-500/30',
    green: 'border-green-500/30',
    orange: 'border-orange-500/30',
    indigo: 'border-indigo-500/30',
    violet: 'border-violet-500/30',
    gray: 'border-gray-500/30',
  };
  return colorMap[tool.color] || 'border-gray-500/30';
}
