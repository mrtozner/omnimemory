import { create } from 'zustand';
import type { ToolConfig } from '../types/metrics';

interface ConfigState {
  tools: ToolConfig[];
  selectedTool: string;
  darkMode: boolean;
  setSelectedTool: (tool: string) => void;
  toggleTool: (toolId: string) => void;
  toggleDarkMode: () => void;
}

export const useConfigStore = create<ConfigState>((set) => ({
  tools: [
    {
      id: 'claude-code',
      name: 'Claude Code',
      enabled: true,
      color: '#E07B39',
      icon: 'Bot',
    },
    {
      id: 'cursor',
      name: 'Cursor',
      enabled: false,
      color: '#00A3FF',
      icon: 'Zap',
    },
    {
      id: 'codex',
      name: 'Codex',
      enabled: false,
      color: '#10B981',
      icon: 'Sparkles',
    },
    {
      id: 'vscode',
      name: 'VS Code',
      enabled: false,
      color: '#007ACC',
      icon: 'FileText',
    },
  ],
  selectedTool: 'claude-code',
  darkMode: true,
  setSelectedTool: (tool) => set({ selectedTool: tool }),
  toggleTool: (toolId) =>
    set((state) => ({
      tools: state.tools.map((tool) =>
        tool.id === toolId ? { ...tool, enabled: !tool.enabled } : tool
      ),
    })),
  toggleDarkMode: () => set((state) => ({ darkMode: !state.darkMode })),
}));
