import { useState, useEffect } from 'react';

const GATEWAY_BASE = 'http://localhost:8009/api/v1';

export interface ToolInfo {
  tool_id: string;
  tool_type: string;
  last_activity: string;
  capabilities: Record<string, unknown>;
}

export interface UseToolRegistryReturn {
  ideTools: ToolInfo[];
  agents: ToolInfo[];
  loading: boolean;
  error: string | null;
  refetch: () => Promise<void>;
}

/**
 * Hook to fetch and manage tool registry data from the multi-tool gateway
 *
 * Provides access to both IDE tools and agents registered in the system.
 * Supports manual refetch for real-time updates.
 *
 * @example
 * ```tsx
 * const { ideTools, agents, loading, error, refetch } = useToolRegistry('my-project');
 *
 * return (
 *   <div>
 *     {loading ? (
 *       <div>Loading tools...</div>
 *     ) : error ? (
 *       <div>Error: {error}</div>
 *     ) : (
 *       <>
 *         <h3>IDE Tools ({ideTools.length})</h3>
 *         {ideTools.map(tool => <ToolCard key={tool.tool_id} tool={tool} />)}
 *
 *         <h3>Agents ({agents.length})</h3>
 *         {agents.map(agent => <AgentCard key={agent.tool_id} agent={agent} />)}
 *       </>
 *     )}
 *   </div>
 * );
 * ```
 */
export function useToolRegistry(projectId: string): UseToolRegistryReturn {
  const [ideTools, setIdeTools] = useState<ToolInfo[]>([]);
  const [agents, setAgents] = useState<ToolInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchTools = async () => {
    try {
      setLoading(true);
      setError(null);

      const response = await fetch(
        `${GATEWAY_BASE}/projects/${projectId}/all-tools`
      );

      if (!response.ok) {
        throw new Error(`Failed to fetch tools: ${response.statusText}`);
      }

      const data = await response.json();
      setIdeTools(data.ide_tools || []);
      setAgents(data.agents || []);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
      console.error('[useToolRegistry] Error fetching tools:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (!projectId) {
      setError('Project ID is required');
      setLoading(false);
      return;
    }

    fetchTools();
    // fetchTools is wrapped in useCallback to prevent infinite loops
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [projectId]);

  return {
    ideTools,
    agents,
    loading,
    error,
    refetch: fetchTools
  };
}
