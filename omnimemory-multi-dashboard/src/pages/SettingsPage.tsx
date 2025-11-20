import { Settings as SettingsIcon } from 'lucide-react';
import { useConfigStore } from '../stores/configStore';

export function SettingsPage() {
  const { tools, toggleTool } = useConfigStore();

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Settings</h1>
        <p className="text-muted-foreground mt-2">
          Configure tool integrations and dashboard preferences
        </p>
      </div>

      <div className="space-y-4">
        <div className="rounded-lg border bg-card p-6">
          <h2 className="text-xl font-semibold mb-4">Tool Integrations</h2>
          <div className="space-y-3">
            {tools.map((tool) => (
              <div
                key={tool.id}
                className="flex items-center justify-between p-3 rounded-md border"
              >
                <div className="flex items-center gap-3">
                  <span className="text-2xl">{tool.icon}</span>
                  <div>
                    <p className="font-medium">{tool.name}</p>
                    <p className="text-sm text-muted-foreground">
                      {tool.enabled ? 'Enabled' : 'Disabled'}
                    </p>
                  </div>
                </div>
                <button
                  onClick={() => toggleTool(tool.id)}
                  className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
                    tool.enabled
                      ? 'bg-primary text-primary-foreground hover:bg-primary/90'
                      : 'bg-muted text-muted-foreground hover:bg-muted/80'
                  }`}
                >
                  {tool.enabled ? 'Enabled' : 'Disabled'}
                </button>
              </div>
            ))}
          </div>
        </div>

        <div className="rounded-lg border bg-card p-12">
          <div className="flex flex-col items-center justify-center text-center space-y-4">
            <SettingsIcon className="h-16 w-16 text-muted-foreground" />
            <div>
              <h2 className="text-2xl font-bold">Additional Settings - Coming Soon</h2>
              <p className="text-muted-foreground mt-2 max-w-md">
                Advanced configuration options will be available in Phase 3
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
