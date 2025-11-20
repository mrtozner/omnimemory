/**
 * Example usage of the ToolsList component
 * This file demonstrates how to use the ToolsList component in a page
 */

import { ToolsList } from './ToolsList';

export function ToolsListExample() {
  // Example: Using with a specific project ID
  const projectId = 'example-project-123';

  return (
    <div className="container mx-auto p-6">
      <ToolsList projectId={projectId} />
    </div>
  );
}

// Example: Using in a page with custom layout
export function ToolsPageExample() {
  return (
    <div className="min-h-screen bg-background">
      <header className="border-b border-border p-6">
        <h1 className="text-3xl font-bold text-foreground">Connected Tools</h1>
        <p className="text-muted-foreground mt-2">
          Manage IDE tools and autonomous agents connected to your projects
        </p>
      </header>

      <main className="container mx-auto p-6">
        <ToolsList projectId="my-project" />
      </main>
    </div>
  );
}
