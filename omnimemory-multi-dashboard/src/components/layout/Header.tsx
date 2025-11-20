import { memo } from 'react';
import { useConfigStore } from '../../stores/configStore';
import { Moon, Sun } from 'lucide-react';

export const Header = memo(function Header() {
  const { darkMode, toggleDarkMode } = useConfigStore();

  return (
    <header className="sticky top-0 z-10 flex h-16 items-center border-b bg-card px-6">
      <div className="flex-1">
        <h2 className="text-lg font-semibold">Dashboard</h2>
      </div>
      <div className="flex items-center gap-4">
        <button
          onClick={toggleDarkMode}
          className="rounded-md p-2 hover:bg-accent transition-colors"
          aria-label="Toggle dark mode"
        >
          {darkMode ? (
            <Sun className="h-5 w-5" />
          ) : (
            <Moon className="h-5 w-5" />
          )}
        </button>
      </div>
    </header>
  );
});
