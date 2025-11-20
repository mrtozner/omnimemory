import { memo, useState, useEffect } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { Settings, Sparkles, Award, FileText, Brain, DollarSign, History, Menu, X, Layers, Users } from 'lucide-react';
import { cn } from '../../lib/utils';

const navigation = [
  { name: 'Overview', href: '/overview', icon: Sparkles },
  { name: 'File Context', href: '/file-context', icon: FileText },
  { name: 'API Savings', href: '/api-savings', icon: DollarSign },
  { name: 'Cache Performance', href: '/cache', icon: Layers },
  { name: 'Unified Intelligence', href: '/unified-intelligence', icon: Brain },
  { name: 'Team Collaboration', href: '/team', icon: Users },
  { name: 'Session History', href: '/sessions', icon: History },
  { name: 'Benchmarks', href: '/benchmark', icon: Award },
  { name: 'Settings', href: '/settings', icon: Settings },
];

export const Sidebar = memo(function Sidebar() {
  const location = useLocation();
  const [isOpen, setIsOpen] = useState(false);

  // Auto-close sidebar on route change (mobile only)
  useEffect(() => {
    setIsOpen(false);
  }, [location.pathname]);

  return (
    <>
      {/* Mobile Menu Button */}
      <button
        onClick={() => setIsOpen(true)}
        className="md:hidden fixed top-4 left-4 z-40 p-3 rounded-lg bg-card border border-border hover:bg-accent transition-colors min-w-[44px] min-h-[44px] flex items-center justify-center"
        aria-label="Open menu"
      >
        <Menu className="h-6 w-6" />
      </button>

      {/* Mobile Backdrop */}
      {isOpen && (
        <div
          className="md:hidden fixed inset-0 bg-black/50 z-40 transition-opacity"
          onClick={() => setIsOpen(false)}
          aria-hidden="true"
        />
      )}

      {/* Sidebar */}
      <div
        className={cn(
          'fixed md:static inset-y-0 left-0 z-50',
          'flex h-screen w-64 flex-col border-r bg-card',
          'transform transition-transform duration-300 ease-in-out',
          isOpen ? 'translate-x-0' : '-translate-x-full md:translate-x-0'
        )}
      >
        {/* Header with Close Button */}
        <div className="flex h-16 items-center border-b px-6 relative">
          <h1 className="text-xl font-bold text-foreground">OmniMemory</h1>

          {/* Mobile Close Button */}
          <button
            onClick={() => setIsOpen(false)}
            className="md:hidden absolute top-4 right-4 p-2 rounded-lg hover:bg-white/10 min-w-[44px] min-h-[44px] flex items-center justify-center"
            aria-label="Close menu"
          >
            <X className="w-6 h-6" />
          </button>
        </div>

        {/* Navigation */}
        <nav className="flex-1 space-y-1 px-3 py-4">
          {navigation.map((item) => {
            const isActive = location.pathname === item.href;
            return (
              <Link
                key={item.name}
                to={item.href}
                className={cn(
                  'flex items-center gap-3 rounded-md px-3 py-3 text-sm font-medium transition-colors min-h-[44px]',
                  isActive
                    ? 'bg-primary text-primary-foreground'
                    : 'text-muted-foreground hover:bg-accent hover:text-foreground'
                )}
              >
                <item.icon className="h-5 w-5" />
                {item.name}
              </Link>
            );
          })}
        </nav>

        {/* Footer */}
        <div className="border-t p-4">
          <div className="text-xs text-muted-foreground">
            <p className="font-medium">Multi-Tool Dashboard</p>
            <p className="mt-1">v1.0.0 - Phase 1</p>
          </div>
        </div>
      </div>
    </>
  );
});
