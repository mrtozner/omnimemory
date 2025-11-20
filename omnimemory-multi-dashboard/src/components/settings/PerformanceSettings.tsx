import { useEffect, useState } from 'react';
import {
  Card,
  CardHeader,
  CardTitle,
  CardDescription,
  CardContent
} from '../shared/Card';
import { Button } from '../shared/Button';
import { Switch } from '../shared/Switch';
import { Input } from '../shared/Input';
import { Label } from '../shared/Label';
import { cn } from '../../lib/utils';
import {
  Activity,
  Settings,
  Zap,
  Clock,
  Filter,
  CheckCircle,
  XCircle,
  AlertCircle,
  Save,
  RotateCcw
} from 'lucide-react';

// Type Definitions
interface FeatureSettings {
  compression: boolean;
  embeddings: boolean;
  workflows: boolean;
  response_cache: boolean;
}

interface Settings {
  metrics_streaming: boolean;
  collection_interval_seconds: number;
  max_events_per_minute: number;
  features: FeatureSettings;
  performance_profile: string;
}

interface PerformanceProfile {
  name: string;
  display_name: string;
  description: string;
  settings: Partial<Settings>;
  color: 'green' | 'blue' | 'gray' | 'red';
  icon: 'zap' | 'activity' | 'clock' | 'x';
}

interface MessageState {
  type: 'success' | 'error' | 'info';
  text: string;
}

const API_BASE_URL = 'http://localhost:8003';

// Profile icon mapping
const ProfileIcon = ({ icon }: { icon: string }) => {
  switch (icon) {
    case 'zap':
      return <Zap className="h-5 w-5" />;
    case 'activity':
      return <Activity className="h-5 w-5" />;
    case 'clock':
      return <Clock className="h-5 w-5" />;
    case 'x':
      return <XCircle className="h-5 w-5" />;
    default:
      return <Settings className="h-5 w-5" />;
  }
};

// Profile color mapping
const getProfileColorClass = (color: string, selected: boolean) => {
  const baseClasses = 'transition-all duration-200';

  if (selected) {
    switch (color) {
      case 'green':
        return `${baseClasses} bg-green-600 border-green-500 text-white`;
      case 'blue':
        return `${baseClasses} bg-blue-600 border-blue-500 text-white`;
      case 'gray':
        return `${baseClasses} bg-gray-600 border-gray-500 text-white`;
      case 'red':
        return `${baseClasses} bg-red-600 border-red-500 text-white`;
      default:
        return `${baseClasses} bg-purple-600 border-purple-500 text-white`;
    }
  }

  return `${baseClasses} bg-gray-800 border-gray-600 text-gray-300 hover:bg-gray-700 hover:border-gray-500`;
};

export function PerformanceSettings() {
  // State Management
  const [settings, setSettings] = useState<Settings | null>(null);
  const [profiles, setProfiles] = useState<PerformanceProfile[]>([]);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [resetting, setResetting] = useState(false);
  const [message, setMessage] = useState<MessageState | null>(null);

  // Fetch current settings on mount
  useEffect(() => {
    fetchSettings();
    fetchProfiles();
  }, []);

  // Auto-dismiss messages after 5 seconds
  useEffect(() => {
    if (message) {
      const timer = setTimeout(() => setMessage(null), 5000);
      return () => clearTimeout(timer);
    }
  }, [message]);

  // API Functions
  const fetchSettings = async () => {
    setLoading(true);
    try {
      const response = await fetch(`${API_BASE_URL}/settings`);
      if (!response.ok) {
        throw new Error(`Failed to fetch settings: ${response.statusText}`);
      }
      const data = await response.json();
      setSettings(data.settings);
    } catch (error) {
      setMessage({
        type: 'error',
        text: error instanceof Error ? error.message : 'Failed to fetch settings'
      });
    } finally {
      setLoading(false);
    }
  };

  const fetchProfiles = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/settings/profiles`);
      if (!response.ok) {
        throw new Error(`Failed to fetch profiles: ${response.statusText}`);
      }
      const data = await response.json();
      setProfiles(data.profiles || []);
    } catch (error) {
      console.error('Failed to fetch profiles:', error);
      // Don't show error to user, profiles are optional
    }
  };

  const saveSettings = async () => {
    if (!settings) return;

    setSaving(true);
    setMessage(null);

    try {
      const response = await fetch(`${API_BASE_URL}/settings`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(settings),
      });

      if (!response.ok) {
        throw new Error(`Failed to save settings: ${response.statusText}`);
      }

      const data = await response.json();
      setSettings(data.settings);
      setMessage({
        type: 'success',
        text: 'Settings saved successfully! Changes will take effect immediately.'
      });
    } catch (error) {
      setMessage({
        type: 'error',
        text: error instanceof Error ? error.message : 'Failed to save settings'
      });
    } finally {
      setSaving(false);
    }
  };

  const resetSettings = async () => {
    setResetting(true);
    setMessage(null);

    try {
      const response = await fetch(`${API_BASE_URL}/settings/reset`, {
        method: 'POST',
      });

      if (!response.ok) {
        throw new Error(`Failed to reset settings: ${response.statusText}`);
      }

      const data = await response.json();
      setSettings(data.settings);
      setMessage({
        type: 'success',
        text: 'Settings reset to defaults successfully!'
      });
    } catch (error) {
      setMessage({
        type: 'error',
        text: error instanceof Error ? error.message : 'Failed to reset settings'
      });
    } finally {
      setResetting(false);
    }
  };

  const applyProfile = async (profileName: string) => {
    setMessage(null);

    try {
      const profile = profiles.find(p => p.name === profileName);
      if (!profile) {
        throw new Error(`Profile "${profileName}" not found`);
      }

      // Update local settings with profile settings
      setSettings(prev => {
        if (!prev) return prev;
        return {
          ...prev,
          ...profile.settings,
          performance_profile: profileName,
        };
      });

      setMessage({
        type: 'info',
        text: `Applied "${profile.display_name}" profile. Click Save to persist changes.`
      });
    } catch (error) {
      setMessage({
        type: 'error',
        text: error instanceof Error ? error.message : 'Failed to apply profile'
      });
    }
  };

  // Update Handlers
  const updateStreamingEnabled = (enabled: boolean) => {
    setSettings(prev => prev ? { ...prev, metrics_streaming: enabled } : null);
  };

  const updateCollectionInterval = (value: number) => {
    setSettings(prev => prev ? { ...prev, collection_interval_seconds: value } : null);
  };

  const updateMaxEvents = (value: number) => {
    setSettings(prev => prev ? { ...prev, max_events_per_minute: value } : null);
  };

  const updateFeature = (feature: keyof FeatureSettings, enabled: boolean) => {
    setSettings(prev => {
      if (!prev) return prev;
      return {
        ...prev,
        features: {
          ...prev.features,
          [feature]: enabled,
        }
      };
    });
  };

  // Loading State
  if (loading || !settings) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <div className="flex flex-col items-center gap-4">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-purple-600"></div>
          <p className="text-gray-400">Loading settings...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold tracking-tight flex items-center gap-2">
          <Settings className="h-8 w-8 text-purple-500" />
          Performance Settings
        </h1>
        <p className="text-muted-foreground mt-2">
          Configure real-time metrics streaming, collection intervals, and feature toggles
        </p>
      </div>

      {/* Message Banner */}
      {message && (
        <div
          className={cn(
            'rounded-lg border p-4 flex items-start gap-3',
            {
              'bg-green-900/20 border-green-700 text-green-200': message.type === 'success',
              'bg-red-900/20 border-red-700 text-red-200': message.type === 'error',
              'bg-blue-900/20 border-blue-700 text-blue-200': message.type === 'info',
            }
          )}
          role="alert"
          aria-live="polite"
        >
          {message.type === 'success' && <CheckCircle className="h-5 w-5 flex-shrink-0 mt-0.5" />}
          {message.type === 'error' && <XCircle className="h-5 w-5 flex-shrink-0 mt-0.5" />}
          {message.type === 'info' && <AlertCircle className="h-5 w-5 flex-shrink-0 mt-0.5" />}
          <p className="text-sm">{message.text}</p>
        </div>
      )}

      {/* Performance Profiles */}
      {profiles.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Zap className="h-5 w-5 text-purple-500" />
              Quick Profiles
            </CardTitle>
            <CardDescription>
              Apply pre-configured performance profiles with one click
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              {profiles.map((profile) => {
                const isSelected = settings.performance_profile === profile.name;
                return (
                  <button
                    key={profile.name}
                    onClick={() => applyProfile(profile.name)}
                    className={cn(
                      'p-4 rounded-lg border-2 text-left',
                      'focus:outline-none focus:ring-2 focus:ring-purple-500 focus:ring-offset-2 focus:ring-offset-gray-900',
                      getProfileColorClass(profile.color, isSelected)
                    )}
                    aria-pressed={isSelected}
                  >
                    <div className="flex items-center gap-2 mb-2">
                      <ProfileIcon icon={profile.icon} />
                      <h3 className="font-semibold">{profile.display_name}</h3>
                    </div>
                    <p className={cn(
                      'text-sm',
                      isSelected ? 'text-white/90' : 'text-gray-400'
                    )}>
                      {profile.description}
                    </p>
                  </button>
                );
              })}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Streaming Settings */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Activity className="h-5 w-5 text-purple-500" />
            Streaming Settings
          </CardTitle>
          <CardDescription>
            Control real-time metrics collection and streaming behavior
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* Enable Streaming Toggle */}
          <div className="flex items-center justify-between p-4 rounded-lg bg-gray-800/50 border border-gray-700">
            <div className="flex-1">
              <Label htmlFor="streaming-toggle" className="text-base font-semibold">
                Enable Metrics Streaming
              </Label>
              <p className="text-sm text-gray-400 mt-1">
                Stream real-time metrics from all OmniMemory services
              </p>
            </div>
            <Switch
              id="streaming-toggle"
              checked={settings.metrics_streaming}
              onCheckedChange={updateStreamingEnabled}
            />
          </div>

          {/* Collection Interval Slider */}
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <Label htmlFor="interval-slider" className="text-base font-semibold">
                Collection Interval
              </Label>
              <span className="text-lg font-mono text-purple-400">
                {settings.collection_interval_seconds}s
              </span>
            </div>
            <p className="text-sm text-gray-400">
              How often to collect and send metrics data
            </p>
            <input
              id="interval-slider"
              type="range"
              min="1"
              max="60"
              value={settings.collection_interval_seconds}
              onChange={(e) => updateCollectionInterval(Number(e.target.value))}
              disabled={!settings.metrics_streaming}
              className={cn(
                'w-full h-2 rounded-lg appearance-none cursor-pointer',
                'bg-gray-700',
                '[&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-4 [&::-webkit-slider-thumb]:h-4',
                '[&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-purple-600',
                '[&::-webkit-slider-thumb]:cursor-pointer [&::-webkit-slider-thumb]:hover:bg-purple-500',
                '[&::-moz-range-thumb]:w-4 [&::-moz-range-thumb]:h-4 [&::-moz-range-thumb]:rounded-full',
                '[&::-moz-range-thumb]:bg-purple-600 [&::-moz-range-thumb]:border-0',
                '[&::-moz-range-thumb]:cursor-pointer [&::-moz-range-thumb]:hover:bg-purple-500',
                'disabled:opacity-50 disabled:cursor-not-allowed'
              )}
              aria-label="Collection interval in seconds"
            />
            <div className="flex justify-between text-xs text-gray-500">
              <span>1s (fast)</span>
              <span>30s (balanced)</span>
              <span>60s (slow)</span>
            </div>
          </div>

          {/* Max Events Per Minute */}
          <div className="space-y-3">
            <Label htmlFor="max-events" className="text-base font-semibold">
              Max Events Per Minute
            </Label>
            <p className="text-sm text-gray-400">
              Maximum number of metric events to send per minute (rate limiting)
            </p>
            <Input
              id="max-events"
              type="number"
              min="1"
              max="10000"
              value={settings.max_events_per_minute}
              onChange={(e) => updateMaxEvents(Number(e.target.value))}
              disabled={!settings.metrics_streaming}
              className="max-w-xs"
              aria-label="Maximum events per minute"
            />
          </div>
        </CardContent>
      </Card>

      {/* Feature Toggles */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Filter className="h-5 w-5 text-purple-500" />
            Feature Toggles
          </CardTitle>
          <CardDescription>
            Enable or disable specific OmniMemory features
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Compression */}
          <div className="flex items-center justify-between p-4 rounded-lg bg-gray-800/50 border border-gray-700">
            <div className="flex-1">
              <Label htmlFor="feature-compression" className="text-base font-semibold">
                Compression
              </Label>
              <p className="text-sm text-gray-400 mt-1">
                Enable intelligent text compression for cost savings
              </p>
            </div>
            <Switch
              id="feature-compression"
              checked={settings.features.compression}
              onCheckedChange={(enabled) => updateFeature('compression', enabled)}
              disabled={!settings.metrics_streaming}
            />
          </div>

          {/* Embeddings */}
          <div className="flex items-center justify-between p-4 rounded-lg bg-gray-800/50 border border-gray-700">
            <div className="flex-1">
              <Label htmlFor="feature-embeddings" className="text-base font-semibold">
                Embeddings
              </Label>
              <p className="text-sm text-gray-400 mt-1">
                Enable semantic search and similarity matching
              </p>
            </div>
            <Switch
              id="feature-embeddings"
              checked={settings.features.embeddings}
              onCheckedChange={(enabled) => updateFeature('embeddings', enabled)}
              disabled={!settings.metrics_streaming}
            />
          </div>

          {/* Workflows */}
          <div className="flex items-center justify-between p-4 rounded-lg bg-gray-800/50 border border-gray-700">
            <div className="flex-1">
              <Label htmlFor="feature-workflows" className="text-base font-semibold">
                Workflows
              </Label>
              <p className="text-sm text-gray-400 mt-1">
                Enable workflow learning and pattern recognition
              </p>
            </div>
            <Switch
              id="feature-workflows"
              checked={settings.features.workflows}
              onCheckedChange={(enabled) => updateFeature('workflows', enabled)}
              disabled={!settings.metrics_streaming}
            />
          </div>

          {/* Response Cache */}
          <div className="flex items-center justify-between p-4 rounded-lg bg-gray-800/50 border border-gray-700">
            <div className="flex-1">
              <Label htmlFor="feature-cache" className="text-base font-semibold">
                Response Cache
              </Label>
              <p className="text-sm text-gray-400 mt-1">
                Cache responses for faster retrieval and reduced costs
              </p>
            </div>
            <Switch
              id="feature-cache"
              checked={settings.features.response_cache}
              onCheckedChange={(enabled) => updateFeature('response_cache', enabled)}
              disabled={!settings.metrics_streaming}
            />
          </div>

          {!settings.metrics_streaming && (
            <div className="flex items-center gap-2 p-3 rounded-lg bg-yellow-900/20 border border-yellow-700/50 text-yellow-200">
              <AlertCircle className="h-4 w-4 flex-shrink-0" />
              <p className="text-sm">
                Enable Metrics Streaming to configure feature toggles
              </p>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Actions */}
      <Card>
        <CardContent className="pt-6">
          <div className="flex flex-col sm:flex-row gap-3 justify-between items-center">
            <p className="text-sm text-gray-400">
              Changes will take effect immediately after saving
            </p>
            <div className="flex gap-3">
              <Button
                variant="outline"
                onClick={resetSettings}
                disabled={resetting || saving}
              >
                {resetting ? (
                  <>
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-gray-400 mr-2"></div>
                    Resetting...
                  </>
                ) : (
                  <>
                    <RotateCcw className="h-4 w-4 mr-2" />
                    Reset to Defaults
                  </>
                )}
              </Button>
              <Button
                variant="primary"
                onClick={saveSettings}
                disabled={saving || resetting}
              >
                {saving ? (
                  <>
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                    Saving...
                  </>
                ) : (
                  <>
                    <Save className="h-4 w-4 mr-2" />
                    Save Settings
                  </>
                )}
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
