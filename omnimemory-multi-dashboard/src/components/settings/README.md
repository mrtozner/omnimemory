# PerformanceSettings Component

A comprehensive UI component for managing OmniMemory tenant performance and feature settings.

## Usage

### Basic Import

```tsx
import { PerformanceSettings } from '@/components/settings';

// In your page/component
export function SettingsPage() {
  return (
    <div>
      <PerformanceSettings />
    </div>
  );
}
```

### Integration Example

```tsx
import { PerformanceSettings } from '@/components/settings/PerformanceSettings';

export function SettingsPage() {
  return (
    <div className="container mx-auto p-6">
      <PerformanceSettings />
    </div>
  );
}
```

## Features

### 1. Performance Profiles
- Quick preset buttons for common configurations
- Visual feedback for selected profile
- Color-coded profiles (green=high, blue=low, gray=batch, red=disabled)
- One-click application of settings

### 2. Streaming Settings
- **Enable/Disable Metrics Streaming**: Toggle real-time metrics collection
- **Collection Interval**: Slider control (1-60 seconds) with visual feedback
- **Max Events Per Minute**: Input field for rate limiting

### 3. Feature Toggles
- **Compression**: Enable intelligent text compression
- **Embeddings**: Enable semantic search and similarity matching
- **Workflows**: Enable workflow learning and pattern recognition
- **Response Cache**: Enable response caching for cost savings
- All toggles automatically disabled when streaming is off

### 4. Actions
- **Save Button**: Persists settings to API with loading state
- **Reset Button**: Restores default settings with confirmation
- **Auto-save notifications**: Success/error messages with auto-dismiss

## API Endpoints

The component communicates with the following endpoints:

- `GET /settings` - Fetch current settings
- `GET /settings/profiles` - Fetch available performance profiles
- `PUT /settings` - Save updated settings
- `POST /settings/reset` - Reset to default settings

## State Management

```typescript
interface Settings {
  metrics_streaming: boolean;
  collection_interval_seconds: number;
  max_events_per_minute: number;
  features: {
    compression: boolean;
    embeddings: boolean;
    workflows: boolean;
    response_cache: boolean;
  };
  performance_profile: string;
}
```

## Accessibility

- Proper ARIA labels and roles
- Keyboard navigation support
- Screen reader friendly
- Focus management
- Live region for status messages

## Styling

- Matches existing dashboard dark mode theme
- Purple accent colors (brand color)
- Responsive layout (mobile, tablet, desktop)
- Smooth transitions and animations
- Loading states with spinners

## Error Handling

- Network error handling with user-friendly messages
- Loading states during async operations
- Disabled states for dependent fields
- Auto-dismissing notifications (5 seconds)

## Testing Checklist

- [x] Loads current settings on mount
- [x] Displays all settings correctly
- [x] Profile selection updates all related settings
- [x] Toggles work correctly
- [x] Slider updates interval value
- [x] Save button sends correct data to API
- [x] Reset button restores defaults
- [x] Shows success/error messages
- [x] TypeScript compiles without errors
- [x] Accessibility attributes present
- [x] Responsive on all screen sizes
- [x] Dark mode compatible
