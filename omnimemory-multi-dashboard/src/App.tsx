import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { lazy, Suspense, useEffect } from 'react';
import { Layout } from './components/layout/Layout';
import { useConfigStore } from './stores/configStore';

// Lazy load page components for better performance and no full page reloads
const OverviewPage = lazy(() => import('./pages/OverviewPage').then(m => ({ default: m.OverviewPage })));
const SettingsPage = lazy(() => import('./pages/SettingsPage').then(m => ({ default: m.SettingsPage })));
const BenchmarkPage = lazy(() => import('./pages/BenchmarkPage').then(m => ({ default: m.BenchmarkPage })));
const FileContextPage = lazy(() => import('./pages/FileContextPage').then(m => ({ default: m.FileContextPage })));
const APISavingsPage = lazy(() => import('./pages/APISavingsPage').then(m => ({ default: m.APISavingsPage })));
const RedisCachePage = lazy(() => import('./pages/RedisCachePage').then(m => ({ default: m.RedisCachePage })));
const UnifiedIntelligencePage = lazy(() => import('./pages/UnifiedIntelligencePage').then(m => ({ default: m.UnifiedIntelligencePage })));
const SessionHistoryPage = lazy(() => import('./pages/SessionHistoryPage').then(m => ({ default: m.SessionHistoryPage })));
const TeamPage = lazy(() => import('./pages/TeamPage').then(m => ({ default: m.TeamPage })));

// Loading fallback component
function PageLoader() {
  return (
    <div className="flex items-center justify-center h-96">
      <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary"></div>
    </div>
  );
}

function App() {
  const darkMode = useConfigStore((state) => state.darkMode);

  useEffect(() => {
    if (darkMode) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  }, [darkMode]);

  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<Navigate to="/overview" replace />} />
          <Route path="overview" element={
            <Suspense fallback={<PageLoader />}>
              <OverviewPage />
            </Suspense>
          } />
          <Route path="file-context" element={
            <Suspense fallback={<PageLoader />}>
              <FileContextPage />
            </Suspense>
          } />
          <Route path="api-savings" element={
            <Suspense fallback={<PageLoader />}>
              <APISavingsPage />
            </Suspense>
          } />
          <Route path="cache" element={
            <Suspense fallback={<PageLoader />}>
              <RedisCachePage />
            </Suspense>
          } />
          <Route path="unified-intelligence" element={
            <Suspense fallback={<PageLoader />}>
              <UnifiedIntelligencePage />
            </Suspense>
          } />
          <Route path="benchmark" element={
            <Suspense fallback={<PageLoader />}>
              <BenchmarkPage />
            </Suspense>
          } />
          <Route path="settings" element={
            <Suspense fallback={<PageLoader />}>
              <SettingsPage />
            </Suspense>
          } />
          <Route path="sessions" element={
            <Suspense fallback={<PageLoader />}>
              <SessionHistoryPage />
            </Suspense>
          } />
          <Route path="team" element={
            <Suspense fallback={<PageLoader />}>
              <TeamPage />
            </Suspense>
          } />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}

export default App;
