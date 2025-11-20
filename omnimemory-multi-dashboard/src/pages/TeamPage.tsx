'use client';

import { useEffect, useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../components/shared/Card';
import { Users, Activity, TrendingUp, Clock } from 'lucide-react';

interface TeamMember {
  user_id: string;
  role: string;
  cache_size_mb: number;
  last_active: string;
}

interface Repository {
  repo_id: string;
  name: string;
  cache_size_mb: number;
  files_cached: number;
}

interface TeamSavings {
  without_sharing_tokens: number;
  with_sharing_tokens: number;
  savings_percent: number;
  cost_saved_monthly: number;
}

interface TeamActivity {
  user: string;
  file: string;
  time_ago: string;
}

interface TeamStats {
  team_id: string;
  members: TeamMember[];
  repositories: Repository[];
  savings: TeamSavings;
  recent_activity: TeamActivity[];
}

export function TeamPage() {
  const [stats, setStats] = useState<TeamStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [teamId, setTeamId] = useState('engineering'); // Default team

  useEffect(() => {
    const fetchTeamStats = async () => {
      try {
        setLoading(true);
        setError(null);
        const response = await fetch(`http://localhost:8003/api/team/${teamId}/stats`);

        if (!response.ok) {
          throw new Error(`Failed to fetch team stats: ${response.statusText}`);
        }

        const data = await response.json();
        setStats(data);
      } catch (error) {
        console.error('Failed to fetch team stats:', error);
        setError(error instanceof Error ? error.message : 'Failed to load team data');
      } finally {
        setLoading(false);
      }
    };

    fetchTeamStats();
    const interval = setInterval(fetchTeamStats, 10000); // Update every 10s

    return () => clearInterval(interval);
  }, [teamId]);

  if (loading) {
    return (
      <div className="p-8">
        <div className="flex items-center justify-center h-96">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary"></div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-8">
        <Card className="border-red-500/30 bg-red-900/20">
          <CardHeader>
            <CardTitle className="text-red-400">Error Loading Team Data</CardTitle>
            <CardDescription className="text-red-300">{error}</CardDescription>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-gray-400 mb-4">
              Make sure the team API endpoint is available and the team ID is correct.
            </p>
            <p className="text-sm text-gray-400">
              To initialize a team, run: <code className="bg-gray-800 px-2 py-1 rounded">omni repo init --team {teamId}</code>
            </p>
          </CardContent>
        </Card>
      </div>
    );
  }

  if (!stats) {
    return (
      <div className="p-8">
        <Card className="border-yellow-500/30 bg-yellow-900/20">
          <CardHeader>
            <CardTitle className="text-yellow-400">No Team Data</CardTitle>
            <CardDescription className="text-yellow-300">
              Initialize a team with: <code className="bg-gray-800 px-2 py-1 rounded">omni repo init --team {teamId}</code>
            </CardDescription>
          </CardHeader>
        </Card>
      </div>
    );
  }

  return (
    <div className="p-8 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Team Collaboration</h1>
          <p className="text-muted-foreground mt-2">
            Team: {stats.team_id} • {stats.members.length} members • {stats.repositories.length} repositories
          </p>
        </div>

        {/* Team selector */}
        <select
          value={teamId}
          onChange={(e) => setTeamId(e.target.value)}
          className="px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
        >
          <option value="engineering">Engineering</option>
          <option value="design">Design</option>
          <option value="product">Product</option>
        </select>
      </div>

      {/* Key Metrics */}
      <div className="grid gap-4 md:grid-cols-4">
        {/* Team Members */}
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Team Members</CardTitle>
            <Users className="h-4 w-4 text-blue-600" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-blue-600">{stats.members.length}</div>
            <p className="text-xs text-muted-foreground">
              {stats.repositories.length} shared repositories
            </p>
          </CardContent>
        </Card>

        {/* Token Savings */}
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Token Savings</CardTitle>
            <TrendingUp className="h-4 w-4 text-green-600" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-green-600">
              {stats.savings.savings_percent.toFixed(1)}%
            </div>
            <p className="text-xs text-muted-foreground">
              {((stats.savings.without_sharing_tokens - stats.savings.with_sharing_tokens) / 1000).toFixed(0)}K tokens saved
            </p>
          </CardContent>
        </Card>

        {/* Cost Saved */}
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Cost Saved</CardTitle>
            <TrendingUp className="h-4 w-4 text-purple-600" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-purple-600">
              ${stats.savings.cost_saved_monthly.toFixed(2)}
            </div>
            <p className="text-xs text-muted-foreground">
              per month with L2 sharing
            </p>
          </CardContent>
        </Card>

        {/* Repositories */}
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Repositories</CardTitle>
            <Activity className="h-4 w-4 text-orange-600" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-orange-600">
              {stats.repositories.length}
            </div>
            <p className="text-xs text-muted-foreground">
              {stats.repositories.reduce((sum, r) => sum + r.files_cached, 0)} files cached
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Team Members Table */}
      <Card>
        <CardHeader>
          <CardTitle>Team Members</CardTitle>
          <CardDescription>Active members and their cache usage</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-gray-700">
                  <th className="text-left py-2 px-4">Member</th>
                  <th className="text-left py-2 px-4">Role</th>
                  <th className="text-right py-2 px-4">Cache Size</th>
                  <th className="text-right py-2 px-4">Last Active</th>
                </tr>
              </thead>
              <tbody>
                {stats.members.map((member) => (
                  <tr key={member.user_id} className="border-b border-gray-700/50">
                    <td className="py-2 px-4 font-medium">{member.user_id}</td>
                    <td className="py-2 px-4">
                      <span className="px-2 py-1 bg-blue-500/20 text-blue-400 rounded text-xs">
                        {member.role}
                      </span>
                    </td>
                    <td className="text-right py-2 px-4">{member.cache_size_mb.toFixed(2)} MB</td>
                    <td className="text-right py-2 px-4 text-muted-foreground">{member.last_active}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>

      {/* Repositories Table */}
      <Card>
        <CardHeader>
          <CardTitle>Team Repositories</CardTitle>
          <CardDescription>Shared repositories and their cache statistics</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-gray-700">
                  <th className="text-left py-2 px-4">Repository</th>
                  <th className="text-right py-2 px-4">Files Cached</th>
                  <th className="text-right py-2 px-4">Cache Size</th>
                </tr>
              </thead>
              <tbody>
                {stats.repositories.map((repo) => (
                  <tr key={repo.repo_id} className="border-b border-gray-700/50">
                    <td className="py-2 px-4">
                      <div className="font-medium">{repo.name}</div>
                      <div className="text-xs text-muted-foreground">{repo.repo_id}</div>
                    </td>
                    <td className="text-right py-2 px-4">{repo.files_cached}</td>
                    <td className="text-right py-2 px-4">{repo.cache_size_mb.toFixed(2)} MB</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>

      {/* Recent Team Activity */}
      <Card>
        <CardHeader>
          <CardTitle>Recent Activity</CardTitle>
          <CardDescription>What your team has been working on</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-2">
            {stats.recent_activity.slice(0, 10).map((activity, idx) => (
              <div key={idx} className="flex items-center justify-between py-2 border-b border-gray-700/50 last:border-0">
                <div className="flex items-center gap-3">
                  <Clock className="h-4 w-4 text-muted-foreground" />
                  <div>
                    <span className="font-medium">{activity.user}</span>
                    <span className="text-muted-foreground"> worked on </span>
                    <span className="font-mono text-sm">{activity.file}</span>
                  </div>
                </div>
                <span className="text-sm text-muted-foreground">{activity.time_ago}</span>
              </div>
            ))}
            {stats.recent_activity.length === 0 && (
              <div className="text-center py-8 text-muted-foreground">
                No recent activity
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Team Savings Summary */}
      <Card className="border-green-500/30 bg-green-900/20">
        <CardHeader>
          <CardTitle className="text-green-400">Team Sharing Impact</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
              <div className="text-xs text-green-400 mb-1">Without Sharing</div>
              <div className="text-lg font-bold text-green-300">{(stats.savings.without_sharing_tokens / 1000).toFixed(0)}K tokens</div>
              <div className="text-xs text-green-400">per month</div>
            </div>
            <div>
              <div className="text-xs text-green-400 mb-1">With L2 Sharing</div>
              <div className="text-lg font-bold text-green-300">{(stats.savings.with_sharing_tokens / 1000).toFixed(0)}K tokens</div>
              <div className="text-xs text-green-400">per month</div>
            </div>
            <div>
              <div className="text-xs text-green-400 mb-1">Team Saves</div>
              <div className="text-lg font-bold text-green-300">${stats.savings.cost_saved_monthly.toFixed(2)}</div>
              <div className="text-xs text-green-400">per month</div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

export default TeamPage;
