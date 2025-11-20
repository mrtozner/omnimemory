import { useState, useEffect, useCallback } from 'react';
import { Card } from '../shared/Card';
import { Input } from '../shared/Input';
import { Label } from '../shared/Label';
import { DollarSign, TrendingDown } from 'lucide-react';
import { api } from '../../services/api';

interface CostAnalysisData {
  omnimemory_cost?: number;
  mem0_savings?: number;
  mem0_savings_pct?: number;
  openai_savings?: number;
  openai_savings_pct?: number;
  cohere_savings?: number;
  cohere_savings_pct?: number;
}

export function CostSavingsCalculator() {
  const [monthlyOps, setMonthlyOps] = useState(100000);
  const [costData, setCostData] = useState<CostAnalysisData | null>(null);
  const [loading, setLoading] = useState(true);

  const fetchCostData = useCallback(async () => {
    setLoading(true);
    try {
      const response = await api.getCostAnalysis(monthlyOps);
      setCostData((response.data || null) as CostAnalysisData | null);
    } catch (error) {
      console.error('Failed to fetch cost data:', error);
    } finally {
      setLoading(false);
    }
  }, [monthlyOps]);

  useEffect(() => {
    fetchCostData();
  }, [fetchCostData]);

  if (loading || !costData) {
    return (
      <Card>
        <div className="flex items-center justify-center h-64">
          <div className="text-gray-600 dark:text-gray-400">Loading cost analysis...</div>
        </div>
      </Card>
    );
  }

  return (
    <Card>
      <div className="flex items-center gap-2 mb-6">
        <DollarSign className="w-5 h-5 text-green-600" />
        <h3 className="text-lg font-semibold">Cost Savings Calculator</h3>
      </div>

      <div className="mb-6">
        <Label htmlFor="monthlyOps">Monthly Operations</Label>
        <Input
          id="monthlyOps"
          type="number"
          value={monthlyOps}
          onChange={(e) => setMonthlyOps(parseInt(e.target.value) || 0)}
          className="mt-2"
        />
      </div>

      <div className="grid grid-cols-2 gap-4">
        <div className="bg-gray-50 dark:bg-gray-800 p-4 rounded-lg">
          <div className="text-sm text-gray-600 dark:text-gray-400 mb-1">OmniMemory Cost</div>
          <div className="text-2xl font-bold text-gray-900 dark:text-white">
            ${costData.omnimemory_cost?.toFixed(2) || '0.00'}
          </div>
          <div className="text-xs text-gray-500 mt-1">
            {monthlyOps <= 10000 ? 'Free Tier' : 'Pro Plan ($9/mo)'}
          </div>
        </div>

        <div className="bg-green-50 dark:bg-green-900/20 p-4 rounded-lg border border-green-200 dark:border-green-800">
          <div className="text-sm text-gray-600 dark:text-gray-400 mb-1">Savings vs mem0</div>
          <div className="text-2xl font-bold text-green-600 flex items-center gap-1">
            <TrendingDown className="w-5 h-5" />
            ${costData.mem0_savings?.toFixed(2) || '0.00'}
          </div>
          <div className="text-xs text-gray-500 mt-1">
            {costData.mem0_savings_pct?.toFixed(1) || '0.0'}% savings
          </div>
        </div>

        <div className="bg-blue-50 dark:bg-blue-900/20 p-4 rounded-lg border border-blue-200 dark:border-blue-800">
          <div className="text-sm text-gray-600 dark:text-gray-400 mb-1">Savings vs OpenAI</div>
          <div className="text-2xl font-bold text-blue-600 flex items-center gap-1">
            <TrendingDown className="w-5 h-5" />
            ${costData.openai_savings?.toFixed(2) || '0.00'}
          </div>
          <div className="text-xs text-gray-500 mt-1">
            {costData.openai_savings_pct?.toFixed(1) || '0.0'}% savings
          </div>
        </div>

        <div className="bg-purple-50 dark:bg-purple-900/20 p-4 rounded-lg border border-purple-200 dark:border-purple-800">
          <div className="text-sm text-gray-600 dark:text-gray-400 mb-1">Savings vs Cohere</div>
          <div className="text-2xl font-bold text-purple-600 flex items-center gap-1">
            <TrendingDown className="w-5 h-5" />
            ${costData.cohere_savings?.toFixed(2) || '0.00'}
          </div>
          <div className="text-xs text-gray-500 mt-1">
            {costData.cohere_savings_pct?.toFixed(1) || '0.0'}% savings
          </div>
        </div>
      </div>
    </Card>
  );
}
