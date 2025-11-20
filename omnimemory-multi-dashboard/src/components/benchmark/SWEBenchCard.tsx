import { Award, CheckCircle, TrendingUp } from 'lucide-react';
import { Card } from '../shared/Card';

interface SWEBenchCardProps {
  passRate: number;
  totalTests: number;
  testsPassed: number;
  contextRetention: number;
}

export function SWEBenchCard({ passRate, totalTests, testsPassed, contextRetention }: SWEBenchCardProps) {
  return (
    <Card className="bg-gradient-to-br from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20 border-green-200 dark:border-green-800">
      <div className="flex items-start justify-between">
        <div className="flex-1">
          <div className="flex items-center gap-2 mb-2">
            <Award className="w-5 h-5 text-green-600 dark:text-green-400" />
            <h3 className="text-sm font-medium text-gray-600 dark:text-gray-300">
              SWE-Bench Validation
            </h3>
          </div>

          <div className="mt-4">
            <div className="text-4xl font-bold text-green-600 dark:text-green-400">
              {passRate}%
            </div>
            <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
              Pass@1 Rate
            </p>
          </div>

          <div className="mt-6 grid grid-cols-2 gap-4">
            <div>
              <div className="flex items-center gap-1">
                <CheckCircle className="w-4 h-4 text-green-600" />
                <span className="text-2xl font-semibold text-gray-900 dark:text-white">
                  {testsPassed.toLocaleString()}
                </span>
              </div>
              <p className="text-xs text-gray-600 dark:text-gray-400">
                Tests Passed
              </p>
            </div>

            <div>
              <div className="flex items-center gap-1">
                <TrendingUp className="w-4 h-4 text-green-600" />
                <span className="text-2xl font-semibold text-gray-900 dark:text-white">
                  {contextRetention}%
                </span>
              </div>
              <p className="text-xs text-gray-600 dark:text-gray-400">
                Context Retention
              </p>
            </div>
          </div>

          <div className="mt-4 pt-4 border-t border-green-200 dark:border-green-800">
            <p className="text-xs text-gray-600 dark:text-gray-400">
              Validated on {totalTests.toLocaleString()} total tests from SWE-bench Lite
            </p>
          </div>
        </div>
      </div>
    </Card>
  );
}
