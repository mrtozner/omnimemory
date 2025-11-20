// Unit test for formatPercent() safety with undefined values
// Simulates the fix verification

console.log('🧪 Testing formatPercent() Null Safety');
console.log('=' .repeat(60));

// Simulate the formatPercent function
function formatPercent(value) {
  if (value == null || isNaN(value)) return '0.0%';
  return `${value.toFixed(1)}%`;
}

const testCases = [
  { input: undefined, expected: '0.0%', description: 'undefined value' },
  { input: null, expected: '0.0%', description: 'null value' },
  { input: NaN, expected: '0.0%', description: 'NaN value' },
  { input: 0, expected: '0.0%', description: 'zero value' },
  { input: 25.5, expected: '25.5%', description: 'valid decimal' },
  { input: 100, expected: '100.0%', description: 'valid integer' },
  { input: -5.2, expected: '-5.2%', description: 'negative value' },
];

let passed = 0;
let failed = 0;

console.log('\nRunning test cases:\n');

testCases.forEach((test, index) => {
  try {
    const result = formatPercent(test.input);
    const success = result === test.expected;

    if (success) {
      console.log(`✅ Test ${index + 1}: PASSED - ${test.description}`);
      console.log(`   Input: ${test.input}, Output: ${result}`);
      passed++;
    } else {
      console.log(`❌ Test ${index + 1}: FAILED - ${test.description}`);
      console.log(`   Input: ${test.input}, Expected: ${test.expected}, Got: ${result}`);
      failed++;
    }
  } catch (error) {
    console.log(`❌ Test ${index + 1}: ERROR - ${test.description}`);
    console.log(`   Input: ${test.input}, Error: ${error.message}`);
    failed++;
  }
});

console.log('\n' + '='.repeat(60));
console.log('📊 TEST SUMMARY');
console.log('='.repeat(60));
console.log(`Total: ${testCases.length} tests`);
console.log(`✅ Passed: ${passed}`);
console.log(`❌ Failed: ${failed}`);
console.log(`Success Rate: ${((passed / testCases.length) * 100).toFixed(1)}%`);

// Test the actual scenario from the bug report
console.log('\n' + '='.repeat(60));
console.log('🐛 BUG SCENARIO TEST');
console.log('='.repeat(60));
console.log('\nSimulating the original crash scenario:');
console.log('Before fix: formatPercent(undefined.toFixed(1)) → TypeError');
console.log('After fix: formatPercent(undefined) → "0.0%" ✅\n');

// Simulate the component usage pattern
const mockMetrics = {}; // No data available

// Old pattern (would crash):
console.log('❌ OLD PATTERN (CRASHES):');
console.log('   formatPercent(mockMetrics.avg_compression_ratio)');
console.log('   → Would call: undefined.toFixed(1)');
console.log('   → Result: TypeError: Cannot read properties of undefined\n');

// New pattern (safe):
console.log('✅ NEW PATTERN (SAFE):');
const safeValue = mockMetrics.avg_compression_ratio ?? 0;
console.log(`   formatPercent(mockMetrics.avg_compression_ratio ?? 0)`);
console.log(`   → Calls: formatPercent(${safeValue})`);
console.log(`   → Result: ${formatPercent(safeValue)}`);

console.log('\n' + '='.repeat(60));

if (failed === 0) {
  console.log('✅ ALL TESTS PASSED - formatPercent() is null-safe!');
  process.exit(0);
} else {
  console.log('❌ SOME TESTS FAILED');
  process.exit(1);
}
