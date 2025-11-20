// Test script to verify React error fixes and performance
// Run with: node test-pages.js

import { exec } from 'child_process';
import { promisify } from 'util';
import { writeFileSync } from 'fs';

const execAsync = promisify(exec);

const BASE_URL = 'http://localhost:8004';
const PAGES = [
  { path: '/overview', name: 'Overview' },
  { path: '/claude-code', name: 'Claude Code' },
  { path: '/codex', name: 'Codex' },
  { path: '/cursor', name: 'Cursor' },
  { path: '/global', name: 'Global' }
];

const results = {
  timestamp: new Date().toISOString(),
  pages: []
};

async function testPage(page) {
  const url = `${BASE_URL}${page.path}`;
  console.log(`\n📊 Testing: ${page.name} (${url})`);

  const result = {
    name: page.name,
    path: page.path,
    url: url,
    accessible: false,
    responseTime: 0,
    screenshot: null,
    errors: []
  };

  try {
    // Test if page is accessible
    const startTime = Date.now();
    const { stdout, stderr } = await execAsync(`curl -s -o /dev/null -w "%{http_code}" "${url}"`);
    const responseTime = Date.now() - startTime;
    const statusCode = stdout.trim();

    result.accessible = statusCode === '200';
    result.responseTime = responseTime;

    console.log(`  ✓ HTTP Status: ${statusCode} (${responseTime}ms)`);

    if (!result.accessible) {
      result.errors.push(`HTTP ${statusCode} - Page not accessible`);
      return result;
    }

    // Capture screenshot with Chrome headless
    const screenshotPath = `/tmp/test-${page.name.toLowerCase().replace(/\s+/g, '-')}.png`;
    const screenshotCmd = `/Applications/Google\\ Chrome.app/Contents/MacOS/Google\\ Chrome --headless --disable-gpu --screenshot="${screenshotPath}" --window-size=1920,1080 "${url}" 2>&1`;

    try {
      await execAsync(screenshotCmd, { timeout: 10000 });
      result.screenshot = screenshotPath;
      console.log(`  ✓ Screenshot captured: ${screenshotPath}`);
    } catch (screenshotError) {
      console.log(`  ⚠ Screenshot warning: ${screenshotError.message}`);
    }

    // Check for common React error patterns in the page source
    const { stdout: pageSource } = await execAsync(`curl -s "${url}"`);

    if (pageSource.includes('TypeError') || pageSource.includes('Uncaught')) {
      result.errors.push('Page source contains error indicators');
    }

    console.log(`  ✓ Page loaded successfully`);

  } catch (error) {
    result.errors.push(error.message);
    console.log(`  ✗ Error: ${error.message}`);
  }

  return result;
}

async function runTests() {
  console.log('🚀 Starting Dashboard Test Suite');
  console.log('=' .repeat(60));

  // Test server availability
  try {
    const { stdout } = await execAsync(`curl -s -o /dev/null -w "%{http_code}" "${BASE_URL}"`);
    if (stdout.trim() !== '200') {
      console.error('❌ Server is not running at', BASE_URL);
      process.exit(1);
    }
    console.log('✓ Server is running at', BASE_URL);
  } catch (error) {
    console.error('❌ Cannot connect to server:', error.message);
    process.exit(1);
  }

  // Test each page
  for (const page of PAGES) {
    const result = await testPage(page);
    results.pages.push(result);
  }

  // Generate report
  console.log('\n' + '='.repeat(60));
  console.log('📋 TEST SUMMARY');
  console.log('='.repeat(60));

  let passCount = 0;
  let failCount = 0;

  results.pages.forEach(page => {
    const status = page.errors.length === 0 && page.accessible ? '✅ PASS' : '❌ FAIL';
    const responseTimeStr = `${page.responseTime}ms`;

    if (page.errors.length === 0 && page.accessible) {
      passCount++;
    } else {
      failCount++;
    }

    console.log(`\n${status} - ${page.name}`);
    console.log(`  URL: ${page.url}`);
    console.log(`  Response Time: ${responseTimeStr}`);
    console.log(`  Screenshot: ${page.screenshot || 'N/A'}`);

    if (page.errors.length > 0) {
      console.log(`  Errors:`);
      page.errors.forEach(err => console.log(`    - ${err}`));
    }
  });

  console.log('\n' + '='.repeat(60));
  console.log(`Total: ${results.pages.length} pages`);
  console.log(`Passed: ${passCount}`);
  console.log(`Failed: ${failCount}`);
  console.log('='.repeat(60));

  // Save results to file
  const reportPath = '/tmp/dashboard-test-report.json';
  writeFileSync(reportPath, JSON.stringify(results, null, 2));
  console.log(`\n📄 Full report saved to: ${reportPath}`);

  // Exit with appropriate code
  process.exit(failCount > 0 ? 1 : 0);
}

runTests().catch(error => {
  console.error('Fatal error:', error);
  process.exit(1);
});
