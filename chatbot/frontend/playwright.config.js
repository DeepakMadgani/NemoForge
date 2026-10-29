import { defineConfig, devices } from '@playwright/test';

/**
 * Playwright Configuration for Ticket-to-Code E2E Testing
 * 
 * Tests the complete Ticket-to-Code workflow:
 * 1. Application loads
 * 2. Projects are created/managed
 * 3. Projects are indexed (AST, Vector, Neo4j)
 * 4. Tickets are submitted
 * 5. Workflow executes and generates code
 * 6. Results are validated
 */

export default defineConfig({
  testDir: './tests/e2e',
  
  // Test configuration
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  workers: process.env.CI ? 1 : undefined,
  
  // Reporter configuration
  reporter: [
    ['html', { outputFolder: 'test-results/html' }],
    ['json', { outputFile: 'test-results/results.json' }],
    ['junit', { outputFile: 'test-results/junit.xml' }],
    ['list']
  ],
  
  // Global configuration
  use: {
    // Base URL for all tests
    baseURL: 'http://localhost:3000',
    
    // Screenshot/video on failure
    screenshot: 'only-on-failure',
    video: 'retain-on-failure',
    trace: 'on-first-retry',
  },

  // Web server configuration
  webServer: {
    command: 'npm run dev',
    url: 'http://localhost:3000',
    reuseExistingServer: !process.env.CI,
    timeout: 120000,
  },

  // Projects (browser configurations)
  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] },
    },

    // Uncomment to test on Firefox
    // {
    //   name: 'firefox',
    //   use: { ...devices['Desktop Firefox'] },
    // },

    // Uncomment to test on Safari
    // {
    //   name: 'webkit',
    //   use: { ...devices['Desktop Safari'] },
    // },

    /* Test against mobile viewports. */
    // {
    //   name: 'Mobile Chrome',
    //   use: { ...devices['Pixel 5'] },
    // },
    // {
    //   name: 'Mobile Safari',
    //   use: { ...devices['iPhone 12'] },
    // },
  ],
});
