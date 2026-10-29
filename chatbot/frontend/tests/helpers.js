/**
 * Test Helpers and Utilities for Aviator E2E Tests
 */

/**
 * Wait for API request/response
 * @param {Page} page - Playwright page object
 * @param {string} pattern - URL pattern to match
 * @param {number} timeout - Timeout in ms
 */
export async function waitForAPIResponse(page, pattern, timeout = 30000) {
  return await Promise.race([
    page.waitForResponse(
      response => response.url().includes(pattern),
      { timeout }
    ),
    new Promise((_, reject) => 
      setTimeout(() => reject(new Error(`Timeout waiting for API: ${pattern}`)), timeout)
    )
  ]);
}

/**
 * Wait for WebSocket message
 * @param {Page} page - Playwright page object
 * @param {number} timeout - Timeout in ms
 */
export async function waitForWebSocket(page, timeout = 30000) {
  return await Promise.race([
    page.waitForEvent('websocket', { timeout }),
    new Promise((_, reject) => 
      setTimeout(() => reject(new Error('Timeout waiting for WebSocket')), timeout)
    )
  ]);
}

/**
 * Wait for element with text
 * @param {Page} page - Playwright page object
 * @param {string} selector - Element selector
 * @param {string} text - Text to wait for
 */
export async function waitForElementWithText(page, selector, text) {
  await page.waitForSelector(selector, { state: 'visible' });
  await page.locator(selector).filter({ hasText: text }).first().waitFor({ state: 'visible' });
}

/**
 * Fill form field
 * @param {Page} page - Playwright page object
 * @param {string} label - Label text
 * @param {string} value - Value to fill
 */
export async function fillFormField(page, label, value) {
  const input = page.locator(`label:has-text("${label}") ~ input, label:has-text("${label}") ~ textarea, label:has-text("${label}") ~ select`);
  await input.fill(value);
}

/**
 * Click button by text
 * @param {Page} page - Playwright page object
 * @param {string} text - Button text
 */
export async function clickButton(page, text) {
  await page.locator(`button:has-text("${text}")`).click();
}

/**
 * Check if element is visible
 * @param {Page} page - Playwright page object
 * @param {string} selector - Element selector
 */
export async function isElementVisible(page, selector) {
  return await page.locator(selector).isVisible();
}

/**
 * Get element text
 * @param {Page} page - Playwright page object
 * @param {string} selector - Element selector
 */
export async function getElementText(page, selector) {
  return await page.locator(selector).textContent();
}

/**
 * Check if element contains text
 * @param {Page} page - Playwright page object
 * @param {string} selector - Element selector
 * @param {string} text - Text to check
 */
export async function elementContainsText(page, selector, text) {
  const element = page.locator(selector);
  const content = await element.textContent();
  return content.includes(text);
}

/**
 * Wait for loading to complete
 * @param {Page} page - Playwright page object
 */
export async function waitForLoadingComplete(page) {
  // Wait for any loading spinners to disappear
  await page.locator('.loading-spinner, [class*="loading"], [aria-busy="true"]').all().then(async (spinners) => {
    for (const spinner of spinners) {
      await spinner.waitFor({ state: 'hidden' });
    }
  }).catch(() => {
    // No loading spinners found, continue
  });
  
  // Wait a bit for DOM to settle
  await page.waitForTimeout(500);
}

/**
 * Screenshot with timestamp
 * @param {Page} page - Playwright page object
 * @param {string} name - Screenshot name
 */
export async function takeScreenshot(page, name) {
  const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
  await page.screenshot({ path: `test-results/screenshots/${name}-${timestamp}.png` });
}

/**
 * Get current URL
 * @param {Page} page - Playwright page object
 */
export async function getCurrentURL(page) {
  return page.url();
}

/**
 * Navigate to URL
 * @param {Page} page - Playwright page object
 * @param {string} url - URL to navigate to
 */
export async function navigateTo(page, url) {
  await page.goto(url, { waitUntil: 'networkidle' });
}

/**
 * Wait for network to be idle
 * @param {Page} page - Playwright page object
 */
export async function waitForNetworkIdle(page) {
  await page.waitForLoadState('networkidle');
}

/**
 * Get all console messages during execution
 * @param {Page} page - Playwright page object
 */
export function captureConsoleLogs(page) {
  const logs = {
    errors: [],
    warnings: [],
    info: [],
    debug: []
  };
  
  page.on('console', msg => {
    if (msg.type() === 'error') {
      logs.errors.push(msg.text());
    } else if (msg.type() === 'warning') {
      logs.warnings.push(msg.text());
    } else if (msg.type() === 'info') {
      logs.info.push(msg.text());
    } else if (msg.type() === 'log') {
      logs.debug.push(msg.text());
    }
  });
  
  return logs;
}

/**
 * Intercept and log API calls
 * @param {Page} page - Playwright page object
 */
export function logAPICalls(page) {
  const calls = [];
  
  page.on('request', request => {
    if (request.url().includes('/api/')) {
      calls.push({
        method: request.method(),
        url: request.url(),
        timestamp: new Date()
      });
    }
  });
  
  return calls;
}

/**
 * Wait for specific workflow status
 * @param {Page} page - Playwright page object
 * @param {string} status - Status to wait for
 * @param {number} timeout - Timeout in ms
 */
export async function waitForWorkflowStatus(page, status, timeout = 60000) {
  const startTime = Date.now();
  
  while (Date.now() - startTime < timeout) {
    const currentStatus = await page.locator('[class*="status"], [class*="workflow-status"]').textContent();
    if (currentStatus && currentStatus.includes(status)) {
      return true;
    }
    await page.waitForTimeout(1000);
  }
  
  throw new Error(`Workflow status "${status}" not reached within ${timeout}ms`);
}

/**
 * Get project list
 * @param {Page} page - Playwright page object
 */
export async function getProjectList(page) {
  await page.waitForSelector('[class*="project-list"], [class*="project-item"]');
  const items = await page.locator('[class*="project-item"], [class*="project-row"]').all();
  
  const projects = [];
  for (const item of items) {
    const text = await item.textContent();
    projects.push(text.trim());
  }
  
  return projects;
}

/**
 * Find project in list
 * @param {Page} page - Playwright page object
 * @param {string} projectName - Project name to find
 */
export async function findProject(page, projectName) {
  const projects = await getProjectList(page);
  return projects.find(p => p.includes(projectName));
}
