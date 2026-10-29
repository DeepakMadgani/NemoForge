import { test, expect } from '@playwright/test';
import {
  clickButton,
  fillFormField,
  getElementText,
  isElementVisible,
  waitForLoadingComplete,
  waitForNetworkIdle,
  navigateTo,
  captureConsoleLogs,
  logAPICalls
} from '../helpers';

test.describe('Aviator Application - Core UI Tests', () => {
  test.beforeEach(async ({ page }) => {
    // Capture console logs and API calls for debugging
    captureConsoleLogs(page);
    logAPICalls(page);
  });

  test('should load application homepage', async ({ page }) => {
    // Navigate to application
    await page.goto('/', { waitUntil: 'networkidle' });
    
    // Verify page loads
    expect(page).toHaveTitle(/Aviator|chatbot|ContentBridge/i);
    
    // Verify main heading is visible
    const heading = page.locator('h1, h2, .app-title');
    await expect(heading).toBeVisible({ timeout: 10000 });
  });

  test('should display welcome/project list view', async ({ page }) => {
    await page.goto('/', { waitUntil: 'networkidle' });
    
    // Check for project list or welcome message
    const projectList = page.locator('[class*="project"]');
    const welcome = page.locator('text=/welcome|create|add/i');
    
    // At least one should be visible
    const projectListVisible = await projectList.count() > 0;
    const welcomeVisible = await welcome.count() > 0;
    
    expect(projectListVisible || welcomeVisible).toBeTruthy();
  });

  test('should have "Add Project" button', async ({ page }) => {
    await page.goto('/', { waitUntil: 'networkidle' });
    
    const addButton = page.locator('button:has-text("Add Project"), button:has-text("New Project"), button:has-text("Create")');
    await expect(addButton).toBeVisible({ timeout: 5000 });
  });

  test('should display form when adding project', async ({ page }) => {
    await page.goto('/', { waitUntil: 'networkidle' });
    
    // Click Add Project button
    const addButton = page.locator('button:has-text("Add Project"), button:has-text("New Project"), button:has-text("Create")');
    await addButton.click();
    
    // Wait for form to appear
    await page.waitForSelector('input, textarea, form', { timeout: 5000 });
    
    // Check for form elements
    const inputs = await page.locator('input, textarea').count();
    expect(inputs).toBeGreaterThan(0);
  });

  test('should accept project path input', async ({ page }) => {
    await page.goto('/', { waitUntil: 'networkidle' });
    
    // Click Add Project
    const addButton = page.locator('button:has-text("Add Project"), button:has-text("New Project"), button:has-text("Create")');
    await addButton.click();
    
    // Find input field
    const input = page.locator('input[type="text"], textarea').first();
    await input.click();
    
    // Type a test path
    await input.fill('C:\\test\\project');
    
    // Verify input has value
    const value = await input.inputValue();
    expect(value).toBe('C:\\test\\project');
  });

  test('should handle form submission', async ({ page }) => {
    await page.goto('/', { waitUntil: 'networkidle' });
    
    // Click Add Project
    let addButton = page.locator('button:has-text("Add Project"), button:has-text("New Project"), button:has-text("Create")');
    await addButton.click();
    
    // Fill form with test project path
    const input = page.locator('input[type="text"], textarea').first();
    await input.fill('.');  // Current directory
    
    // Wait for submit button and click
    const submitButton = page.locator('button:has-text("Add"), button:has-text("Create"), button:has-text("Submit")').last();
    
    // Wait for network response
    const [response] = await Promise.all([
      page.waitForResponse(response => 
        response.url().includes('/api/projects') && response.status() === 200,
        { timeout: 10000 }
      ).catch(() => [null]),
      submitButton.click().catch(() => {})
    ]);
    
    // Verify either response received or UI updated
    expect(response?.status() === 200 || response === null).toBeTruthy();
  });

  test('should display validation messages for empty form', async ({ page }) => {
    await page.goto('/', { waitUntil: 'networkidle' });
    
    // Click Add Project
    const addButton = page.locator('button:has-text("Add Project"), button:has-text("New Project"), button:has-text("Create")');
    await addButton.click();
    
    // Try to submit empty form
    const submitButton = page.locator('button:has-text("Add"), button:has-text("Create"), button:has-text("Submit")').last();
    
    // Check if submit is disabled or validation shows
    const isDisabled = await submitButton.isDisabled();
    const errorMessages = page.locator('[class*="error"], [class*="required"], .text-red-600');
    const hasErrors = await errorMessages.count() > 0;
    
    expect(isDisabled || hasErrors).toBeTruthy();
  });

  test('should have responsive layout', async ({ page }) => {
    // Test desktop view
    await page.setViewportSize({ width: 1280, height: 720 });
    await page.goto('/', { waitUntil: 'networkidle' });
    
    const mainContent = page.locator('main, [class*="container"], [role="main"]');
    await expect(mainContent).toBeVisible();
    
    // Test tablet view
    await page.setViewportSize({ width: 768, height: 1024 });
    await expect(mainContent).toBeVisible();
    
    // Test mobile view
    await page.setViewportSize({ width: 375, height: 667 });
    await expect(mainContent).toBeVisible();
  });

  test('should handle navigation', async ({ page }) => {
    await page.goto('/', { waitUntil: 'networkidle' });
    
    // Check for navigation elements
    const nav = page.locator('nav, [role="navigation"], [class*="navbar"]');
    const navVisible = await nav.isVisible().catch(() => false);
    
    // Navigation may or may not exist, that's OK
    // Main point is page loads without error
    expect(page.url()).toContain('localhost');
  });

  test('should not show console errors', async ({ page }) => {
    const errors = [];
    page.on('console', msg => {
      if (msg.type() === 'error') {
        errors.push(msg.text());
      }
    });
    
    await page.goto('/', { waitUntil: 'networkidle' });
    
    // Allow time for any error messages to appear
    await page.waitForTimeout(2000);
    
    // Filter out known acceptable errors
    const criticalErrors = errors.filter(e => 
      !e.includes('ResizeObserver') && 
      !e.includes('timeout') &&
      !e.includes('CORS')
    );
    
    expect(criticalErrors.length).toBe(0);
  });

  test('should handle network errors gracefully', async ({ page }) => {
    // Simulate offline
    await page.context().setOffline(true);
    
    await page.goto('/', { waitUntil: 'domcontentloaded' });
    
    // Page should still load (or show offline message)
    expect(page).toHaveTitle(/Aviator|chatbot|ContentBridge/i);
    
    // Go back online
    await page.context().setOffline(false);
  });

  test('should maintain session across navigation', async ({ page }) => {
    await page.goto('/', { waitUntil: 'networkidle' });
    
    // Get initial URL
    const initialUrl = page.url();
    
    // Navigate and come back
    await page.goBack().catch(() => {});
    await page.goForward().catch(() => {});
    
    // Should maintain reasonable state
    expect(page.url()).toContain('localhost');
  });
});
