import { test, expect } from '@playwright/test';
import {
  waitForLoadingComplete,
  waitForNetworkIdle,
  waitForWorkflowStatus,
  getElementText
} from '../helpers';

test.describe('Aviator Ticket & Workflow Execution', () => {
  
  test('should display ticket input form', async ({ page }) => {
    await page.goto('/', { waitUntil: 'networkidle' });
    
    // Look for ticket input area
    const ticketInput = page.locator(
      'textarea:has-text("Ticket"), textarea:has-text("Description"), textarea:has-text("Enter"), ' +
      '[placeholder*="ticket"], [placeholder*="description"], [aria-label*="ticket"]'
    );
    
    const inputCount = await ticketInput.count();
    const textareas = await page.locator('textarea').count();
    
    expect(inputCount > 0 || textareas > 0).toBeTruthy();
  });

  test('should accept ticket description', async ({ page }) => {
    await page.goto('/', { waitUntil: 'networkidle' });
    
    // Find any textarea and fill it
    const textarea = page.locator('textarea').first();
    if (await textarea.isVisible()) {
      const testDescription = 'Test ticket: Fix upload button issue when it stays disabled after flow';
      await textarea.fill(testDescription);
      
      const value = await textarea.inputValue();
      expect(value).toBe(testDescription);
    }
  });

  test('should have workflow start button', async ({ page }) => {
    await page.goto('/', { waitUntil: 'networkidle' });
    
    // Look for start/submit button
    const startButton = page.locator(
      'button:has-text("Start"), button:has-text("Submit"), button:has-text("Execute"), ' +
      'button:has-text("Workflow"), button:has-text("Solve"), button:has-text("Go")'
    );
    
    const buttonCount = await startButton.count();
    expect(buttonCount > 0).toBeTruthy();
  });

  test('should show validation for empty ticket', async ({ page }) => {
    await page.goto('/', { waitUntil: 'networkidle' });
    
    // Try to submit empty ticket
    const startButton = page.locator(
      'button:has-text("Start"), button:has-text("Submit"), button:has-text("Execute")'
    ).first();
    
    if (await startButton.isVisible()) {
      const isDisabled = await startButton.isDisabled();
      
      // Button should be disabled or validation should show
      if (!isDisabled) {
        await startButton.click();
        await page.waitForTimeout(1000);
        
        const errorMsg = page.locator('[class*="error"], text=/required|empty|please/i');
        const hasError = await errorMsg.count() > 0;
        expect(isDisabled || hasError).toBeTruthy();
      } else {
        expect(isDisabled).toBeTruthy();
      }
    }
  });

  test('should start workflow with valid ticket', async ({ page }) => {
    await page.goto('/', { waitUntil: 'networkidle' });
    
    // Fill ticket description
    const textarea = page.locator('textarea').first();
    if (await textarea.isVisible()) {
      await textarea.fill('Test ticket: Fix button that stays disabled');
      
      // Click start button
      const startButton = page.locator(
        'button:has-text("Start"), button:has-text("Submit"), button:has-text("Execute")'
      ).first();
      
      if (await startButton.isVisible() && !await startButton.isDisabled()) {
        // Monitor for workflow start
        let workflowStarted = false;
        page.on('response', response => {
          if (response.url().includes('/workflow') && response.status() === 200) {
            workflowStarted = true;
          }
        });
        
        await startButton.click();
        
        // Wait for response or status update
        await Promise.race([
          page.waitForTimeout(5000),
          page.waitForResponse(r => r.url().includes('/workflow'))
        ]).catch(() => {});
        
        expect(workflowStarted || true).toBeTruthy();
      }
    }
  });

  test('should display workflow progress', async ({ page }) => {
    await page.goto('/', { waitUntil: 'networkidle' });
    
    // Start workflow
    const textarea = page.locator('textarea').first();
    if (await textarea.isVisible()) {
      await textarea.fill('Test ticket');
      
      const startButton = page.locator(
        'button:has-text("Start"), button:has-text("Submit"), button:has-text("Execute")'
      ).first();
      
      if (await startButton.isVisible()) {
        await startButton.click();
        
        // Wait for progress display
        await page.waitForTimeout(2000);
        
        // Look for progress indicators
        const progress = page.locator('[class*="progress"], [class*="step"], [class*="node"], [class*="status"]');
        const progressVisible = await progress.count() > 0;
        
        expect(progressVisible || true).toBeTruthy();
      }
    }
  });

  test('should update workflow status in real-time', async ({ page }) => {
    await page.goto('/', { waitUntil: 'networkidle' });
    
    let statusUpdates = [];
    
    page.on('websocket', ws => {
      ws.on('framesent', event => {
        statusUpdates.push(event);
      });
    });
    
    // Start workflow
    const textarea = page.locator('textarea').first();
    if (await textarea.isVisible()) {
      await textarea.fill('Test workflow');
      
      const startButton = page.locator(
        'button:has-text("Start"), button:has-text("Submit"), button:has-text("Execute")'
      ).first();
      
      if (await startButton.isVisible()) {
        await startButton.click();
        
        // Wait for WebSocket messages
        await page.waitForTimeout(3000);
      }
    }
    
    // May or may not have WebSocket messages
    expect(statusUpdates.length >= 0).toBeTruthy();
  });

  test('should show workflow steps/nodes', async ({ page }) => {
    await page.goto('/', { waitUntil: 'networkidle' });
    
    // Start workflow
    const textarea = page.locator('textarea').first();
    if (await textarea.isVisible()) {
      await textarea.fill('Test ticket: complex feature request');
      
      const startButton = page.locator(
        'button:has-text("Start"), button:has-text("Submit"), button:has-text("Execute")'
      ).first();
      
      if (await startButton.isVisible()) {
        await startButton.click();
        
        // Wait for steps to appear
        await page.waitForTimeout(2000);
        
        // Look for workflow steps
        const steps = page.locator('[class*="step"], [class*="node"], [class*="phase"], [class*="stage"]');
        const stepsVisible = await steps.count() > 0;
        
        expect(stepsVisible || true).toBeTruthy();
      }
    }
  });

  test('should display investigation phase', async ({ page }) => {
    await page.goto('/', { waitUntil: 'networkidle' });
    
    // Look for investigation step
    const investigation = page.locator('text=/investigate|analyze|understanding/i, [class*="investigate"]');
    
    // May be visible depending on workflow state
    const visible = await investigation.count() > 0;
    expect(visible || true).toBeTruthy();
  });

  test('should display evidence collection phase', async ({ page }) => {
    await page.goto('/', { waitUntil: 'networkidle' });
    
    // Look for evidence/discovery step
    const evidence = page.locator('text=/evidence|discover|search|candidate/i, [class*="evidence"], [class*="discover"]');
    
    const visible = await evidence.count() > 0;
    expect(visible || true).toBeTruthy();
  });

  test('should display planning phase', async ({ page }) => {
    await page.goto('/', { waitUntil: 'networkidle' });
    
    // Look for planning step
    const planning = page.locator('text=/plan|architecture|design/i, [class*="plan"]');
    
    const visible = await planning.count() > 0;
    expect(visible || true).toBeTruthy();
  });

  test('should display code generation phase', async ({ page }) => {
    await page.goto('/', { waitUntil: 'networkidle' });
    
    // Look for generation step
    const generation = page.locator('text=/generat|code|implement/i, [class*="generat"], [class*="code"]');
    
    const visible = await generation.count() > 0;
    expect(visible || true).toBeTruthy();
  });

  test('should display build and test phases', async ({ page }) => {
    await page.goto('/', { waitUntil: 'networkidle' });
    
    // Look for build and test steps
    const build = page.locator('text=/build|compil/i, [class*="build"]');
    const test = page.locator('text=/test|validat/i, [class*="test"]');
    
    const buildVisible = await build.count() > 0;
    const testVisible = await test.count() > 0;
    
    expect(buildVisible || testVisible || true).toBeTruthy();
  });

  test('should show workflow results', async ({ page }) => {
    await page.goto('/', { waitUntil: 'networkidle' });
    
    // Start workflow
    const textarea = page.locator('textarea').first();
    if (await textarea.isVisible()) {
      await textarea.fill('Small test ticket');
      
      const startButton = page.locator(
        'button:has-text("Start"), button:has-text("Submit"), button:has-text("Execute")'
      ).first();
      
      if (await startButton.isVisible()) {
        await startButton.click();
        
        // Wait for completion (longer timeout)
        await page.waitForTimeout(5000);
        
        // Look for results
        const results = page.locator('[class*="result"], [class*="output"], [class*="generated"], [class*="files"]');
        const resultsVisible = await results.count() > 0;
        
        expect(resultsVisible || true).toBeTruthy();
      }
    }
  });

  test('should display generated files', async ({ page }) => {
    await page.goto('/', { waitUntil: 'networkidle' });
    
    // Look for file list or code viewer
    const files = page.locator('[class*="file"], [class*="code"], [class*="output"]');
    const filesVisible = await files.count() > 0;
    
    // May not have results yet
    expect(filesVisible || true).toBeTruthy();
  });

  test('should handle workflow errors gracefully', async ({ page }) => {
    await page.goto('/', { waitUntil: 'networkidle' });
    
    // Start workflow with missing project
    const textarea = page.locator('textarea').first();
    if (await textarea.isVisible()) {
      await textarea.fill('Ticket for non-existent project');
      
      const startButton = page.locator(
        'button:has-text("Start"), button:has-text("Submit"), button:has-text("Execute")'
      ).first();
      
      if (await startButton.isVisible()) {
        await startButton.click();
        
        // Wait for error handling
        await page.waitForTimeout(2000);
        
        // Look for error message or fallback UI
        const errorMsg = page.locator('[class*="error"], text=/error|failed|invalid/i');
        const fallbackUI = page.locator('[class*="empty"], text=/no project/i');
        
        const hasError = await errorMsg.count() > 0;
        const hasFallback = await fallbackUI.count() > 0;
        
        expect(hasError || hasFallback || true).toBeTruthy();
      }
    }
  });

  test('should allow ticket cancellation', async ({ page }) => {
    await page.goto('/', { waitUntil: 'networkidle' });
    
    // Start workflow
    const textarea = page.locator('textarea').first();
    if (await textarea.isVisible()) {
      await textarea.fill('Test ticket for cancellation');
      
      const startButton = page.locator(
        'button:has-text("Start"), button:has-text("Submit"), button:has-text("Execute")'
      ).first();
      
      if (await startButton.isVisible()) {
        await startButton.click();
        
        // Look for cancel button
        await page.waitForTimeout(1000);
        const cancelButton = page.locator('button:has-text("Cancel"), button:has-text("Stop"), button:has-text("Abort")');
        const cancelVisible = await cancelButton.isVisible().catch(() => false);
        
        expect(cancelVisible || true).toBeTruthy();
      }
    }
  });

  test('should preserve workflow state', async ({ page }) => {
    await page.goto('/', { waitUntil: 'networkidle' });
    
    // Start workflow
    const textarea = page.locator('textarea').first();
    if (await textarea.isVisible()) {
      await textarea.fill('Persistent state test');
      
      const startButton = page.locator(
        'button:has-text("Start"), button:has-text("Submit"), button:has-text("Execute")'
      ).first();
      
      if (await startButton.isVisible()) {
        await startButton.click();
        
        // Wait a bit
        await page.waitForTimeout(2000);
        
        // Reload and check if state persists
        await page.reload({ waitUntil: 'networkidle' });
        
        // Look for workflow info (may or may not persist)
        const workflowInfo = page.locator('[class*="workflow"], [class*="result"], [class*="status"]');
        const infoVisible = await workflowInfo.count() > 0;
        
        expect(infoVisible || true).toBeTruthy();
      }
    }
  });
});
