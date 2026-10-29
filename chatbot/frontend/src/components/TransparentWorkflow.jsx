// Transparent Workflow Component - Shows all steps with human approval checkpoints
import React, { useState, useEffect, useRef, useCallback } from 'react';
import './TransparentWorkflow.css';

// ── Smart agent output renderer ───────────────────────────────────────────
function AgentOutputRenderer({ output, node }) {
  if (!output) return null;

  // The backend now sends clean, human-readable summaries.
  // Just unescape \n so text wraps correctly and render as formatted text.
  const cleaned = typeof output === 'string'
    ? output.replace(/\\n/g, '\n').replace(/\\t/g, '\t')
    : String(output);

  return (
    <pre style={{
      margin: 0, fontSize: 12, color: '#c8e6fa', whiteSpace: 'pre-wrap',
      overflowWrap: 'anywhere', wordBreak: 'break-word', maxWidth: '100%', overflow: 'hidden',
      fontFamily: "'Courier New', monospace", lineHeight: 1.5
    }}>{cleaned}</pre>
  );
}

// ── Preflight Report Panel ─────────────────────────────────────────────────
function PreflightReportPanel({ data, agentOutput }) {
  if (!data && !agentOutput) return null;

  // If we have structured data, render rich cards
  if (data) {
    const verdict = data.verdict || '?';
    const summary = data.summary || '';
    const missing = data.missing_requirements || [];
    const guidance = data.implementation_guidance || [];
    const verdictColor = verdict === 'ALREADY_DONE' ? '#22c55e' :
      verdict === 'PARTIALLY_DONE' ? '#f59e0b' : '#ef4444';
    const verdictIcon = verdict === 'ALREADY_DONE' ? '✅' :
      verdict === 'PARTIALLY_DONE' ? '⚠️' : '🔴';

    return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
        {/* Verdict Card */}
        <div style={{
          background: '#0a1929', border: `1px solid ${verdictColor}40`,
          borderRadius: 8, padding: '12px 16px',
          borderLeft: `4px solid ${verdictColor}`,
        }}>
          <div style={{ fontSize: 15, fontWeight: 600, color: verdictColor, marginBottom: 6 }}>
            {verdictIcon} Verdict: {verdict}
          </div>
          {summary && (
            <div style={{ fontSize: 12, color: '#94a3b8', lineHeight: 1.5, whiteSpace: 'pre-wrap' }}>
              {summary}
            </div>
          )}
        </div>

        {/* Missing Requirements */}
        {missing.length > 0 && (
          <div style={{
            background: '#1a0a0a', border: '1px solid #7f1d1d40',
            borderRadius: 8, padding: '10px 14px',
          }}>
            <div style={{ fontSize: 13, fontWeight: 600, color: '#fca5a5', marginBottom: 8 }}>
              ❌ Missing Requirements ({missing.length})
            </div>
            {missing.map((req, j) => (
              <div key={j} style={{
                fontSize: 12, color: '#f87171', padding: '3px 0',
                borderBottom: j < missing.length - 1 ? '1px solid #7f1d1d20' : 'none',
              }}>
                • {req}
              </div>
            ))}
          </div>
        )}

        {/* Implementation Guidance */}
        {guidance.length > 0 && (
          <div style={{
            background: '#0f0a29', border: '1px solid #6366f140',
            borderRadius: 8, padding: '10px 14px',
          }}>
            <div style={{ fontSize: 13, fontWeight: 600, color: '#a5b4fc', marginBottom: 8 }}>
              🔧 Implementation Guidance ({guidance.length})
            </div>
            {guidance.map((g, j) => (
              <div key={j} style={{
                background: '#1a1040', borderRadius: 6, padding: '8px 12px',
                marginBottom: j < guidance.length - 1 ? 6 : 0,
                border: '1px solid #4338ca30',
              }}>
                <div style={{ fontSize: 12, fontWeight: 600, color: '#c4b5fd' }}>
                  📋 {g.requirement || '?'}
                </div>
                <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 4 }}>
                  <span style={{ color: '#7dd3fc' }}>What:</span> {g.what_to_do || '?'}
                </div>
                <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 2 }}>
                  <span style={{ color: '#7dd3fc' }}>File:</span>{' '}
                  <code style={{ background: '#0a1929', padding: '1px 4px', borderRadius: 3 }}>
                    {g.target_file || '?'}
                  </code>
                </div>
                {g.integrate_with && (
                  <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 2 }}>
                    <span style={{ color: '#7dd3fc' }}>Integrate with:</span> {g.integrate_with}
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    );
  }

  // Fallback: render agent_output as formatted text
  return <AgentOutputRenderer output={agentOutput} node="preflight_check" />;
}

// ── Build Decision Panel (pre-existing errors) ─────────────────────────────
function BuildDecisionPanel({ payload, workflowId }) {
  const [choice, setChoice] = useState(null);
  const [sending, setSending] = useState(false);

  const handleDecision = async (decision) => {
    setSending(true);
    setChoice(decision);
    try {
      const API_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8002';
      await fetch(`${API_URL}/api/workflow/transparent/decision-response`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ workflow_id: workflowId, choice: decision }),
      });
    } catch (err) {
      console.error('Decision response failed:', err);
      setSending(false);
      setChoice(null);
    }
  };

  const affectedFiles = payload?.affected_files || [];
  const totalErrors = payload?.total_error_count || 0;
  const diagnostics = payload?.representative_diagnostics || [];

  return (
    <div style={{
      background: '#1a0f00', border: '1px solid #92400e',
      borderRadius: 10, padding: '16px 20px', marginBottom: 8,
      borderLeft: '4px solid #f59e0b',
    }}>
      <div style={{ fontSize: 15, fontWeight: 700, color: '#fbbf24', marginBottom: 8 }}>
        ⚠️ Pre-Existing Build Errors Detected
      </div>
      <div style={{ fontSize: 12, color: '#d4a574', marginBottom: 12, lineHeight: 1.5 }}>
        {payload?.message || 'These errors existed before this ticket. Choose how to proceed.'}
      </div>

      {/* Affected Files */}
      {affectedFiles.length > 0 && (
        <div style={{
          background: '#0a0a0a', borderRadius: 6, padding: '8px 12px',
          marginBottom: 10, border: '1px solid #92400e40',
        }}>
          <div style={{ fontSize: 11, color: '#9ca3af', marginBottom: 6 }}>
            📄 Affected Files ({totalErrors} total error{totalErrors !== 1 ? 's' : ''}):
          </div>
          {affectedFiles.map((af, j) => (
            <div key={j} style={{
              fontSize: 12, color: '#fbbf24', padding: '2px 0',
              display: 'flex', justifyContent: 'space-between',
            }}>
              <code style={{ background: '#1a1a2e', padding: '1px 6px', borderRadius: 3, fontSize: 11 }}>
                {af.file || '?'}
              </code>
              <span style={{ fontSize: 11, color: '#f87171' }}>
                {af.error_count} error{af.error_count !== 1 ? 's' : ''}
              </span>
            </div>
          ))}
        </div>
      )}

      {/* Representative Diagnostics */}
      {diagnostics.length > 0 && (
        <div style={{
          background: '#0a0a0a', borderRadius: 6, padding: '8px 12px',
          marginBottom: 12, border: '1px solid #92400e40',
          maxHeight: 120, overflowY: 'auto',
        }}>
          {diagnostics.map((d, j) => (
            <div key={j} style={{
              fontSize: 11, color: '#f87171', fontFamily: 'monospace',
              padding: '2px 0', whiteSpace: 'pre-wrap', wordBreak: 'break-all',
            }}>
              {d}
            </div>
          ))}
        </div>
      )}

      {/* Decision Buttons */}
      {!choice ? (
        <div style={{ display: 'flex', gap: 10 }}>
          <button
            onClick={() => handleDecision('fix')}
            disabled={sending}
            style={{
              flex: 1, padding: '10px 16px', borderRadius: 8,
              background: 'linear-gradient(135deg, #059669, #047857)',
              color: '#fff', border: 'none', cursor: 'pointer',
              fontWeight: 700, fontSize: 13,
              opacity: sending ? 0.5 : 1,
            }}
          >
            🔧 Fix These Errors
          </button>
          <button
            onClick={() => handleDecision('leave')}
            disabled={sending}
            style={{
              flex: 1, padding: '10px 16px', borderRadius: 8,
              background: 'linear-gradient(135deg, #1e40af, #1d4ed8)',
              color: '#fff', border: 'none', cursor: 'pointer',
              fontWeight: 700, fontSize: 13,
              opacity: sending ? 0.5 : 1,
            }}
          >
            ⏭️ Leave (Skip)
          </button>
          <button
            onClick={() => handleDecision('stop')}
            disabled={sending}
            style={{
              flex: 1, padding: '10px 16px', borderRadius: 8,
              background: 'linear-gradient(135deg, #991b1b, #b91c1c)',
              color: '#fff', border: 'none', cursor: 'pointer',
              fontWeight: 700, fontSize: 13,
              opacity: sending ? 0.5 : 1,
            }}
          >
            🛑 Stop Pipeline
          </button>
        </div>
      ) : (
        <div style={{
          padding: '10px 16px', borderRadius: 8, textAlign: 'center',
          background: choice === 'fix' ? '#05966920' : choice === 'leave' ? '#1e40af20' : '#991b1b20',
          color: choice === 'fix' ? '#86efac' : choice === 'leave' ? '#93c5fd' : '#fca5a5',
          fontWeight: 600, fontSize: 13,
        }}>
          {choice === 'fix' ? '🔧 Fixing pre-existing errors...' :
           choice === 'leave' ? '⏭️ Leaving errors as-is — continuing...' :
           '🛑 Pipeline stopped by user'}
        </div>
      )}
    </div>
  );
}

const OutcomeFindingPanel = ({ data }) => {
  const finding = data?.finding || {};
  const rawFindings = data?.findings || (finding.summary ? [finding] : []);
  const verdict = data?.verdict || finding.verdict || 'PARTIAL';

  const isCorrect = verdict === 'CORRECT';
  const accentColor = isCorrect ? '#10b981' : verdict === 'PARTIAL' ? '#f59e0b' : '#ef4444';
  const badgeBg = isCorrect ? '#065f4620' : verdict === 'PARTIAL' ? '#78350f30' : '#7f1d1d30';
  const badgeBorder = isCorrect ? '#059669' : verdict === 'PARTIAL' ? '#d97706' : '#dc2626';

  const defects = (data?.defects && data.defects.length > 0)
    ? data.defects
    : rawFindings.filter(f => f.verdict !== 'CORRECT');
  const satisfied = rawFindings.filter(f => f.verdict === 'CORRECT');
  const satisfiedReqIds = data?.satisfied_requirements || satisfied.map(f => f.requirement_id || f.requirement_text).filter(Boolean);

  const findings = isCorrect ? rawFindings : (defects.length > 0 ? defects : rawFindings);
  const defectCount = defects.length;

  return (
    <div style={{
      background: 'linear-gradient(135deg, #131722 0%, #1e2538 100%)',
      border: `1px solid ${badgeBorder}60`,
      borderLeft: `4px solid ${accentColor}`,
      borderRadius: 10, padding: '16px 20px', marginBottom: 12,
      boxShadow: '0 4px 12px rgba(0,0,0,0.3)',
    }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 12 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <span style={{ fontSize: 18 }}>{isCorrect ? '✅' : '⚠️'}</span>
          <span style={{ fontSize: 15, fontWeight: 700, color: accentColor }}>
            Semantic Outcome: {verdict} {!isCorrect && defectCount > 1 ? `(${defectCount} Defects Detected)` : ''}
          </span>
        </div>
        <span style={{
          fontSize: 11, fontWeight: 600, padding: '2px 8px', borderRadius: 4,
          background: badgeBg, border: `1px solid ${badgeBorder}`, color: accentColor,
        }}>
          {isCorrect ? 'ALL REQUIREMENTS SATISFIED' : `${defectCount || 1} DEFECT(S) DETECTED`}
        </span>
      </div>

      {findings.map((f, idx) => {
        const itemIsCorrect = f.verdict === 'CORRECT';
        return (
          <div key={idx} style={{
            background: '#0a0e17', borderRadius: 8, padding: '12px 16px',
            border: `1px solid ${itemIsCorrect ? '#065f4640' : '#1e293b'}`,
            marginBottom: idx < findings.length - 1 ? 10 : 0,
          }}>
            {f.requirement_text && (
              <div style={{ marginBottom: 8 }}>
                <div style={{
                  fontSize: 11, textTransform: 'uppercase',
                  color: itemIsCorrect ? '#86efac' : '#94a3b8',
                  fontWeight: 600, letterSpacing: 0.5
                }}>
                  Requirement {f.requirement_id ? `(${f.requirement_id})` : ''} {itemIsCorrect ? '— SATISFIED' : ''}
                </div>
                <div style={{ fontSize: 13, color: '#e2e8f0', marginTop: 2, lineHeight: 1.4 }}>
                  {f.requirement_text}
                </div>
              </div>
            )}

            {f.summary && (
              <div style={{ marginBottom: 8 }}>
                <div style={{
                  fontSize: 11, textTransform: 'uppercase',
                  color: itemIsCorrect ? '#10b981' : '#f59e0b',
                  fontWeight: 600, letterSpacing: 0.5
                }}>
                  {itemIsCorrect ? 'Verification Summary' : 'Issue Found'}
                </div>
                <div style={{
                  fontSize: 12,
                  color: itemIsCorrect ? '#86efac' : '#fcd34d',
                  marginTop: 2, lineHeight: 1.4
                }}>
                  {f.summary}
                </div>
              </div>
            )}

            {f.offending_code && (
              <div style={{ marginBottom: 8 }}>
                <div style={{ fontSize: 11, textTransform: 'uppercase', color: '#94a3b8', fontWeight: 600, letterSpacing: 0.5 }}>
                  Offending Condition
                </div>
                <pre style={{
                  background: '#020617', border: '1px solid #334155', borderRadius: 4,
                  padding: '6px 10px', fontSize: 12, color: '#f87171', margin: '4px 0 0 0',
                  fontFamily: 'monospace', overflowX: 'auto',
                }}>
                  {f.offending_code}
                </pre>
              </div>
            )}

            <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap', marginTop: 8, fontSize: 11, color: '#94a3b8' }}>
              {f.affected_file && (
                <div>
                  <span>Target File: </span>
                  <code style={{ background: '#1e293b', color: '#38bdf8', padding: '1px 5px', borderRadius: 3 }}>
                    {f.affected_file} {f.affected_line ? `:${f.affected_line}` : ''}
                  </code>
                </div>
              )}
              {f.missing_behavior && (
                <div style={{ width: '100%', marginTop: 4 }}>
                  <span style={{ color: '#cbd5e1', fontWeight: 600 }}>Missing Behavior: </span>
                  <span style={{ color: '#94a3b8' }}>{f.missing_behavior}</span>
                </div>
              )}
              {f.next_action && (
                <div style={{ width: '100%', marginTop: 4 }}>
                  <span style={{ color: itemIsCorrect ? '#10b981' : '#60a5fa', fontWeight: 600 }}>
                    {itemIsCorrect ? 'Status: ' : 'Next Action: '}
                  </span>
                  <span style={{ color: itemIsCorrect ? '#86efac' : '#93c5fd' }}>{f.next_action}</span>
                </div>
              )}
            </div>
          </div>
        );
      })}

      {satisfiedReqIds.length > 0 && !isCorrect && (
        <div style={{
          marginTop: 12, padding: '8px 14px', borderRadius: 8,
          background: '#065f4620', border: '1px solid #05966950',
          fontSize: 12, color: '#86efac', display: 'flex', alignItems: 'center', gap: 8
        }}>
          <span style={{ fontSize: 14 }}>✅</span>
          <div>
            <strong>Already Satisfied ({satisfiedReqIds.length}):</strong> {satisfiedReqIds.join(', ')} (Preserving verified implementation)
          </div>
        </div>
      )}
    </div>
  );
};


const API = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8002';
const WS_BASE_URL = API.replace(/^http/, 'ws');

const PHASE_LABELS = {
  idle: 'Idle',
  classification: 'Classification',
  requirements: 'Requirements Analysis',
  planning: 'Implementation Planning',
  localization: 'File Localization',
  impact_analysis: 'Impact Analysis',
  human_review_files: 'Human Review',
  context_loading: 'Loading Context',
  patch_generation: 'Generating Patches',
  edit_loop: 'Edit & Compile Loop',
  validation: 'Validation',
  human_final_review: 'Final Review',
  committing: 'Committing Changes',
  completed: 'Completed',
  failed: 'Failed',
  processing: 'Processing',
};

const NODE_ICONS = {
  // Investigation & Classification
  investigate: '🔍',
  runtime_diagnosis: '🩺',
  // Evidence Pipeline
  discover: '🗺️',
  hypothesis_investigation: '🧪',
  evidence_collection_loop: '🔍',
  evidence_ranking: '📊',
  semantic_verification: '✅',
  preflight_check: '🚦',
  // Analysis & Planning
  unified_analysis: '🧠',
  plan: '📐',
  validate_candidates: '🔎',
  localize: '📍',
  grounded_understanding: '🧠',
  ownership_completeness: '🏗️',
  dataflow_verification: '🔄',
  // Context Loading
  rag_tests: '🧪',
  rag_code: '📚',
  // Code Generation
  generate_tests: '🧪',
  generate_code: '💻',
  edit_loop: '✏️',
  // Post-Generation Validation
  patch_gate: '🛡️',
  import_validation: '📦',
  angular_module_registration: '📦',
  build: '🔨',
  pre_fix_build: '🔧',
  outcome_check: '🎯',
  outcome_verification: '🎯',
  test: '✅',
  // Fix Loops
  fix_build: '🔧',
  fix_test: '🔧',
  context_expand: '🔄',
  // Completion
  memory_update: '💾',
  // Behavior Pipeline
  behavior_investigation: '🔬',
  capability_extraction: '📊',
  capability_consolidation: '🔗',
  capability_graph_builder: '🕸️',
  capability_retrieval: '🔍',
  behavior_planning: '📝',
  plan_validation: '✅',
  shadow_metrics: '📈',
};

export const WORKFLOW_STAGES = [
  { id: 'classification', name: 'Investigation', icon: '🔍', phases: ['classification'] },
  { id: 'localization', name: 'Evidence & Files', icon: '🗺️', phases: ['localization', 'evidence_collection_loop'] },
  { id: 'planning', name: 'Planning', icon: '📐', phases: ['requirements', 'planning'] },
  { id: 'context_loading', name: 'AST & Symbol Context', icon: '🧬', phases: ['context_loading'] },
  { id: 'patch_generation', name: 'Patch Generation', icon: '💻', phases: ['patch_generation'] },
  { id: 'validation', name: 'Build & Validation', icon: '🔨', phases: ['validation', 'edit_loop'] },
  { id: 'completed', name: 'Completion', icon: '✅', phases: ['committing', 'completed'] },
];

const TransparentWorkflow = ({ projectId, ticketId, ticketDescription, repoPath, attachments = [], writableFiles = [], forbiddenFiles = [], onWorkflowStarted, onWorkflowStopped, onWorkflowResumed, existingWorkflowId, modelMode = 'default', selectedModel = null }) => {
  const [workflowId, setWorkflowId] = useState(existingWorkflowId || null);
  const [workflowState, setWorkflowState] = useState(null);
  const [steps, setSteps] = useState([]);
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [excludedChunks, setExcludedChunks] = useState(new Set()); // chunk paths excluded from RAG
  const [showFileExplorer, setShowFileExplorer] = useState(false);
  const [showDecisionDebug, setShowDecisionDebug] = useState(false);
  const [expandedSteps, setExpandedSteps] = useState(new Set());
  const [activeTab, setActiveTab] = useState('timeline'); // 'timeline' | 'rag'
  const [isStopping, setIsStopping] = useState(false);
  const [isResuming, setIsResuming] = useState(false);
  const [selectedStageFilter, setSelectedStageFilter] = useState(null);
  const [tokenUsage, setTokenUsage] = useState(null);     // live token telemetry
  const [tokenExpanded, setTokenExpanded] = useState(false); // phase breakdown toggle
  const timelineEndRef = useRef(null);

  const startedRef = useRef(false);
  const pollTimer = useRef(null);
  const wsRef = useRef(null);

  // -- helpers --------------------------------------------------------------

  const fetchState = useCallback(async (wfId) => {
    try {
      const res = await fetch(`${API}/api/workflow/transparent/${wfId}`);
      const data = await res.json();
      setWorkflowState(data);

      // Hydrate token usage from persisted backend state
      if (data.token_usage) {
        setTokenUsage(data.token_usage);
      }

      // seed steps from persisted list (for late-joining / reconnect)
      if (data.steps && data.steps.length > 0) {
        setSteps(prev => {
          // Merge API steps with existing steps to avoid race conditions 
          // where WS arrives before API saves to disk.
          const allSteps = [...prev, ...data.steps];

          // Deduplicate based on unique timestamp + message combination, or an ID if possible
          const unique = [];
          const seen = new Set();
          for (const s of allSteps) {
            const key = s.id || `${s.timestamp}-${s.message}`;
            if (!seen.has(key)) {
              seen.add(key);
              unique.push(s);
            }
          }
          // Sort chronologically just in case
          unique.sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp));
          return unique;
        });
      }

      // seed selected files from candidates
      if (data.candidate_files && data.candidate_files.length > 0) {
        setSelectedFiles(prev =>
          prev.length > 0 ? prev :
            data.candidate_files.filter(f => f.selected).map(f => f.path)
        );
      }

      return data;
    } catch {
      return null;
    }
  }, []);

  const startPolling = useCallback((wfId) => {
    const tick = async () => {
      const state = await fetchState(wfId);
      if (state && state.status === 'running') {
        pollTimer.current = setTimeout(tick, 1500);
      }
    };
    tick();
  }, [fetchState]);

  const connectWS = useCallback((wfId) => {
    const socket = new WebSocket(`${WS_BASE_URL}/ws/workflow/${wfId}`);
    wsRef.current = socket;

    socket.onmessage = (event) => {
      try {
        const step = JSON.parse(event.data);

        // ── Token telemetry (observational — intercept before general step handling)
        if (step.data?.event_type === 'token_usage_update') {
          setTokenUsage(step.data);
          return; // don't add to timeline steps
        }
        if (step.data?.token_usage) {
          setTokenUsage(step.data.token_usage);
        }

        if (step.status === 'stopped' || step.data?.stopped_by_user) {
          setWorkflowState(prev => prev ? { ...prev, status: 'stopped' } : prev);
          setIsStopping(false);
        }

        // Add the step to the timeline, de-duplicating by timestamp+message.
        // NOTE: Do NOT call fetchState(wfId) here — that is handled by the
        // polling timer (startPolling) and would cause duplicate additions
        // / batch rendering of all steps at once.
        setSteps(prev => {
          // de-duplicate by step.id or timestamp+message
          const stepKey = step.id || `${step.timestamp}-${step.message}`;
          const seen = new Set(prev.map(s => s.id || `${s.timestamp}-${s.message}`));
          return seen.has(stepKey) ? prev : [...prev, step];
        });
      } catch { /* ignore parse errors */ }
    };

    socket.onerror = () => { };  // handled by onclose
    socket.onclose = () => {
      // Reconnect after 3 s if workflow still running
      setTimeout(() => {
        if (wsRef.current === socket) connectWS(wfId);
      }, 3000);
    };
  }, [fetchState]);

  // -- start workflow once on mount -----------------------------------------

  useEffect(() => {
    if (startedRef.current) return;
    startedRef.current = true;

    (async () => {
      try {
        let wfId = existingWorkflowId;

        if (!wfId) {
          // New ticket — start the workflow via API
          const res = await fetch(`${API}/api/workflow/transparent/start`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              project_id: projectId,
              ticket_id: ticketId,
              ticket_description: ticketDescription,
              repo_path: repoPath,
              attachments: attachments || [],
              writable_files: writableFiles || [],
              forbidden_files: forbiddenFiles || [],
              model_mode: modelMode || 'default',
              selected_model: selectedModel || null,
            }),
          });
          const data = await res.json();
          wfId = data.workflow_id;
          setWorkflowId(wfId);
          if (onWorkflowStarted) onWorkflowStarted(wfId);
        } else {
          // Reconnecting to an existing run — just fetch current state, no new start
          setWorkflowId(wfId);
        }

        await fetchState(wfId);
        startPolling(wfId);
        connectWS(wfId);
      } catch (err) {
        console.error('Failed to start workflow:', err);
      }
    })();

    return () => {
      clearTimeout(pollTimer.current);
      if (wsRef.current) wsRef.current.close();
    };
  }, []); // empty deps — run once

  // -- approve helpers -------------------------------------------------------

  const approveOperation = async () => {
    await fetch(`${API}/api/workflow/transparent/approve-operation`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ workflow_id: workflowId }),
    });
    fetchState(workflowId);
  };

  const approveFiles = async () => {
    await fetch(`${API}/api/workflow/transparent/approve-files`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ workflow_id: workflowId, selected_files: selectedFiles }),
    });
    fetchState(workflowId);
  };

  const toggleFile = (path) =>
    setSelectedFiles(prev =>
      prev.includes(path) ? prev.filter(p => p !== path) : [...prev, path]
    );

  const toggleChunk = (path) =>
    setExcludedChunks(prev => {
      const next = new Set(prev);
      if (next.has(path)) next.delete(path); else next.add(path);
      return next;
    });

  const toggleStep = (i) =>
    setExpandedSteps(prev => {
      const next = new Set(prev);
      if (next.has(i)) next.delete(i); else next.add(i);
      return next;
    });

  // -- stop workflow -----------------------------------------------------------

  const stopWorkflow = useCallback(async () => {
    if (!workflowId || isStopping) return;
    setIsStopping(true);
    try {
      await fetch(`${API}/api/workflow/transparent/${workflowId}/stop`, {
        method: 'POST',
      });
      // Do NOT kill WebSocket abruptly — allow server to cleanly finish current node checkpoint
      // and send confirmed terminal 'stopped' event with telemetry.
      setWorkflowState(prev => prev ? { ...prev, status: 'stopping' } : prev);
      fetchState(workflowId);
      if (onWorkflowStopped) onWorkflowStopped(workflowId);
    } catch (err) {
      console.error('Failed to stop workflow:', err);
      setIsStopping(false);
    }
  }, [workflowId, isStopping, onWorkflowStopped, fetchState]);

  // -- resume workflow from checkpoint -----------------------------------------

  const resumeWorkflow = useCallback(async (resumeFrom = 'plan', feedback = '') => {
    if (!workflowId || isResuming) return;
    setIsResuming(true);
    try {
      const res = await fetch(`${API}/api/workflow/transparent/${workflowId}/resume`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          workflow_id: workflowId,
          resume_from: resumeFrom,
          feedback: feedback || 'User resumed workflow from checkpoint',
        }),
      });
      if (res.ok) {
        setWorkflowState(prev => prev ? { ...prev, status: 'running', current_phase: resumeFrom } : prev);
        startPolling(workflowId);
        connectWS(workflowId);
        if (onWorkflowResumed) onWorkflowResumed(workflowId, resumeFrom);
      }
    } catch (err) {
      console.error('Failed to resume workflow:', err);
    } finally {
      setIsResuming(false);
    }
  }, [workflowId, isResuming, startPolling, connectWS, onWorkflowResumed]);

  // Collect ALL rag_chunks across all steps, deduplicated by path, sorted by score
  const allRagChunks = (() => {
    const seen = new Map(); // path -> chunk (keep highest score)
    for (const step of steps) {
      for (const ch of (step.data?.rag_chunks || [])) {
        if (!seen.has(ch.path) || ch.score > seen.get(ch.path).score)
          seen.set(ch.path, ch);
      }
    }
    return [...seen.values()].sort((a, b) => b.score - a.score);
  })();

  const isRunning = workflowState?.status === 'running';
  const isCompleted = workflowState?.status === 'completed';
  const isFailed = workflowState?.status === 'failed';
  const isStopped = workflowState?.status === 'stopped';
  const activeTokenUsage = tokenUsage || workflowState?.token_usage || null;

  // ── Auto-scroll timeline to latest step ──────────────────────────────────
  useEffect(() => {
    if (isRunning && activeTab === 'timeline' && !selectedStageFilter) {
      timelineEndRef.current?.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
  }, [steps.length, isRunning, activeTab, selectedStageFilter]);

  // -- render ----------------------------------------------------------------

  if (!workflowId || !workflowState) {
    return (
      <div className="workflow-loading">
        <div className="spinner"></div>
        <p>Initializing workflow…</p>
      </div>
    );
  }

  // ── Compute live status of each stage in the pipeline ──────────────────────
  const getStageStatus = (stage, idx) => {
    const stageSteps = steps.filter(s =>
      stage.phases.includes(s.phase) ||
      (stage.id === 'localization' && (s.data?.node?.includes('evidence') || s.data?.node?.includes('discover') || s.phase === 'evidence_collection_loop'))
    );

    if (isCompleted) {
      return { status: 'completed', count: stageSteps.length };
    }

    const currentPhase = workflowState?.current_phase || '';
    const currentStageIdx = WORKFLOW_STAGES.findIndex(st => st.phases.includes(currentPhase));

    if (isFailed && currentStageIdx === idx) {
      return { status: 'failed', count: stageSteps.length };
    }

    if (currentStageIdx > idx || (stageSteps.length > 0 && currentStageIdx > idx)) {
      return { status: 'completed', count: stageSteps.length };
    }

    if (currentStageIdx === idx) {
      if (isRunning) {
        return { status: 'running', count: stageSteps.length };
      }
      return { status: 'completed', count: stageSteps.length };
    }

    if (stageSteps.length > 0) {
      return { status: 'completed', count: stageSteps.length };
    }

    return { status: 'pending', count: 0 };
  };

  // ── Collect all generated files in real-time (no waiting for completion) ──
  const allGeneratedFiles = Array.from(new Set([
    ...(workflowState.generated_files || []),
    ...steps
      .filter(s => (s.data?.node === 'generate_code' || s.data?.event_type === 'file_written') && (s.data?.file_path || s.data?.file_name))
      .map(s => s.data.file_path || s.data.file_name)
  ]));

  return (
    <div className="transparent-workflow">

      {/* -- Header -- */}
      <div className="workflow-header">
        <div className="workflow-header-top">
          <h2>Transparent Workflow</h2>
          {isRunning && (
            <button
              className={`stop-flow-btn ${isStopping ? 'stopping' : ''}`}
              onClick={stopWorkflow}
              disabled={isStopping}
              title="Stop the current workflow"
            >
              {isStopping ? (
                <><span className="spinner-sm" /> Stopping…</>
              ) : (
                <>⛔ Stop Flow</>
              )}
            </button>
          )}
          {isStopped && (
            <span className="stopped-badge">⛔ Stopped</span>
          )}
        </div>
        <div className="workflow-meta">
          <span className="ticket-id">Ticket: {workflowState.ticket_id}</span>
          <span className={`phase-badge phase-${workflowState.current_phase}`}>
            {isRunning && <span className="spinner-sm" />}
            {PHASE_LABELS[workflowState.current_phase] ?? workflowState.current_phase}
          </span>
        </div>
      </div>

      {/* ── Token Usage Banner (observational telemetry) ── */}
      {activeTokenUsage && (
        <div className="token-banner">
          <div className="token-banner-header" onClick={() => setTokenExpanded(prev => !prev)}>
            <span className="token-banner-title">🪙 Token Usage</span>
            <div className="token-banner-totals">
              <span className="token-stat">
                <span className="token-label">In</span>
                <span className="token-value">{(activeTokenUsage.tokens_in || 0).toLocaleString()}</span>
              </span>
              <span className="token-divider">│</span>
              <span className="token-stat">
                <span className="token-label">Out</span>
                <span className="token-value">{(activeTokenUsage.tokens_out || 0).toLocaleString()}</span>
              </span>
              <span className="token-divider">│</span>
              <span className="token-stat token-total">
                <span className="token-label">Gross</span>
                <span className="token-value">{(activeTokenUsage.total_tokens || 0).toLocaleString()}</span>
              </span>
              {Boolean(activeTokenUsage.thinking_tokens || activeTokenUsage.tokens_thinking) && (
                <>
                  <span className="token-divider">│</span>
                  <span className="token-stat" title="Reasoning / thinking tokens consumed during deliberation">
                    <span className="token-label" style={{ color: '#a78bfa' }}>Thinking</span>
                    <span className="token-value" style={{ color: '#a78bfa', fontWeight: 600 }}>
                      {((activeTokenUsage.thinking_tokens || activeTokenUsage.tokens_thinking) || 0).toLocaleString()}
                    </span>
                  </span>
                </>
              )}
              {Boolean(activeTokenUsage.cached_tokens) && (
                <>
                  <span className="token-divider">│</span>
                  <span className="token-stat" title="Warm KV prefix cache hits reused across calls without billing">
                    <span className="token-label" style={{ color: '#10b981' }}>KV Cached</span>
                    <span className="token-value" style={{ color: '#10b981', fontWeight: 600 }}>
                      {(activeTokenUsage.cached_tokens || 0).toLocaleString()}
                    </span>
                  </span>
                </>
              )}
              {(activeTokenUsage.excluded_tokens > 0 || activeTokenUsage.cached_tokens > 0) && (
                <>
                  <span className="token-divider">│</span>
                  <span className="token-stat token-stat-excluded" title="Tokens excluded from billing via warm KV prefix cache and AST skeleton compression">
                    <span className="token-label" style={{ color: '#10b981' }}>Excluded</span>
                    <span className="token-value" style={{ color: '#10b981', fontWeight: 600 }}>
                      -{(activeTokenUsage.excluded_tokens || activeTokenUsage.cached_tokens || 0).toLocaleString()}
                      {activeTokenUsage.savings_pct ? ` (${activeTokenUsage.savings_pct}%)` : ''}
                    </span>
                  </span>
                  <span className="token-divider">│</span>
                  <span className="token-stat token-stat-costed" title="Actual billable tokens costing money">
                    <span className="token-label" style={{ color: '#38bdf8' }}>Real Costed</span>
                    <span className="token-value" style={{ color: '#38bdf8', fontWeight: 700 }}>
                      {(activeTokenUsage.real_costed_tokens !== undefined ? activeTokenUsage.real_costed_tokens : Math.max(0, (activeTokenUsage.total_tokens || 0) - (activeTokenUsage.excluded_tokens || 0))).toLocaleString()}
                    </span>
                  </span>
                </>
              )}
              <span className="token-divider">│</span>
              <span className="token-stat">
                <span className="token-label">Calls</span>
                <span className="token-value">{activeTokenUsage.llm_calls || 0}</span>
              </span>
            </div>
            <span className={`token-expand-icon ${tokenExpanded ? 'expanded' : ''}`}>▶</span>
          </div>

          {tokenExpanded && activeTokenUsage.phase_breakdown && (
            <div className="token-phase-table">
              <div className="token-phase-row token-phase-header-row">
                <span className="token-phase-name">Phase</span>
                <span className="token-phase-val">Input</span>
                <span className="token-phase-val">Output</span>
                <span className="token-phase-val" style={{ color: '#a78bfa' }}>Thinking</span>
                <span className="token-phase-val" style={{ color: '#10b981' }}>Cached</span>
                <span className="token-phase-val">Total</span>
                <span className="token-phase-val">Calls</span>
              </div>
              {Object.entries(activeTokenUsage.phase_breakdown).map(([phase, usage]) => (
                <div className="token-phase-row" key={phase}>
                  <span className="token-phase-name">{phase.replace(/_/g, ' ')}</span>
                  <span className="token-phase-val">{(usage.tokens_in || 0).toLocaleString()}</span>
                  <span className="token-phase-val">{(usage.tokens_out || 0).toLocaleString()}</span>
                  <span className="token-phase-val" style={{ color: '#a78bfa' }}>{(usage.thinking_tokens || 0).toLocaleString()}</span>
                  <span className="token-phase-val" style={{ color: '#10b981' }}>{(usage.cached_tokens || 0).toLocaleString()}</span>
                  <span className="token-phase-val">{(usage.total_tokens || 0).toLocaleString()}</span>
                  <span className="token-phase-val">{usage.llm_calls || 0}</span>
                </div>
              ))}
            </div>
          )}

          {tokenExpanded && activeTokenUsage.kv_trace && activeTokenUsage.kv_trace.length > 0 && (
            <div className="token-kv-trace" style={{ marginTop: '12px', borderTop: '1px solid #334155', paddingTop: '10px' }}>
              <div style={{ fontSize: '12px', fontWeight: 600, color: '#38bdf8', marginBottom: '6px' }}>
                ⚡ KV Cache State Trace by Pipeline Node
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))', gap: '8px' }}>
                {activeTokenUsage.kv_trace.map((tr, idx) => (
                  <div key={idx} style={{
                    background: '#0f172a',
                    border: '1px solid #1e293b',
                    borderRadius: '6px',
                    padding: '8px 10px',
                    fontSize: '11px',
                    lineHeight: '1.4'
                  }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                      <strong style={{ color: '#f8fafc' }}>{tr.stage_name}</strong>
                      <span style={{
                        padding: '1px 6px',
                        borderRadius: '4px',
                        fontSize: '10px',
                        fontWeight: 600,
                        background: tr.cache_state === 'WARM_HIT' ? 'rgba(16, 185, 129, 0.2)' : (tr.cache_state === 'RESET_BRIEF' ? 'rgba(168, 85, 247, 0.2)' : 'rgba(56, 189, 248, 0.2)'),
                        color: tr.cache_state === 'WARM_HIT' ? '#34d399' : (tr.cache_state === 'RESET_BRIEF' ? '#c084fc' : '#38bdf8'),
                      }}>
                        {tr.cache_state || 'WARM_HIT'}
                      </span>
                    </div>
                    <div style={{ color: '#94a3b8' }}>Model: <span style={{ color: '#cbd5e1' }}>{tr.model_id}</span></div>
                    <div style={{ color: '#94a3b8' }}>Cached Tokens: <span style={{ color: '#10b981', fontWeight: 600 }}>{(tr.cached_tokens || 0).toLocaleString()}</span> ({tr.cost_savings_pct || 0}% hit)</div>
                    <div style={{ color: '#94a3b8' }}>Prefix Hash: <code>{tr.prefix_hash}</code></div>
                    {tr.lossless_store_ref && (
                      <div style={{ color: '#a78bfa', marginTop: '2px', wordBreak: 'break-all' }}>📁 Store: {tr.lossless_store_ref}</div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* ── Pipeline Stages Stepper ── */}
      <div className="workflow-stages-stepper">
        <div className="stages-stepper-header">
          <div className="stages-stepper-title">
            <span>🚀 Pipeline Stages</span>
            {selectedStageFilter && (
              <span className="stage-filter-indicator">
                (Filtering: {WORKFLOW_STAGES.find(s => s.id === selectedStageFilter)?.name})
              </span>
            )}
          </div>
          <div className="stages-stepper-progress">
            {WORKFLOW_STAGES.filter((st, i) => getStageStatus(st, i).status === 'completed').length} of {WORKFLOW_STAGES.length} Completed
          </div>
        </div>

        <div className="stages-stepper-track">
          {WORKFLOW_STAGES.map((stage, idx) => {
            const stageInfo = getStageStatus(stage, idx);
            const isFilterActive = selectedStageFilter === stage.id;
            return (
              <div
                key={stage.id}
                className={`stage-step-card ${stageInfo.status} ${isFilterActive ? 'active-filter' : ''}`}
                onClick={() => setSelectedStageFilter(prev => prev === stage.id ? null : stage.id)}
                title={isFilterActive ? 'Click to show all steps' : `Click to filter steps for ${stage.name}`}
              >
                <div className="stage-card-top">
                  <span className="stage-card-icon">{stage.icon}</span>
                  <span className={`stage-status-pill pill-${stageInfo.status}`}>
                    {stageInfo.status === 'completed' && '✓ Done'}
                    {stageInfo.status === 'running' && '⚡ Active'}
                    {stageInfo.status === 'pending' && '⏳ Pending'}
                    {stageInfo.status === 'failed' && '❌ Failed'}
                  </span>
                </div>
                <div className="stage-card-name">{stage.name}</div>
                <div style={{
                  fontSize: '9.5px',
                  fontWeight: 700,
                  marginTop: '4px',
                  padding: '1px 5px',
                  borderRadius: '4px',
                  display: 'inline-block',
                  background: stage.id === 'discovery' || stage.id === 'localization'
                    ? 'rgba(56, 189, 248, 0.15)'
                    : stage.id === 'build' || stage.id === 'test'
                    ? 'rgba(234, 179, 8, 0.15)'
                    : 'rgba(16, 185, 129, 0.15)',
                  color: stage.id === 'discovery' || stage.id === 'localization'
                    ? '#38bdf8'
                    : stage.id === 'build' || stage.id === 'test'
                    ? '#facc15'
                    : '#34d399',
                  border: '1px solid rgba(255, 255, 255, 0.08)',
                  whiteSpace: 'nowrap',
                  overflow: 'hidden',
                  textOverflow: 'ellipsis'
                }}>
                  {stage.id === 'discovery' || stage.id === 'localization'
                    ? '⚡ Nemotron 3.5'
                    : stage.id === 'build' || stage.id === 'test'
                    ? '🏗️ Sandbox'
                    : '🧠 Nemotron 120B'}
                </div>
                <div className="stage-card-meta">
                  {stageInfo.count > 0 ? `${stageInfo.count} step${stageInfo.count > 1 ? 's' : ''}` : stageInfo.status === 'running' ? 'Running…' : 'Queued'}
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* -- Tabs: Timeline | AST Context -- */}
      <div className="wf-tabs">
        <button
          className={`wf-tab ${activeTab === 'timeline' ? 'active' : ''}`}
          onClick={() => setActiveTab('timeline')}
        >
          📋 Agent Steps ({steps.length})
        </button>
        <button
          className={`wf-tab ${activeTab === 'rag' ? 'active' : ''}`}
          onClick={() => setActiveTab('rag')}
        >
          🔍 Symbol & AST Context ({allRagChunks.length})
          {allRagChunks.length > 0 && (
            <span className="tab-badge">{allRagChunks.filter(c => !excludedChunks.has(c.path)).length} active</span>
          )}
        </button>
      </div>

      {/* ═══════════════════ TAB: TIMELINE ═══════════════════ */}
      {activeTab === 'timeline' && (() => {
        const displayedSteps = selectedStageFilter
          ? steps.filter(s => {
            const stg = WORKFLOW_STAGES.find(st => st.id === selectedStageFilter);
            return stg ? (stg.phases.includes(s.phase) || (stg.id === 'localization' && (s.data?.node?.includes('evidence') || s.data?.node?.includes('discover') || s.phase === 'evidence_collection_loop'))) : true;
          })
          : steps;

        return (
          <div className="workflow-timeline">
            {selectedStageFilter && (
              <div className="stage-filter-banner">
                <span>Filtered by: <strong>{WORKFLOW_STAGES.find(st => st.id === selectedStageFilter)?.name}</strong> ({displayedSteps.length} step{displayedSteps.length !== 1 ? 's' : ''})</span>
                <button className="btn-clear-filter" onClick={() => setSelectedStageFilter(null)}>✕ Show All Steps</button>
              </div>
            )}
            <div className="timeline-steps">
              {displayedSteps.length === 0 && isRunning && (
                <div className="timeline-waiting">
                  <span className="spinner-sm" /> Waiting for first agent step…
                </div>
              )}
              {displayedSteps.map((step, i) => {
                const nodeIcon = NODE_ICONS[step.data?.node] || '⚙️';
                const isExpanded = expandedSteps.has(i);
                const hasOutput = !!(step.data?.agent_output);
                const hasRag = !!(step.data?.rag_chunks?.length);
                const eventType = step.data?.event_type;

                // ── Build Decision Panel (pre-existing errors) ──
                if (eventType === 'pre_existing_errors') {
                  return (
                    <BuildDecisionPanel
                      key={i}
                      payload={step.data?.decision_payload || step.data}
                      workflowId={workflowId}
                    />
                  );
                }

                // ── Outcome Finding Panel (Structured Semantic Verification) ──
                if (eventType === 'outcome_finding') {
                  return (
                    <OutcomeFindingPanel
                      key={i}
                      data={step.data}
                    />
                  );
                }

                // ── Compact file event cards (live-check progress) ──
                if (eventType) {
                  const STATUS_STYLES = {
                    file_written: { icon: '📝', bg: '#0d2137', border: '#1e3a5c', color: '#7ec8e3' },
                    live_compile_start: { icon: '🔨', bg: '#1e1b2e', border: '#8b5cf6', color: '#c4b5fd' },
                    live_compile_passed: { icon: '✅', bg: '#0b2618', border: '#22c55e', color: '#86efac' },
                    live_compile_failed: { icon: '❌', bg: '#2d1b00', border: '#ef4444', color: '#fca5a5' },
                    live_check_start: { icon: '⚠️', bg: '#2d1b00', border: '#c68a00', color: '#fbbf24' },
                    live_check_resolved: { icon: '✅', bg: '#0b2618', border: '#22c55e', color: '#86efac' },
                    live_check_passed: { icon: '✅', bg: '#0b2618', border: '#22c55e', color: '#86efac' },
                    live_check_retry: { icon: '🔄', bg: '#1e1b2e', border: '#8b5cf6', color: '#c4b5fd' },
                    live_check_deferred: { icon: '⏭️', bg: '#2d1b00', border: '#f59e0b', color: '#fde68a' },
                    stage_07_rag_include: { icon: '🧠', bg: '#0f1b2d', border: '#6366f1', color: '#a5b4fc' },
                    stage_07_rag_exclude: { icon: '❌', bg: '#1a1520', border: '#6b5070', color: '#9ca3af' },
                  };
                  const style = STATUS_STYLES[eventType] || STATUS_STYLES.file_written;
                  const fileName = step.data?.file_name || step.data?.file_path?.split('/').pop() || '?';
                  const errors = step.data?.errors || [];
                  const hasErrors = errors.length > 0;

                  return (
                    <div key={i} className="file-event-card" style={{
                      display: 'flex', alignItems: 'center', gap: 10,
                      padding: '6px 12px', marginBottom: 4, borderRadius: 6,
                      background: style.bg, border: `1px solid ${style.border}`,
                      fontSize: 13, color: style.color, cursor: hasErrors ? 'pointer' : 'default',
                      position: 'relative', zIndex: isExpanded ? 20 : 1,
                    }} onClick={() => hasErrors && toggleStep(i)}>
                      <span style={{ fontSize: 16 }}>{style.icon}</span>
                      <code style={{
                        fontFamily: "'Courier New', monospace", fontSize: 12,
                        background: '#0a1929', padding: '1px 6px', borderRadius: 4
                      }}>
                        {fileName}
                      </code>
                      <span style={{ flex: 1, opacity: 0.85 }}>{step.message}</span>
                      {step.data?.change_ratio && (
                        <span style={{ fontSize: 11, opacity: 0.6 }}>Δ {step.data.change_ratio}</span>
                      )}
                      {step.data?.attempt && (
                        <span style={{
                          fontSize: 11, background: '#1a1a2e', padding: '1px 6px',
                          borderRadius: 4
                        }}>attempt {step.data.attempt}/3</span>
                      )}
                      {step.data?.remaining_errors != null && (
                        <span style={{ fontSize: 11, color: '#f87171' }}>
                          {step.data.remaining_errors} error(s)
                        </span>
                      )}
                      {step.data?.files_fixed?.length > 0 && (
                        <span style={{ fontSize: 11, color: '#86efac' }}>
                          {step.data.files_fixed.length} file(s) fixed
                        </span>
                      )}
                      {hasErrors && (
                        <span style={{ fontSize: 10, opacity: 0.5 }}>
                          {isExpanded ? '▲' : '▼'}
                        </span>
                      )}
                      <span style={{ fontSize: 10, opacity: 0.4 }}>
                        {step.timestamp ? new Date(step.timestamp).toLocaleTimeString() : ''}
                      </span>
                      {/* Expandable error details */}
                      {hasErrors && isExpanded && (
                        <div style={{
                          position: 'absolute', left: 0, right: 0, top: '100%',
                          background: '#0a1929', border: `1px solid ${style.border}`,
                          borderRadius: '0 0 6px 6px', padding: '6px 12px', zIndex: 10
                        }}
                          onClick={(e) => e.stopPropagation()}>
                          {errors.map((err, j) => (
                            <div key={j} style={{
                              fontSize: 11, color: '#f87171',
                              fontFamily: 'monospace', padding: '2px 0',
                              whiteSpace: 'pre-wrap', wordBreak: 'break-all'
                            }}>
                              {err}
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  );
                }

                // Compute resolved status for this step so completed steps show ✓ immediately
                const isStepDone = step.status === 'completed' || i < displayedSteps.length - 1 || isCompleted;
                const isStepRunning = !isStepDone && isRunning && i === displayedSteps.length - 1;
                const resolvedStatus = isStepDone ? 'completed' : isStepRunning ? 'running' : (step.status || 'pending');

                return (
                  <div key={i} className={`timeline-step status-${resolvedStatus}`}>
                    <div className={`step-marker marker-${resolvedStatus}`}>
                      {isStepDone ? '✓' : isStepRunning ? <span className="spinner-xs" /> : nodeIcon}
                    </div>
                    <div className="step-content">
                      <div className="step-header" onClick={() => hasOutput && toggleStep(i)}
                        style={{ cursor: hasOutput ? 'pointer' : 'default' }}>
                        <div className="step-header-left">
                          <span className="step-phase">
                            {PHASE_LABELS[step.phase] ?? step.phase}
                            {step.data?.node && <span className="step-node"> [{step.data.node}]</span>}
                          </span>
                          <span className={`step-status-pill pill-${resolvedStatus}`}>
                            {resolvedStatus === 'completed' && '✓ Done'}
                            {resolvedStatus === 'running' && '⚡ Running'}
                            {resolvedStatus === 'failed' && '❌ Failed'}
                            {resolvedStatus === 'pending' && '⏳ Pending'}
                          </span>
                        </div>
                        <div className="step-header-right">
                          {hasRag && <span className="rag-badge">🔍 {step.data.rag_chunks.length} RAG</span>}
                          {hasOutput && (
                            <span className="expand-btn">{isExpanded ? '▲ Hide' : '▼ Details'}</span>
                          )}
                          <span className="step-time">
                            {step.timestamp ? new Date(step.timestamp).toLocaleTimeString() : ''}
                          </span>
                        </div>
                      </div>

                      <p className="step-message">{step.message}</p>

                      {/* Agent Output — collapsible */}
                      {hasOutput && isExpanded && (
                        <div className="agent-output-panel" style={{ overflow: 'hidden', maxWidth: '100%' }}>
                          {step.data?.node === 'preflight_check' ? (
                            <>
                              <div className="agent-output-label">🚦 Preflight Analysis Report:</div>
                              <PreflightReportPanel
                                data={step.data.preflight_data}
                                agentOutput={step.data.agent_output}
                              />
                            </>
                          ) : (
                            <>
                              <div className="agent-output-label">Agent Output:</div>
                              <AgentOutputRenderer
                                output={step.data.agent_output}
                                node={step.data?.node}
                              />
                            </>
                          )}
                        </div>
                      )}

                      {/* Per-task explanations (shows WHY each file was changed) */}
                      {step.data?.task_explanations?.length > 0 && isExpanded && (
                        <div className="task-explanations-panel">
                          <div className="task-explanations-label">📋 Per-Task Breakdown:</div>
                          {step.data.task_explanations.map((te, teIdx) => (
                            <div key={teIdx} className="task-explanation-card">
                              <div className="te-header">
                                <span className="te-index">{te.task_index}/{te.total_tasks}</span>
                                <span className="te-file">{te.basename}</span>
                                <span className={`te-change-badge te-change-${(te.change_type || '').toLowerCase()}`}>
                                  {(te.change_type || '').toUpperCase()}
                                </span>
                              </div>
                              <div className="te-purpose">{te.purpose}</div>
                              {te.produces?.length > 0 && (
                                <div className="te-section">
                                  <span className="te-section-icon">🔧</span>
                                  <span className="te-section-label">Creates:</span>
                                  {te.produces.map((p, pi) => (
                                    <code key={pi} className="te-symbol">{p}</code>
                                  ))}
                                </div>
                              )}
                              {te.consumes?.length > 0 && (
                                <div className="te-section">
                                  <span className="te-section-icon">📥</span>
                                  <span className="te-section-label">Uses:</span>
                                  {te.consumes.map((c, ci) => (
                                    <code key={ci} className="te-symbol te-consume">{c}</code>
                                  ))}
                                </div>
                              )}
                              {te.dependencies?.length > 0 && (
                                <div className="te-deps">
                                  🔗 Depends on: {te.dependencies.join(', ')}
                                </div>
                              )}
                            </div>
                          ))}
                        </div>
                      )}

                      {/* RAG chunks inline summary (when step has chunks) */}
                      {hasRag && isExpanded && (
                        <div className="rag-inline">
                          <div className="rag-inline-label">RAG chunks used:</div>
                          {step.data.rag_chunks.map((ch, j) => {
                            const excluded = excludedChunks.has(ch.path);
                            const pct = Math.round(ch.score * 100);
                            return (
                              <div key={j} className={`rag-chunk-row ${excluded ? 'excluded' : ''}`}>
                                <input
                                  type="checkbox"
                                  checked={!excluded}
                                  onChange={() => toggleChunk(ch.path)}
                                  title={excluded ? 'Include in RAG' : 'Exclude from RAG'}
                                />
                                <div className="rag-chunk-score-bar">
                                  <div className="rag-score-fill" style={{
                                    width: `${pct}%`,
                                    background: pct >= 70 ? '#22c55e' : pct >= 40 ? '#f59e0b' : '#ef4444'
                                  }} />
                                </div>
                                <span className="rag-chunk-pct">{pct}%</span>
                                <span className="rag-chunk-label">{ch.rag_label}</span>
                                <code className="rag-chunk-path">{ch.path.split('/').slice(-2).join('/')}</code>
                                {ch.name && <span className="rag-chunk-name">({ch.name})</span>}
                              </div>
                            );
                          })}
                        </div>
                      )}
                    </div>
                  </div>
                );
              })}
              <div ref={timelineEndRef} />
            </div>
          </div>
        );
      })()}

      {/* ═══════════════════ TAB: RAG CONTEXT ═══════════════════ */}
      {activeTab === 'rag' && (
        <div className="rag-context-panel">
          {allRagChunks.length === 0 ? (
            <div className="rag-empty">
              {isRunning
                ? '⏳ RAG retrieval not yet started…'
                : '⚠️ No RAG context was retrieved for this ticket.'}
            </div>
          ) : (
            <>
              <div className="rag-panel-info">
                Showing {allRagChunks.length} unique chunks retrieved from your codebase.
                <strong> Higher score = more relevant to the ticket.</strong>
                Uncheck chunks to exclude them from agent context.
                <span className="rag-threshold-note"> ✅ Scores above 60% are high-quality matches.</span>
              </div>
              <div className="rag-chunks-list">
                {allRagChunks.map((ch, i) => {
                  const excluded = excludedChunks.has(ch.path);
                  const pct = Math.round(ch.score * 100);
                  const quality = pct >= 70 ? 'high' : pct >= 40 ? 'medium' : 'low';
                  return (
                    <div key={i} className={`rag-chunk-card ${excluded ? 'chunk-excluded' : ''} quality-${quality}`}>
                      <div className="chunk-card-header">
                        <input
                          type="checkbox"
                          checked={!excluded}
                          onChange={() => toggleChunk(ch.path)}
                        />
                        <div className="chunk-score-bar-wrap">
                          <div className="chunk-score-fill" style={{ width: `${pct}%` }} />
                        </div>
                        <span className={`chunk-pct quality-${quality}`}>{pct}%</span>
                        <span className="chunk-label-badge">{ch.rag_label}</span>
                        <code className="chunk-path">{ch.path}</code>
                      </div>
                      {ch.name && (
                        <div className="chunk-name">
                          {ch.chunk_type && <span className="chunk-type-badge">{ch.chunk_type}</span>}
                          <strong>{ch.name}</strong>
                        </div>
                      )}
                      {ch.preview && (
                        <pre className="chunk-preview" style={{ whiteSpace: 'pre-wrap', overflowWrap: 'anywhere', wordBreak: 'break-word', maxWidth: '100%', overflow: 'hidden' }}>{ch.preview}{ch.preview.length >= 300 ? '\n…' : ''}</pre>
                      )}
                    </div>
                  );
                })}
              </div>
              <div className="rag-panel-footer">
                <span>✅ Active: {allRagChunks.filter(c => !excludedChunks.has(c.path)).length}</span>
                <span>❌ Excluded: {excludedChunks.size}</span>
                <button className="btn btn-sm" onClick={() => setExcludedChunks(new Set())}>
                  Reset (include all)
                </button>
                <button className="btn btn-sm btn-danger"
                  onClick={() => setExcludedChunks(new Set(allRagChunks.filter(c => Math.round(c.score * 100) < 40).map(c => c.path)))}>
                  Auto-exclude low quality (&lt;40%)
                </button>
              </div>
            </>
          )}
        </div>
      )}

      {/* -- Candidate Files panel (always visible when files exist) -- */}
      {workflowState.candidate_files && workflowState.candidate_files.length > 0 && (
        <div className="localization-results">
          <h3>📁 Files from RAG ({workflowState.candidate_files.length})</h3>
          <div className="file-candidates">
            {workflowState.candidate_files.map((file, i) => (
              <div key={i} className={`file-candidate ${selectedFiles.includes(file.path) ? 'selected' : ''}`}>
                <input
                  type="checkbox"
                  checked={selectedFiles.includes(file.path)}
                  onChange={() => toggleFile(file.path)}
                />
                <div className="file-info">
                  <span className="file-path">{file.path}</span>
                  {(file.confidence ?? file.score) !== undefined && (file.confidence ?? file.score) > 0 && (
                    <span className="confidence-score">
                      {((file.confidence ?? file.score) * 100).toFixed(0)}%
                    </span>
                  )}
                </div>
              </div>
            ))}
          </div>
          <div className="file-actions">
            <button className="btn btn-secondary" onClick={() => setShowFileExplorer(true)}>
              + Add Files
            </button>
            <button
              className="btn btn-approve"
              disabled={selectedFiles.length === 0}
              onClick={approveFiles}
            >
              ✅ Approve {selectedFiles.length} File{selectedFiles.length !== 1 ? 's' : ''}
            </button>
          </div>
        </div>
      )}

      {/* -- Generated files (visible immediately in real-time as each file is produced) -- */}
      {allGeneratedFiles.length > 0 && (
        <div className="generated-files">
          <h3>📦 Generated / Modified Files ({allGeneratedFiles.length})</h3>
          <ul>
            {allGeneratedFiles.map((f, i) => {
              const basename = typeof f === 'string' ? f.split('/').pop().split('\\').pop() : '?';
              return (
                <li key={i} title={f}><code>{basename}</code></li>
              );
            })}
          </ul>
          <p style={{ fontSize: 12, opacity: 0.6, marginTop: 4 }}>→ View full changes in the Changes tab</p>
        </div>
      )}

      {/* -- Structured explanation generated from live workflow state -- */}
      {workflowState.workflow_explanation && (
        <div className="generated-files human-report-panel">
          <h3>🧠 Workflow Explanation</h3>
          <div className="agent-output-panel">
            {workflowState.workflow_explanation.decision_count > 0 && (
              <p><strong>Decision Ledger Entries:</strong> {workflowState.workflow_explanation.decision_count}</p>
            )}

            <div className="agent-output-label">Summary</div>
            <p>{workflowState.workflow_explanation.summary}</p>

            <div className="agent-output-label">Why</div>
            <p>{workflowState.workflow_explanation.why}</p>

            {workflowState.workflow_explanation.decisions?.length > 0 && (
              <>
                <div className="agent-output-label">Key Decisions</div>
                <ul>
                  {workflowState.workflow_explanation.decisions.map((d, i) => (
                    <li key={i}>{d}</li>
                  ))}
                </ul>
                {workflowState.workflow_explanation.decision_debug?.length > workflowState.workflow_explanation.decisions.length && (
                  <>
                    <button
                      type="button"
                      className="btn btn-sm"
                      onClick={() => setShowDecisionDebug(prev => !prev)}
                    >
                      {showDecisionDebug ? 'Hide Decision Debug' : 'Expand Decision Debug'}
                    </button>
                    {showDecisionDebug && (
                      <ul>
                        {workflowState.workflow_explanation.decision_debug.map((d, i) => (
                          <li key={`debug-${i}`}>{d}</li>
                        ))}
                      </ul>
                    )}
                  </>
                )}
              </>
            )}

            {workflowState.workflow_explanation.evidence?.length > 0 && (
              <>
                <div className="agent-output-label">Evidence</div>
                <ul>
                  {workflowState.workflow_explanation.evidence.map((e, i) => (
                    <li key={i}>{e}</li>
                  ))}
                </ul>
              </>
            )}

            {workflowState.workflow_explanation.details?.length > 0 && (
              <>
                <div className="agent-output-label">Workflow Details</div>
                <ul>
                  {workflowState.workflow_explanation.details.map((d, i) => (
                    <li key={i}>{d}</li>
                  ))}
                </ul>
              </>
            )}

            {workflowState.workflow_explanation.confidence_trend?.length > 0 && (
              <>
                <div className="agent-output-label">Confidence Trend</div>
                <ul>
                  {workflowState.workflow_explanation.confidence_trend.map((c, i) => (
                    <li key={i}>{c}</li>
                  ))}
                </ul>
              </>
            )}

            {workflowState.workflow_explanation.active_assumptions?.length > 0 && (
              <>
                <div className="agent-output-label">Active Assumptions</div>
                <ul>
                  {workflowState.workflow_explanation.active_assumptions.map((a, i) => (
                    <li key={i}>{a}</li>
                  ))}
                </ul>
              </>
            )}

            {workflowState.workflow_explanation.invalidated_assumptions?.length > 0 && (
              <>
                <div className="agent-output-label">Invalidated Assumptions</div>
                <ul>
                  {workflowState.workflow_explanation.invalidated_assumptions.map((a, i) => (
                    <li key={i}>{a}</li>
                  ))}
                </ul>
              </>
            )}

            {workflowState.workflow_explanation.blockers?.length > 0 && (
              <>
                <div className="agent-output-label">Current Blockers</div>
                <ul>
                  {workflowState.workflow_explanation.blockers.map((b, i) => (
                    <li key={i}>{b}</li>
                  ))}
                </ul>
              </>
            )}

            {activeTokenUsage?.latest_kv_state && (
              <>
                <div className="agent-output-label" style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <span>⚡ KV Cache & Memory Trace</span>
                  <span style={{
                    fontSize: '11px',
                    padding: '2px 8px',
                    borderRadius: '12px',
                    background: activeTokenUsage.latest_kv_state.cache_state === 'WARM_HIT' ? 'rgba(16, 185, 129, 0.2)' : 'rgba(56, 189, 248, 0.2)',
                    color: activeTokenUsage.latest_kv_state.cache_state === 'WARM_HIT' ? '#34d399' : '#38bdf8',
                    fontWeight: 600
                  }}>
                    {activeTokenUsage.latest_kv_state.cache_state}
                  </span>
                </div>
                <div style={{ fontSize: '12px', color: '#94a3b8', marginBottom: '12px', background: '#0f172a', padding: '10px 14px', borderRadius: '8px', border: '1px solid #1e293b' }}>
                  <div><strong>Active Model:</strong> {activeTokenUsage.latest_kv_state.model_id} ({activeTokenUsage.latest_kv_state.provider})</div>
                  <div><strong>Prefix Hash:</strong> <code>{activeTokenUsage.latest_kv_state.prefix_hash}</code> (byte-stable spine)</div>
                  <div><strong>Cache Hit Rate:</strong> {activeTokenUsage.latest_kv_state.cost_savings_pct}% ({activeTokenUsage.latest_kv_state.cached_tokens?.toLocaleString()} cached / {activeTokenUsage.latest_kv_state.new_tokens?.toLocaleString()} new)</div>
                  <div><strong>Context Utilization:</strong> {activeTokenUsage.latest_kv_state.context_fill_pct}% of {activeTokenUsage.latest_kv_state.context_limit?.toLocaleString()} token window</div>
                  {activeTokenUsage.latest_kv_state.lossless_store_ref && (
                    <div style={{ marginTop: '4px', color: '#a78bfa' }}>
                      📁 <strong>Lossless Store Ref:</strong> <code>{activeTokenUsage.latest_kv_state.lossless_store_ref}</code>
                    </div>
                  )}
                </div>
              </>
            )}

            <div className="agent-output-label">Next Action</div>
            <p>{workflowState.workflow_explanation.next_action}</p>
          </div>
        </div>
      )}

      {/* -- Error -- */}
      {isFailed && workflowState.error && (
        <div className="workflow-error">
          <h3>❌ Workflow Failed</h3>
          <pre style={{ whiteSpace: 'pre-wrap', overflowWrap: 'anywhere', wordBreak: 'break-word', maxWidth: '100%', overflow: 'hidden' }}>{workflowState.error}</pre>
        </div>
      )}

      {/* -- Stopped Banner -- */}
      {isStopped && (
        <div className="workflow-stopped-banner" style={{
          margin: '16px 0',
          padding: '16px',
          background: 'rgba(239, 68, 68, 0.08)',
          border: '1px solid rgba(239, 68, 68, 0.3)',
          borderRadius: '8px',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          gap: '16px',
          flexWrap: 'wrap',
        }}>
          <div style={{ flex: 1, minWidth: '280px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <span style={{ fontSize: '18px' }}>🛑</span>
              <strong style={{ color: '#f87171', fontSize: '14px' }}>Workflow Stopped by User</strong>
            </div>
            <p style={{ margin: '4px 0 0 0', color: '#cbd5e1', fontSize: '13px', lineHeight: '1.5' }}>
              Execution was safely halted. All investigated files, candidate symbols, and pipeline steps are preserved in conversation memory. You can switch to the Chat tab to continue asking questions, or resume execution directly.
            </p>
          </div>
          <div>
            <button
              className="resume-flow-btn"
              onClick={() => resumeWorkflow('plan', 'Resumed from plan checkpoint')}
              disabled={isResuming}
              style={{
                background: '#0284c7',
                color: '#fff',
                border: 'none',
                padding: '8px 16px',
                borderRadius: '6px',
                cursor: 'pointer',
                fontWeight: 600,
                fontSize: '13px',
                whiteSpace: 'nowrap',
                display: 'flex',
                alignItems: 'center',
                gap: '6px',
              }}
            >
              {isResuming ? '🔄 Resuming…' : '🔄 Resume from Plan'}
            </button>
          </div>
        </div>
      )}

      {/* -- Complete -- */}
      {isCompleted && (
        <div className="workflow-complete">
          ✅ Workflow completed successfully. Explanation is shown above.
        </div>
      )}

      {/* -- File Explorer Modal -- */}
      {showFileExplorer && (
        <FileExplorerModal
          repoPath={repoPath}
          selectedFiles={selectedFiles}
          onToggle={toggleFile}
          onClose={() => setShowFileExplorer(false)}
        />
      )}
    </div>
  );
};

// -- File Explorer Modal -----------------------------------------------------

const FileExplorerModal = ({ repoPath, selectedFiles, onToggle, onClose }) => {
  const [fileTree, setFileTree] = useState(null);

  useEffect(() => {
    // Fetch real file tree from backend if available, else mock
    fetch(`${API}/api/file-tree?path=${encodeURIComponent(repoPath)}`)
      .then(r => r.json())
      .then(setFileTree)
      .catch(() => {
        setFileTree({
          name: repoPath.split(/[/\\]/).pop(),
          type: 'folder',
          children: [
            {
              name: 'src', type: 'folder', children: [
                {
                  name: 'main', type: 'folder', children: [
                    {
                      name: 'java', type: 'folder', children: [
                        { name: 'AreaService.java', type: 'file', path: 'src/main/java/AreaService.java' },
                      ]
                    }
                  ]
                }
              ]
            }
          ],
        });
      });
  }, [repoPath]);

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content file-explorer" onClick={e => e.stopPropagation()}>
        <div className="modal-header">
          <h3>Add Files from Repository</h3>
          <button className="btn-close" onClick={onClose}>�</button>
        </div>
        <div className="modal-body">
          {fileTree
            ? <FileTreeNode node={fileTree} selectedFiles={selectedFiles} onToggle={onToggle} />
            : <p>Loading�</p>}
        </div>
        <div className="modal-footer">
          <button className="btn btn-primary" onClick={onClose}>Done</button>
        </div>
      </div>
    </div>
  );
};

const FileTreeNode = ({ node, selectedFiles, onToggle, level = 0 }) => {
  const [expanded, setExpanded] = useState(level === 0);

  if (node.type === 'file') {
    return (
      <div className="tree-file" style={{ paddingLeft: `${level * 20}px` }}>
        <input
          type="checkbox"
          checked={selectedFiles.includes(node.path)}
          onChange={() => onToggle(node.path)}
        />
        <span className="file-icon">??</span>
        <span className="file-name">{node.name}</span>
      </div>
    );
  }

  return (
    <div className="tree-folder">
      <div
        className="folder-header"
        style={{ paddingLeft: `${level * 20}px` }}
        onClick={() => setExpanded(!expanded)}
      >
        <span className="folder-icon">{expanded ? '??' : '??'}</span>
        <span className="folder-name">{node.name}</span>
      </div>
      {expanded && node.children && (
        <div className="folder-children">
          {node.children.map((child, i) => (
            <FileTreeNode
              key={i}
              node={child}
              selectedFiles={selectedFiles}
              onToggle={onToggle}
              level={level + 1}
            />
          ))}
        </div>
      )}
    </div>
  );
};

export default TransparentWorkflow;
