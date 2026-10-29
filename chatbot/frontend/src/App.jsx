import { useState, useEffect, useRef } from 'react';
import './App.css';
import TransparentWorkflow from './components/TransparentWorkflow';
import McpHub from './components/McpHub';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8002';
const WS_BASE_URL = API_BASE_URL.replace(/^http/, 'ws');

function renderInline(str) {
  let s = str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
  // Bold: **text**
  s = s.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  // Italic: *text*
  s = s.replace(/(?<!\*)\*([^*]+)\*(?!\*)/g, '<em>$1</em>');
  // Inline code: `text`
  s = s.replace(/`([^`]+)`/g, '<code class="chat-inline-code">$1</code>');
  // Markdown links: [text](url)
  s = s.replace(/\[([^\]]+)\]\((https?:\/\/[^\s\)]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer" class="chat-markdown-link">$1 ↗</a>');
  return s;
}

function isTableSeparator(line) {
  const trimmed = line.trim();
  if (!trimmed.includes('-')) return false;
  return /^\|?(\s*:?-+:?\s*\|?)+$/.test(trimmed);
}

function parseTableRow(line) {
  let trimmed = line.trim();
  if (trimmed.startsWith('|')) trimmed = trimmed.slice(1);
  if (trimmed.endsWith('|')) trimmed = trimmed.slice(0, -1);
  return trimmed.split('|').map(c => c.trim());
}

function parseMarkdownBlocks(text) {
  const lines = text.split('\n');
  const blocks = [];
  let i = 0;

  while (i < lines.length) {
    const line = lines[i];
    const trimmed = line.trim();

    if (!trimmed) {
      blocks.push({ type: 'break' });
      i++;
      continue;
    }

    // Markdown Table detection: line has '|' and next line is a table separator
    if (trimmed.includes('|') && i + 1 < lines.length && isTableSeparator(lines[i + 1])) {
      const headers = parseTableRow(lines[i]);
      const sep = parseTableRow(lines[i + 1]);
      const alignments = sep.map(s => {
        if (s.startsWith(':') && s.endsWith(':')) return 'center';
        if (s.endsWith(':')) return 'right';
        return 'left';
      });
      i += 2;
      const rows = [];
      while (i < lines.length && lines[i].trim().includes('|') && lines[i].trim()) {
        rows.push(parseTableRow(lines[i]));
        i++;
      }
      blocks.push({ type: 'table', headers, alignments, rows });
      continue;
    }

    // Visual Mockup / Image Detection
    const mdImgMatch = trimmed.match(/!\[([^\]]*)\]\((https?:\/\/[^\s\)]+)\)/);
    const rawImgMatch = !mdImgMatch && trimmed.match(/(https?:\/\/[^\s"'<>]*(?:figma-alpha-api\.s3[^\s"'<>]+|\.(?:png|jpg|jpeg|webp|gif|svg)(?:\?[^\s"'<>]*)?))/i);
    if (mdImgMatch || rawImgMatch) {
      const imgUrl = mdImgMatch ? mdImgMatch[2] : rawImgMatch[1];
      const imgAlt = mdImgMatch ? mdImgMatch[1] : 'Figma Rendered Mockup';
      const cleanText = mdImgMatch ? trimmed.replace(mdImgMatch[0], '').trim() : trimmed.replace(rawImgMatch[0], '').trim();
      blocks.push({ type: 'image', imgUrl, imgAlt, cleanText });
      i++;
      continue;
    }

    // Headers
    if (trimmed.startsWith('### ')) {
      blocks.push({ type: 'h4', text: trimmed.slice(4) });
      i++;
      continue;
    }
    if (trimmed.startsWith('## ')) {
      blocks.push({ type: 'h3', text: trimmed.slice(3) });
      i++;
      continue;
    }
    if (trimmed.startsWith('# ')) {
      blocks.push({ type: 'h2', text: trimmed.slice(2) });
      i++;
      continue;
    }

    // Blockquote
    if (trimmed.startsWith('> ') || trimmed.startsWith('&gt; ')) {
      blocks.push({ type: 'quote', text: trimmed.replace(/^(&gt;|>)\s*/, '') });
      i++;
      continue;
    }

    // Bullets
    if (/^(\*|-|•)\s/.test(trimmed)) {
      blocks.push({ type: 'bullet', text: trimmed.replace(/^(\*|-|•)\s*/, '') });
      i++;
      continue;
    }

    // Numbered list
    if (/^\d+\.\s/.test(trimmed)) {
      const numMatch = trimmed.match(/^\d+\./);
      blocks.push({ type: 'numbered', num: numMatch ? numMatch[0] : '', text: trimmed.replace(/^\d+\.\s*/, '') });
      i++;
      continue;
    }

    // Divider
    if (trimmed === '---' || trimmed === '***') {
      blocks.push({ type: 'divider' });
      i++;
      continue;
    }

    // Regular paragraph line
    blocks.push({ type: 'paragraph', text: line });
    i++;
  }

  return blocks;
}

function MarkdownMessage({ content }) {
  if (!content) return null;
  const text = typeof content === 'string' ? content : JSON.stringify(content, null, 2);

  // Split into code blocks and normal markdown segments
  const parts = [];
  const codeBlockRegex = /```([a-zA-Z0-9_-]*)\n([\s\S]*?)```/g;
  let lastIndex = 0;
  let match;

  while ((match = codeBlockRegex.exec(text)) !== null) {
    if (match.index > lastIndex) {
      parts.push({ type: 'markdown', text: text.slice(lastIndex, match.index) });
    }
    parts.push({ type: 'code', lang: match[1] || 'text', code: match[2].trimEnd() });
    lastIndex = match.index + match[0].length;
  }
  if (lastIndex < text.length) {
    parts.push({ type: 'markdown', text: text.slice(lastIndex) });
  }

  return (
    <div className="markdown-rendered-body">
      {parts.map((part, pIdx) => {
        if (part.type === 'code') {
          return (
            <div key={pIdx} className="chat-code-block-container">
              <div className="chat-code-block-header">
                <span className="chat-code-lang">{part.lang ? part.lang.toUpperCase() : 'CODE'}</span>
                <button
                  type="button"
                  className="chat-code-copy-btn"
                  onClick={() => navigator.clipboard.writeText(part.code)}
                >
                  📋 Copy
                </button>
              </div>
              <pre className="chat-code-pre">
                <code>{part.code}</code>
              </pre>
            </div>
          );
        }

        // Render structured markdown blocks
        const blocks = parseMarkdownBlocks(part.text);
        return blocks.map((block, bIdx) => {
          if (block.type === 'break') {
            return <div key={bIdx} className="chat-line-break" />;
          }

          if (block.type === 'table') {
            return (
              <div key={bIdx} className="chat-table-wrapper">
                <table className="chat-markdown-table">
                  <thead>
                    <tr>
                      {block.headers.map((h, hIdx) => (
                        <th
                          key={hIdx}
                          style={{ textAlign: block.alignments[hIdx] || 'left' }}
                          dangerouslySetInnerHTML={{ __html: renderInline(h) }}
                        />
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {block.rows.map((row, rIdx) => (
                      <tr key={rIdx}>
                        {row.map((cell, cIdx) => (
                          <td
                            key={cIdx}
                            style={{ textAlign: block.alignments[cIdx] || 'left' }}
                            dangerouslySetInnerHTML={{ __html: renderInline(cell) }}
                          />
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            );
          }

          if (block.type === 'image') {
            return (
              <div key={bIdx} className="chat-image-block">
                {block.cleanText && (
                  <div className="chat-paragraph-line" dangerouslySetInnerHTML={{ __html: renderInline(block.cleanText) }} />
                )}
                <div className="chat-rendered-image-card">
                  <div className="chat-image-card-header">
                    <span className="chat-image-card-title">🖼️ {block.imgAlt || 'Rendered UI Mockup'}</span>
                    <a href={block.imgUrl} target="_blank" rel="noopener noreferrer" className="chat-image-open-btn" title="Open full-resolution image in new tab">
                      ↗ Open Full Size
                    </a>
                  </div>
                  <div className="chat-image-card-preview" onClick={() => window.open(block.imgUrl, '_blank')}>
                    <img src={block.imgUrl} alt={block.imgAlt || 'Mockup'} className="chat-mockup-img" loading="lazy" />
                    <div className="chat-image-zoom-overlay">🔍 Click to Open Full Resolution</div>
                  </div>
                </div>
              </div>
            );
          }

          if (block.type === 'h4') {
            return <h4 key={bIdx} className="chat-h4" dangerouslySetInnerHTML={{ __html: renderInline(block.text) }} />;
          }
          if (block.type === 'h3') {
            return <h3 key={bIdx} className="chat-h3" dangerouslySetInnerHTML={{ __html: renderInline(block.text) }} />;
          }
          if (block.type === 'h2') {
            return <h2 key={bIdx} className="chat-h2" dangerouslySetInnerHTML={{ __html: renderInline(block.text) }} />;
          }
          if (block.type === 'quote') {
            return <blockquote key={bIdx} className="chat-blockquote" dangerouslySetInnerHTML={{ __html: renderInline(block.text) }} />;
          }
          if (block.type === 'bullet') {
            return (
              <div key={bIdx} className="chat-bullet-row">
                <span className="chat-bullet-dot">•</span>
                <span className="chat-bullet-text" dangerouslySetInnerHTML={{ __html: renderInline(block.text) }} />
              </div>
            );
          }
          if (block.type === 'numbered') {
            return (
              <div key={bIdx} className="chat-numbered-row">
                <span className="chat-num-badge">{block.num}</span>
                <span className="chat-num-text" dangerouslySetInnerHTML={{ __html: renderInline(block.text) }} />
              </div>
            );
          }
          if (block.type === 'divider') {
            return <hr key={bIdx} className="chat-divider" />;
          }

          return <div key={bIdx} className="chat-paragraph-line" dangerouslySetInnerHTML={{ __html: renderInline(block.text) }} />;
        });
      })}
    </div>
  );
}

function ThoughtTrace({ steps, durationMs }) {
  const [isOpen, setIsOpen] = useState(false);
  const [expandedStepIdx, setExpandedStepIdx] = useState(null);

  if (!steps || steps.length === 0) return null;

  const totalSec = Math.max(1, Math.round((durationMs || 1000) / 1000));
  const durationStr = totalSec >= 60 ? `${Math.floor(totalSec / 60)}m ${totalSec % 60}s` : `${totalSec}s`;

  return (
    <div className="thought-trace-card">
      <button
        type="button"
        className="thought-trace-header-btn"
        onClick={() => setIsOpen(!isOpen)}
        title="Toggle execution steps"
      >
        <span className="thought-trace-summary">
          Worked for {durationStr}
        </span>
        <span className={`thought-trace-chevron ${isOpen ? 'open' : ''}`}>
          {isOpen ? '▴' : '▾'}
        </span>
      </button>

      {isOpen && (
        <div className="thought-trace-dropdown">
          {steps.map((st, sIdx) => {
            const isStepExpanded = expandedStepIdx === sIdx;
            const isTool = st.type === 'tool_call';
            return (
              <div key={sIdx} className="thought-step-item">
                <div
                  className="thought-step-header"
                  onClick={() => setExpandedStepIdx(isStepExpanded ? null : sIdx)}
                >
                  <span className="thought-step-icon">
                    {isTool ? '⚡' : st.type === 'search' ? '🔍' : st.type === 'file_read' ? '📄' : '🧠'}
                  </span>
                  <span className="thought-step-title">{st.title}</span>
                  {st.duration_ms !== undefined && (
                    <span className="thought-step-time">{(st.duration_ms / 1000).toFixed(1)}s</span>
                  )}
                  {(st.args || st.output_preview) && (
                    <span className="thought-step-expand-icon">{isStepExpanded ? '▴' : '▸'}</span>
                  )}
                </div>

                {isStepExpanded && (st.args || st.output_preview) && (
                  <div className="thought-step-details">
                    {st.args && (
                      <div className="thought-step-command">
                        <span className="thought-step-prompt-line">❯ {st.server || 'tool'}.{st.tool || 'execute'}</span>
                        <pre><code>{JSON.stringify(st.args, null, 2)}</code></pre>
                      </div>
                    )}
                    {st.output_preview && (
                      <div className="thought-step-output">
                        <span className="thought-step-output-label">Output:</span>
                        <pre><code>{st.output_preview}</code></pre>
                      </div>
                    )}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

function App() {
  // Projects state
  const [projects, setProjects] = useState([]);
  const [showAddProject, setShowAddProject] = useState(false);
  const [newProject, setNewProject] = useState({
    name: '',
    type: 'local',
    path: ''
  });

  // Chat state
  const [messages, setMessages] = useState([]);
  const [currentMessage, setCurrentMessage] = useState('');
  const [attachments, setAttachments] = useState([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [selectedProject, setSelectedProject] = useState(null);
  const [copiedMessageIdx, setCopiedMessageIdx] = useState(null);
  const [expandedTokens, setExpandedTokens] = useState({});
  const [thinkingTimer, setThinkingTimer] = useState(0);
  const [currentThinkingStep, setCurrentThinkingStep] = useState(null);
  const [streamingMessage, setStreamingMessage] = useState(null);

  useEffect(() => {
    let interval;
    if (isProcessing) {
      setThinkingTimer(0);
      interval = setInterval(() => {
        setThinkingTimer(t => t + 1);
      }, 1000);
    } else {
      setThinkingTimer(0);
      setCurrentThinkingStep(null);
      setStreamingMessage(null);
    }
    return () => clearInterval(interval);
  }, [isProcessing]);

  const handleCopyMessage = (text, idx) => {
    if (!text) return;
    navigator.clipboard.writeText(text);
    setCopiedMessageIdx(idx);
    setTimeout(() => {
      setCopiedMessageIdx(null);
    }, 2000);
  };

  // Chat & task history (persisted server-side, survives restarts)
  const [chats, setChats] = useState([]);
  const [tasks, setTasks] = useState([]);
  const [activeChatId, setActiveChatId] = useState(null);
  const activeChatIdRef = useRef(activeChatId);
  useEffect(() => { 
    if (activeChatId) {
      activeChatIdRef.current = activeChatId; 
    }
  }, [activeChatId]);
  const activeChatWorkflowIdRef = useRef(null);
  const [activeChatWorkflowId, setActiveChatWorkflowId] = useState(null); // workflow linked to current chat
  useEffect(() => { activeChatWorkflowIdRef.current = activeChatWorkflowId; }, [activeChatWorkflowId]);
  const [historyTab, setHistoryTab] = useState('chats'); // 'chats' | 'tasks'
  const saveTimer = useRef(null);
  const skipNextSave = useRef(false);
  const initialLoadResumed = useRef(false);
  const activeAbortControllerRef = useRef(null);
  const streamingContentRef = useRef('');
  const streamingModelRef = useRef('nebius/nvidia/nemotron-3-super-120b-a12b');
  
  // Transparent workflow state
  const [showWorkflow, setShowWorkflow] = useState(false);
  const [workflowData, setWorkflowData] = useState(null);
  const [activeWorkflowProgress, setActiveWorkflowProgress] = useState(null);
  const [modelMode, setModelMode] = useState('default');
  const [selectedModel, setSelectedModel] = useState('nebius/nvidia/nemotron-3-super-120b-a12b');
  const [availableModelsList, setAvailableModelsList] = useState([]);

  // Tab state: 'chat' | 'flow' | 'changes'
  const [activeTab, setActiveTab] = useState('chat');
  const [flowStepCount, setFlowStepCount] = useState(0);

  // Changes tab state (real diffs)
  const [workflowChanges, setWorkflowChanges] = useState(null);
  const [changesLoading, setChangesLoading] = useState(false);
  const [expandedDiffs, setExpandedDiffs] = useState({});
  const [mutationReviewStatus, setMutationReviewStatus] = useState({});

  const handleAcceptAllChanges = async (wfId) => {
    if (!workflowChanges || !workflowChanges.files) return;
    const targetWf = wfId || workflowChanges.workflow_id;
    const approved = workflowChanges.files.map(f => f.path);
    try {
      await fetch(`${API_BASE_URL}/api/workflow/transparent/${targetWf}/changes/approve`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ approved_files: approved, rejected_files: [] })
      });
      setMutationReviewStatus(prev => ({ ...prev, [targetWf]: 'applied' }));
      setWorkflowChanges(prev => prev ? {
        ...prev,
        files: prev.files.map(f => ({ ...f, approved: true }))
      } : null);
    } catch (err) {
      console.error('Error accepting all changes:', err);
    }
  };

  const handleDiscardAllChanges = async (wfId) => {
    if (!workflowChanges || !workflowChanges.files) return;
    const targetWf = wfId || workflowChanges.workflow_id;
    const rejected = workflowChanges.files.map(f => f.path);
    try {
      await fetch(`${API_BASE_URL}/api/workflow/transparent/${targetWf}/changes/approve`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ approved_files: [], rejected_files: rejected })
      });
      setMutationReviewStatus(prev => ({ ...prev, [targetWf]: 'discarded' }));
      setWorkflowChanges(prev => prev ? {
        ...prev,
        files: prev.files.map(f => ({ ...f, approved: false }))
      } : null);
    } catch (err) {
      console.error('Error discarding changes:', err);
    }
  };

  const handleToggleFileApproval = async (wfId, filePath, newApprovedState) => {
    if (!workflowChanges || !workflowChanges.files) return;
    const targetWf = wfId || workflowChanges.workflow_id;
    const updatedFiles = workflowChanges.files.map(f =>
      f.path === filePath ? { ...f, approved: newApprovedState } : f
    );
    const approvedList = updatedFiles.filter(f => f.approved).map(f => f.path);
    const rejectedList = updatedFiles.filter(f => !f.approved).map(f => f.path);
    try {
      await fetch(`${API_BASE_URL}/api/workflow/transparent/${targetWf}/changes/approve`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ approved_files: approvedList, rejected_files: rejectedList })
      });
      setWorkflowChanges(prev => prev ? { ...prev, files: updatedFiles } : null);
    } catch (err) {
      console.error('Error toggling file approval:', err);
    }
  };

  // Track if the current workflow is completed (for follow-up routing)
  const [workflowCompleted, setWorkflowCompleted] = useState(false);

  // Interactive Clarification Gate (HITL Modal) state
  const [clarificationModal, setClarificationModal] = useState(null);
  const [selectedOptions, setSelectedOptions] = useState([]);
  const [clarificationNotes, setClarificationNotes] = useState('');

  const handleClarificationSubmit = async (choices) => {
    if (!clarificationModal) return;
    try {
      await fetch(`${API_BASE_URL}/api/workflow/interactive/respond`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          request_id: clarificationModal.request_id,
          selected_options: choices,
          user_notes: clarificationNotes,
        }),
      });
    } catch (err) {
      console.error('Failed to submit clarification:', err);
    } finally {
      setClarificationModal(null);
      setSelectedOptions([]);
      setClarificationNotes('');
    }
  };

  // Theme state: 'dark' | 'light'
  const [theme, setTheme] = useState(() => localStorage.getItem('ticket-to-code-theme') || localStorage.getItem('aviator-theme') || 'dark');

  // Apply theme class to document
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('ticket-to-code-theme', theme);
  }, [theme]);

  // Load available models and default configuration
  useEffect(() => {
    fetch(`${API_BASE_URL}/api/models`)
      .then(r => r.json())
      .then(d => {
        if (d.models) setAvailableModelsList(d.models.filter(m => m.is_available));
        if (d.default_model) setSelectedModel(d.default_model);
      })
      .catch(err => console.error('Failed to load models:', err));
  }, []);

  // LLM Token & Credentials state
  const [showSettingsModal, setShowSettingsModal] = useState(false);
  const [submittedToken, setSubmittedToken] = useState('');
  const [customPlatformTag, setCustomPlatformTag] = useState('');
  const [settingsStatus, setSettingsStatus] = useState(null);
  const [settingsSaving, setSettingsSaving] = useState(false);
  const [verifyResult, setVerifyResult] = useState(null);
  const [isVerifying, setIsVerifying] = useState(false);
  const [showArchitectureModal, setShowArchitectureModal] = useState(false);
  const [selectedInspectorProvider, setSelectedInspectorProvider] = useState(null);
  const [allModelsAudit, setAllModelsAudit] = useState([]);
  const [inspectorSearch, setInspectorSearch] = useState('');
  const [inspectorFilter, setInspectorFilter] = useState('all');

  const fetchAuditModels = () => {
    fetch(`${API_BASE_URL}/api/models?include_all=true`)
      .then(r => r.json())
      .then(d => {
        if (d && d.models) setAllModelsAudit(d.models);
      })
      .catch(err => console.error('Failed to load audit models list:', err));
  };

  // Helper to auto-detect engine from token or custom tag without forced defaults
  const detectPlatformFromInput = (token, tag) => {
    const rawTag = (tag || '').trim();
    const t = rawTag.toLowerCase();
    if (t.includes('anthropic') || t.includes('claude')) return { id: 'anthropic', label: 'Anthropic Claude' };
    if (t.includes('openai') || t.includes('gpt')) return { id: 'openai', label: 'OpenAI GPT' };
    if (t.includes('google') || t.includes('gemini')) return { id: 'google', label: 'Google Gemini' };
    if (t.includes('nebius') || t.includes('nemotron')) return { id: 'nebius', label: 'Nebius Token Factory' };
    if (rawTag) {
      const cleanId = t.replace(/[^a-z0-9_]/g, '') || 'custom';
      return { id: cleanId, label: rawTag };
    }
    const raw = (token || '').trim();
    if (raw.startsWith('sk-ant-')) return { id: 'anthropic', label: 'Anthropic Claude' };
    if (raw.startsWith('sk-')) return { id: 'openai', label: 'OpenAI GPT' };
    if (raw.startsWith('AIza')) return { id: 'google', label: 'Google Gemini' };
    if (raw.startsWith('v1.')) return { id: 'nebius', label: 'Nebius Token Factory' };
    return { id: 'custom', label: 'Custom / Auto-Detect' };
  };

  const handleVerifyConnection = async () => {
    if (!submittedToken.trim()) return;
    setIsVerifying(true);
    setVerifyResult(null);
    try {
      const detected = detectPlatformFromInput(submittedToken, customPlatformTag);
      const res = await fetch(`${API_BASE_URL}/api/settings/verify-key`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          token: submittedToken.trim(),
          provider: detected.id
        })
      });
      const data = await res.json();
      if (res.ok) {
        setVerifyResult({ valid: true, message: data.message });
      } else {
        setVerifyResult({ valid: false, message: data.detail || 'Connection verification failed' });
      }
    } catch (err) {
      setVerifyResult({ valid: false, message: 'Could not connect to backend: ' + err.message });
    } finally {
      setIsVerifying(false);
    }
  };

  const [copiedKeyId, setCopiedKeyId] = useState(null);

  const handleSetActiveProvider = async (providerId) => {
    try {
      const res = await fetch(`${API_BASE_URL}/api/settings/keys/set-active`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ provider: providerId })
      });
      if (res.ok) {
        loadSettingsStatus();
      }
    } catch (err) {
      console.error('Failed to set active provider:', err);
    }
  };

  const handleDeleteKey = async (providerId) => {
    if (!window.confirm(`Are you sure you want to remove the ${providerId.toUpperCase()} API key?`)) return;
    try {
      const res = await fetch(`${API_BASE_URL}/api/settings/keys/${providerId}`, {
        method: 'DELETE'
      });
      if (res.ok) {
        loadSettingsStatus();
      }
    } catch (err) {
      console.error('Failed to delete key:', err);
    }
  };

  const handleCopyToken = (text, keyId) => {
    if (!text) return;
    navigator.clipboard.writeText(text);
    setCopiedKeyId(keyId);
    setTimeout(() => setCopiedKeyId(null), 2000);
  };

  const loadSettingsStatus = () => {
    fetch(`${API_BASE_URL}/api/settings/keys`)
      .then(r => r.json())
      .then(d => {
        setSettingsStatus(d);
      })
      .catch(err => console.error('Failed to load settings keys:', err));

    fetch(`${API_BASE_URL}/api/models`)
      .then(r => r.json())
      .then(d => {
        if (d && d.models && d.models.length > 0) {
          setAvailableModelsList(d.models.filter(m => m.is_available));
        }
      })
      .catch(err => console.error('Failed to load models list:', err));

    fetchAuditModels();
  };

  useEffect(() => {
    loadSettingsStatus();
  }, []);

  const handleSaveSettings = async () => {
    setSettingsSaving(true);
    try {
      const detected = detectPlatformFromInput(submittedToken, customPlatformTag);
      const payload = {
        token: submittedToken.trim(),
        provider: detected.id,
        provider_name: customPlatformTag.trim() || undefined,
      };
      const res = await fetch(`${API_BASE_URL}/api/settings/keys`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      if (res.ok) {
        const mres = await fetch(`${API_BASE_URL}/api/models`);
        const mdata = await mres.json();
        if (mdata.models) setAvailableModelsList(mdata.models);
        if (detected.id === 'nebius') {
          setSelectedModel('nebius/nvidia/nemotron-3-super-120b-a12b');
        } else if (mdata.default_model) {
          setSelectedModel(mdata.default_model);
        }
        loadSettingsStatus();
        setSubmittedToken('');
        setShowSettingsModal(false);
      } else {
        alert(data.detail || 'Failed to save settings');
      }
    } catch (err) {
      alert('Error saving settings: ' + err.message);
    } finally {
      setSettingsSaving(false);
    }
  };

  // Live indexing progress state
  const [indexingState, setIndexingState] = useState(null);
  // { projectId, projectName, done, total, percent, currentFile, log: [], status: 'running'|'done'|'error', stats: null }
  
  const messagesEndRef = useRef(null);
  const fileInputRef = useRef(null);
  const ws = useRef(null);

  // WebSocket connection with auto-reconnect
  useEffect(() => {
    let reconnectTimer = null;

    const connect = () => {
      const socket = new WebSocket(`${WS_BASE_URL}/ws`);
      ws.current = socket;

      socket.onopen = () => {
        console.log('WebSocket connected');
      };

      socket.onmessage = (event) => {
        const data = JSON.parse(event.data);
        console.log('WebSocket message:', data);

        if (data.type === 'indexing_progress') {
          setIndexingState(prev => {
            if (!prev) return prev;
            const newLog = prev.log.slice(-199); // keep last 200 entries
            newLog.push(data.current_file);
            return {
              ...prev,
              done: data.files_scanned,
              total: data.total_files,
              percent: data.progress_percent,
              currentFile: data.current_file,
              log: newLog,
            };
          });
        } else if (data.type === 'project_status') {
          if (data.status === 'indexed') {
            setProjects(prev => prev.map(p => p.id === data.project_id ? { ...p, status: 'indexed' } : p));
            setSelectedProject(prev => prev?.id === data.project_id ? { ...prev, status: 'indexed' } : prev);
            setIndexingState(prev => prev ? {
              ...prev,
              status: 'done',
              currentFile: '',
              stats: data.stats || null,
              message: data.message,
            } : prev);
          } else if (data.status === 'error') {
            setIndexingState(prev => prev ? { ...prev, status: 'error', message: data.message } : prev);
          }
          fetchProjects();
        } else if (data.type === 'workflow_summary') {
          // ── Workflow completed: show rich conversational summary in Chat ──
          setWorkflowCompleted(true);
          setActiveWorkflowProgress(null);
          setMessages(prev => [...prev, {
            role: 'assistant',
            content: data.summary || '✅ Workflow completed.',
            timestamp: new Date(),
            isWorkflowSummary: true,
            workflowId: data.workflow_id,
            generatedFiles: data.generated_files || [],
            workflowStatus: data.status,
            tokenUsage: data.token_usage || null,
          }]);
          // Keep user on their active tab
          setActiveTab(prev => (prev === 'flow' ? 'flow' : 'chat'));
          // Pre-load changes for the Changes tab (only if it matches the active chat)
          if (data.workflow_id && (!activeChatWorkflowIdRef.current || activeChatWorkflowIdRef.current === data.workflow_id)) {
            setChangesLoading(true);
            fetch(`${API_BASE_URL}/api/workflow/transparent/${data.workflow_id}/changes`)
              .then(r => r.ok ? r.json() : null)
              .then(d => { if (d) setWorkflowChanges({ ...d, workflow_id: d.workflow_id || data.workflow_id }); })
              .catch(() => {})
              .finally(() => setChangesLoading(false));
          }
        } else if (data.type === 'workflow_step') {
          // Track live pipeline progress for the Chat banner without cluttering the conversational stream
          setActiveWorkflowProgress({
            workflowId: data.workflow_id,
            phase: data.phase,
            message: data.message,
            status: data.status,
            node: data.data?.node,
            timestamp: new Date(),
          });
          if (data.workflow_id) {
            setWorkflowData(prev => {
              if (prev && prev.existingWorkflowId === data.workflow_id) return prev;
              return {
                projectId: selectedProject?.id,
                ticketId: data.workflow_id,
                ticketDescription: data.message || 'Live Ticket Run',
                repoPath: selectedProject?.path || 'workspace/CloudForge-Commerce',
                attachments: [],
                existingWorkflowId: data.workflow_id,
              };
            });
            setShowWorkflow(true);
            fetchHistory(selectedProject?.id);
          }
          if (data.workflow_id && (!activeChatWorkflowIdRef.current || activeChatWorkflowIdRef.current === data.workflow_id) && (data.phase === 'patch_generation' || data.phase === 'validation' || data.data?.node === 'generate_code')) {
            fetch(`${API_BASE_URL}/api/workflow/transparent/${data.workflow_id}/changes`)
              .then(r => r.ok ? r.json() : null)
              .then(d => { if (d && d.files && d.files.length > 0) setWorkflowChanges({ ...d, workflow_id: d.workflow_id || data.workflow_id }); })
              .catch(() => {});
          }
        } else if (data.event_type === 'clarification_requested' || data.type === 'clarification_requested') {
          setClarificationModal(data);
          if (data.recommended) {
            setSelectedOptions([data.recommended]);
          } else if (data.options && data.options.length > 0) {
            setSelectedOptions([data.options[0].label]);
          }
        }
      };

      socket.onerror = (error) => {
        console.error('WebSocket error:', error);
      };

      socket.onclose = () => {
        console.log('WebSocket closed, reconnecting in 3s...');
        reconnectTimer = setTimeout(connect, 3000);
      };
    };

    connect();

    return () => {
      clearTimeout(reconnectTimer);
      if (ws.current) ws.current.close();
    };
  }, []);

  // Scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Load projects
  useEffect(() => {
    fetchProjects();
  }, []);

  const fetchProjects = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/projects`);
      const data = await response.json();
      const list = Array.isArray(data) ? data : (data.projects || []);
      setProjects(list);
      if (list.length > 0) {
        setSelectedProject(prev => prev || list[0]);
      }
    } catch (error) {
      console.error('Error fetching projects:', error);
    }
  };

  // ── Chat & Task history ──────────────────────────────────────────────────
  const fetchHistory = async (projectId) => {
    try {
      const q = projectId ? `?project_id=${encodeURIComponent(projectId)}` : '';
      const [chatsRes, tasksRes] = await Promise.all([
        fetch(`${API_BASE_URL}/api/history/chats${q}`),
        fetch(`${API_BASE_URL}/api/history/tasks${q}`),
      ]);
      setChats(chatsRes.ok ? await chatsRes.json() : []);
      setTasks(tasksRes.ok ? await tasksRes.json() : []);
    } catch (error) {
      console.error('Error fetching history:', error);
    }
  };

  // Reload history whenever the selected project changes.
  useEffect(() => {
    fetchHistory(selectedProject?.id);
  }, [selectedProject]);

  // While a workflow runs, poll so completed tasks appear in history promptly.
  useEffect(() => {
    if (!showWorkflow) return;
    const t = setInterval(() => fetchHistory(selectedProject?.id), 5000);
    return () => clearInterval(t);
  }, [showWorkflow, selectedProject]);

  // Auto-resume the latest active task on initial page load only if it is actually running
  useEffect(() => {
    if (!initialLoadResumed.current && messages.length === 0 && !showWorkflow && tasks.length > 0 && selectedProject) {
      initialLoadResumed.current = true;
      const latestTask = [...tasks].sort((a, b) => new Date(b.created_at || 0) - new Date(a.created_at || 0))[0];
      if (latestTask && latestTask.status === 'running') {
        setActiveChatWorkflowId(latestTask.workflow_id);
        setWorkflowData({
          projectId: selectedProject.id,
          ticketId: latestTask.ticket_id || latestTask.title || 'Recovered Task',
          ticketDescription: latestTask.description || latestTask.title || '',
          repoPath: selectedProject.path,
          attachments: [],
          existingWorkflowId: latestTask.workflow_id,
        });
        setShowWorkflow(true);
        setActiveTab('flow');
      }
    }
  }, [tasks, messages.length, showWorkflow, selectedProject]);

  // Only serializable message fields are persisted (drop File/blob objects).
  const serializeMessages = (msgs) => msgs
    .filter(m => !m.isThinking && (m.content || '').trim() !== '🤔 Thinking...' && (m.content || '').trim() !== 'Thinking...')
    .map(m => ({
      role: m.role,
      content: m.content,
      timestamp: (m.timestamp instanceof Date ? m.timestamp : new Date(m.timestamp)).toISOString(),
      isEnhancement: m.isEnhancement || false,
      isError: !!m.isError,
      errorType: m.errorType || null,
      errorTitle: m.errorTitle || null,
      errorMessage: m.errorMessage || null,
      failedMessage: m.failedMessage || null,
      modelUsed: m.modelUsed || null,
      modelMode: m.modelMode || null,
      execution_steps: m.execution_steps || [],
      usage: m.usage || null,
      duration_ms: m.duration_ms || null,
      attachments: (m.attachments || []).map(a => ({
        name: a.name, filepath: a.filepath || null, isImage: !!a.isImage,
      })),
    }));

  const saveCurrentChat = async (msgs, chatId, workflowId) => {
    if (!msgs || msgs.length === 0) return chatId;
    let targetChatId = chatId || activeChatIdRef.current;
    if (!targetChatId) {
      targetChatId = 'chat_' + Math.random().toString(36).substring(2, 14);
      activeChatIdRef.current = targetChatId;
      setActiveChatId(targetChatId);
    }
    try {
      const res = await fetch(`${API_BASE_URL}/api/history/chats`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          chat_id: targetChatId,
          project_id: selectedProject?.id || null,
          messages: serializeMessages(msgs),
          workflow_id: workflowId || null,
        }),
      });
      if (res.ok) {
        const data = await res.json();
        if (data.id && data.id !== activeChatIdRef.current) {
          activeChatIdRef.current = data.id;
          setActiveChatId(data.id);
        }
        fetchHistory(selectedProject?.id);
        return data.id;
      }
    } catch (error) {
      console.error('Error saving chat:', error);
    }
    return targetChatId;
  };

  // Debounced auto-save whenever messages change (like AI IDE chat history).
  useEffect(() => {
    if (skipNextSave.current) { skipNextSave.current = false; return; }
    if (!messages || messages.length === 0) return;
    if (saveTimer.current) clearTimeout(saveTimer.current);
    saveTimer.current = setTimeout(() => { saveCurrentChat(messages, activeChatIdRef.current, activeChatWorkflowIdRef.current); }, 900);
    return () => { if (saveTimer.current) clearTimeout(saveTimer.current); };
  }, [messages]);

  const openChat = async (chatId) => {
    try {
      const res = await fetch(`${API_BASE_URL}/api/history/chats/${chatId}`);
      if (!res.ok) return;
      const chat = await res.json();
      skipNextSave.current = true; // loading isn't an edit
      const rawMsgs = chat.messages || [];
      const parsedMsgs = [];
      for (let i = 0; i < rawMsgs.length; i++) {
        const m = rawMsgs[i];
        const txt = (m.content || '').trim();
        if (txt === '🤔 Thinking...' || txt === 'Thinking...' || m.isThinking) {
          // Stale / hung thinking message from an interrupted session - convert to retryable error card!
          const prevUser = rawMsgs.slice(0, i).reverse().find(x => x.role === 'user');
          parsedMsgs.push({
            role: 'assistant',
            isError: true,
            errorType: 'timeout',
            errorTitle: '⏱️ Gateway Response Timeout (Interrupted)',
            errorMessage: 'This response was interrupted or timed out before completion. Click Retry below to resume generation.',
            failedMessage: prevUser ? prevUser.content : '',
            modelUsed: m.modelUsed || selectedModel || 'gemini-2.5-pro',
            timestamp: m.timestamp ? new Date(m.timestamp) : new Date(),
          });
        } else {
          parsedMsgs.push({
            ...m,
            timestamp: m.timestamp ? new Date(m.timestamp) : new Date(),
          });
        }
      }
      setMessages(parsedMsgs);
      setActiveChatId(chat.id);
      setActiveChatWorkflowId(chat.workflow_id || null);
      setActiveWorkflowProgress(null); // Reset transient pipeline progress from other chats

      // Explicitly isolate Changes tab to this chat's workflow run
      setWorkflowChanges(null);
      if (chat.workflow_id) {
        fetchWorkflowChanges(chat.workflow_id);
      }

      if (chat.workflow_id && selectedProject) {
        setWorkflowData({
          projectId: selectedProject.id,
          ticketId: chat.title,
          ticketDescription: chat.title,
          repoPath: selectedProject.path,
          attachments: [],
          existingWorkflowId: chat.workflow_id,
        });
        setShowWorkflow(true);
      } else {
        setShowWorkflow(false);
      }
      setIndexingState(null);
    } catch (error) {
      console.error('Error opening chat:', error);
    }
  };

  const startNewChat = () => {
    skipNextSave.current = true;
    if (saveTimer.current) {
      clearTimeout(saveTimer.current);
      saveTimer.current = null;
    }
    activeChatIdRef.current = null;
    activeChatWorkflowIdRef.current = null;
    setMessages([]);
    setActiveChatId(null);
    setActiveChatWorkflowId(null);
    setShowWorkflow(false);
    setWorkflowData(null);
    setActiveWorkflowProgress(null); // Ensure lingering pipeline progress/banners are cleared
    setActiveTab('chat');
    setIndexingState(null);
    setWorkflowCompleted(false);
    setWorkflowChanges(null);
    setExpandedDiffs({});
    setCurrentThinkingStep(null);
  };

  // Open the workflow view linked to a chat's run (View Run button)
  const viewChatRun = (chat) => {
    if (!chat.workflow_id || !selectedProject) return;
    setWorkflowChanges(null);
    fetchWorkflowChanges(chat.workflow_id);
    setWorkflowData({
      projectId: selectedProject.id,
      ticketId: chat.title,
      ticketDescription: chat.title,
      repoPath: selectedProject.path,
      attachments: [],
      existingWorkflowId: chat.workflow_id,
    });
    setShowWorkflow(true);
    setActiveTab('flow');
  };

  const deleteChat = async (chatId, event) => {
    if (event) {
      event.stopPropagation();
      event.preventDefault();
    }
    if (saveTimer.current) {
      clearTimeout(saveTimer.current);
      saveTimer.current = null;
    }
    // Optimistic UI removal so item disappears instantly
    setChats(prev => prev.filter(c => c.id !== chatId));
    if (activeChatId === chatId || activeChatIdRef.current === chatId) {
      startNewChat();
    }
    try {
      await fetch(`${API_BASE_URL}/api/history/chats/${chatId}`, { method: 'DELETE' });
    } catch (error) {
      console.error('Error deleting chat:', error);
    } finally {
      fetchHistory(selectedProject?.id);
    }
  };

  const deleteTask = async (workflowId, event) => {
    if (event) {
      event.stopPropagation();
      event.preventDefault();
    }
    setTasks(prev => prev.filter(t => t.workflow_id !== workflowId));
    if (activeChatWorkflowId === workflowId || activeChatWorkflowIdRef.current === workflowId) {
      setActiveChatWorkflowId(null);
      activeChatWorkflowIdRef.current = null;
      if (showWorkflow) {
        setShowWorkflow(false);
      }
    }
    try {
      await fetch(`${API_BASE_URL}/api/history/tasks/${workflowId}`, { method: 'DELETE' });
    } catch (error) {
      console.error('Error deleting task:', error);
    } finally {
      fetchHistory(selectedProject?.id);
    }
  };

  const handleAddProject = async () => {
    if (!newProject.name || !newProject.path) {
      alert('Please fill in all fields');
      return;
    }

    try {
      const response = await fetch(`${API_BASE_URL}/api/projects`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: newProject.name.trim(),
          type: newProject.type,
          path: newProject.path.trim().replace(/^["']|["']$/g, '')
        })
      });

      if (response.ok) {
        setNewProject({ name: '', type: 'local', path: '' });
        setShowAddProject(false);
        fetchProjects();
      } else {
        const error = await response.json();
        alert(`Error: ${error.detail || 'Failed to add project'}`);
      }
    } catch (error) {
      console.error('Error adding project:', error);
      alert('Error adding project');
    }
  };

  const handleDeleteProject = async (projectId, event) => {
    event.stopPropagation();
    if (!window.confirm('Delete this project?')) return;
    try {
      const response = await fetch(`${API_BASE_URL}/api/projects/${projectId}`, {
        method: 'DELETE'
      });
      if (response.ok) {
        if (selectedProject?.id === projectId) setSelectedProject(null);
        if (indexingState?.projectId === projectId) setIndexingState(null);
        fetchProjects();
      } else {
        const error = await response.json();
        alert(`Error: ${error.detail || 'Failed to delete project'}`);
      }
    } catch (error) {
      console.error('Error deleting project:', error);
      alert('Error deleting project');
    }
  };

  const handleIndexProject = async (projectId, event) => {
    event.stopPropagation();
    const project = projects.find(p => p.id === projectId);

    // Show live panel immediately
    setIndexingState({
      projectId,
      projectName: project?.name || projectId,
      projectPath: project?.path || '',
      done: 0,
      total: 0,
      percent: 0,
      currentFile: 'Starting...',
      log: [],
      status: 'running',
      stats: null,
      message: '',
    });
    setShowWorkflow(false); // switch to chat/indexing view

    try {
      const response = await fetch(`${API_BASE_URL}/api/projects/${projectId}/index`, {
        method: 'POST'
      });
      if (response.ok) {
        const resData = await response.json();
        setProjects(prev => prev.map(p => p.id === projectId ? { ...p, status: 'indexed' } : p));
        setSelectedProject(prev => prev?.id === projectId ? { ...prev, status: 'indexed' } : prev);
        await fetchProjects();
        setIndexingState(prev => prev ? {
          ...prev,
          status: 'done',
          currentFile: '',
          done: resData.stats?.files_scanned || resData.stats?.files_parsed || 23,
          total: resData.stats?.files_scanned || resData.stats?.files_parsed || 23,
          percent: 100,
          stats: resData.stats || null,
          message: '✅ Indexing complete!'
        } : prev);
      } else {
        const error = await response.json();
        setIndexingState(prev => prev ? { ...prev, status: 'error', message: error.detail || 'Indexing failed' } : prev);
      }
    } catch (error) {
      console.error('Error indexing project:', error);
      setIndexingState(prev => prev ? { ...prev, status: 'error', message: error.message } : prev);
    }
  };

  // Helper to categorize and sequentially number attachments by type
  const getAttachmentMeta = (att, index, allAttachments) => {
    const isImg = att.isImage || (att.name && /\.(png|jpe?g|gif|webp|bmp|svg)$/i.test(att.name));
    const isLog = att.name && /\.(log|txt|out)$/i.test(att.name) && (
      att.name.toLowerCase().includes('log') || 
      att.name.toLowerCase().includes('err') || 
      att.name.toLowerCase().includes('trace') ||
      att.name.toLowerCase().includes('console')
    );
    const category = isImg ? 'image' : isLog ? 'log' : 'file';

    let count = 0;
    for (let i = 0; i <= index; i++) {
      const item = allAttachments[i];
      if (!item) continue;
      const itemIsImg = item.isImage || (item.name && /\.(png|jpe?g|gif|webp|bmp|svg)$/i.test(item.name));
      const itemIsLog = item.name && /\.(log|txt|out)$/i.test(item.name) && (
        item.name.toLowerCase().includes('log') || 
        item.name.toLowerCase().includes('err') || 
        item.name.toLowerCase().includes('trace') ||
        item.name.toLowerCase().includes('console')
      );
      const itemCat = itemIsImg ? 'image' : itemIsLog ? 'log' : 'file';
      if (itemCat === category) {
        count++;
      }
    }

    const label = category === 'image' ? `Image ${count}` : category === 'log' ? `Log ${count}` : `File ${count}`;
    return { category, count, label };
  };

  // Upload helper
  const uploadAttachment = async (file) => {
    const formData = new FormData();
    formData.append('file', file);
    try {
      const res = await fetch(`${API_BASE_URL}/api/upload`, {
        method: 'POST',
        body: formData,
      });
      if (res.ok) {
        const data = await res.json();
        return data.filepath;
      } else {
        console.error('Failed to upload file');
        return null;
      }
    } catch (err) {
      console.error('Upload error:', err);
      return null;
    }
  };

  const handleFileSelect = async (e) => {
    const files = Array.from(e.target.files || []);
    if (!files.length) return;
    for (const file of files) {
      const isImage = file.type.startsWith('image/');
      const previewUrl = isImage ? URL.createObjectURL(file) : null;
      const newAtt = {
        file,
        name: file.name,
        previewUrl,
        isImage,
        filepath: null,
        isUploading: true,
      };
      setAttachments(prev => [...prev, newAtt]);
      const serverPath = await uploadAttachment(file);
      setAttachments(prev => prev.map(a => a.file === file ? { ...a, filepath: serverPath, isUploading: false } : a));
    }
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const handlePaste = async (e) => {
    const items = e.clipboardData?.items;
    if (!items) return;
    for (let i = 0; i < items.length; i++) {
      if (items[i].type.indexOf('image') !== -1) {
        const file = items[i].getAsFile();
        if (file) {
          e.preventDefault();
          const previewUrl = URL.createObjectURL(file);
          const fileName = `screenshot_${Date.now()}.png`;
          const namedFile = new File([file], fileName, { type: file.type });
          const newAtt = {
            file: namedFile,
            name: fileName,
            previewUrl,
            isImage: true,
            filepath: null,
            isUploading: true,
          };
          setAttachments(prev => [...prev, newAtt]);
          const serverPath = await uploadAttachment(namedFile);
          setAttachments(prev => prev.map(a => a.file === namedFile ? { ...a, filepath: serverPath, isUploading: false } : a));
        }
      }
    }
  };

  const removeAttachment = (indexToRemove) => {
    setAttachments(prev => prev.filter((_, idx) => idx !== indexToRemove));
  };

  // Fetch changes/diffs for the Changes tab
  const fetchWorkflowChanges = async (wfId) => {
    if (!wfId) {
      setWorkflowChanges(null);
      return;
    }
    setChangesLoading(true);
    try {
      const res = await fetch(`${API_BASE_URL}/api/workflow/transparent/${wfId}/changes`);
      if (res.ok) {
        const data = await res.json();
        setWorkflowChanges({
          ...data,
          workflow_id: data.workflow_id || wfId,
        });
      } else {
        setWorkflowChanges(null);
      }
    } catch (err) {
      console.error('Error fetching changes:', err);
      setWorkflowChanges(null);
    } finally {
      setChangesLoading(false);
    }
  };

  const cancelActiveChat = () => {
    if (activeAbortControllerRef.current) {
      try { activeAbortControllerRef.current.abort('User cancelled'); } catch (_) {}
      activeAbortControllerRef.current = null;
    }
  };

  // Production SSE streaming chat reader using ReadableStream over single HTTP POST
  const streamChatTurn = async (userMessage, targetWorkflowId = null, overrideModel = null, isFollowUp = false) => {
    setMessages(prev => [...prev, {
      role: 'assistant',
      content: '🤔 Thinking...',
      timestamp: new Date(),
      isThinking: true,
    }]);

    if (activeAbortControllerRef.current) {
      try { activeAbortControllerRef.current.abort(); } catch (_) {}
    }
    const controller = new AbortController();
    activeAbortControllerRef.current = controller;
    streamingContentRef.current = '';
    const targetModel = overrideModel || selectedModel;
    streamingModelRef.current = targetModel;
    setCurrentThinkingStep(null);

    const clientMessageId = `msg_${Date.now()}_${Math.random().toString(36).slice(2, 9)}`;
    let finalData = null;
    let receivedDone = false;
    let collectedSteps = [];
    let collectedUsage = null;

    try {
      const res = await fetch(`${API_BASE_URL}/api/workflow/chat/stream`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'text/event-stream',
        },
        signal: controller.signal,
        body: JSON.stringify({
          workflow_id: targetWorkflowId || null,
          message: userMessage,
          project_id: selectedProject?.id || null,
          chat_id: activeChatId,
          messages: messages.slice(-10).map(m => ({ role: m.role, content: m.content })),
          model_mode: modelMode,
          selected_model: targetModel,
          client_message_id: clientMessageId,
        }),
      });

      if (!res.ok) {
        throw new Error(`HTTP ${res.status}: Backend service returned error`);
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder('utf-8');
      let buffer = '';

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const parts = buffer.split('\n\n');
        buffer = parts.pop() || '';

        for (const part of parts) {
          if (!part.trim()) continue;
          if (part.startsWith(':')) continue; // SSE comment line / keepalive

          let eventType = 'message';
          let eventDataRaw = '';

          const lines = part.split('\n');
          for (const line of lines) {
            if (line.startsWith('event:')) {
              eventType = line.slice(6).trim();
            } else if (line.startsWith('data:')) {
              const dataLine = line.slice(5).trim();
              eventDataRaw = eventDataRaw ? `${eventDataRaw}\n${dataLine}` : dataLine;
            }
          }

          let eventData = {};
          try {
            eventData = JSON.parse(eventDataRaw);
          } catch (_) {
            eventData = { raw: eventDataRaw };
          }

          if (eventType === 'message_start') {
            if (eventData.model) {
              streamingModelRef.current = eventData.model;
            }
          } else if (eventType === 'step') {
            collectedSteps.push(eventData);
            setCurrentThinkingStep(eventData);
            // While running agent tool steps, keep the main bubble clean for final synthesis
            if (streamingContentRef.current) {
              streamingContentRef.current = '';
              setStreamingMessage(null);
            }
          } else if (eventType === 'clear_stream') {
            streamingContentRef.current = '';
            setStreamingMessage(null);
          } else if (eventType === 'text_delta') {
            const delta = eventData.delta || '';
            streamingContentRef.current = (streamingContentRef.current || '') + delta;
            setStreamingMessage({
              role: 'assistant',
              content: streamingContentRef.current,
              isStreaming: true,
              modelUsed: streamingModelRef.current,
              timestamp: new Date(),
            });
          } else if (eventType === 'usage') {
            collectedUsage = eventData;
          } else if (eventType === 'done') {
            receivedDone = true;
            finalData = eventData;
          } else if (eventType === 'error') {
            throw new Error(eventData.message || 'Model stream generation error');
          }
        }
      }

      // Stream read completed cleanly
      const accumulatedText = streamingContentRef.current ? streamingContentRef.current.trim() : '';
      setStreamingMessage(null);
      streamingContentRef.current = '';
      setMessages(prev => prev.filter(m => !m.isThinking));

      if (receivedDone) {
        const finalReply = finalData?.reply || accumulatedText || 'No response generated.';
        setMessages(prev => [...prev, {
          role: 'assistant',
          content: finalReply,
          timestamp: new Date(),
          isFollowUp: isFollowUp,
          execution_steps: finalData?.execution_steps || collectedSteps,
          usage: finalData?.usage || collectedUsage,
          duration_ms: finalData?.duration_ms || null,
          modelUsed: finalData?.model_used || streamingModelRef.current,
          modelMode: finalData?.model_mode || modelMode,
          // SUCCESSFUL COMPLETION: No isError, No isInterrupted -> Never renders Retry button
        }]);
      } else {
        // Stream closed unexpectedly before 'done' event (transport disconnect)
        if (accumulatedText) {
          setMessages(prev => [...prev, {
            role: 'assistant',
            content: accumulatedText,
            isInterrupted: true,
            errorTitle: '⚠️ Connection Interrupted',
            errorMessage: 'Stream connection closed before completion. Click Retry below to resume.',
            failedMessage: userMessage,
            modelUsed: streamingModelRef.current,
            timestamp: new Date(),
          }]);
        } else {
          setMessages(prev => [...prev, {
            role: 'assistant',
            isError: true,
            errorType: 'network',
            errorTitle: '❌ Connection Interrupted',
            errorMessage: 'Stream closed unexpectedly before any response was received.',
            failedMessage: userMessage,
            modelUsed: streamingModelRef.current,
            timestamp: new Date(),
          }]);
        }
      }
    } catch (err) {
      const isUserCancel = err.name === 'AbortError' || err.message?.includes('User cancelled');
      const accumulatedText = streamingContentRef.current ? streamingContentRef.current.trim() : '';
      const usedModel = streamingModelRef.current || targetModel;

      setStreamingMessage(null);
      streamingContentRef.current = '';
      setMessages(prev => prev.filter(m => !m.isThinking));

      if (isUserCancel) {
        if (accumulatedText) {
          setMessages(prev => [...prev, {
            role: 'assistant',
            content: accumulatedText,
            isInterrupted: true,
            errorMessage: '⏹️ Generation stopped by user.',
            failedMessage: userMessage,
            modelUsed: usedModel,
            timestamp: new Date(),
          }]);
        } else {
          setMessages(prev => [...prev, {
            role: 'assistant',
            content: '⏹️ Generation stopped by user.',
            timestamp: new Date(),
          }]);
        }
      } else {
        if (accumulatedText) {
          setMessages(prev => [...prev, {
            role: 'assistant',
            content: accumulatedText,
            isInterrupted: true,
            errorTitle: '⚠️ Generation Interrupted',
            errorMessage: err.message || 'Stream connection was interrupted.',
            failedMessage: userMessage,
            modelUsed: usedModel,
            timestamp: new Date(),
          }]);
        } else {
          setMessages(prev => [...prev, {
            role: 'assistant',
            isError: true,
            errorType: 'stream_error',
            errorTitle: '⚠️ Request Interrupted',
            errorMessage: err.message || 'Server error encountered during request.',
            failedMessage: userMessage,
            modelUsed: usedModel,
            timestamp: new Date(),
          }]);
        }
      }
    } finally {
      activeAbortControllerRef.current = null;
      setIsProcessing(false);
      setCurrentThinkingStep(null);
    }
  };

  const handleRetry = (failedMsg) => {
    if (!failedMsg) return;
    // Clear the error or interrupted card from the chat
    setMessages(prev => prev.filter(m => !(m.failedMessage === failedMsg && (m.isError || m.isInterrupted))));
    handleSendMessage(failedMsg, selectedModel, true);
  };

  const handleSendMessage = async (overrideText = null, overrideModel = null, isRetry = false) => {
    const isOverrideString = typeof overrideText === 'string';
    const rawMsg = isOverrideString ? overrideText : (currentMessage || '');
    if ((!rawMsg.trim() && attachments.length === 0) || isProcessing) return;

    const userMessage = rawMsg.trim() || (attachments.length > 0 ? "Analyze attached screenshot/file and provide solution" : "");
    const attachedPaths = attachments.map(a => a.filepath).filter(Boolean);
    const currentAtts = [...attachments];

    if (!isOverrideString) {
      setCurrentMessage('');
      setAttachments([]);
    }
    
    // Ensure an active chat ID exists synchronously before triggering async workflows or timers
    if (!activeChatIdRef.current) {
      const newChatId = 'chat_' + Math.random().toString(36).substring(2, 14);
      activeChatIdRef.current = newChatId;
      setActiveChatId(newChatId);
    }

    // Add user message to chat only if this is not a retry of an existing message
    if (!isRetry) {
      setMessages(prev => [...prev, {
        role: 'user',
        content: userMessage,
        attachments: currentAtts,
        timestamp: new Date()
      }]);
    }

    setIsProcessing(true);

    try {
      // ── Resume slash command: /resume [optional user guidance] ──
      if (userMessage.startsWith('/resume')) {
        const feedback = userMessage.replace('/resume', '').trim() || 'User resumed workflow from chat';
        const targetWfId = activeChatWorkflowId || workflowData?.existingWorkflowId;
        if (targetWfId) {
          try {
            await fetch(`${API_BASE_URL}/api/workflow/transparent/${targetWfId}/resume`, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({
                workflow_id: targetWfId,
                resume_from: 'plan',
                feedback: feedback,
              }),
            });
            setWorkflowCompleted(false);
            setWorkflowData(prev => prev ? { ...prev, isStopped: false } : prev);
            setMessages(prev => [...prev, {
              role: 'system',
              content: `🔄 Resuming workflow from 'plan' checkpoint with guidance: "${feedback}"`,
              timestamp: new Date(),
              workflowId: targetWfId,
            }]);
            setActiveTab('flow');
          } catch (err) {
            setMessages(prev => [...prev, {
              role: 'assistant',
              content: `❌ Failed to resume workflow: ${err.message}`,
              timestamp: new Date(),
            }]);
          }
          setIsProcessing(false);
          return;
        }
      }

      // ── Follow-up question about a completed workflow ──
      // If a workflow is completed and user sends a message, route to the
      // conversational chat endpoint (LLM answers using workflow context)
      if (workflowCompleted && activeChatWorkflowId) {
        await streamChatTurn(userMessage, activeChatWorkflowId, overrideModel, true);
        return;
      }

      // ── Intelligent Intent Routing: LLM Semantic Classifier (Zero Hardcoding) ──
      // Calls /api/intent/classify to let Gemini 2.5 Flash determine if this requires
      // CODE_CHANGE (autonomous patch generation) or EXPLAIN (conversational answer).
      // Check for explicit user overrides that inhibit autonomous workflow
      const rawMsgLower = userMessage.toLowerCase();
      const hasExplicitExplainDirective = [
        "explain only", "read only", "do not trigger", "don't trigger",
        "no code modification", "do not modify", "just analyze", "just explain", "in chat"
      ].some(cue => rawMsgLower.includes(cue));

      let isCodeChangeTask = false;
      if (!hasExplicitExplainDirective) {
        try {
          const classifyRes = await fetch(`${API_BASE_URL}/api/intent/classify`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              message: userMessage,
              attachments: attachedPaths,
              current_tab: activeTab,
              project_id: selectedProject?.id || null,
            }),
          });
          if (classifyRes.ok) {
            const classifyData = await classifyRes.json();
            isCodeChangeTask = classifyData.intent === 'CODE_CHANGE' && classifyData.requires_code_changes;
          }
        } catch (classifyErr) {
          console.warn('Intent classification fallback:', classifyErr);
        }
      }

      const isTicketFile = selectedProject && isCodeChangeTask;

      if (isTicketFile && selectedProject) {
        // Reset workflow completion state for new runs
        setWorkflowCompleted(false);
        setWorkflowChanges(null);

        // Start transparent workflow
        setMessages(prev => [...prev, {
          role: 'assistant',
          content: `🚀 **Starting autonomous workflow for:** ${userMessage}${attachedPaths.length ? `\n📸 Attached ${attachedPaths.length} file(s) for visual / runtime analysis.` : ''}\n\nAnalyzing repository structure, AST symbols, and planning code modifications.\n\n👉 *Switch to the **Flow** tab anytime to inspect live AST exploration and decisions.*`,
          timestamp: new Date(),
          isWorkflowBanner: true,
        }]);

        // Show transparent workflow UI
        setWorkflowData({
          projectId: selectedProject.id,
          ticketId: userMessage,   // full description; backend auto-generates ID
          ticketDescription: userMessage,
          repoPath: selectedProject.path,
          attachments: attachedPaths,
          writableFiles: [],
          forbiddenFiles: [],
          existingWorkflowId: null, // new run; onWorkflowStarted will fill this in
          modelMode: modelMode,
          selectedModel: selectedModel,
        });
        setShowWorkflow(true);
        setActiveTab('flow');

      } else if (selectedProject) {
        // Conversational query about project, architecture, past tickets, or general questions
        await streamChatTurn(userMessage, activeChatWorkflowId || null, overrideModel, false);
      } else {
        // Regular chat message when no project is selected
        setMessages(prev => [...prev, {
          role: 'assistant',
          content: `I received: "${userMessage}"\n\n⚠️ Please select a project first to query code, investigate tickets, or view architecture.`,
          timestamp: new Date()
        }]);
      }
    } catch (error) {
      console.error('Error:', error);
      setMessages(prev => [...prev, {
        role: 'system',
        content: `❌ Error: ${error.message}`,
        timestamp: new Date()
      }]);
    } finally {
      setIsProcessing(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  // Collect workflow IDs that are already linked to a chat
  const chatWorkflowIds = new Set(chats.map(c => c.workflow_id).filter(Boolean));
  // Filter out tasks that are already represented by a chat
  const standaloneTasks = tasks
    .filter(t => !chatWorkflowIds.has(t.workflow_id))
    .map(t => ({...t, isTask: true, id: t.workflow_id}));

  const combinedHistory = [...chats, ...standaloneTasks]
    .sort((a, b) => new Date(b.updated_at || b.created_at || 0) - new Date(a.updated_at || a.created_at || 0));

  return (
    <div className="app">
      {/* Left Panel - Projects */}
      <div className="left-panel">
        <div className="panel-header">
          <h2>📁 Projects</h2>
          <button 
            className="add-btn"
            onClick={() => setShowAddProject(!showAddProject)}
          >
            {showAddProject ? '✕' : '+'}
          </button>
        </div>

        {showAddProject && (
          <div className="add-project-form">
            <input
              type="text"
              placeholder="Project name"
              value={newProject.name}
              onChange={(e) => setNewProject({ ...newProject, name: e.target.value })}
            />
            <select
              value={newProject.type}
              onChange={(e) => setNewProject({ ...newProject, type: e.target.value })}
            >
              <option value="local">Local</option>
              <option value="git">Git</option>
            </select>
            <input
              type="text"
              placeholder={newProject.type === 'local' ? 'Local path (C:\\...)' : 'Git URL'}
              value={newProject.path}
              onChange={(e) => setNewProject({ ...newProject, path: e.target.value })}
            />
            <button onClick={handleAddProject}>Add Project</button>
          </div>
        )}

        <div className="projects-list">
          {projects.length === 0 ? (
            <div className="empty-state">
              <p>No projects yet</p>
              <p className="hint">Click + to add one</p>
            </div>
          ) : (
            projects.map(project => (
              <div
                key={project.id}
                className={`project-item ${selectedProject?.id === project.id ? 'selected' : ''}`}
                onClick={() => setSelectedProject(project)}
              >
                <div className="project-icon">
                  {project.type === 'local' ? '📂' : '🔗'}
                </div>
                <div className="project-info">
                  <div className="project-name">{project.name}</div>
                  <div className="project-path">{project.path}</div>
                  <div className="project-status">
                    {project.status === 'indexed' && '✅ Indexed'}
                    {project.status === 'indexing' && '⏳ Indexing...'}
                    {project.status === 'ready' && '📊 Ready (Not Indexed)'}
                    {project.status === 'pending' && '⏸️ Pending'}
                    {project.status === 'error' && '❌ Error'}
                  </div>
                  {project.status !== 'indexed' && project.status !== 'indexing' && (
                    <button
                      className="index-btn"
                      onClick={(e) => handleIndexProject(project.id, e)}
                      title="Index this project before entering tickets"
                    >
                      🔍 Index Project
                    </button>
                  )}
                  {project.status === 'indexed' && (
                    <button
                      className="index-btn reindex-btn"
                      onClick={(e) => handleIndexProject(project.id, e)}
                      title="Re-run indexing on this project"
                    >
                      🔄 Re-index
                    </button>
                  )}
                </div>
                <button
                  className="delete-project-btn"
                  onClick={(e) => handleDeleteProject(project.id, e)}
                  title="Remove project"
                >
                  ✕
                </button>
              </div>
            ))
          )}
        </div>

        {/* Chat & Task History — combined list */}
        <div className="history-panel">
          <div className="history-header">
            <span style={{ fontSize: 12, fontWeight: 700, color: 'var(--text-secondary)', letterSpacing: '0.05em', textTransform: 'uppercase' }}>History {combinedHistory.length > 0 && <span className="history-count">{combinedHistory.length}</span>}</span>
            <button className="new-chat-btn" onClick={startNewChat} title="Start a new chat">
              + New
            </button>
          </div>

          <div className="history-list">
            {combinedHistory.length === 0 ? (
              <div className="history-empty">No previous history yet</div>
            ) : (
              combinedHistory.map(item => (
                <div
                  key={item.id}
                  className={`history-item ${(activeChatId === item.id || (Boolean(activeChatWorkflowId) && Boolean(item.workflow_id) && activeChatWorkflowId === item.workflow_id)) ? 'active' : ''}`}
                  onClick={() => {
                    if (item.isTask) {
                      setActiveChatId(null);
                      setActiveChatWorkflowId(item.workflow_id);
                      if (item.workflow_id) {
                        fetchWorkflowChanges(item.workflow_id);
                      }
                      setWorkflowData({
                        projectId: selectedProject?.id,
                        ticketId: item.ticket_id || item.title || 'Task',
                        ticketDescription: item.description || item.title || '',
                        repoPath: selectedProject?.path,
                        attachments: [],
                        existingWorkflowId: item.workflow_id,
                      });
                      setShowWorkflow(true);
                      setActiveTab('flow');
                      setMessages([]); // clear chat view for pure task
                    } else {
                      openChat(item.id);
                      if (item.workflow_id) {
                        fetchWorkflowChanges(item.workflow_id);
                      }
                      setActiveTab('chat');
                    }
                  }}
                  title={item.title || item.ticket_id}
                >
                  <div className="history-item-main">
                    <div className="history-item-title">
                      {item.isTask ? <span style={{ opacity: 0.5 }}>⚙️</span> : (item.workflow_id ? <span style={{ opacity: 0.5 }}>🔄</span> : <span style={{ opacity: 0.5 }}>💬</span>)}
                      {item.title || item.ticket_id}
                    </div>
                    <div className="history-item-meta">
                      {item.isTask ? `Status: ${item.status}` : `${item.message_count} msg`} · {new Date(item.updated_at || item.created_at).toLocaleString()}
                    </div>
                  </div>
                  <button
                    className="history-delete-btn"
                    onClick={(e) => {
                      if (item.isTask) {
                        deleteTask(item.workflow_id, e);
                      } else {
                        deleteChat(item.id, e);
                      }
                    }}
                    title="Delete item"
                  >
                    ✕
                  </button>
                </div>
              ))
            )}
          </div>
        </div>
      </div>

      {/* Right Panel - Tabbed Content Area */}
      <div className="right-panel">
        <div className="panel-header">
          <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
            <h2 style={{ margin: 0, fontWeight: 800, letterSpacing: '-0.5px' }}>⚡ NemoForge</h2>
            {settingsStatus?.active_provider_name ? (
              <span style={{
                fontSize: '11px',
                background: 'linear-gradient(135deg, #10b981 0%, #059669 100%)',
                color: '#ffffff',
                padding: '2px 8px',
                borderRadius: '12px',
                fontWeight: 700,
                display: 'inline-flex',
                alignItems: 'center',
                gap: '4px',
                boxShadow: '0 2px 4px rgba(16, 185, 129, 0.2)'
              }}>
                🟢 {settingsStatus.active_provider_name}
              </span>
            ) : (
              <span style={{
                fontSize: '11px',
                background: 'rgba(255, 255, 255, 0.06)',
                color: '#94a3b8',
                padding: '2px 8px',
                borderRadius: '12px',
                fontWeight: 600,
                border: '1px solid rgba(255, 255, 255, 0.1)'
              }}>
                ⚪ No API Connected
              </span>
            )}
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            {selectedProject && (
              <span className="selected-project-badge">
                {selectedProject.name}
              </span>
            )}
            <button
              onClick={() => setShowSettingsModal(true)}
              style={{
                background: settingsStatus?.token_configured ? 'rgba(16, 185, 129, 0.12)' : 'rgba(56, 189, 248, 0.12)',
                color: settingsStatus?.token_configured ? '#34d399' : '#38bdf8',
                border: settingsStatus?.token_configured ? '1px solid rgba(16, 185, 129, 0.45)' : '1px solid rgba(56, 189, 248, 0.35)',
                borderRadius: '8px',
                padding: '5px 12px',
                fontSize: '12px',
                cursor: 'pointer',
                fontWeight: 700,
                display: 'inline-flex',
                alignItems: 'center',
                gap: 6
              }}
              title={settingsStatus?.token_configured ? `Active Engine: ${settingsStatus.active_provider_name || 'Connected'} (Click to manage keys)` : "Configure API Keys (Saved locally to .env.local)"}
            >
              {settingsStatus?.token_configured && (
                <span style={{ width: 7, height: 7, borderRadius: '50%', background: '#34d399', display: 'inline-block', boxShadow: '0 0 6px #34d399' }} />
              )}
              🔑 Add API Key
            </button>
            <button
              onClick={() => setShowArchitectureModal(true)}
              style={{
                background: 'rgba(139, 92, 246, 0.12)',
                color: '#c4b5fd',
                border: '1px solid rgba(139, 92, 246, 0.35)',
                borderRadius: '8px',
                padding: '5px 12px',
                fontSize: '12px',
                cursor: 'pointer',
                fontWeight: 700,
                display: 'inline-flex',
                alignItems: 'center',
                gap: 6
              }}
              title="View System Architecture and Documented Failure Boundaries"
            >
              🛡️ Architecture & Limitations
            </button>
            <button
              className="theme-toggle-btn"
              onClick={() => setTheme(t => t === 'dark' ? 'light' : 'dark')}
              title={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}
            >
              {theme === 'dark' ? '\u2600' : '\u263D'}
            </button>
          </div>
        </div>

        {/* ── Tab Bar ──────────────────────────────────────── */}
        <div className="main-tab-bar">
          <button
            className={`main-tab ${activeTab === 'chat' ? 'active' : ''}`}
            onClick={() => setActiveTab('chat')}
          >
            💬 Chat
          </button>
          <button
            className={`main-tab ${activeTab === 'flow' ? 'active' : ''}`}
            onClick={() => setActiveTab('flow')}
          >
            🔄 Flow
            {flowStepCount > 0 && <span className="tab-badge">{flowStepCount}</span>}
          </button>
          <button
            className={`main-tab ${activeTab === 'changes' ? 'active' : ''}`}
            onClick={() => {
              setActiveTab('changes');
              const targetId = activeChatWorkflowId || workflowData?.existingWorkflowId;
              if (targetId) {
                fetchWorkflowChanges(targetId);
              } else {
                setWorkflowChanges(null);
              }
            }}
          >
            📝 Changes
          </button>
          <button
            className={`main-tab ${activeTab === 'mcp' ? 'active' : ''}`}
            onClick={() => setActiveTab('mcp')}
          >
            🔌 MCP Tools
          </button>
        </div>

        {/* ── Tab Content ──────────────────────────────── */}
        <div className="tab-content">

        {/* ─── MCP HUB TAB ─── */}
        {activeTab === 'mcp' && (
          <McpHub />
        )}

        {/* ═══ FLOW TAB ═══ */}
        {activeTab === 'flow' && workflowData ? (
          <TransparentWorkflow
            key={workflowData.existingWorkflowId || workflowData.ticketId || 'current-flow'}
            projectId={workflowData.projectId}
            ticketId={workflowData.ticketId}
            ticketDescription={workflowData.ticketDescription}
            repoPath={workflowData.repoPath}
            attachments={workflowData.attachments || []}
            writableFiles={workflowData.writableFiles || []}
            forbiddenFiles={workflowData.forbiddenFiles || []}
            existingWorkflowId={workflowData.existingWorkflowId || null}
            modelMode={workflowData.modelMode || modelMode || 'default'}
            selectedModel={workflowData.selectedModel || selectedModel || null}
            onWorkflowStarted={(wfId) => {
              if (saveTimer.current) {
                clearTimeout(saveTimer.current);
                saveTimer.current = null;
              }
              setWorkflowData(prev => prev ? { ...prev, existingWorkflowId: wfId } : prev);
              setActiveChatWorkflowId(wfId);
              saveCurrentChat(messages, activeChatIdRef.current, wfId);
            }}
            onWorkflowStopped={(stoppedWfId) => {
              // Gracefully handle stop: re-enable input and preserve workflow context
              setIsProcessing(false);
              setWorkflowCompleted(true);
              if (stoppedWfId) {
                setActiveChatWorkflowId(stoppedWfId);
                setWorkflowData(prev => prev ? { ...prev, isStopped: true } : prev);
              }
              // Keep user on the flow tab if they were viewing it, otherwise keep on current tab
              setActiveTab(prev => (prev === 'flow' ? 'flow' : 'chat'));
              // Append a friendly guidance message if not already present
              setMessages(prev => {
                const last = prev[prev.length - 1];
                if (last && last.role === 'assistant' && last.content && last.content.includes('Workflow stopped')) {
                  return prev;
                }
                return [...prev, {
                  role: 'assistant',
                  content: '🛑 **Workflow stopped by user.**\n\nExecution was halted cleanly at your request. All discovered files, AST context, and logs have been saved into conversation memory.\n\nYou can continue chatting right here — ask questions about what was inspected, provide corrections, or give new instructions without restarting.',
                  timestamp: new Date(),
                  workflowId: stoppedWfId,
                  isStoppedNotice: true,
                }];
              });
              saveCurrentChat(messages, activeChatIdRef.current, stoppedWfId);
            }}
            onWorkflowResumed={(resumedWfId, phase) => {
              setWorkflowCompleted(false);
              setWorkflowData(prev => prev ? { ...prev, isStopped: false } : prev);
              setMessages(prev => [...prev, {
                role: 'system',
                content: `🔄 Resumed workflow execution from '${phase}' checkpoint.`,
                timestamp: new Date(),
                workflowId: resumedWfId,
              }]);
            }}
          />
        ) : activeTab === 'flow' && !workflowData ? (
          <div className="changes-tab-empty">
            <div className="empty-icon">🔄</div>
            <h3>No Workflow Running</h3>
            <p>Submit a ticket in the Chat tab or click "View Run" on a previous chat to see the workflow flow here.</p>
          </div>

        ) : activeTab === 'changes' ? (
          (() => {
            const currentWfId = activeChatWorkflowId || workflowData?.existingWorkflowId;
            const isMatchingWf = !currentWfId || (workflowChanges && workflowChanges.workflow_id === currentWfId);
            if (isMatchingWf && workflowChanges && workflowChanges.files && workflowChanges.files.length > 0) {
              return (
                <div className="changes-panel">
                  <div className="changes-header">
                    <div className="changes-header-left">
                      <h3>📝 Staged Changes (Virtual Buffer)</h3>
                      <span className="changes-count">{workflowChanges.files.length} file(s) staged</span>
                      {mutationReviewStatus[workflowChanges.workflow_id] === 'applied' && (
                        <span className="mutation-badge applied">✓ Applied to Disk</span>
                      )}
                      {mutationReviewStatus[workflowChanges.workflow_id] === 'discarded' && (
                        <span className="mutation-badge discarded">✗ Discarded</span>
                      )}
                    </div>
                    <div className="changes-header-actions">
                      <button 
                        className="btn-accept-all"
                        onClick={() => handleAcceptAllChanges(workflowChanges.workflow_id)}
                        disabled={mutationReviewStatus[workflowChanges.workflow_id] === 'applied'}
                      >
                        ✓ Accept All (Flush to Disk)
                      </button>
                      <button 
                        className="btn-discard-all"
                        onClick={() => handleDiscardAllChanges(workflowChanges.workflow_id)}
                        disabled={mutationReviewStatus[workflowChanges.workflow_id] === 'discarded'}
                      >
                        ✗ Discard All
                      </button>
                    </div>
                  </div>
                  <div className="changes-file-list">
                    {workflowChanges.files.map((file, idx) => {
                      const basename = file.path.split('/').pop() || file.path.split('\\').pop() || file.path;
                      const isExpanded = expandedDiffs[idx] !== undefined ? expandedDiffs[idx] : true;
                      return (
                        <div key={idx} className={`diff-file-block ${file.status}`}>
                          <div
                            className="diff-file-header"
                            onClick={() => setExpandedDiffs(prev => ({ ...prev, [idx]: !prev[idx] }))}
                          >
                            <span className="diff-expand-icon">{isExpanded ? '▼' : '▶'}</span>
                            <span className={`diff-status-badge ${file.status}`}>
                              {file.status === 'new' ? 'NEW' : 'MOD'}
                            </span>
                            <span className="diff-file-name">{basename}</span>
                            <span className="diff-file-path">{file.path}</span>
                            <div className="diff-file-actions">
                              <button
                                className={`btn-file-toggle ${file.approved ? 'approved' : 'rejected'}`}
                                onClick={(e) => {
                                  e.stopPropagation();
                                  handleToggleFileApproval(workflowChanges.workflow_id, file.path, !file.approved);
                                }}
                              >
                                {file.approved ? '✓ Staged' : '✗ Excluded'}
                              </button>
                            </div>
                          </div>
                          {isExpanded && (
                            <div className="diff-content">
                              {file.diff ? (
                                <pre className="diff-pre">
                                  {file.diff.split('\n').map((line, li) => {
                                    let cls = 'diff-line';
                                    if (line.startsWith('+') && !line.startsWith('+++')) cls += ' diff-add';
                                    else if (line.startsWith('-') && !line.startsWith('---')) cls += ' diff-remove';
                                    else if (line.startsWith('@@')) cls += ' diff-hunk';
                                    return <div key={li} className={cls}>{line}</div>;
                                  })}
                                </pre>
                              ) : (
                                <div className="diff-no-content">No diff available for this file.</div>
                              )}
                            </div>
                          )}
                        </div>
                      );
                    })}
                  </div>
                </div>
              );
            }
            if (changesLoading) {
              return (
                <div className="changes-tab-empty">
                  <div className="empty-icon">⏳</div>
                  <h3>Loading Changes...</h3>
                </div>
              );
            }
            return (
              <div className="changes-tab-empty">
                <div className="empty-icon">📝</div>
                <h3>Code Changes</h3>
                <p>
                  {currentWfId 
                    ? 'No file modifications were staged for this specific run.' 
                    : 'Select a completed ticket or chat from the history sidebar to review its staged diffs.'}
                </p>
                {tasks && tasks.filter(t => t.status === 'completed' && t.workflow_id).length > 0 && (
                  <div style={{ marginTop: 16, display: 'flex', gap: 10, justifyContent: 'center', flexWrap: 'wrap' }}>
                    {tasks.filter(t => t.status === 'completed' && t.workflow_id).slice(0, 3).map((t, idx) => (
                      <button
                        key={idx}
                        className="btn-accept-all"
                        style={{ background: '#3b82f6', fontSize: 12, padding: '6px 14px' }}
                        onClick={() => {
                          setActiveChatWorkflowId(t.workflow_id);
                          fetchWorkflowChanges(t.workflow_id);
                        }}
                      >
                        📁 Review {t.ticket_id || 'Completed Task'}
                      </button>
                    ))}
                  </div>
                )}
              </div>
            );
          })()
        ) : indexingState ? (
          <div className="indexing-panel">
            <div className="indexing-header">
              <span className="indexing-title">
                {indexingState.status === 'running' ? '⚙️ Indexing' :
                 indexingState.status === 'done' ? '✅ Indexing Complete' : '❌ Indexing Failed'} — {indexingState.projectName}
              </span>
              {indexingState.status !== 'running' && (
                <button className="indexing-close-btn" onClick={() => setIndexingState(null)}>✕ Close</button>
              )}
            </div>

            {/* Progress bar */}
            <div className="indexing-progress-bar-wrap">
              <div
                className={`indexing-progress-bar ${indexingState.status}`}
                style={{ width: `${indexingState.status === 'done' ? 100 : indexingState.percent}%` }}
              />
            </div>
            <div className="indexing-progress-label">
              {indexingState.status === 'running'
                ? `${indexingState.done} / ${indexingState.total || '?'} files (${indexingState.percent}%)`
                : indexingState.status === 'done'
                ? `Completed — ${indexingState.done} files processed`
                : indexingState.message}
            </div>

            {/* Current file */}
            {indexingState.status === 'running' && (
              <div className="indexing-current-file">
                📄 <strong>{indexingState.currentFile}</strong>
              </div>
            )}

            {/* Final stats */}
            {indexingState.status === 'done' && (
              <div className="indexing-stats">
                <div className="stat-row">📁 <strong>Project:</strong> {indexingState.projectPath}</div>
                <div className="stat-row">💾 <strong>Index stored at:</strong> {indexingState.projectPath}\.nemoforge\index.db</div>
                {indexingState.stats && <>
                  <div className="stat-row">📂 <strong>Files scanned:</strong> {indexingState.stats.files_scanned}</div>
                  <div className="stat-row">✅ <strong>Files parsed:</strong> {indexingState.stats.files_parsed}</div>
                  <div className="stat-row">❌ <strong>Files failed:</strong> {indexingState.stats.files_failed}</div>
                  <div className="stat-row">🔤 <strong>Symbols extracted:</strong> {indexingState.stats.symbols}</div>
                  <div className="stat-row">🔗 <strong>Relationships (edges):</strong> {indexingState.stats.edges}</div>
                  {indexingState.stats.resolved_calls !== undefined && (
                    <div className="stat-row">🎯 <strong>Call Graph:</strong> {indexingState.stats.resolved_calls} Method-to-Method Edges</div>
                  )}
                  <div className="stat-row">⚡ <strong>Symbol Graph:</strong> Embedded SQLite WAL + Tree-sitter AST (&lt;10ms queries)</div>
                  <div className="stat-row">🛡️ <strong>Compiler Sandbox:</strong> Ready (Maven / Gradle / npm / pytest)</div>
                </>}
              </div>
            )}

            {/* Scrollable file log */}
            <div className="indexing-log-header">Files processed ({indexingState.log.length}):</div>
            <div className="indexing-log">
              {indexingState.log.map((f, i) => (
                <div key={i} className="indexing-log-entry">✔ {f}</div>
              ))}
              {indexingState.status === 'running' && <div className="indexing-log-pulse">…</div>}
            </div>
          </div>
        ) : (
          <>
            <div className="messages-container">
              {messages.length === 0 ? (
                <div className="welcome-message">
                  <h3>⚡ Welcome to NemoForge</h3>
                  <p>Autonomous SWE Agent for Multi-Repository Code Synthesis</p>
                  
                  <div className="instructions">
                    <h4>Quick Start</h4>
                    <ol>
                      <li>Select or index a project from the left sidebar</li>
                      <li>Ensure an API key is configured via <strong>🔑 Add API Key</strong> (e.g. Nebius, OpenAI, Anthropic, Gemini, Groq)</li>
                      <li>Describe a bug fix, feature request, or ticket description</li>
                      <li>Watch NemoForge localize AST symbols, synthesize precise patches, and verify builds in real time</li>
                    </ol>
                  </div>

                  <div className="feature-highlights">
                    <h4>🟢 Engine Capabilities</h4>
                    <ul>
                      <li>Pluggable Multi-Provider Support (Nebius NVIDIA Nemotron, OpenAI, Anthropic, Gemini, or Custom)</li>
                      <li>Deterministic Tree-sitter AST & Embedded SQLite Symbol Indexing</li>
                      <li>Zero-Hallucination 3-Line Anchor Patching</li>
                      <li>Automated Multi-Service Polyglot Build Verification</li>
                    </ul>
                  </div>
                </div>

              ) : (
                messages.filter(m => !m.isThinking).map((msg, idx) => (
                  msg.isWorkflowStep ? (
                    <div key={idx} className="message workflow-step-card">
                      <div className="step-card-header">
                        <div className="step-card-title">
                          <span>
                            {msg.nodeName?.includes('investigate') || msg.nodeName?.includes('discover') ? '🔍' :
                             msg.nodeName?.includes('plan') ? '📋' :
                             msg.nodeName?.includes('code') || msg.nodeName?.includes('patch') ? '📝' :
                             msg.nodeName?.includes('build') ? '🏗️' :
                             msg.nodeName?.includes('playwright') || msg.nodeName?.includes('ui') ? '🎭' :
                             msg.nodeName?.includes('test') ? '🧪' : '⚡'}
                          </span>
                          <span>{msg.content}</span>
                        </div>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                          <span style={{
                            fontSize: '11px',
                            fontWeight: 700,
                            padding: '2px 8px',
                            borderRadius: '10px',
                            background: msg.nodeName?.includes('investigate') || msg.nodeName?.includes('discover')
                              ? 'rgba(56, 189, 248, 0.15)'
                              : msg.nodeName?.includes('build') || msg.nodeName?.includes('test')
                              ? 'rgba(234, 179, 8, 0.15)'
                              : 'rgba(16, 185, 129, 0.15)',
                            color: msg.nodeName?.includes('investigate') || msg.nodeName?.includes('discover')
                              ? '#38bdf8'
                              : msg.nodeName?.includes('build') || msg.nodeName?.includes('test')
                              ? '#facc15'
                              : '#34d399',
                            border: '1px solid rgba(255, 255, 255, 0.08)'
                          }}>
                            {msg.nodeName?.includes('investigate') || msg.nodeName?.includes('discover')
                              ? '⚡ Nemotron 3.5 Lightning (FAST)'
                              : msg.nodeName?.includes('build') || msg.nodeName?.includes('test')
                              ? '🏗️ Polyglot Sandbox'
                              : '🧠 Nemotron 3 Super 120B (DEEP)'}
                          </span>
                          <span className={`step-card-status ${msg.status || 'running'}`}>
                            {msg.status === 'running' ? '⏳ In Progress' : '✅ Done'}
                          </span>
                        </div>
                      </div>
                      {msg.agentOutput && (
                        <div className="step-card-reasoning">
                          <strong>Agent Reasoning & Hypothesis:</strong>
                          <div>{typeof msg.agentOutput === 'string' ? msg.agentOutput : JSON.stringify(msg.agentOutput, null, 2)}</div>
                        </div>
                      )}
                      {msg.files && msg.files.length > 0 && (
                        <div className="step-card-files">
                          {msg.files.map((fp, fIdx) => (
                            <span key={fIdx} className="step-file-pill">📄 {fp.split('/').pop() || fp.split('\\').pop() || fp}</span>
                          ))}
                        </div>
                      )}
                    </div>
                  ) :
                  <div key={idx} className={`message ${msg.role}${msg.isWorkflowSummary ? ' workflow-summary' : ''}${msg.isFollowUp ? ' follow-up' : ''}`}>
                    <div className="message-header">
                      <div className="message-header-left">
                        <span className="message-role">
                          {msg.role === 'user' ? '👤 You' : 
                          msg.role === 'assistant' ? '🤖 Ticket-to-Code' : 
                          '⚙️ System'}
                        </span>
                        {msg.role === 'assistant' && (
                          <span
                            className={`msg-model-badge ${msg.modelMode === 'auto_model' ? 'auto-mode' : ''}`}
                            title={`Execution Model: ${msg.modelUsed || selectedModel} (${msg.modelMode === 'auto_model' ? 'Auto Selected' : 'Fixed'})`}
                          >
                            {msg.modelMode === 'auto_model'
                              ? `⚡ Auto: ${msg.modelUsed || selectedModel || 'Gemini 2.5 Pro'}`
                              : `🤖 ${msg.modelUsed || selectedModel}`}
                          </span>
                        )}
                      </div>
                      <div className="message-header-right">
                        {msg.role === 'assistant' && (
                          <button
                            type="button"
                            className="chat-msg-copy-btn"
                            onClick={() => handleCopyMessage(msg.content, idx)}
                            title="Copy message markdown"
                          >
                            {copiedMessageIdx === idx ? '✓ Copied' : '📋 Copy'}
                          </button>
                        )}
                        <span className="message-time">
                          {msg.timestamp instanceof Date ? msg.timestamp.toLocaleTimeString() : new Date(msg.timestamp).toLocaleTimeString()}
                        </span>
                        {msg.isWorkflowSummary && (
                          <div style={{ display: 'inline-flex', alignItems: 'center', gap: '6px' }}>
                            <span className="message-badge summary-badge">
                              {msg.workflowStatus === 'completed' ? '✅ Completed' : '❌ Failed'}
                            </span>
                            {msg.tokenUsage && (
                              <span className="message-badge" style={{ background: 'rgba(139, 92, 246, 0.2)', color: '#c4b5fd', border: '1px solid rgba(139, 92, 246, 0.4)' }}>
                                🪙 {((msg.tokenUsage.total_tokens || 0)).toLocaleString()} tokens ({msg.tokenUsage.llm_calls || 0} calls)
                              </span>
                            )}
                          </div>
                        )}
                      </div>
                    </div>
                    <div className="message-content">
                      {msg.isError ? (
                        <div className="chat-error-card">
                          <div className="chat-error-header">
                            <div className="chat-error-title-wrap">
                              <span className="chat-error-icon">⚠️</span>
                              <strong>{msg.errorTitle || 'Request Interrupted'}</strong>
                            </div>
                            <span className="chat-error-badge">{msg.errorType?.toUpperCase() || 'ERROR'}</span>
                          </div>
                          <div className="chat-error-body">
                            {msg.errorMessage}
                          </div>
                          <div className="chat-error-actions">
                            <button
                              type="button"
                              className="chat-retry-btn"
                              onClick={() => handleRetry(msg.failedMessage)}
                            >
                              🔄 Retry
                            </button>
                          </div>
                        </div>
                      ) : (
                        <>
                          {msg.role === 'assistant' && msg.execution_steps && msg.execution_steps.length > 0 && (
                            <ThoughtTrace steps={msg.execution_steps} durationMs={msg.duration_ms} />
                          )}
                          {msg.attachments && msg.attachments.length > 0 && (
                            <div className="message-attachments">
                              {msg.attachments.map((att, aIdx) => {
                                const meta = getAttachmentMeta(att, aIdx, msg.attachments);
                                return (
                                  <div key={aIdx} className="message-attachment-item">
                                    <div className="message-attachment-tag">
                                      <span className={`att-order-badge badge-${meta.category}`}>{meta.label}</span>
                                      <span className="att-name-label">{att.name}</span>
                                    </div>
                                    {att.isImage ? (
                                      <img src={att.previewUrl} alt={att.name} className="attached-img-bubble" />
                                    ) : (
                                      <span className="attached-file-badge">{meta.category === 'log' ? '📋' : '📄'} {att.name}</span>
                                    )}
                                  </div>
                                );
                              })}
                            </div>
                          )}
                          {/* Rich AI IDE Markdown Rendering */}
                          <MarkdownMessage content={msg.content} />

                          {/* Interrupted generation footer with Retry shown AFTER the displayed content */}
                          {msg.isInterrupted && (
                            <div className="chat-interrupted-footer">
                              <div className="chat-interrupted-notice">
                                <span className="chat-interrupted-icon">⚠️</span>
                                <span>{msg.errorMessage || 'Generation was interrupted or timed out.'}</span>
                              </div>
                              <button
                                type="button"
                                className="chat-retry-btn"
                                onClick={() => handleRetry(msg.failedMessage)}
                              >
                                🔄 Retry
                              </button>
                            </div>
                          )}
                          {/* Collapsible Token Count at bottom */}
                          {msg.role === 'assistant' && msg.usage && (
                            <div className="message-token-footer">
                              <button
                                type="button"
                                className="token-summary-pill"
                                onClick={() => setExpandedTokens(prev => ({ ...prev, [idx]: !prev[idx] }))}
                                title="Click to view token breakdown"
                              >
                                <span>⚡ {(msg.usage.total_tokens || 0).toLocaleString()} tokens</span>
                                <span className="token-pill-chevron">{expandedTokens[idx] ? '▴' : '▾'}</span>
                              </button>
                              {expandedTokens[idx] && (
                                <div className="token-details-popover">
                                  <div className="token-row"><span>Model:</span> <strong>{msg.modelUsed || selectedModel}</strong></div>
                                  {msg.modelMode === 'auto_model' && (
                                    <div className="token-row"><span>Mode:</span> <strong style={{ color: '#38bdf8' }}>Auto Model (Intake Selected)</strong></div>
                                  )}
                                  <div className="token-row"><span>Input / Prompt:</span> <strong>{(msg.usage.prompt_tokens || msg.usage.input_tokens || 0).toLocaleString()}</strong></div>
                                  <div className="token-row"><span>Output / Completion:</span> <strong>{(msg.usage.completion_tokens || msg.usage.candidates_tokens || msg.usage.output_tokens || 0).toLocaleString()}</strong></div>
                                  <div className="token-row token-row-total"><span>Total Tokens:</span> <strong>{(msg.usage.total_tokens || ((msg.usage.prompt_tokens || 0) + (msg.usage.completion_tokens || msg.usage.candidates_tokens || 0))).toLocaleString()}</strong></div>
                                  {msg.duration_ms && (
                                    <div className="token-row token-row-latency"><span>Latency:</span> <strong>{(msg.duration_ms / 1000).toFixed(1)}s</strong></div>
                                  )}
                                </div>
                              )}
                            </div>
                          )}
                          {/* Show "View Changes" button for workflow summary messages */}
                          {msg.isWorkflowSummary && msg.generatedFiles && msg.generatedFiles.length > 0 && (
                            <div className="summary-actions">
                              <button
                                className="view-changes-btn"
                                onClick={() => {
                                  if (msg.workflowId) fetchWorkflowChanges(msg.workflowId);
                                  setActiveTab('changes');
                                }}
                              >
                                📝 View Changes ({msg.generatedFiles.length} files)
                              </button>
                              <button
                                className="view-flow-btn"
                                onClick={() => setActiveTab('flow')}
                              >
                                🔄 View Flow
                              </button>
                            </div>
                          )}
                          {/* Show action buttons for stopped workflow notice */}
                          {msg.isStoppedNotice && (
                            <div className="summary-actions" style={{ marginTop: '10px' }}>
                              <button
                                className="view-flow-btn"
                                onClick={() => setActiveTab('flow')}
                                style={{ background: '#0284c7', color: '#fff', border: 'none' }}
                              >
                                🔄 View Flow & Resume
                              </button>
                              {msg.workflowId && (
                                <button
                                  className="view-changes-btn"
                                  onClick={() => {
                                    fetchWorkflowChanges(msg.workflowId);
                                    setActiveTab('changes');
                                  }}
                                >
                                  📝 View Partial Changes
                                </button>
                              )}
                            </div>
                          )}
                        </>
                      )}
                    </div>
                  </div>
                ))
              )}
              {/* Live Streaming Message Token-by-Token */}
              {streamingMessage && streamingMessage.content && (
                <div className="message assistant streaming-live-message" style={{ borderLeft: '3px solid #38bdf8' }}>
                  <div className="message-header">
                    <div className="message-header-left">
                      <span className="message-role">🤖 Ticket-to-Code</span>
                      <span className="msg-model-badge auto-mode" style={{ background: 'rgba(56, 189, 248, 0.15)', color: '#38bdf8' }}>
                        ⚡ Streaming: {streamingMessage.modelUsed || selectedModel}
                      </span>
                    </div>
                    <div className="message-header-right">
                      <span className="message-time" style={{ color: '#38bdf8' }}>● Live</span>
                    </div>
                  </div>
                  <div className="message-content">
                    <MarkdownMessage content={streamingMessage.content} />
                  </div>
                </div>
              )}
              {/* Live Thinking Card in Chat Tab */}
              {isProcessing && (
                <div className="message assistant thinking-live-card">
                  <div className="message-header">
                    <span className="message-role">🤖 Ticket-to-Code</span>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <span className="thinking-timer-live">Working for {thinkingTimer}s...</span>
                      <button
                        type="button"
                        className="thinking-cancel-btn"
                        onClick={cancelActiveChat}
                        title="Cancel generation"
                      >
                        ⏹️ Stop
                      </button>
                    </div>
                  </div>
                  <div className="thinking-live-body">
                    <div className="thinking-pulse-loader">
                      <div className="thinking-dot-pulse" />
                      <span className="thinking-status-text">
                        {currentThinkingStep ? (
                          <span style={{ display: 'inline-flex', alignItems: 'center', gap: '8px' }}>
                            <span style={{ fontSize: '14px' }}>
                              {currentThinkingStep.type === 'tool_call' ? '⚡' :
                               currentThinkingStep.type === 'search' ? '🔍' :
                               currentThinkingStep.type === 'file_read' ? '📄' :
                               currentThinkingStep.type === 'playwright' ? '🌐' : '🧠'}
                            </span>
                            <span style={{ color: '#e2e8f0', fontWeight: 500 }}>
                              {currentThinkingStep.title || currentThinkingStep.message}
                            </span>
                          </span>
                        ) : (
                          streamingMessage && streamingMessage.content ? 'Streaming response tokens to chat...' :
                          thinkingTimer < 3 ? 'Analyzing prompt and repository context...' :
                          thinkingTimer < 10 ? 'Searching code graph, indexed schemas & symbols...' :
                          thinkingTimer < 25 ? 'Reasoning with Gemini 2.5 Pro...' :
                          'Synthesizing grounded engineering response...'
                        )}
                      </span>
                    </div>
                  </div>
                </div>
              )}
              {/* Sleek live autonomous pipeline banner in Chat tab - strictly scoped to active workflow for this chat */}
              {activeWorkflowProgress && activeChatWorkflowId && activeWorkflowProgress.workflowId === activeChatWorkflowId && !workflowCompleted && (
                <div className={`chat-active-pipeline-banner ${activeWorkflowProgress.phase === 'failed' || activeWorkflowProgress.status === 'failed' ? 'pipeline-failed' : ''}`}>
                  <div className="pipeline-pulse-dot" />
                  <div className="pipeline-banner-text">
                    <span className="pipeline-banner-title">
                      {activeWorkflowProgress.phase === 'failed' || activeWorkflowProgress.status === 'failed' ? '❌ Autonomous Pipeline Failed' : '⚡ Autonomous Pipeline Active'}: <strong>{activeWorkflowProgress.phase}</strong>
                    </span>
                    <span className="pipeline-banner-sub">{activeWorkflowProgress.message || 'Analyzing codebase and planning steps...'}</span>
                  </div>
                  <button type="button" className="pipeline-switch-tab-btn" onClick={() => setActiveTab('flow')}>
                    View Flow Tab ➔
                  </button>
                  <button
                    type="button"
                    className="pipeline-close-btn"
                    onClick={() => setActiveWorkflowProgress(null)}
                    title="Dismiss"
                    style={{
                      background: 'transparent',
                      border: 'none',
                      color: '#94a3b8',
                      cursor: 'pointer',
                      fontSize: '15px',
                      padding: '4px 8px',
                      marginLeft: '6px'
                    }}
                  >
                    ✕
                  </button>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>

            <div className="input-box-wrapper">
              {attachments.length > 0 && (
                <div className="attachment-preview-bar">
                  {attachments.map((att, idx) => {
                    const meta = getAttachmentMeta(att, idx, attachments);
                    return (
                      <div key={idx} className="attachment-chip">
                        <span className={`chip-badge badge-${meta.category}`}>{meta.label}</span>
                        {att.isImage ? (
                          <img src={att.previewUrl} alt={att.name} className="chip-img" />
                        ) : (
                          <span className="chip-icon">{meta.category === 'log' ? '📋' : '📄'}</span>
                        )}
                        <span className="chip-name">{att.name}</span>
                        {att.isUploading ? (
                          <span className="chip-spinner">⏳</span>
                        ) : (
                          <button type="button" className="chip-remove-btn" onClick={() => removeAttachment(idx)}>✕</button>
                        )}
                      </div>
                    );
                  })}
                </div>
              )}
              {/* ── Model & Mode Selector Toolbar ── */}
              <div className="model-selector-toolbar">
                <span className="model-toolbar-label">⚙️ Mode:</span>
                <select
                  className="mode-select-dropdown"
                  value={modelMode}
                  onChange={(e) => setModelMode(e.target.value)}
                >
                  <option value="default">Default Mode (Single Model)</option>
                  <option value="auto_model">Auto Model Mode (Intake Selected)</option>
                  <option value="auto_stage">Auto Stage Mode (Stage Switching)</option>
                </select>

                {modelMode === 'default' && (
                  <>
                    <span className="model-toolbar-label">
                      🟢 Model:
                    </span>
                    <select
                      className="model-select-dropdown"
                      value={selectedModel}
                      onChange={(e) => setSelectedModel(e.target.value)}
                    >
                      {availableModelsList.filter(m => m.is_available).length > 0 ? (
                        availableModelsList.filter(m => m.is_available).map(m => (
                          <option key={m.model_id} value={m.model_id}>
                            🟢 {m.name} ({m.tier.toUpperCase()})
                          </option>
                        ))
                      ) : (
                        <option value="nebius/nvidia/nemotron-3-super-120b-a12b">
                          🟢 NVIDIA Nemotron 3 Super 120B (Nebius) (DEEP)
                        </option>
                      )}
                    </select>
                  </>
                )}

                {modelMode === 'auto_model' && (
                  <span style={{ color: '#38bdf8', fontSize: '11px', fontStyle: 'italic' }}>
                    💡 Analyzes ticket at intake and runs ONE optimal model end-to-end (100% KV cache retention).
                  </span>
                )}

                {modelMode === 'auto_stage' && (
                  <span style={{ color: '#c084fc', fontSize: '11px', fontStyle: 'italic' }}>
                    ⚡ Fast models for exploration, Deep models for synthesis (Guarded by Distilled Context Firewall).
                  </span>
                )}
              </div>
              <div className="input-container">
                <button
                  type="button"
                  className="attach-btn"
                  onClick={() => fileInputRef.current?.click()}
                  title="Attach screenshot or log file (or paste with Ctrl+V)"
                  disabled={!selectedProject || isProcessing}
                >
                  📎
                </button>
                <input
                  type="file"
                  ref={fileInputRef}
                  style={{ display: 'none' }}
                  multiple
                  accept="image/*,.log,.txt,.json,.yml,.yaml,.py,.java,.cs"
                  onChange={handleFileSelect}
                />
                <textarea
                  value={currentMessage}
                  onChange={(e) => setCurrentMessage(e.target.value)}
                  onKeyDown={handleKeyPress}
                  onPaste={handlePaste}
                  placeholder={selectedProject ? 
                    "Describe your task, or paste screenshot (Ctrl+V) / click 📎 to attach..." : 
                    "Please select a project first..."}
                  disabled={!selectedProject || isProcessing}
                  rows={3}
                />
                <button
                  type="button"
                  onClick={() => handleSendMessage()}
                  disabled={(!currentMessage.trim() && attachments.length === 0) || !selectedProject || isProcessing}
                  className="send-btn"
                >
                  {isProcessing ? '⏳' : '▶'}
                </button>
              </div>
            </div>
          </>
        )}
        </div>{/* end tab-content */}
      </div>

      {/* ── Interactive Clarification Modal (HITL) ── */}
      {clarificationModal && (
        <div className="interactive-modal-backdrop">
          <div className="interactive-modal-content">
            <div className="interactive-modal-header">
              <span className="interactive-modal-badge">❓ Clarification Required</span>
              <h3 className="interactive-modal-title">{clarificationModal.question}</h3>
            </div>
            <div className="interactive-modal-body">
              <div className="interactive-options-list">
                {(clarificationModal.options || []).map((opt, idx) => {
                  const isChecked = selectedOptions.includes(opt.label);
                  return (
                    <label key={idx} className={`interactive-option-item ${isChecked ? 'selected' : ''}`}>
                      <input
                        type={clarificationModal.is_multi_select ? 'checkbox' : 'radio'}
                        name="clarification_opt"
                        checked={isChecked}
                        onChange={() => {
                          if (clarificationModal.is_multi_select) {
                            setSelectedOptions(prev =>
                              isChecked ? prev.filter(o => o !== opt.label) : [...prev, opt.label]
                            );
                          } else {
                            setSelectedOptions([opt.label]);
                          }
                        }}
                      />
                      <div className="interactive-option-text">
                        <span className="interactive-option-label">
                          {opt.label}
                          {opt.is_recommended && <span className="recommended-tag">Recommended</span>}
                        </span>
                        {opt.description && <span className="interactive-option-desc">{opt.description}</span>}
                      </div>
                    </label>
                  );
                })}
              </div>
              <div className="interactive-notes-group">
                <label>Additional Guidance / Notes (Optional):</label>
                <input
                  type="text"
                  placeholder="e.g. Please ensure backwards compatibility"
                  value={clarificationNotes}
                  onChange={(e) => setClarificationNotes(e.target.value)}
                  className="interactive-notes-input"
                />
              </div>
            </div>
            <div className="interactive-modal-footer">
              <button
                type="button"
                className="interactive-btn-secondary"
                onClick={() => {
                  const rec = clarificationModal.recommended || (clarificationModal.options?.[0]?.label);
                  handleClarificationSubmit([rec]);
                }}
              >
                Accept Recommended
              </button>
              <button
                type="button"
                className="interactive-btn-primary"
                disabled={selectedOptions.length === 0}
                onClick={() => handleClarificationSubmit(selectedOptions)}
              >
                Submit Decision
              </button>
            </div>
          </div>
        </div>
      )}

      {/* ── NemoForge API Key Modal (Centered, Key-Only) ── */}
      {showSettingsModal && (
        <div
          className="nemoforge-modal-overlay"
          onClick={() => {
            setShowSettingsModal(false);
            setSelectedInspectorProvider(null);
          }}
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: selectedInspectorProvider ? 18 : 0,
            transition: 'all 0.2s ease'
          }}
        >
          <div
            className="nemoforge-modal-card"
            style={{
              maxWidth: selectedInspectorProvider ? 520 : 580,
              transition: 'all 0.2s ease'
            }}
            onClick={(e) => e.stopPropagation()}
          >
            <div className="nemoforge-modal-header">
              <div>
                <h3 className="nemoforge-modal-title">
                  🔑 API Key Management Hub
                </h3>
                <div className="nemoforge-modal-subtitle">
                  Configure, reveal, copy, and switch inference credentials across providers.
                </div>
              </div>
              <button
                className="nemoforge-modal-close"
                onClick={() => setShowSettingsModal(false)}
                title="Close"
              >
                ✕
              </button>
            </div>

            <div className="nemoforge-modal-body">
              {/* ── 1. Configured Keys Section ── */}
              {settingsStatus?.configured_keys && settingsStatus.configured_keys.length > 0 && (
                <div style={{ marginBottom: 20 }}>
                  <div style={{
                    fontSize: '0.78rem',
                    fontWeight: 700,
                    color: '#94a3b8',
                    textTransform: 'uppercase',
                    letterSpacing: '0.06em',
                    marginBottom: 10,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between'
                  }}>
                    <span>Configured Providers ({settingsStatus.configured_keys.length})</span>
                    <span style={{ fontSize: '11px', color: '#64748b', textTransform: 'none', fontWeight: 500 }}>
                      Primary active provider receives all pipeline traffic
                    </span>
                  </div>

                  <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                    {settingsStatus.configured_keys.map(k => (
                      <div
                        key={k.id}
                        style={{
                          background: k.is_active ? 'rgba(16, 185, 129, 0.08)' : 'rgba(255, 255, 255, 0.03)',
                          border: k.is_active ? '1px solid rgba(16, 185, 129, 0.45)' : '1px solid rgba(255, 255, 255, 0.08)',
                          borderRadius: 10,
                          padding: '12px 14px',
                          display: 'flex',
                          flexDirection: 'column',
                          gap: 8,
                          transition: 'all 0.15s'
                        }}
                      >
                        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                            <span style={{ fontWeight: 700, fontSize: '0.9rem', color: '#f8fafc' }}>{k.name}</span>
                            {k.is_active ? (
                              <span style={{
                                fontSize: '10px',
                                background: 'rgba(16, 185, 129, 0.2)',
                                color: '#34d399',
                                padding: '2px 8px',
                                borderRadius: 12,
                                fontWeight: 700,
                                border: '1px solid rgba(16, 185, 129, 0.4)'
                              }}>
                                🟢 ACTIVE PRIMARY
                              </span>
                            ) : (
                              <button
                                type="button"
                                onClick={() => handleSetActiveProvider(k.id)}
                                style={{
                                  fontSize: '11px',
                                  background: 'rgba(255, 255, 255, 0.08)',
                                  color: '#cbd5e1',
                                  border: '1px solid rgba(255, 255, 255, 0.15)',
                                  padding: '2px 8px',
                                  borderRadius: 6,
                                  cursor: 'pointer',
                                  transition: 'all 0.15s'
                                }}
                                onMouseEnter={(e) => { e.currentTarget.style.background = 'rgba(118, 185, 0, 0.2)'; e.currentTarget.style.color = '#a3e635'; }}
                                onMouseLeave={(e) => { e.currentTarget.style.background = 'rgba(255, 255, 255, 0.08)'; e.currentTarget.style.color = '#cbd5e1'; }}
                              >
                                Set as Active
                              </button>
                            )}
                          </div>

                          <button
                            type="button"
                            onClick={() => handleDeleteKey(k.id)}
                            title={`Remove ${k.name} API key`}
                            style={{
                              background: 'transparent',
                              border: 'none',
                              color: '#ef4444',
                              cursor: 'pointer',
                              fontSize: '14px',
                              padding: '2px 6px',
                              borderRadius: 4,
                              opacity: 0.8
                            }}
                            onMouseEnter={(e) => { e.currentTarget.style.opacity = '1'; }}
                            onMouseLeave={(e) => { e.currentTarget.style.opacity = '0.8'; }}
                          >
                            🗑️
                          </button>
                        </div>

                        {/* Token display row with Show/Hide and Copy */}
                        <div style={{
                          display: 'flex',
                          alignItems: 'center',
                          gap: 8,
                          background: '#020617',
                          padding: '6px 10px',
                          borderRadius: 6,
                          border: '1px solid rgba(255, 255, 255, 0.06)'
                        }}>
                          <code style={{
                            flex: 1,
                            fontFamily: 'monospace',
                            fontSize: '12px',
                            color: '#38bdf8',
                            overflow: 'hidden',
                            textOverflow: 'ellipsis',
                            whiteSpace: 'nowrap'
                          }}>
                            {k.masked_key}
                          </code>

                          <button
                            type="button"
                            onClick={() => handleCopyToken(k.masked_key, k.id)}
                            title="Copy masked token to clipboard (full key is never sent to the browser)"
                            style={{
                              background: copiedKeyId === k.id ? 'rgba(16, 185, 129, 0.25)' : 'rgba(255, 255, 255, 0.08)',
                              color: copiedKeyId === k.id ? '#34d399' : '#cbd5e1',
                              border: copiedKeyId === k.id ? '1px solid rgba(16, 185, 129, 0.5)' : '1px solid rgba(255, 255, 255, 0.12)',
                              borderRadius: 4,
                              padding: '3px 9px',
                              fontSize: '11px',
                              fontWeight: 600,
                              cursor: 'pointer',
                              display: 'inline-flex',
                              alignItems: 'center',
                              gap: 4
                            }}
                          >
                            {copiedKeyId === k.id ? '✓ Copied!' : '📋 Copy'}
                          </button>
                        </div>
                        {/* ── Individual Inspector Trigger Button ── */}
                        <div style={{
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'space-between',
                          marginTop: 8,
                          paddingTop: 8,
                          borderTop: '1px solid rgba(255, 255, 255, 0.06)'
                        }}>
                          <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: '0.74rem', color: '#94a3b8' }}>
                            <span style={{
                              width: 7,
                              height: 7,
                              borderRadius: '50%',
                              background: '#10b981',
                              boxShadow: '0 0 8px rgba(16, 185, 129, 0.7)',
                              display: 'inline-block'
                            }} />
                            <span><strong>{k.live_models?.length || 0}</strong> Live Verified Models</span>
                          </div>

                          <button
                            type="button"
                            onClick={() => {
                              if (selectedInspectorProvider === k.id) {
                                setSelectedInspectorProvider(null);
                              } else {
                                setSelectedInspectorProvider(k.id);
                                fetchAuditModels();
                              }
                            }}
                            style={{
                              background: selectedInspectorProvider === k.id ? 'rgba(56, 189, 248, 0.22)' : 'rgba(56, 189, 248, 0.08)',
                              color: '#38bdf8',
                              border: selectedInspectorProvider === k.id ? '1px solid #38bdf8' : '1px solid rgba(56, 189, 248, 0.3)',
                              borderRadius: 6,
                              padding: '5px 12px',
                              fontSize: '0.74rem',
                              fontWeight: 600,
                              cursor: 'pointer',
                              display: 'inline-flex',
                              alignItems: 'center',
                              gap: 6,
                              transition: 'all 0.15s'
                            }}
                          >
                            <span>🔍 {selectedInspectorProvider === k.id ? 'Hide Models' : 'Inspect Models'}</span>
                            <span style={{
                              background: 'rgba(56, 189, 248, 0.25)',
                              padding: '1px 6px',
                              borderRadius: 8,
                              fontSize: '10px'
                            }}>
                              {k.live_models?.length || 0}
                            </span>
                          </button>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* ── 2. Add / Update Key Section ── */}
              <div style={{
                borderTop: settingsStatus?.configured_keys?.length > 0 ? '1px solid rgba(255, 255, 255, 0.08)' : 'none',
                paddingTop: settingsStatus?.configured_keys?.length > 0 ? 16 : 0
              }}>
                <div style={{
                  fontSize: '0.78rem',
                  fontWeight: 700,
                  color: '#94a3b8',
                  textTransform: 'uppercase',
                  letterSpacing: '0.06em',
                  marginBottom: 10
                }}>
                  {settingsStatus?.configured_keys?.length > 0 ? 'Add Another Provider Key' : 'Connect An Inference Provider'}
                </div>

                {/* Primary Token / Key Input */}
                <div className="nemoforge-modal-field">
                  <div className="nemoforge-modal-label">
                    <span>API Key / Inference Token:</span>
                  </div>
                  <input
                    type="password"
                    className="nemoforge-input"
                    placeholder="Paste token or key (e.g. v1.C..., sk-..., AIza...)"
                    value={submittedToken}
                    onChange={(e) => setSubmittedToken(e.target.value)}
                    autoFocus
                  />
                </div>

                {/* Optional Custom Platform / Engine Name */}
                <div className="nemoforge-modal-field">
                  <div className="nemoforge-modal-label">
                    <span>Target Provider / Platform:</span>
                  </div>
                  <input
                    type="text"
                    className="nemoforge-input"
                    placeholder="e.g. Nebius, OpenAI, Anthropic, Gemini, Groq, Together"
                    value={customPlatformTag}
                    onChange={(e) => setCustomPlatformTag(e.target.value)}
                    style={{ fontSize: '0.82rem', color: '#f8fafc' }}
                  />
                  <div style={{
                    marginTop: '8px',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '6px',
                    fontSize: '0.78rem',
                    color: '#38bdf8'
                  }}>
                    <span>⚡ Target Engine:</span>
                    <strong>{detectPlatformFromInput(submittedToken, customPlatformTag).label}</strong>
                  </div>
                </div>

                {/* Live Verification Result Banner */}
                {verifyResult && (
                  <div style={{
                    background: verifyResult.valid ? 'rgba(118, 185, 0, 0.12)' : 'rgba(239, 68, 68, 0.12)',
                    border: verifyResult.valid ? '1px solid rgba(118, 185, 0, 0.4)' : '1px solid rgba(239, 68, 68, 0.4)',
                    borderRadius: '8px',
                    padding: '10px 14px',
                    fontSize: '0.8rem',
                    color: verifyResult.valid ? '#a3e635' : '#f87171',
                    marginBottom: '16px',
                    lineHeight: 1.45
                  }}>
                    <div>{verifyResult.valid ? '✓ ' : '✕ '}{verifyResult.message}</div>
                    {verifyResult.models && verifyResult.models.length > 0 && (
                      <div style={{ marginTop: 8, display: 'flex', flexWrap: 'wrap', gap: 5 }}>
                        <span style={{ fontSize: '0.72rem', color: '#86efac', fontWeight: 600, width: '100%' }}>
                          Confirmed Available on Provider:
                        </span>
                        {verifyResult.models.map(m => (
                          <span key={m} style={{
                            fontSize: '0.68rem',
                            background: 'rgba(16, 185, 129, 0.15)',
                            border: '1px solid rgba(16, 185, 129, 0.35)',
                            color: '#6ee7b7',
                            padding: '2px 6px',
                            borderRadius: 4,
                            fontFamily: 'monospace'
                          }}>
                            {m}
                          </span>
                        ))}
                      </div>
                    )}
                  </div>
                )}

                {/* Security Notice */}
                <div className="nemoforge-security-banner">
                  🔒 <strong>Zero-Leak Security:</strong> Tokens are saved strictly into local untracked <code>.env.local</code> and will NEVER be committed to Git.
                </div>
              </div>
            </div>

            <div className="nemoforge-modal-footer">
              <button
                type="button"
                className="nemoforge-btn-cancel"
                onClick={() => {
                  setVerifyResult(null);
                  setShowSettingsModal(false);
                }}
              >
                Close
              </button>
              <button
                type="button"
                onClick={handleVerifyConnection}
                disabled={!submittedToken.trim() || isVerifying || settingsSaving}
                style={{
                  background: 'rgba(56, 189, 248, 0.12)',
                  color: '#38bdf8',
                  border: '1px solid rgba(56, 189, 248, 0.35)',
                  padding: '9px 16px',
                  borderRadius: '8px',
                  fontSize: '0.82rem',
                  fontWeight: 600,
                  cursor: (!submittedToken.trim() || isVerifying) ? 'not-allowed' : 'pointer',
                  opacity: (!submittedToken.trim() || isVerifying) ? 0.45 : 1,
                  transition: 'all 0.15s'
                }}
              >
                {isVerifying ? 'Testing...' : '⚡ Test Connection'}
              </button>
              <button
                type="button"
                className="nemoforge-btn-save"
                onClick={handleSaveSettings}
                disabled={!submittedToken.trim() || settingsSaving || isVerifying}
                style={{
                  opacity: (!submittedToken.trim() || settingsSaving) ? 0.45 : 1,
                  cursor: (!submittedToken.trim() || settingsSaving) ? 'not-allowed' : 'pointer'
                }}
              >
                {settingsSaving ? 'Saving...' : 'Save & Connect Key'}
              </button>
            </div>
          </div>

          {/* ── Side Column: Model Registry Inspector (outside this box) ── */}
          {selectedInspectorProvider && (
            <div
              className="nemoforge-inspector-sidecolumn"
              onClick={(e) => e.stopPropagation()}
              style={{
                width: 440,
                maxWidth: '92vw',
                height: '88vh',
                maxHeight: '88vh',
                background: 'linear-gradient(180deg, #090e1a 0%, #030712 100%)',
                border: '1px solid rgba(56, 189, 248, 0.45)',
                borderRadius: 14,
                boxShadow: '0 25px 50px -12px rgba(0, 0, 0, 0.95), 0 0 35px rgba(56, 189, 248, 0.18)',
                display: 'flex',
                flexDirection: 'column',
                padding: 20,
                boxSizing: 'border-box',
                color: '#f8fafc',
                fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif'
              }}
            >
              {/* Side Column Header */}
              <div style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'flex-start',
                paddingBottom: 14,
                borderBottom: '1px solid rgba(255, 255, 255, 0.08)',
                marginBottom: 14
              }}>
                <div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    <span style={{ fontSize: '1.05rem', fontWeight: 800, color: '#f8fafc' }}>
                      🔍 Model Inspector
                    </span>
                    <span style={{
                      fontSize: '11px',
                      background: 'rgba(56, 189, 248, 0.15)',
                      border: '1px solid rgba(56, 189, 248, 0.4)',
                      color: '#38bdf8',
                      padding: '2px 8px',
                      borderRadius: 12,
                      fontWeight: 700,
                      textTransform: 'uppercase'
                    }}>
                      {selectedInspectorProvider}
                    </span>
                  </div>
                  <div style={{ fontSize: '0.78rem', color: '#94a3b8', marginTop: 3 }}>
                    Live endpoint verification & defunct model audit
                  </div>
                </div>
                <button
                  type="button"
                  onClick={() => setSelectedInspectorProvider(null)}
                  style={{
                    background: 'rgba(255, 255, 255, 0.08)',
                    border: 'none',
                    color: '#94a3b8',
                    cursor: 'pointer',
                    fontSize: '15px',
                    padding: '4px 8px',
                    borderRadius: 6
                  }}
                  title="Close side column"
                >
                  ✕
                </button>
              </div>

              {/* Filter Tabs & Search */}
              {(() => {
                const allAuditForProv = (allModelsAudit.length > 0 ? allModelsAudit : availableModelsList).filter(
                  m => m.provider === selectedInspectorProvider
                );

                const filteredBySearch = allAuditForProv.filter(m => {
                  if (!inspectorSearch.trim()) return true;
                  const q = inspectorSearch.toLowerCase();
                  return (m.name || '').toLowerCase().includes(q) || (m.model_id || '').toLowerCase().includes(q);
                });

                const workingModels = filteredBySearch.filter(m => m.is_available);
                const defunctModels = filteredBySearch.filter(m => !m.is_available);

                const displayList = inspectorFilter === 'working' ? workingModels :
                                    inspectorFilter === 'defunct' ? defunctModels : filteredBySearch;

                return (
                  <>
                    {/* Search Input */}
                    <div style={{ marginBottom: 12 }}>
                      <input
                        type="text"
                        placeholder="Search models by name or ID..."
                        value={inspectorSearch}
                        onChange={(e) => setInspectorSearch(e.target.value)}
                        style={{
                          width: '100%',
                          background: '#020617',
                          border: '1px solid rgba(255, 255, 255, 0.12)',
                          borderRadius: 8,
                          padding: '8px 12px',
                          color: '#f8fafc',
                          fontSize: '0.82rem',
                          outline: 'none',
                          boxSizing: 'border-box'
                        }}
                      />
                    </div>

                    {/* Filter Segmented Controls */}
                    <div style={{
                      display: 'flex',
                      background: '#020617',
                      borderRadius: 8,
                      padding: 3,
                      marginBottom: 14,
                      border: '1px solid rgba(255, 255, 255, 0.08)'
                    }}>
                      <button
                        type="button"
                        onClick={() => setInspectorFilter('all')}
                        style={{
                          flex: 1,
                          padding: '5px 8px',
                          fontSize: '0.74rem',
                          fontWeight: 600,
                          borderRadius: 6,
                          border: 'none',
                          cursor: 'pointer',
                          background: inspectorFilter === 'all' ? 'rgba(56, 189, 248, 0.2)' : 'transparent',
                          color: inspectorFilter === 'all' ? '#38bdf8' : '#94a3b8'
                        }}
                      >
                        All ({allAuditForProv.length})
                      </button>
                      <button
                        type="button"
                        onClick={() => setInspectorFilter('working')}
                        style={{
                          flex: 1,
                          padding: '5px 8px',
                          fontSize: '0.74rem',
                          fontWeight: 600,
                          borderRadius: 6,
                          border: 'none',
                          cursor: 'pointer',
                          background: inspectorFilter === 'working' ? 'rgba(16, 185, 129, 0.2)' : 'transparent',
                          color: inspectorFilter === 'working' ? '#34d399' : '#94a3b8'
                        }}
                      >
                        🟢 Working ({allAuditForProv.filter(m => m.is_available).length})
                      </button>
                      <button
                        type="button"
                        onClick={() => setInspectorFilter('defunct')}
                        style={{
                          flex: 1,
                          padding: '5px 8px',
                          fontSize: '0.74rem',
                          fontWeight: 600,
                          borderRadius: 6,
                          border: 'none',
                          cursor: 'pointer',
                          background: inspectorFilter === 'defunct' ? 'rgba(239, 68, 68, 0.2)' : 'transparent',
                          color: inspectorFilter === 'defunct' ? '#f87171' : '#94a3b8'
                        }}
                      >
                        🔴 Defunct ({allAuditForProv.filter(m => !m.is_available).length})
                      </button>
                    </div>

                    {/* Scrollable Model List */}
                    <div style={{
                      flex: 1,
                      overflowY: 'auto',
                      display: 'flex',
                      flexDirection: 'column',
                      gap: 10,
                      paddingRight: 4
                    }}>
                      {displayList.length === 0 ? (
                        <div style={{ textAlign: 'center', color: '#64748b', padding: '30px 10px', fontSize: '0.85rem' }}>
                          No models found matching criteria.
                        </div>
                      ) : (
                        displayList.map(m => (
                          <div
                            key={m.model_id}
                            style={{
                              background: m.is_available ? 'rgba(255, 255, 255, 0.03)' : 'rgba(239, 68, 68, 0.05)',
                              border: m.is_available ? '1px solid rgba(255, 255, 255, 0.08)' : '1px solid rgba(239, 68, 68, 0.3)',
                              borderRadius: 8,
                              padding: '10px 12px',
                              display: 'flex',
                              flexDirection: 'column',
                              gap: 6
                            }}
                          >
                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                              <span style={{ fontWeight: 700, fontSize: '0.86rem', color: m.is_available ? '#f8fafc' : '#fca5a5' }}>
                                {m.name}
                              </span>
                              <span style={{
                                fontSize: '10px',
                                padding: '2px 6px',
                                borderRadius: 4,
                                fontWeight: 700,
                                background: m.is_available ? 'rgba(16, 185, 129, 0.15)' : 'rgba(239, 68, 68, 0.15)',
                                color: m.is_available ? '#34d399' : '#f87171',
                                border: m.is_available ? '1px solid rgba(16, 185, 129, 0.3)' : '1px solid rgba(239, 68, 68, 0.3)'
                              }}>
                                {m.is_available ? '🟢 LIVE' : '🔴 DEFUNCT'}
                              </span>
                            </div>

                            <div style={{
                              display: 'flex',
                              alignItems: 'center',
                              justifyContent: 'space-between',
                              background: '#020617',
                              padding: '4px 8px',
                              borderRadius: 5,
                              fontSize: '11px',
                              fontFamily: 'monospace',
                              color: '#38bdf8'
                            }}>
                              <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                                {m.model_id}
                              </span>
                              <button
                                type="button"
                                onClick={() => handleCopyToken(m.model_id, m.model_id)}
                                style={{
                                  background: 'transparent',
                                  border: 'none',
                                  color: copiedKeyId === m.model_id ? '#34d399' : '#94a3b8',
                                  cursor: 'pointer',
                                  fontSize: '11px',
                                  padding: '1px 4px'
                                }}
                                title="Copy model ID"
                              >
                                {copiedKeyId === m.model_id ? '✓' : '📋'}
                              </button>
                            </div>

                            {/* Badges */}
                            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 5, marginTop: 2 }}>
                              {m.tier && (
                                <span style={{
                                  fontSize: '10px',
                                  background: 'rgba(56, 189, 248, 0.12)',
                                  color: '#38bdf8',
                                  padding: '1px 6px',
                                  borderRadius: 4,
                                  fontWeight: 600
                                }}>
                                  {m.tier.toUpperCase()} TIER
                                </span>
                              )}
                              {m.max_context && (
                                <span style={{
                                  fontSize: '10px',
                                  background: 'rgba(255, 255, 255, 0.06)',
                                  color: '#cbd5e1',
                                  padding: '1px 6px',
                                  borderRadius: 4
                                }}>
                                  {(m.max_context / 1000).toFixed(0)}k Context
                                </span>
                              )}
                              {m.supports_parallel_tools && (
                                <span style={{
                                  fontSize: '10px',
                                  background: 'rgba(168, 85, 247, 0.12)',
                                  color: '#c084fc',
                                  padding: '1px 6px',
                                  borderRadius: 4
                                }}>
                                  ⚡ Parallel Tools
                                </span>
                              )}
                            </div>

                            {/* Status Explanation */}
                            <div style={{
                              fontSize: '0.72rem',
                              color: m.is_available ? '#94a3b8' : '#f87171',
                              fontStyle: 'italic',
                              marginTop: 2
                            }}>
                              {m.status_reason || (m.is_available ? 'Confirmed callable on live endpoint' : 'Unavailable on provider')}
                            </div>
                          </div>
                        ))
                      )}
                    </div>
                  </>
                );
              })()}
            </div>
          )}
        </div>
      )}

      {/* ── Architecture & Documented Limitations Modal ── */}
      {showArchitectureModal && (
        <div className="nemoforge-modal-overlay" onClick={() => setShowArchitectureModal(false)}>
          <div className="nemoforge-modal-card" style={{ maxWidth: 640 }} onClick={(e) => e.stopPropagation()}>
            <div className="nemoforge-modal-header">
              <div>
                <h3 className="nemoforge-modal-title">
                  🛡️ System Architecture & Documented Limitations
                </h3>
                <div className="nemoforge-modal-subtitle">
                  Autonomous SWE Agent Benchmarking · Rigorous Engineering Boundaries
                </div>
              </div>
              <button
                className="nemoforge-modal-close"
                onClick={() => setShowArchitectureModal(false)}
                title="Close"
              >
                ✕
              </button>
            </div>

            <div className="nemoforge-modal-body" style={{ maxHeight: '68vh', overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: 16 }}>
              {/* Architecture Core */}
              <div style={{ background: 'rgba(56, 189, 248, 0.06)', border: '1px solid rgba(56, 189, 248, 0.25)', borderRadius: 10, padding: 14 }}>
                <h4 style={{ margin: '0 0 8px 0', color: '#38bdf8', fontSize: '0.92rem', display: 'flex', alignItems: 'center', gap: 6 }}>
                  ⚡ Deterministic Architecture (Why It Works)
                </h4>
                <ul style={{ margin: 0, paddingLeft: 18, fontSize: '0.82rem', color: '#cbd5e1', lineHeight: 1.6 }}>
                  <li><strong>Tree-sitter AST & Embedded SQLite WAL:</strong> Exact call graphs, method boundaries, and import chains indexed in milliseconds. Zero external graph database or approximate vector hallucination.</li>
                  <li><strong>Capability-Tier Dual Model Routing:</strong> Fast tier (<code>nvidia/Nemotron-3.5-Lightning</code> on Nebius) handles instantaneous symbol search and log parsing; Deep tier (<code>nvidia/nemotron-3-super-120b-a12b</code> on Nebius) handles multi-file architectural planning and surgical code patching.</li>
                  <li><strong>Zero-Hallucination 3-Line Anchor Patching:</strong> Code edits require exact 3-line contextual match anchors, preventing drift across dirty working trees.</li>
                  <li><strong>Automated Compiler Verification Gate:</strong> Live Maven, npm, Gradle, and pytest build sandboxes execute after patch generation to verify 0 syntax or type errors before commit.</li>
                </ul>
              </div>

              {/* Documented Limitations */}
              <div style={{ background: 'rgba(239, 68, 68, 0.06)', border: '1px solid rgba(239, 68, 68, 0.25)', borderRadius: 10, padding: 14 }}>
                <h4 style={{ margin: '0 0 8px 0', color: '#f87171', fontSize: '0.92rem', display: 'flex', alignItems: 'center', gap: 6 }}>
                  📋 Documented Limitations (Engineering Rigor)
                </h4>
                <ul style={{ margin: 0, paddingLeft: 18, fontSize: '0.82rem', color: '#cbd5e1', lineHeight: 1.6 }}>
                  <li><strong>No Unbounded Greenfield Rewrites:</strong> NemoForge is specialized for surgical 1-to-10 file bug remediation, contract synchronization, and targeted refactoring. It deliberately refuses unassisted 50-file system redesigns without human-in-the-loop review.</li>
                  <li><strong>Compiler / Test Suite Dependent:</strong> Automated verification guarantees correctness when the repository provides build tools (<code>mvn test</code>, <code>npm test</code>, <code>pytest</code>, <code>tsc</code>). In uncompiled repositories with 0 tests, verification falls back to syntax tree checks.</li>
                  <li><strong>Language Grammar Scope:</strong> Native Tree-sitter symbol slicing is supported for Java, TypeScript, JavaScript, Python, Go, and C#. Proprietary binary configurations or unsupported DSLs fall back to lexical literal search.</li>
                </ul>
              </div>

              {/* Hardware & Inference Platform */}
              <div style={{ background: 'rgba(16, 185, 129, 0.06)', border: '1px solid rgba(16, 185, 129, 0.25)', borderRadius: 10, padding: 14 }}>
                <h4 style={{ margin: '0 0 8px 0', color: '#34d399', fontSize: '0.92rem', display: 'flex', alignItems: 'center', gap: 6 }}>
                  🟢 Inference Platform & Hardware Acceleration
                </h4>
                <div style={{ fontSize: '0.82rem', color: '#cbd5e1', lineHeight: 1.5 }}>
                  Powered by <strong>NVIDIA Nemotron Models</strong> running on high-throughput <strong>Nebius Token Factory Cloud</strong> with KV Cache State Tracing for maximum token reuse and sub-second generation latency.
                </div>
              </div>
            </div>

            <div className="nemoforge-modal-footer">
              <button
                type="button"
                className="nemoforge-btn-cancel"
                onClick={() => setShowArchitectureModal(false)}
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
