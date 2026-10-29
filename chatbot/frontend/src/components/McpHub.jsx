import React, { useState, useEffect } from 'react';
import './McpHub.css';

const API_BASE_URL = 'http://localhost:8002';

export default function McpHub() {
  const [servers, setServers] = useState({});
  const [loading, setLoading] = useState(true);
  const [testingServer, setTestingServer] = useState(null);
  const [testResults, setTestResults] = useState({});
  const [showAddModal, setShowAddModal] = useState(false);
  const [newServer, setNewServer] = useState({
    name: '',
    description: '',
    command: 'npx',
    args: '',
    transport: 'stdio',
    env: ''
  });

  const fetchServers = async () => {
    try {
      setLoading(true);
      const res = await fetch(`${API_BASE_URL}/api/mcp/servers`);
      if (res.ok) {
        const data = await res.json();
        setServers(data || {});
      }
    } catch (err) {
      console.error('Failed to load MCP servers:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchServers();
  }, []);

  const handleTestConnection = async (serverName) => {
    try {
      setTestingServer(serverName);
      const res = await fetch(`${API_BASE_URL}/api/mcp/servers/${serverName}/test`, {
        method: 'POST'
      });
      const data = await res.json();
      setTestResults(prev => ({
        ...prev,
        [serverName]: data
      }));
    } catch (err) {
      setTestResults(prev => ({
        ...prev,
        [serverName]: { status: 'error', message: err.message }
      }));
    } finally {
      setTestingServer(null);
    }
  };

  const handleDeleteServer = async (serverName) => {
    if (!window.confirm(`Delete MCP server '${serverName}'?`)) return;
    try {
      const res = await fetch(`${API_BASE_URL}/api/mcp/servers/${serverName}`, {
        method: 'DELETE'
      });
      if (res.ok) {
        fetchServers();
      }
    } catch (err) {
      alert(`Failed to delete server: ${err.message}`);
    }
  };

  const handleAddServer = async (e) => {
    e.preventDefault();
    if (!newServer.name.trim()) {
      alert('Server name is required');
      return;
    }

    try {
      const parsedArgs = newServer.args
        .split(' ')
        .map(a => a.trim())
        .filter(Boolean);

      let parsedEnv = {};
      if (newServer.env.trim()) {
        try {
          parsedEnv = JSON.parse(newServer.env);
        } catch (pe) {
          alert('Environment variables must be valid JSON');
          return;
        }
      }

      const payload = {
        name: newServer.name.trim(),
        description: newServer.description.trim(),
        command: newServer.command.trim(),
        args: parsedArgs,
        transport: newServer.transport,
        env: parsedEnv,
        enabled: true,
        tools: []
      };

      const res = await fetch(`${API_BASE_URL}/api/mcp/servers`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      if (res.ok) {
        setShowAddModal(false);
        setNewServer({
          name: '',
          description: '',
          command: 'npx',
          args: '',
          transport: 'stdio',
          env: ''
        });
        fetchServers();
      } else {
        const err = await res.json();
        alert(`Error: ${err.detail || 'Failed to save MCP server'}`);
      }
    } catch (err) {
      alert(`Failed to save server: ${err.message}`);
    }
  };

  const serverEntries = Object.entries(servers);
  const totalTools = serverEntries.reduce(
    (acc, [_, s]) => acc + ((s.tools || []).length),
    0
  );

  return (
    <div className="mcp-hub">
      <div className="mcp-hub-header">
        <div>
          <h2>🔌 Model Context Protocol (MCP) Hub</h2>
          <p className="mcp-subtitle">
            Configure external tool servers for live browser execution, DOM inspection, and autonomous workflow capabilities.
          </p>
        </div>
        <div className="mcp-header-actions">
          <span className="mcp-badge tools-badge">
            ⚡ {totalTools} Available Tools
          </span>
          <span className="mcp-badge servers-badge">
            🖥️ {serverEntries.length} Configured Servers
          </span>
          <button
            className="mcp-add-btn"
            onClick={() => setShowAddModal(true)}
          >
            + Add MCP Server
          </button>
        </div>
      </div>

      {loading ? (
        <div className="mcp-loading">
          <div className="mcp-spinner"></div>
          <p>Loading MCP configuration...</p>
        </div>
      ) : serverEntries.length === 0 ? (
        <div className="mcp-empty-state">
          <p>No MCP servers configured yet.</p>
          <button onClick={() => setShowAddModal(true)}>Add your first MCP Server</button>
        </div>
      ) : (
        <div className="mcp-server-grid">
          {serverEntries.map(([key, server]) => {
            const isTesting = testingServer === key;
            const result = testResults[key];
            const isConnected = result?.status === 'connected';
            const isError = result?.status === 'error';

            return (
              <div key={key} className={`mcp-card ${isConnected ? 'connected' : ''}`}>
                <div className="mcp-card-header">
                  <div className="mcp-card-title-row">
                    <span className="mcp-server-icon">
                      {key === 'playwright' ? '🎭' : key === 'filesystem' ? '📁' : '🔌'}
                    </span>
                    <div>
                      <h3 className="mcp-server-name">{server.name || key}</h3>
                      <span className="mcp-transport-tag">{server.transport || 'stdio'}</span>
                    </div>
                  </div>
                  <div className="mcp-status-pill">
                    {isTesting ? (
                      <span className="status-testing">⏳ Testing...</span>
                    ) : isConnected ? (
                      <span className="status-connected">🟢 Connected ({result.latency_ms || 15}ms)</span>
                    ) : isError ? (
                      <span className="status-error">🔴 Error</span>
                    ) : (
                      <span className="status-idle">⚪ Ready</span>
                    )}
                  </div>
                </div>

                <p className="mcp-server-desc">
                  {server.description || 'No description provided.'}
                </p>

                <div className="mcp-command-box">
                  <code>{server.command} {(server.args || []).join(' ')}</code>
                </div>

                {result && (
                  <div className={`mcp-test-feedback ${result.status}`}>
                    <span>{result.message}</span>
                  </div>
                )}

                {/* Exposed Tools */}
                <div className="mcp-tools-section">
                  <div className="mcp-tools-title">
                    <span>Exposed Tools ({server.tools?.length || 0})</span>
                  </div>
                  <div className="mcp-tools-list">
                    {(server.tools || []).map((t, idx) => (
                      <div key={idx} className="mcp-tool-pill" title={t.description}>
                        <span className="tool-dot">●</span>
                        <span className="tool-name">{t.name}</span>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="mcp-card-footer">
                  <button
                    className="mcp-test-btn"
                    onClick={() => handleTestConnection(key)}
                    disabled={isTesting}
                  >
                    {isTesting ? 'Testing...' : '⚡ Test Connection'}
                  </button>
                  <button
                    className="mcp-delete-btn"
                    onClick={() => handleDeleteServer(key)}
                  >
                    🗑️ Delete
                  </button>
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* Add MCP Server Modal */}
      {showAddModal && (
        <div className="mcp-modal-overlay">
          <div className="mcp-modal">
            <div className="mcp-modal-header">
              <h3>Add New MCP Server</h3>
              <button className="mcp-modal-close" onClick={() => setShowAddModal(false)}>✕</button>
            </div>
            <form onSubmit={handleAddServer} className="mcp-modal-form">
              <label>
                Server Name *
                <input
                  type="text"
                  placeholder="e.g. playwright-inspector"
                  value={newServer.name}
                  onChange={e => setNewServer({ ...newServer, name: e.target.value })}
                  required
                />
              </label>

              <label>
                Description
                <input
                  type="text"
                  placeholder="e.g. Browser automation for DOM inspection"
                  value={newServer.description}
                  onChange={e => setNewServer({ ...newServer, description: e.target.value })}
                />
              </label>

              <label>
                Command *
                <input
                  type="text"
                  placeholder="e.g. npx, python, or docker"
                  value={newServer.command}
                  onChange={e => setNewServer({ ...newServer, command: e.target.value })}
                  required
                />
              </label>

              <label>
                Arguments (space-separated)
                <input
                  type="text"
                  placeholder="e.g. -y @modelcontextprotocol/server-playwright"
                  value={newServer.args}
                  onChange={e => setNewServer({ ...newServer, args: e.target.value })}
                />
              </label>

              <label>
                Transport
                <select
                  value={newServer.transport}
                  onChange={e => setNewServer({ ...newServer, transport: e.target.value })}
                >
                  <option value="stdio">stdio (standard I/O)</option>
                  <option value="sse">sse (Server-Sent Events)</option>
                </select>
              </label>

              <label>
                Environment Variables (JSON optional)
                <textarea
                  rows="3"
                  placeholder='{"BROWSER": "msedge", "HEADLESS": "false"}'
                  value={newServer.env}
                  onChange={e => setNewServer({ ...newServer, env: e.target.value })}
                />
              </label>

              <div className="mcp-modal-footer">
                <button type="button" className="mcp-btn-secondary" onClick={() => setShowAddModal(false)}>
                  Cancel
                </button>
                <button type="submit" className="mcp-btn-primary">
                  Add MCP Server
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}
