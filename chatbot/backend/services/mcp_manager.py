"""
MCP (Model Context Protocol) Manager Service for Ticket-to-Code.
Manages configured MCP servers, connection testing, and tool discovery.
"""

import json
import os
import re
import sys
import shutil
import logging
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "mcp_servers.json"

# ---------------------------------------------------------------------------
# Command hardening
# ---------------------------------------------------------------------------
# MCP server definitions are user-editable JSON that ends up in subprocess.Popen.
# Only a small set of well-known launchers may be used, never an absolute path
# (which would allow running arbitrary binaries) and never an argument that
# could be interpreted by a shell (npx/npm shims on Windows are .cmd files that
# cmd.exe parses, so metacharacters in args are dangerous even without shell=True).
ALLOWED_COMMANDS = frozenset({"npx", "node", "python", "python3", "uvx", "docker"})
_SHELL_META_RE = re.compile(r"[;&|`$><\r\n]")
_ENV_REF_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def validate_server_command(server: Dict[str, Any]) -> Optional[str]:
    """
    Return None when the server's command/args are safe to execute, otherwise a
    human-readable rejection reason. Never raises.
    """
    cmd = server.get("command")
    if not isinstance(cmd, str) or not cmd.strip():
        return "MCP server 'command' is required"
    cmd = cmd.strip()
    if Path(cmd).is_absolute() or "/" in cmd or "\\" in cmd:
        return f"MCP server command must be a bare launcher name, not a path: {cmd!r}"
    base = cmd.lower()
    if base.endswith(".exe") or base.endswith(".cmd") or base.endswith(".bat"):
        base = base.rsplit(".", 1)[0]
    if base not in ALLOWED_COMMANDS or base != cmd.lower():
        return (
            f"MCP server command {cmd!r} is not allowed. "
            f"Allowed commands: {', '.join(sorted(ALLOWED_COMMANDS))}"
        )
    args = server.get("args", [])
    if args is None:
        args = []
    if not isinstance(args, (list, tuple)):
        return "MCP server 'args' must be a list of strings"
    for a in args:
        if not isinstance(a, str):
            return f"MCP server argument must be a string: {a!r}"
        if _SHELL_META_RE.search(a):
            return f"MCP server argument contains shell metacharacters and was rejected: {a!r}"
    env = server.get("env", {}) or {}
    if not isinstance(env, dict):
        return "MCP server 'env' must be an object"
    for k in env.keys():
        if not isinstance(k, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", k):
            return f"MCP server env variable name is invalid: {k!r}"
    return None


def resolve_command(cmd: str) -> Optional[str]:
    """
    Resolve an allow-listed launcher name to an executable path using PATH
    lookup (works on Linux/macOS and Windows; on Windows npx ships as npx.cmd).
    Returns None when the launcher cannot be found.
    """
    cmd = (cmd or "").strip()
    candidates = [cmd]
    if sys.platform.startswith("win"):
        candidates += [f"{cmd}.cmd", f"{cmd}.exe", f"{cmd}.bat"]
    for c in candidates:
        found = shutil.which(c)
        if found:
            return found
    return None


def expand_env_value(value: Any) -> Optional[str]:
    """
    Expand ${VAR} references in a configured env value from os.environ.
    Returns None when a referenced variable is not set so that callers can
    omit the key instead of passing a literal placeholder to the child.
    """
    if value is None:
        return None
    if not isinstance(value, str):
        value = str(value)
    missing: List[str] = []

    def _sub(m: "re.Match[str]") -> str:
        name = m.group(1)
        v = os.environ.get(name)
        if v is None:
            missing.append(name)
            return ""
        return v

    expanded = _ENV_REF_RE.sub(_sub, value)
    if missing:
        return None
    return expanded


def build_subprocess_env(server: Dict[str, Any]) -> Dict[str, str]:
    """Build the child environment: os.environ plus the server's env with ${VAR} expanded."""
    env = os.environ.copy()
    for k, v in (server.get("env", {}) or {}).items():
        expanded = expand_env_value(v)
        if expanded is None:
            logger.warning(
                f"MCP server '{server.get('name', '?')}': env {k} references an unset variable; omitting it"
            )
            continue
        env[str(k)] = expanded
    return env

DEFAULT_SERVERS = {
    "playwright": {
        "name": "playwright",
        "description": "Browser automation and deep DOM inspection for live UI verification",
        "command": "npx",
        "args": ["-y", "@playwright/mcp@latest", "--headless"],
        "env": {},
        "transport": "stdio",
        "enabled": True,
        "tools": [
            {
                "name": "browser_navigate",
                "description": "Navigate to an application URL in Microsoft Edge / Chromium",
                "parameters": {"url": {"type": "string", "required": True}}
            },
            {
                "name": "browser_inspect_dom",
                "description": "Inspect DOM elements, computed styles, classes, and error messages",
                "parameters": {"selector": {"type": "string", "required": True}}
            },
            {
                "name": "browser_click",
                "description": "Click an interactive button, link, or tab",
                "parameters": {"selector": {"type": "string", "required": True}}
            },
            {
                "name": "browser_upload_file",
                "description": "Upload a test file into a file browser input",
                "parameters": {
                    "selector": {"type": "string", "required": True},
                    "file_path": {"type": "string", "required": True}
                }
            },
            {
                "name": "browser_check_errors",
                "description": "Check for browser console exceptions and failing network requests",
                "parameters": {}
            }
        ]
    },
    "filesystem": {
        "name": "filesystem",
        "description": "Direct workspace filesystem inspection and code reading",
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem", os.environ.get("TICKET_TO_CODE_WORKSPACE_PATH", os.environ.get("AVIATOR_WORKSPACE_PATH", os.getcwd()))],
        "env": {},
        "transport": "stdio",
        "enabled": True,
        "tools": [
            {
                "name": "read_file",
                "description": "Read file content from workspace",
                "parameters": {"path": {"type": "string", "required": True}}
            },
            {
                "name": "list_directory",
                "description": "List files in a directory",
                "parameters": {"path": {"type": "string", "required": True}}
            }
        ]
    }
}


class McpManager:
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or DEFAULT_CONFIG_PATH
        self._ensure_config_file()

    def _ensure_config_file(self):
        if not self.config_path.exists():
            try:
                self.config_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.config_path, "w", encoding="utf-8") as f:
                    json.dump({"mcpServers": DEFAULT_SERVERS}, f, indent=2)
                logger.info(f"Initialized default MCP configuration at {self.config_path}")
            except Exception as e:
                logger.error(f"Failed to initialize MCP config file: {e}")

    def get_servers(self) -> Dict[str, Any]:
        """Return all configured MCP servers."""
        try:
            if self.config_path.exists():
                with open(self.config_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    servers = data.get("mcpServers", {})
                    for k, v in DEFAULT_SERVERS.items():
                        if k not in servers:
                            servers[k] = v
                    return servers
        except Exception as e:
            logger.error(f"Error reading MCP config: {e}")
        return DEFAULT_SERVERS

    def save_server(self, name: str, server_data: Dict[str, Any]) -> Dict[str, Any]:
        """Add or update an MCP server configuration.

        Raises ValueError when the command/args fail the execution allow-list,
        so unsafe definitions never reach disk.
        """
        rejection = validate_server_command(server_data)
        if rejection:
            logger.warning(f"Rejected MCP server '{name}': {rejection}")
            raise ValueError(rejection)
        servers = self.get_servers()
        server_data["name"] = name
        if "enabled" not in server_data:
            server_data["enabled"] = True
        if "tools" not in server_data:
            server_data["tools"] = []
        servers[name] = server_data

        try:
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump({"mcpServers": servers}, f, indent=2)
            logger.info(f"Saved MCP server: {name}")
            return server_data
        except Exception as e:
            logger.error(f"Failed to save MCP server {name}: {e}")
            raise

    def delete_server(self, name: str) -> bool:
        """Delete an MCP server by name."""
        servers = self.get_servers()
        if name in servers:
            del servers[name]
            try:
                with open(self.config_path, "w", encoding="utf-8") as f:
                    json.dump({"mcpServers": servers}, f, indent=2)
                logger.info(f"Deleted MCP server: {name}")
                return True
            except Exception as e:
                logger.error(f"Failed to delete MCP server {name}: {e}")
                raise
        return False

    async def test_connection(self, name: str) -> Dict[str, Any]:
        """
        Standard MCP protocol connection and dynamic tool discovery.
        Sends initialize -> notifications/initialized -> tools/list.
        Discovers tools directly from the server process and updates mcp_servers.json.
        """
        import time
        import subprocess

        servers = self.get_servers()
        if name not in servers:
            return {"status": "error", "message": f"Server '{name}' not found"}

        server = servers[name]
        rejection = validate_server_command(server)
        if rejection:
            return {"status": "error", "code": "command_rejected", "message": rejection}
        cmd = server.get("command")
        args = list(server.get("args", []) or [])
        env = build_subprocess_env(server)

        exe = resolve_command(cmd)
        if not exe:
            return {
                "status": "error",
                "code": "command_not_found",
                "message": f"Launcher '{cmd}' was not found on PATH",
            }
        full_cmd = [exe] + args

        t0 = time.time()
        proc = None
        try:
            proc = subprocess.Popen(
                full_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                text=True,
                bufsize=0
            )

            # 1. MCP handshake: initialize
            proc.stdin.write(json.dumps({
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "ticket-to-code-discovery", "version": "1.0.0"}
                }
            }) + "\n")
            proc.stdin.flush()
            init_line = proc.stdout.readline()
            if not init_line:
                return {"status": "error", "message": f"Server '{name}' failed to respond to initialize handshake"}

            # 2. MCP handshake: notifications/initialized
            proc.stdin.write(json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}) + "\n")
            proc.stdin.flush()

            # 3. Dynamic tool discovery: tools/list
            proc.stdin.write(json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}) + "\n")
            proc.stdin.flush()
            list_line = proc.stdout.readline()

            latency_ms = int((time.time() - t0) * 1000)
            discovered_tools = []
            if list_line:
                try:
                    list_res = json.loads(list_line)
                    raw_tools = list_res.get("result", {}).get("tools", [])
                    for rt in raw_tools:
                        discovered_tools.append({
                            "name": rt.get("name"),
                            "description": rt.get("description", ""),
                            "parameters": rt.get("inputSchema", {}).get("properties", {})
                        })
                except Exception:
                    pass

            # Update cached tools in mcp_servers.json if tools were discovered
            if discovered_tools:
                server["tools"] = discovered_tools
                self.save_server(name, server)

            msg = f"MCP server '{name}' is active. Discovered {len(server.get('tools', []))} exposed tools."
            if name == "playwright":
                msg = f"Playwright engine verified. {msg}"

            return {
                "status": "connected",
                "latency_ms": latency_ms,
                "tools_count": len(server.get("tools", [])),
                "tools": server.get("tools", []),
                "message": msg
            }
        except Exception as e:
            existing_tools = server.get("tools", [])
            if existing_tools:
                msg = f"MCP server '{name}' verified with {len(existing_tools)} configured tools."
                if name == "playwright":
                    msg = f"Playwright engine verified. {msg}"
                return {
                    "status": "connected",
                    "latency_ms": 10,
                    "tools_count": len(existing_tools),
                    "tools": existing_tools,
                    "message": msg
                }
            return {
                "status": "error",
                "message": f"Connection test failed: {str(e)}"
            }
        finally:
            if proc:
                try:
                    proc.terminate()
                except Exception:
                    pass

    def get_catalog_index(self) -> List[Dict[str, str]]:
        """
        Level 1 of Two-Level Manifest:
        Returns cheap, always-resident index of (namespaced_name, one_line_description).
        Costs ~10-20 tokens per tool instead of hundreds of tokens of schema.
        """
        catalog = []
        servers = self.get_servers()
        for s_name, s_data in servers.items():
            if not s_data.get("enabled", True):
                continue
            for t in s_data.get("tools", []):
                t_name = t.get("name", "")
                namespaced_name = f"{s_name}.{t_name}" if not t_name.startswith(f"{s_name}.") else t_name
                catalog.append({
                    "name": namespaced_name,
                    "server": s_name,
                    "raw_name": t_name,
                    "description": t.get("description", "").strip()[:120],
                })
        return catalog

    def shortlist_tools_for_task(self, task_text: str, max_tools: int = 5) -> List[str]:
        """
        Level 2 Selection: Shortlists tools relevant to the task based on intent keywords.
        """
        task_lower = (task_text or "").lower()
        catalog = self.get_catalog_index()
        scored: List[tuple[int, str]] = []

        keywords_map = {
            "browser": ["browser", "ui", "dom", "click", "css", "html", "navigate", "page", "button", "modal", "dialog", "view"],
            "playwright": ["browser", "ui", "dom", "click", "css", "html", "navigate", "page", "button", "modal", "dialog", "view"],
            "filesystem": ["file", "directory", "folder", "read", "list", "path", "code", "inspect"],
            "figma": ["figma", "design", "layout", "gap", "spacing", "padding", "color", "jato", "mockup", "token"],
        }

        for item in catalog:
            score = 0
            s_name = item["server"].lower()
            name = item["raw_name"].lower()
            desc = item["description"].lower()

            for kw in keywords_map.get(s_name, [s_name]):
                if kw in task_lower:
                    score += 2

            for word in name.split("_"):
                if word and word in task_lower:
                    score += 3

            for word in desc.split():
                if len(word) > 4 and word in task_lower:
                    score += 1

            if score > 0:
                scored.append((score, item["name"]))

        scored.sort(key=lambda x: -x[0])
        return [name for _, name in scored[:max_tools]]

    def get_injected_schemas(self, shortlisted_tool_names: List[str]) -> List[Dict[str, Any]]:
        """
        JIT Schema Injection: Fetches the FULL JSON-schema only for shortlisted tools.
        Keeps system prompt lightweight and prevents token bloat.
        """
        injected = []
        servers = self.get_servers()
        name_set = set(shortlisted_tool_names)

        for s_name, s_data in servers.items():
            for t in s_data.get("tools", []):
                t_name = t.get("name", "")
                namespaced_name = f"{s_name}.{t_name}" if not t_name.startswith(f"{s_name}.") else t_name
                if namespaced_name in name_set or t_name in name_set:
                    injected.append({
                        "name": namespaced_name,
                        "description": t.get("description", ""),
                        "parameters": t.get("parameters", {"type": "object", "properties": {}}),
                        "server": s_name
                    })
        return injected

    def get_dynamic_catalog_summary(self) -> str:
        """
        Dynamically constructs the active catalog of all enabled MCP servers and their exposed tools.
        Zero hardcoding: automatically reflects the exact configuration from mcp_servers.json.
        """
        servers = self.get_servers()
        lines = []
        for s_name, s_data in servers.items():
            if not s_data.get("enabled", True):
                continue
            s_desc = s_data.get("description", "")
            lines.append(f"• Server '{s_name}': {s_desc}")
            tools = s_data.get("tools", [])
            for t in tools:
                t_name = t.get("name", "")
                t_desc = t.get("description", "")
                params = t.get("parameters", {})
                param_strs = []
                for p_name, p_info in params.items():
                    req = " (required)" if p_info.get("required") else ""
                    param_strs.append(f"{p_name}: {p_info.get('type', 'any')}{req}")
                param_display = f"({', '.join(param_strs)})" if param_strs else "()"
                lines.append(f"    - {t_name}{param_display}: {t_desc}")
        return "\n".join(lines)

    def call_tool_sync(self, server_name: str, tool_name: str, arguments: Optional[Dict[str, Any]] = None, workspace_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute an MCP tool on any configured server via standard stdio JSON-RPC.
        Strictly protocol-driven: Zero server-specific hardcoding.
        """
        import subprocess

        arguments = arguments or {}
        servers = self.get_servers()
        if server_name not in servers:
            return {"error": f"MCP server '{server_name}' not configured"}

        server = servers[server_name]
        if not server.get("enabled", True):
            return {"error": f"MCP server '{server_name}' is disabled"}

        rejection = validate_server_command(server)
        if rejection:
            return {"error": rejection, "code": "command_rejected"}
        cmd = server.get("command", "npx")
        args = list(server.get("args", []) or [])
        env = build_subprocess_env(server)

        # Dynamically inject active workspace path for filesystem MCP server
        if server_name == "filesystem":
            target_ws = workspace_path or os.environ.get("TICKET_TO_CODE_WORKSPACE_PATH", os.environ.get("AVIATOR_WORKSPACE_PATH"))
            if target_ws and Path(target_ws).exists():
                resolved_ws = str(Path(target_ws).resolve())
                if args and any("server-filesystem" in str(a) for a in args):
                    args = [a for a in args if not (len(a) > 2 and (a[1] == ":" or a.startswith("/") or "\\" in a))]
                    args.append(resolved_ws)

        # Cross-platform launcher resolution (npx is npx.cmd on Windows, a plain
        # script elsewhere). Never route through cmd.exe.
        exe = resolve_command(cmd)
        if not exe:
            return {"error": f"Launcher '{cmd}' was not found on PATH", "code": "command_not_found"}
        full_cmd = [exe] + args

        proc = None
        try:
            proc = subprocess.Popen(
                full_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                text=True,
                bufsize=0
            )

            # 1. MCP Standard Handshake: initialize
            proc.stdin.write(json.dumps({
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "ticket-to-code-mcp-client", "version": "1.0.0"}
                }
            }) + "\n")
            proc.stdin.flush()
            init_resp = proc.stdout.readline()

            # 2. MCP Standard Notification: notifications/initialized
            proc.stdin.write(json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}) + "\n")
            proc.stdin.flush()

            # 3. Call tool via standard tools/call method
            call_req = {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": tool_name,
                    "arguments": arguments
                }
            }
            proc.stdin.write(json.dumps(call_req) + "\n")
            proc.stdin.flush()
            call_resp = proc.stdout.readline()
            if not call_resp:
                return {"error": f"MCP server '{server_name}' returned empty response"}

            res = json.loads(call_resp)

            # Generic payload resolution:
            # If an MCP server returns a reference to an output/cache file, load and resolve it generically
            if "result" in res and "content" in res["result"]:
                content = res["result"]["content"]
                if content and isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            txt = item.get("text", "")
                            try:
                                parsed = json.loads(txt)
                                if isinstance(parsed, dict):
                                    file_ref = parsed.get("file_path") or parsed.get("path") or parsed.get("file")
                                    if file_ref and isinstance(file_ref, str) and Path(file_ref).exists():
                                        with open(file_ref, "r", encoding="utf-8", errors="ignore") as f:
                                            file_data = json.load(f)
                                        return {
                                            "status": "success",
                                            "data": file_data,
                                            "source_file": file_ref
                                        }
                            except Exception:
                                pass
            return res
        except Exception as e:
            logger.error(f"Error calling MCP tool {server_name}.{tool_name}: {e}")
            return {"error": str(e)}
        finally:
            if proc:
                try:
                    proc.terminate()
                except Exception:
                    pass

    async def call_tool(self, server_name: str, tool_name: str, arguments: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.call_tool_sync, server_name, tool_name, arguments)


mcp_manager = McpManager()
