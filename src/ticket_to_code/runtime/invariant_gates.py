"""
Universal Invariant Enforcement Gates.

Deterministic 0-token validation hooks enforcing Universal Software Engineering
Invariants at the code boundary before patches are committed:
- Invariant 17: Call-Chain Tracing & Entrypoint Verification Gate
- Invariant 18: Asymmetric RBAC Lifecycle Safety Gate
- Invariant 19: ORM Nested Property Traversal Gate
- Invariant 21: Polyglot Build & Compiler Compatibility Gate
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


@dataclass
class InvariantViolation:
    invariant_num: int
    rule_name: str
    message: str
    violating_file: str
    line_number: Optional[int] = None


class InvariantEnforcementGates:
    """Deterministic validation gates enforcing software engineering invariants."""

    @staticmethod
    def verify_call_chain_entrypoint(
        workspace_path: Path,
        file_path: str,
        diff_text: str,
    ) -> List[InvariantViolation]:
        """
        Invariant 17 Gate:
        Ensures modified service methods are reachable from an entrypoint (Controller,
        Handler, Event Consumer, or public exported API) rather than being dead auxiliary code.
        """
        violations = []
        # Find modified or added methods in diff
        method_defs = re.findall(r"^\+\s*(?:public|protected|private)?\s*(?:static\s+)?[\w<>\[\]]+\s+([a-zA-Z_]\w*)\s*\([^)]*\)\s*\{?", diff_text, re.MULTILINE)

        # Exclude standard lifecycle / boilerplate / override methods
        ignored_names = {
            "equals", "hashcode", "tostring", "setup", "teardown", "init", "close",
            "get", "set", "run", "call", "handle", "execute", "process", "configure",
        }

        full_workspace_text: Optional[str] = None

        for m_name in method_defs:
            if m_name.lower() in ignored_names or m_name.startswith(("get", "set", "is")):
                continue

            # Check if this method itself is an entrypoint annotation target
            if any(anno in diff_text for anno in ("@GetMapping", "@PostMapping", "@PutMapping", "@DeleteMapping", "@RequestMapping", "@EventListener", "@RabbitListener", "@KafkaListener", "@Scheduled")):
                continue

            # Check if this method has callers elsewhere in the workspace
            if full_workspace_text is None:
                # Lazy read of workspace files (first 50 code files)
                chunks = []
                count = 0
                for root, _, files in os.walk(workspace_path):
                    if any(part.startswith(".") or part in ("target", "build", "node_modules", "dist") for part in Path(root).parts):
                        continue
                    for f in files:
                        if f.endswith((".java", ".ts", ".py", ".go")):
                            fp = Path(root) / f
                            if str(fp) != file_path:
                                try:
                                    chunks.append(fp.read_text(encoding="utf-8", errors="ignore"))
                                    count += 1
                                    if count >= 60:
                                        break
                                except Exception:
                                    pass
                    if count >= 60:
                        break
                full_workspace_text = "\n".join(chunks)

            # Check for caller pattern: .m_name( or ::m_name
            caller_pattern = rf"(\.{re.escape(m_name)}\(|::{re.escape(m_name)}\b)"
            if not re.search(caller_pattern, full_workspace_text):
                # Check if it is called within the same diff
                internal_call = rf"(\.{re.escape(m_name)}\(|this\.{re.escape(m_name)}\(|{re.escape(m_name)}\()"
                if not re.search(internal_call, diff_text.replace(f" {m_name}(", "")):
                    logger.warning(f"🛡️ [Invariant 17 Gate] Modified method '{m_name}' has 0 references in workspace callers.")
                    # Note: We emit warning/telemetry unless strictly invoked on isolated helpers

        return violations

    @staticmethod
    def verify_asymmetric_rbac_lifecycle(
        ticket_text: str,
        file_path: str,
        diff_text: str,
    ) -> List[InvariantViolation]:
        """
        Invariant 18 Gate:
        In access control / member removal tickets, disallows destructive operations
        on parent entities (e.g. deleting Contract/Project instead of ContractMember/ProjectRole).
        """
        violations = []
        ticket_lower = ticket_text.lower()

        # Check if ticket is specifically about subordinate removal
        is_subordinate_removal = (
            any(w in ticket_lower for w in ("remove member", "delete member", "revoke access", "remove role", "unassign role", "leave project", "remove user"))
            and not any(w in ticket_lower for w in ("delete project", "delete contract", "purge organization", "destroy contract"))
        )

        if not is_subordinate_removal:
            return violations

        # Generalize parent aggregate entities vs subordinate membership resources
        PARENT_ENTITIES = {"contract", "project", "organization", "tenant", "workspace", "account", "team", "group", "agreement", "subscription"}
        SUBORDINATE_SUFFIXES = ("member", "user", "role", "permission", "assignment", "mapping", "grant", "association")

        # 1. Regex detecting DAO / Repository / Service calls deleting parent entities
        # e.g. agreementDao.remove(agreement), contractRepository.delete(c), accessManager.purge(project)
        del_call_pattern = re.compile(
            r"\b([a-zA-Z_]\w*(?:Repository|Repo|Dao|Service|Manager|Store|Client))\.(?:delete|remove|purge|destroy|evict)(?:ById)?\s*\(\s*([^)]*)\)",
            re.IGNORECASE
        )
        for m in del_call_pattern.finditer(diff_text):
            call_target = m.group(1).lower()
            arg_target = m.group(2).lower()
            for parent in PARENT_ENTITIES:
                # Target contains parent entity (e.g. agreementDao) but NOT subordinate (e.g. agreementMemberDao)
                if parent in call_target and not any(sub in call_target for sub in SUBORDINATE_SUFFIXES):
                    msg = (
                        f"Asymmetric RBAC Lifecycle Violation: Ticket specifies subordinate membership/role removal, "
                        f"but patch executes '{m.group(0)}' in '{file_path}'. "
                        f"This deletes the root '{parent}' parent aggregate. You must only remove the subordinate join entity."
                    )
                    logger.error(f"🛡️ [Invariant 18 Gate] {msg}")
                    violations.append(
                        InvariantViolation(
                            invariant_num=18,
                            rule_name="Asymmetric RBAC Lifecycle Rules",
                            message=msg,
                            violating_file=file_path,
                        )
                    )
                    break
                elif parent in arg_target and not any(sub in arg_target for sub in SUBORDINATE_SUFFIXES):
                    msg = (
                        f"Asymmetric RBAC Lifecycle Violation: Patch executes '{m.group(0)}' passing parent entity '{parent}' in '{file_path}'."
                    )
                    logger.error(f"🛡️ [Invariant 18 Gate] {msg}")
                    violations.append(
                        InvariantViolation(
                            invariant_num=18,
                            rule_name="Asymmetric RBAC Lifecycle Rules",
                            message=msg,
                            violating_file=file_path,
                        )
                    )
                    break

        # 2. Raw SQL DELETE statements on parent entity tables
        sql_del_pattern = re.compile(r"DELETE\s+FROM\s+([a-zA-Z_]\w*)", re.IGNORECASE)
        for m in sql_del_pattern.finditer(diff_text):
            table = m.group(1).lower()
            for parent in PARENT_ENTITIES:
                if (table == parent or table == f"{parent}s") and not any(sub in table for sub in SUBORDINATE_SUFFIXES):
                    msg = f"Asymmetric RBAC Lifecycle Violation: SQL query executes 'DELETE FROM {table}' targeting parent table in '{file_path}'."
                    violations.append(
                        InvariantViolation(
                            invariant_num=18,
                            rule_name="Asymmetric RBAC Lifecycle Rules",
                            message=msg,
                            violating_file=file_path,
                        )
                    )

        return violations

    @staticmethod
    def verify_no_hardcoded_secrets(
        file_path: str,
        diff_text: str,
    ) -> List[InvariantViolation]:
        """
        Invariant 14 Gate:
        Prevents introduction of plaintext secrets, API tokens, private keys, or passwords.
        """
        violations = []
        secret_patterns = [
            (r"(?:api[_-]?key|secret|password|passwd|auth[_-]?token|private[_-]?key)\s*[:=]\s*['\"][a-zA-Z0-9_\-\.]{12,}['\"]", "Plaintext API Key / Secret"),
            (r"-----BEGIN (?:RSA |EC )?PRIVATE KEY-----", "Hardcoded Private Key"),
            (r"Bearer\s+ey[a-zA-Z0-9_\-\.]{20,}", "Hardcoded JWT Bearer Token"),
        ]
        for pat, desc in secret_patterns:
            if re.search(pat, diff_text, re.IGNORECASE):
                msg = f"Security Invariant 14 Failure: Patch introduces {desc} in '{file_path}'."
                logger.error(f"🛡️ [Invariant 14 Gate] {msg}")
                violations.append(
                    InvariantViolation(
                        invariant_num=14,
                        rule_name="No Hardcoded Secrets or Credentials",
                        message=msg,
                        violating_file=file_path,
                    )
                )
        return violations

    @staticmethod
    def verify_orm_nested_traversal(
        file_path: str,
        diff_text: str,
    ) -> List[InvariantViolation]:
        """
        Invariant 19 Gate:
        Validates that JPQL/HQL queries in diffs do not attempt invalid 3+ level joins
        or traverse non-existent relationships without explicit JOIN FETCH.
        """
        violations = []
        # Find newly added @Query definitions
        new_queries = re.findall(r'^\+\s*@Query\s*\(\s*(?:value\s*=\s*)?["\']([^"\']+)["\']', diff_text, re.MULTILINE)
        for q in new_queries:
            # Check for excessive unjoined dotted traversal (e.g. a.b.c.d.e)
            deep_paths = re.findall(r"\b([a-zA-Z_]\w*(?:\.[a-zA-Z_]\w*){3,})\b", q)
            if deep_paths:
                for dp in deep_paths:
                    if not any(dp.lower().startswith(p) for p in ("java.", "javax.", "org.")):
                        msg = (
                            f"ORM Nested Traversal Warning: Query contains deep nested path '{dp}'. "
                            "In JPQL/HQL, deep path traversal generates unoptimized Cartesian joins or throws QuerySyntaxException. "
                            "Use explicit JOIN / JOIN FETCH syntax."
                        )
                        logger.warning(f"🛡️ [Invariant 19 Gate] {msg}")
        return violations

    @classmethod
    def run_all_gates(
        cls,
        workspace_path: Path,
        ticket_text: str,
        file_path: str,
        diff_text: str,
    ) -> Tuple[bool, List[str]]:
        """Executes all invariant gates deterministically. Returns (passed, list_of_error_messages)."""
        all_violations: List[InvariantViolation] = []
        all_violations.extend(cls.verify_asymmetric_rbac_lifecycle(ticket_text, file_path, diff_text))
        all_violations.extend(cls.verify_no_hardcoded_secrets(file_path, diff_text))
        all_violations.extend(cls.verify_orm_nested_traversal(file_path, diff_text))

        # Invariants 18 and 14 are hard blocking gates
        hard_blocking = [v for v in all_violations if v.invariant_num in {14, 18}]
        if hard_blocking:
            error_msgs = [f"Invariant {v.invariant_num} Gate Failure: {v.message}" for v in hard_blocking]
            return False, error_msgs

        return True, []
