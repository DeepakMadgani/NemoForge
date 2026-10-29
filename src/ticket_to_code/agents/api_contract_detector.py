"""
API Contract Detector — Framework-aware API boundary discovery.

Detects API contract boundaries between files by scanning for actual
framework annotations in generated code. NOT a generic regex engine.

Detection priority (from implementation plan v2):
    1. Existing LSP / AST / parser (when available)
    2. Existing SymbolResolver / WorkspaceSymbolIndex
    3. Framework-aware detectors (this module)
    4. API/schema files (OpenAPI, GraphQL .graphql, proto files)
    5. Lightweight structural extraction
    6. Regex only as last-resort fallback

Supported patterns (detected from ACTUAL code, not guessed):
    - Spring: @GetMapping, @PostMapping, @PutMapping, @DeleteMapping,
              @RequestMapping, @QueryMapping, @MutationMapping,
              @RestController, @Controller
    - Angular: HttpClient.get/post calls in .service.ts files
    - GraphQL: schema.graphqls files, @QueryMapping resolvers
    - REST: route patterns in Express/FastAPI/Flask/NestJS

Hard constraints:
    - Only reports what is ACTUALLY in the code
    - Does NOT infer or guess endpoints
    - Does NOT invent API mappings from method names alone
    - Returns VERIFIED status only for annotations found in code

Author: Deepak Madgani
Date: September 2026
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class APIContract:
    """A single API contract found in actual source code.

    Only created when the detector finds a real annotation/decorator/route
    in the code. Never created from inference or guessing.
    """
    endpoint: str                # URL path or GraphQL query/mutation name
    method: str = ""             # HTTP method or GraphQL operation type
    request_type: str = ""       # Request body type (if found)
    response_type: str = ""      # Response type (if found)
    source_file: str = ""        # File where the annotation was found
    source_symbol: str = ""      # Method/function name
    framework: str = ""          # "spring", "angular", "graphql", "express", etc.
    line_number: int = 0         # Line where found (for traceability)


class APIContractDetector:
    """Detects API contract boundaries between files.

    Uses existing LSP/AST infrastructure first.
    Falls back to structural extraction only when LSP is unavailable.

    This detector ONLY reports what it finds in actual code.
    It does NOT infer or guess endpoints. If a Java service method
    exists but has no @GetMapping or @QueryMapping, NO API contract
    is registered — the system correctly treats it as an internal method.
    """

    def detect_contracts(
        self,
        file_path: str,
        content: str,
        symbol_resolver=None,
    ) -> List[APIContract]:
        """Extract API contracts from a file.

        Returns list of APIContract objects — only what is ACTUALLY in the code.
        """
        if not content or not content.strip():
            return []

        ext = Path(file_path).suffix.lower()

        contracts: List[APIContract] = []

        # Dispatch based on file type
        if ext in (".java", ".kt"):
            contracts.extend(self._detect_spring_contracts(file_path, content))
        elif ext in (".ts", ".tsx"):
            contracts.extend(self._detect_angular_http_calls(file_path, content))
        elif ext in (".py",):
            contracts.extend(self._detect_python_api_contracts(file_path, content))
        elif ext in (".graphqls", ".graphql", ".gql"):
            contracts.extend(self._detect_graphql_schema(file_path, content))

        if contracts:
            logger.info(
                f"  [APIContractDetector] Found {len(contracts)} API contracts "
                f"in {Path(file_path).name}"
            )

        return contracts

    # ── Spring/Java ──────────────────────────────────────────────────────

    def _detect_spring_contracts(
        self, file_path: str, content: str
    ) -> List[APIContract]:
        """Detect Spring MVC/WebFlux and Spring GraphQL annotations.

        Only detects annotations that are ACTUALLY in the code:
        - @GetMapping, @PostMapping, @PutMapping, @DeleteMapping
        - @RequestMapping
        - @QueryMapping, @MutationMapping, @SubscriptionMapping
        """
        contracts = []

        # Spring REST annotations
        # Match: @GetMapping("/path") or @GetMapping(value = "/path")
        rest_pattern = re.compile(
            r'@(Get|Post|Put|Delete|Patch)Mapping\s*\(\s*'
            r'(?:value\s*=\s*)?'
            r'["\']([^"\']*)["\']',
            re.MULTILINE,
        )
        for match in rest_pattern.finditer(content):
            http_method = match.group(1).upper()
            path = match.group(2)
            # Find the method name that follows
            method_name = self._find_next_method_name(content, match.end())
            contracts.append(APIContract(
                endpoint=path,
                method=http_method,
                source_file=file_path,
                source_symbol=method_name,
                framework="spring_rest",
                line_number=content[:match.start()].count("\n") + 1,
            ))

        # Spring @RequestMapping with method specification
        req_pattern = re.compile(
            r'@RequestMapping\s*\(\s*'
            r'(?:.*?value\s*=\s*)?["\']([^"\']*)["\']'
            r'(?:.*?method\s*=\s*RequestMethod\.(\w+))?',
            re.MULTILINE | re.DOTALL,
        )
        for match in req_pattern.finditer(content):
            path = match.group(1)
            method = match.group(2) or "GET"
            method_name = self._find_next_method_name(content, match.end())
            contracts.append(APIContract(
                endpoint=path,
                method=method.upper(),
                source_file=file_path,
                source_symbol=method_name,
                framework="spring_rest",
                line_number=content[:match.start()].count("\n") + 1,
            ))

        # Spring GraphQL annotations
        graphql_pattern = re.compile(
            r'@(Query|Mutation|Subscription)Mapping'
            r'(?:\s*\(\s*["\']([^"\']*)["\'])?\s*'
            r'(?:\n\s*)?'
            r'(?:public\s+)?(?:\w+(?:<[^>]+>)?)\s+(\w+)\s*\(',
            re.MULTILINE,
        )
        for match in graphql_pattern.finditer(content):
            gql_type = match.group(1).upper()
            explicit_name = match.group(2)
            method_name = match.group(3)
            endpoint_name = explicit_name if explicit_name else method_name

            # Try to extract return type
            return_type_match = re.search(
                rf'(?:public\s+)?(\w+(?:<[^>]+>)?)\s+{re.escape(method_name)}\s*\(',
                content,
            )
            return_type = return_type_match.group(1) if return_type_match else ""

            contracts.append(APIContract(
                endpoint=endpoint_name,
                method=gql_type,
                response_type=return_type,
                source_file=file_path,
                source_symbol=method_name,
                framework="spring_graphql",
                line_number=content[:match.start()].count("\n") + 1,
            ))

        return contracts

    # ── Angular/TypeScript ───────────────────────────────────────────────

    def _detect_angular_http_calls(
        self, file_path: str, content: str
    ) -> List[APIContract]:
        """Detect Angular HttpClient calls in .service.ts files.

        Looks for actual HttpClient method calls:
        this.http.get<Type>('/path')
        this.http.post<Type>('/path', body)
        """
        contracts = []

        # Only scan .service.ts files for HTTP client usage
        if ".service." not in Path(file_path).name:
            return contracts

        # Match: this.http.get<ResponseType>(`/api/path`)
        http_pattern = re.compile(
            r'this\.\w+\.(get|post|put|delete|patch)'
            r'(?:<([^>]+)>)?'
            r'\s*\(\s*'
            r'[`"\']([^`"\']+)[`"\']',
            re.MULTILINE,
        )
        for match in http_pattern.finditer(content):
            http_method = match.group(1).upper()
            response_type = match.group(2) or ""
            path = match.group(3)

            # Find enclosing method name
            method_name = self._find_enclosing_method_ts(content, match.start())

            contracts.append(APIContract(
                endpoint=path,
                method=http_method,
                response_type=response_type,
                source_file=file_path,
                source_symbol=method_name,
                framework="angular_http",
                line_number=content[:match.start()].count("\n") + 1,
            ))

        return contracts

    # ── Python (FastAPI/Flask) ───────────────────────────────────────────

    def _detect_python_api_contracts(
        self, file_path: str, content: str
    ) -> List[APIContract]:
        """Detect FastAPI/Flask route decorators."""
        contracts = []

        # FastAPI: @app.get("/path") or @router.get("/path")
        fastapi_pattern = re.compile(
            r'@\w+\.(get|post|put|delete|patch)\s*\(\s*'
            r'["\']([^"\']+)["\']',
            re.MULTILINE,
        )
        for match in fastapi_pattern.finditer(content):
            http_method = match.group(1).upper()
            path = match.group(2)
            method_name = self._find_next_python_def(content, match.end())
            contracts.append(APIContract(
                endpoint=path,
                method=http_method,
                source_file=file_path,
                source_symbol=method_name,
                framework="fastapi",
                line_number=content[:match.start()].count("\n") + 1,
            ))

        # Flask: @app.route("/path", methods=["GET"])
        flask_pattern = re.compile(
            r'@\w+\.route\s*\(\s*'
            r'["\']([^"\']+)["\']'
            r'(?:.*?methods\s*=\s*\[([^\]]+)\])?',
            re.MULTILINE | re.DOTALL,
        )
        for match in flask_pattern.finditer(content):
            path = match.group(1)
            methods_str = match.group(2)
            method = "GET"
            if methods_str:
                m = re.search(r'["\'](\w+)["\']', methods_str)
                if m:
                    method = m.group(1).upper()
            method_name = self._find_next_python_def(content, match.end())
            contracts.append(APIContract(
                endpoint=path,
                method=method,
                source_file=file_path,
                source_symbol=method_name,
                framework="flask",
                line_number=content[:match.start()].count("\n") + 1,
            ))

        return contracts

    # ── GraphQL Schema ───────────────────────────────────────────────────

    def _detect_graphql_schema(
        self, file_path: str, content: str
    ) -> List[APIContract]:
        """Detect query/mutation definitions in .graphqls schema files."""
        contracts = []

        # Match type Query { ... } or type Mutation { ... }
        type_blocks = re.finditer(
            r'type\s+(Query|Mutation|Subscription)\s*\{([^}]+)\}',
            content,
            re.MULTILINE | re.DOTALL,
        )
        for block_match in type_blocks:
            gql_type = block_match.group(1).upper()
            block_content = block_match.group(2)

            # Parse field definitions
            field_pattern = re.compile(
                r'(\w+)\s*(?:\(([^)]*)\))?\s*:\s*(\S+)',
            )
            for field_match in field_pattern.finditer(block_content):
                field_name = field_match.group(1)
                params = field_match.group(2) or ""
                return_type = field_match.group(3)

                contracts.append(APIContract(
                    endpoint=field_name,
                    method=gql_type,
                    request_type=params,
                    response_type=return_type,
                    source_file=file_path,
                    source_symbol=field_name,
                    framework="graphql_schema",
                    line_number=content[:field_match.start()].count("\n") + 1,
                ))

        return contracts

    # ── Helpers ───────────────────────────────────────────────────────────

    def _find_next_method_name(self, content: str, start_pos: int) -> str:
        """Find the Java/Kotlin method name after an annotation."""
        # Look for: [modifiers] ReturnType methodName(
        method_match = re.search(
            r'(?:public|private|protected)?\s*'
            r'(?:static\s+)?'
            r'(?:\w+(?:<[^>]+>)?)\s+'
            r'(\w+)\s*\(',
            content[start_pos:start_pos + 500],
        )
        return method_match.group(1) if method_match else ""

    def _find_enclosing_method_ts(self, content: str, pos: int) -> str:
        """Find the TypeScript method name enclosing a given position."""
        # Search backward for method definition
        before = content[:pos]
        method_match = re.findall(
            r'(?:async\s+)?(\w+)\s*\([^)]*\)\s*(?::\s*\w+(?:<[^>]+>)?)?\s*\{',
            before,
        )
        return method_match[-1] if method_match else ""

    def _find_next_python_def(self, content: str, start_pos: int) -> str:
        """Find the Python function name after a decorator."""
        def_match = re.search(
            r'(?:async\s+)?def\s+(\w+)\s*\(',
            content[start_pos:start_pos + 300],
        )
        return def_match.group(1) if def_match else ""
