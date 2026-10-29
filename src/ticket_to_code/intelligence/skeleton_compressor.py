"""
Skeleton Compressor (Tiered Fidelity Context Assembly)

Grounded in AI IDE & Harness Architecture (Claude Opus, Copilot Workspace, OpenAI Codex):
1. Tiered Fidelity: Edit-target files receive Full Fidelity. Surrounding dependencies,
   services, and interfaces receive Skeleton Fidelity (signatures only, bodies elided).
2. Algorithmic Body Elision: Replaces method and function bodies with `{ /* elided */ }`
   or `;`, reducing a 2,000-line service to a clean 40-line interface surface.
3. Decorator & Signature Preservation: Preserves `@Component`, `@Injectable`, `@RestController`,
   field declarations, types, generics, parameters, and leading docstrings.
4. Token Compression Telemetry: Measures character and token savings ratio.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple


@dataclass(frozen=True)
class SkeletonResult:
    """Compressed skeleton code representation and compression telemetry."""
    file_path: str
    language: str
    original_lines: int
    compressed_lines: int
    original_chars: int
    compressed_chars: int
    skeleton_code: str

    @property
    def compression_ratio(self) -> float:
        """Ratio of compressed characters to original characters (e.g. 0.15 = 85% saved)."""
        if self.original_chars == 0:
            return 1.0
        return round(self.compressed_chars / self.original_chars, 3)

    @property
    def estimated_tokens_saved(self) -> int:
        """Heuristic token savings estimate (approx 4 chars per token)."""
        saved_chars = max(0, self.original_chars - self.compressed_chars)
        return saved_chars // 4


class SkeletonCompressor:
    """Extracts signature-only interface skeletons from TypeScript and Java files."""

    @classmethod
    def compress_file(cls, file_path: str, content: Optional[str] = None) -> SkeletonResult:
        """Reads and compresses a source code file into its interface skeleton."""
        path_obj = Path(file_path)
        ext = path_obj.suffix.lower()

        if content is None:
            content = path_obj.read_text(encoding="utf-8", errors="replace")

        original_lines = len(content.splitlines())
        original_chars = len(content)

        if ext in {".ts", ".tsx", ".js", ".jsx"}:
            skeleton = cls.compress_typescript(content)
            lang = "typescript"
        elif ext in {".java"}:
            skeleton = cls.compress_java(content)
            lang = "java"
        elif ext in {".py"}:
            skeleton = cls.compress_python(content)
            lang = "python"
        elif ext in {".html", ".htm"}:
            skeleton = cls.compress_html(content)
            lang = "html"
        elif ext in {".yml", ".yaml"}:
            skeleton = cls.compress_yaml(content)
            lang = "yaml"
        else:
            # Fallback for other file types
            skeleton = content
            lang = "unknown"

        compressed_lines = len(skeleton.splitlines())
        compressed_chars = len(skeleton)

        return SkeletonResult(
            file_path=str(file_path).replace("\\", "/"),
            language=lang,
            original_lines=original_lines,
            compressed_lines=compressed_lines,
            original_chars=original_chars,
            compressed_chars=compressed_chars,
            skeleton_code=skeleton,
        )

    @classmethod
    def compress_typescript(cls, code: str) -> str:
        """Compresses TypeScript / Angular code by eliding function and method bodies."""
        lines = code.splitlines()
        result_lines = []
        i = 0
        n = len(lines)

        while i < n:
            line = lines[i]
            stripped = line.strip()

            # Preserve imports, exports, decorators, comments, and empty lines
            if (
                stripped.startswith("import ")
                or stripped.startswith("export type ")
                or stripped.startswith("export interface ")
                or stripped.startswith("interface ")
                or stripped.startswith("type ")
                or stripped.startswith("@")
                or stripped.startswith("//")
                or stripped.startswith("/*")
                or stripped.startswith("*")
                or stripped == ""
            ):
                result_lines.append(line)
                i += 1
                continue

            # Class/Enum declarations
            if (
                re.match(r'^(export\s+)?(default\s+)?(abstract\s+)?class\s+\w+', stripped)
                or re.match(r'^(export\s+)?enum\s+\w+', stripped)
            ):
                result_lines.append(line)
                i += 1
                continue

            # Member variables / field declarations (e.g. `uploadForm: FormGroup;` or `selectedFile?: File;`)
            if (
                re.match(r'^(public\s+|private\s+|protected\s+|readonly\s+|static\s+)*\w+(\?|!)?:\s*[^;{=]+;', stripped)
                or re.match(r'^(public\s+|private\s+|protected\s+|readonly\s+|static\s+)*\w+\s*=\s*[^;{=]+;', stripped)
            ):
                result_lines.append(line)
                i += 1
                continue

            # Method / constructor / function with body:
            # Look for line with { or leading up to {
            if "{" in line and not line.rstrip().endswith(";"):
                # Check if it looks like a method/function/constructor signature
                sig_match = re.search(r'^(.*?)\s*\{', line)
                if sig_match:
                    sig_prefix = sig_match.group(1).rstrip()
                    # Skip if it's an object literal or class declaration
                    if not re.search(r'\b(class|enum|interface|type)\b', sig_prefix):
                        indent = " " * (len(line) - len(line.lstrip()))
                        result_lines.append(f"{sig_prefix} {{ /* elided */ }}")

                        # Skip lines until matching closing brace of this body
                        brace_count = line.count("{") - line.count("}")
                        i += 1
                        while i < n and brace_count > 0:
                            brace_count += lines[i].count("{") - lines[i].count("}")
                            i += 1
                        continue

            result_lines.append(line)
            i += 1

        # Clean multiple consecutive blank lines
        clean_code = re.sub(r'\n{3,}', '\n\n', "\n".join(result_lines)).strip() + "\n"
        return clean_code

    @classmethod
    def compress_java(cls, code: str) -> str:
        """Compresses Java / Spring Boot code by eliding method and constructor bodies."""
        lines = code.splitlines()
        result_lines = []
        i = 0
        n = len(lines)

        while i < n:
            line = lines[i]
            stripped = line.strip()

            # Preserve package, imports, annotations, comments, empty lines
            if (
                stripped.startswith("package ")
                or stripped.startswith("import ")
                or stripped.startswith("@")
                or stripped.startswith("//")
                or stripped.startswith("/*")
                or stripped.startswith("*")
                or stripped == ""
            ):
                result_lines.append(line)
                i += 1
                continue

            # Class/Interface/Enum/Record declarations
            if re.match(r'^(public\s+|protected\s+|private\s+|abstract\s+|static\s+|final\s+)*(class|interface|enum|record)\s+\w+', stripped):
                result_lines.append(line)
                i += 1
                continue

            # Field declarations ending in semicolon
            if stripped.endswith(";") and not ("(" in stripped and ")" in stripped):
                result_lines.append(line)
                i += 1
                continue

            # Method or constructor declaration opening with {
            if "{" in line and ("(" in line and ")" in line):
                sig_match = re.search(r'^(.*?)\s*\{', line)
                if sig_match:
                    sig_prefix = sig_match.group(1).rstrip()
                    # Skip if class declaration on same line
                    if not re.search(r'\b(class|interface|enum|record)\b', sig_prefix):
                        indent = " " * (len(line) - len(line.lstrip()))
                        result_lines.append(f"{sig_prefix} {{ /* elided */ }}")

                        # Skip lines until matching closing brace of this body
                        brace_count = line.count("{") - line.count("}")
                        i += 1
                        while i < n and brace_count > 0:
                            brace_count += lines[i].count("{") - lines[i].count("}")
                            i += 1
                        continue

            result_lines.append(line)
            i += 1

        clean_code = re.sub(r'\n{3,}', '\n\n', "\n".join(result_lines)).strip() + "\n"
        return clean_code

    @classmethod
    def compress_python(cls, code: str) -> str:
        """Compresses Python code by replacing method bodies with docstring + '...'."""
        lines = code.splitlines()
        result_lines = []
        i = 0
        n = len(lines)

        while i < n:
            line = lines[i]
            stripped = line.strip()

            # Function / method definition
            if re.match(r'^(async\s+)?def\s+\w+\s*\(', stripped):
                result_lines.append(line)
                indent_len = len(line) - len(line.lstrip())
                body_indent = " " * (indent_len + 4)

                # Collect def until ':'
                while not line.rstrip().endswith(":") and i < n - 1:
                    i += 1
                    line = lines[i]
                    result_lines.append(line)

                i += 1
                # Check if first line of body is a docstring
                has_docstring = False
                if i < n and lines[i].strip().startswith(('"""', "'''")):
                    doc_line = lines[i]
                    result_lines.append(doc_line)
                    if not (doc_line.count('"""') >= 2 or doc_line.count("'''") >= 2):
                        # multi-line docstring
                        quote = '"""' if '"""' in doc_line else "'''"
                        i += 1
                        while i < n:
                            result_lines.append(lines[i])
                            if quote in lines[i]:
                                break
                            i += 1
                    has_docstring = True
                    i += 1

                # Add ellipsis and skip remaining body
                result_lines.append(f"{body_indent}...")

                # Skip rest of body lines that are indented deeper than def
                while i < n:
                    curr = lines[i]
                    if curr.strip() == "":
                        i += 1
                        continue
                    curr_indent = len(curr) - len(curr.lstrip())
                    if curr_indent <= indent_len:
                        break
                    i += 1
                continue

            result_lines.append(line)
            i += 1

        clean_code = re.sub(r'\n{3,}', '\n\n', "\n".join(result_lines)).strip() + "\n"
        return clean_code

    @classmethod
    def compress_html(cls, code: str) -> str:
        """Compresses Angular / HTML templates by slicing along DOM tag boundaries,

        preserving structural directives (*ngIf, *ngFor) and container tags without corrupting tag balance.
        """
        lines = code.splitlines()
        if len(lines) <= 25:
            return code

        result_lines = []
        i = 0
        n = len(lines)

        while i < n:
            line = lines[i]
            stripped = line.strip()

            # Detect large table body blocks or repeating containers
            if re.search(r'<tbody\b', stripped, re.IGNORECASE):
                result_lines.append(line)
                indent = " " * (len(line) - len(line.lstrip()) + 2)
                j = i + 1
                while j < n and "</tbody>" not in lines[j].lower():
                    j += 1
                if j < n and (j - i) > 10:
                    for k in range(i + 1, min(i + 4, j)):
                        result_lines.append(lines[k])
                    elided_lines = j - (i + 4)
                    if elided_lines > 0:
                        result_lines.append(f"{indent}<!-- ... elided {elided_lines} lines of inner DOM rows ... -->")
                    result_lines.append(lines[j])
                    i = j + 1
                    continue

            result_lines.append(line)
            i += 1

        return "\n".join(result_lines) + "\n"

    @classmethod
    def compress_yaml(cls, code: str) -> str:
        """Compresses YAML configurations (Spring Gateway, k8s, properties)

        by preserving hierarchy keys and route IDs while collapsing deep configuration values.
        """
        lines = code.splitlines()
        if len(lines) <= 30:
            return code

        result_lines = []
        for line in lines:
            stripped = line.strip()
            indent = len(line) - len(line.lstrip())

            # Always keep comments, blank lines, root keys, and second-level keys
            if (
                not stripped
                or stripped.startswith("#")
                or indent <= 4
                or stripped.startswith("- id:")
                or stripped.startswith("uri:")
                or stripped.startswith("predicates:")
                or stripped.startswith("- Path=")
                or stripped.startswith("filters:")
            ):
                result_lines.append(line)
            elif indent >= 10 and not stripped.endswith(":"):
                continue
            else:
                result_lines.append(line)

        return "\n".join(result_lines) + "\n"

