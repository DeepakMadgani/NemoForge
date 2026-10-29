"""
Template Expression Parser — Structural AST-level reference extractor for templates.

Framework/language adapter for modern template expressions (Angular, Vue, JSX).
Replaces brittle regex matching with lexical and syntactic expression parsing:
- Naturally classifies literals (true, false, null, undefined, strings, numbers)
  as literals rather than identifiers without relying on fragile string blacklists.
- Extracts identifier chains (receiver.prop.subprop), method calls, and bare identifiers.
- Recognizes Angular pipes (`expr | pipe`), safe navigation (`a?.b`), and structural directives.

Invariant:
    Only syntactically identified identifiers/member accesses become controller references.
"""

from __future__ import annotations

import enum
import re
from dataclasses import dataclass
from typing import Optional, Sequence


class TokenType(enum.Enum):
    BOOLEAN_LITERAL = "BOOLEAN_LITERAL"
    NIL_LITERAL = "NIL_LITERAL"
    STRING_LITERAL = "STRING_LITERAL"
    NUMBER_LITERAL = "NUMBER_LITERAL"
    IDENTIFIER = "IDENTIFIER"
    OPERATOR = "OPERATOR"
    KEYWORD = "KEYWORD"
    PUNCTUATION = "PUNCTUATION"
    PIPE = "PIPE"


@dataclass(frozen=True)
class Token:
    type: TokenType
    value: str
    start: int
    end: int


@dataclass(frozen=True)
class TemplateReference:
    receiver: str
    member: str
    is_method: bool = False
    source_syntax: str = "template_expression"
    line_hint: str = ""


# Angular structural directive microsyntax keywords (*ngIf, *ngFor, etc.)
ANGULAR_MICROSYNTAX_KEYWORDS: frozenset[str] = frozenset({
    "let", "as", "of", "in", "else", "then", "trackBy",
})

# JavaScript / TypeScript expression keywords allowed in templates
JS_EXPRESSION_KEYWORDS: frozenset[str] = frozenset({
    "this", "typeof", "instanceof", "void", "new", "var", "const",
})

# Combined set of reserved keywords in template expressions
RESERVED_TEMPLATE_KEYWORDS: frozenset[str] = ANGULAR_MICROSYNTAX_KEYWORDS | JS_EXPRESSION_KEYWORDS


class TemplateLexer:
    """Tokenizer for Angular/JS template expressions."""

    # Lexical regular expressions
    _RULES: Sequence[tuple[TokenType, re.Pattern]] = [
        # Strings: single quote, double quote, backtick
        (TokenType.STRING_LITERAL, re.compile(r"'([^'\\]*(?:\\.[^'\\]*)*)'")),
        (TokenType.STRING_LITERAL, re.compile(r'"([^"\\]*(?:\\.[^"\\]*)*)"')),
        (TokenType.STRING_LITERAL, re.compile(r'`([^`\\]*(?:\\.[^`\\]*)*)`')),
        # Numbers: floats, ints, hex
        (TokenType.NUMBER_LITERAL, re.compile(r'\b0x[0-9a-fA-F]+\b|\b\d+(?:\.\d+)?(?:[eE][+-]?\d+)?\b')),
        # Booleans
        (TokenType.BOOLEAN_LITERAL, re.compile(r'\b(?:true|false)\b')),
        # Nil values
        (TokenType.NIL_LITERAL, re.compile(r'\b(?:null|undefined)\b')),
        # Language keywords and Angular template microsyntax keywords
        (TokenType.KEYWORD, re.compile(r'\b(?:' + '|'.join(sorted(RESERVED_TEMPLATE_KEYWORDS)) + r')\b')),
        # Pipe operator (Angular specific: | but not ||)
        (TokenType.PIPE, re.compile(r'\|(?!=)')),
        # Identifiers: variable, method, property names
        (TokenType.IDENTIFIER, re.compile(r'[a-zA-Z_$][a-zA-Z0-9_$]*')),
        # Multi-char operators
        (TokenType.OPERATOR, re.compile(r'\?\?|\?\.|&&|\|\||===|!==|==|!=|<=|>=|\+\+|--|=>')),
        # Single-char operators
        (TokenType.OPERATOR, re.compile(r'[+\-*/%<>=!?:~^&]')),
        # Punctuation / structural delimiters
        (TokenType.PUNCTUATION, re.compile(r'[()\[\]{},;.]')),
    ]

    @classmethod
    def tokenize(cls, expr: str) -> list[Token]:
        tokens: list[Token] = []
        pos = 0
        n = len(expr)

        while pos < n:
            # Skip whitespace
            if expr[pos].isspace():
                pos += 1
                continue

            matched = False
            for tok_type, pattern in cls._RULES:
                m = pattern.match(expr, pos)
                if m:
                    # Special check: pipe vs ||
                    if tok_type == TokenType.PIPE and pos + 1 < n and expr[pos + 1] == '|':
                        continue  # Let multi-char operator handle ||
                    tokens.append(Token(tok_type, m.group(0), pos, m.end()))
                    pos = m.end()
                    matched = True
                    break

            if not matched:
                # Skip unknown single character safely
                pos += 1

        return tokens


class TemplateExpressionParser:
    """Parses token sequences to extract member access and method call references."""

    @classmethod
    def extract_references(
        cls,
        expression_text: str,
        syntax_type: str = "template_expression",
        local_vars: Optional[set[str]] = None,
        ngfor_vars: Optional[set[str]] = None,
    ) -> list[TemplateReference]:
        """Extract all valid outbound identifier references from a template expression.

        Rules:
        - Literals (true, false, null, undefined, numbers, strings) are ignored.
        - `receiver.member` produces receiver='receiver', member='member'
        - `receiver.method()` produces receiver='receiver', member='method', is_method=True
        - Bare identifiers (e.g. `isDisabled`, `user`) produce receiver='this', member='...'
        - Bare method calls (e.g. `onSave()`) produce receiver='this', member='onSave', is_method=True
        - Angular pipes (`expr | pipeName`) ignore `pipeName` (transform, not controller member).
        - Framework/template context variables (starting with '$', e.g. $event, $index, $any, $emit)
          and locally declared template variables (#ref, let, as) are NEVER controller members.
        """
        scoped_vars = set(local_vars or ()) | set(ngfor_vars or ())

        def is_template_scoped(name: str) -> bool:
            """Dynamically check if identifier belongs to framework or local template context."""
            if not name:
                return True
            # Framework/template context variables start with '$' ($event, $index, $emit, etc.)
            if name.startswith("$"):
                return True
            # Template-local variables (ngFor item, template ref #var, let-bindings, as-bindings)
            if name in scoped_vars:
                return True
            if name == "this":
                return True
            return False

        tokens = TemplateLexer.tokenize(expression_text)
        if not tokens:
            return []

        refs: list[TemplateReference] = []
        seen: set[tuple[str, str]] = set()

        def add_ref(rec: str, mem: str, is_meth: bool = False):
            if not rec or not mem:
                return
            if rec == "this" and mem == "this":
                return
            key = (rec, mem)
            if key not in seen:
                seen.add(key)
                refs.append(TemplateReference(
                    receiver=rec,
                    member=mem,
                    is_method=is_meth,
                    source_syntax=syntax_type,
                    line_hint=expression_text.strip(),
                ))

        i = 0
        num_tokens = len(tokens)

        # Track when we are immediately following a pipe operator or directive keyword
        in_pipe = False
        after_directive_var = False

        while i < num_tokens:
            tok = tokens[i]

            # Angular Pipe handling: `expr | pipeName:arg`
            if tok.type == TokenType.PIPE:
                in_pipe = True
                i += 1
                continue

            if in_pipe:
                # The first identifier after a pipe is the pipe name (e.g., `translate`, `date`)
                if tok.type == TokenType.IDENTIFIER:
                    in_pipe = False
                    i += 1
                    continue
                elif tok.type == TokenType.OPERATOR and tok.value == ':':
                    # Pipe arguments follow
                    in_pipe = False
                    i += 1
                    continue

            # Angular structural directive keywords: else, then, let, as, trackBy
            if tok.type == TokenType.KEYWORD:
                if tok.value in ANGULAR_MICROSYNTAX_KEYWORDS:
                    after_directive_var = True
                i += 1
                continue

            if after_directive_var:
                after_directive_var = False
                if tok.type == TokenType.IDENTIFIER:
                    # Skip the template reference or local alias identifier
                    i += 1
                    continue

            # Check for identifiers
            if tok.type == TokenType.IDENTIFIER:
                ident = tok.value

                # Check if this identifier is part of an access chain: `ident.prop` or `ident?.prop`
                j = i + 1
                current_receiver = ident
                chain_handled = False

                while j < num_tokens and (
                    (tokens[j].type == TokenType.PUNCTUATION and tokens[j].value == '.')
                    or (tokens[j].type == TokenType.OPERATOR and tokens[j].value == '?.')
                ):
                    j += 1
                    if j < num_tokens and tokens[j].type == TokenType.IDENTIFIER:
                        member_name = tokens[j].value
                        is_method = (j + 1 < num_tokens and tokens[j + 1].type == TokenType.PUNCTUATION and tokens[j + 1].value == '(')
                        
                        # Add reference
                        add_ref(current_receiver, member_name, is_meth=is_method)
                        current_receiver = member_name
                        j += 1
                        chain_handled = True
                    else:
                        break

                if chain_handled:
                    i = j
                    continue

                # Not a chain. Check if it's a bare method call: `ident(`
                if i + 1 < num_tokens and tokens[i + 1].type == TokenType.PUNCTUATION and tokens[i + 1].value == '(':
                    if not is_template_scoped(ident):
                        rec = "this" if ident != "this" else "this"
                        add_ref(rec, ident, is_meth=True)
                    i += 2
                    continue

                # Bare identifier reference (e.g. `isDisabled`, `user`, `selectedOrg`)
                # Only if not preceded by a dot/safe navigation
                is_after_dot = (i > 0 and (
                    (tokens[i - 1].type == TokenType.PUNCTUATION and tokens[i - 1].value == '.')
                    or (tokens[i - 1].type == TokenType.OPERATOR and tokens[i - 1].value == '?.')
                ))

                if not is_after_dot and not is_template_scoped(ident):
                    # Emit bare identifier as a member on "this"
                    add_ref("this", ident, is_meth=False)

            i += 1

        return refs
