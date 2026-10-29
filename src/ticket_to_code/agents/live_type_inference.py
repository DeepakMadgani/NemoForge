"""
Live Type Inference Engine — Provide Type Hints While Generating

Infers and suggests types for properties/methods as code is being generated.
Useful for resolving ambiguous types and ensuring type safety.

Key insight: As LLM generates code, agent can ask TypeScript/Pylance/LSP
for type hints to ensure generated code is type-safe before writing.

Author: Deepak Madgani
Date: August 2026
"""

import logging
import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class TypeInferenceMethod(str, Enum):
    """How type was inferred."""
    LSP_HOVER = "lsp_hover"  # From LSP hover information
    CONTEXT_ANALYSIS = "context_analysis"  # From usage context
    PATTERN_MATCHING = "pattern_matching"  # From naming/patterns
    USER_ANNOTATION = "user_annotation"  # Explicit user annotation
    FALLBACK_ANY = "fallback_any"  # Default to any


@dataclass
class TypeHint:
    """A suggested type for a symbol."""
    symbol_name: str
    suggested_type: str  # "boolean", "string", "Observable<boolean>", etc.
    confidence: float  # 0.0-1.0
    method: TypeInferenceMethod
    reasoning: str
    alternatives: List[str] = None  # Other possible types
    required_imports: List[str] = None  # import statements needed

    def to_dict(self):
        return {
            "symbol_name": self.symbol_name,
            "suggested_type": self.suggested_type,
            "confidence": self.confidence,
            "method": self.method.value,
            "reasoning": self.reasoning,
            "alternatives": self.alternatives or [],
            "required_imports": self.required_imports or [],
        }


@dataclass
class TypeContext:
    """Context for type inference."""
    file_path: str
    class_name: Optional[str] = None
    method_name: Optional[str] = None
    property_name: Optional[str] = None
    usage_in_template: Optional[str] = None  # How it's used in HTML ({{ prop }}, etc.)
    usage_in_code: Optional[str] = None  # How it's used in TS (await, subscribe, etc.)
    existing_imports: List[str] = None  # Already imported types


class LiveTypeInferencer:
    """
    Infers types for generated code in real-time.

    Usage:
        inferencer = LiveTypeInferencer(lsp_client, symbol_index)
        
        # While generating a property, infer its type
        hint = inferencer.infer_type(
            symbol_name="isLoading",
            context=TypeContext(
                file_path="add-members.component.ts",
                class_name="AddMembersComponent",
                property_name="isLoading",
                usage_in_template="{{ isLoading }}",  # How it's used in HTML
                usage_in_code="await this.isLoading",  # How it's used in TS
            ),
        )
        
        # Type hint: { suggested_type: "boolean", confidence: 0.9, ... }
        print(f"Type: {hint.suggested_type} (confidence: {hint.confidence})")
    """

    def __init__(
        self,
        lsp_client=None,
        symbol_index=None,
    ):
        self.lsp_client = lsp_client
        self.symbol_index = symbol_index
        self._type_cache: Dict[str, TypeHint] = {}

    def infer_type(self, symbol_name: str, context: TypeContext) -> TypeHint:
        """
        Infer type for a symbol based on context.

        Returns: TypeHint with suggested type, confidence, and reasoning
        """
        cache_key = f"{context.file_path}:{context.class_name}:{symbol_name}"
        if cache_key in self._type_cache:
            return self._type_cache[cache_key]

        logger.debug(f"Inferring type for {symbol_name} in {context.class_name}")

        # Try multiple inference methods
        hints = [
            self._infer_from_lsp(symbol_name, context),
            self._infer_from_usage(symbol_name, context),
            self._infer_from_patterns(symbol_name, context),
        ]

        # Filter out None results
        hints = [h for h in hints if h is not None]

        if not hints:
            # Fallback to any
            hints = [
                TypeHint(
                    symbol_name=symbol_name,
                    suggested_type="any",
                    confidence=0.0,
                    method=TypeInferenceMethod.FALLBACK_ANY,
                    reasoning="Unable to infer type, using 'any' as fallback",
                )
            ]

        # Return highest confidence hint
        best_hint = max(hints, key=lambda h: h.confidence)
        self._type_cache[cache_key] = best_hint

        logger.debug(
            f"  ✅ {symbol_name}: {best_hint.suggested_type} (confidence: {best_hint.confidence:.2f})"
        )

        return best_hint

    def _infer_from_lsp(
        self,
        symbol_name: str,
        context: TypeContext,
    ) -> Optional[TypeHint]:
        """
        Infer type using LSP hover information.

        Query TypeScript/Pylance LSP to get exact type of symbol.
        """
        if not self.lsp_client:
            return None

        try:
            # Try to get type from LSP
            hover_info = self.lsp_client.get_hover_type(
                context.file_path,
                symbol_name,
            )

            if hover_info:
                return TypeHint(
                    symbol_name=symbol_name,
                    suggested_type=hover_info,
                    confidence=0.99,
                    method=TypeInferenceMethod.LSP_HOVER,
                    reasoning=f"Type from LSP: {hover_info}",
                )

        except Exception as e:
            logger.debug(f"LSP hover failed for {symbol_name}: {e}")

        return None

    def _infer_from_usage(
        self,
        symbol_name: str,
        context: TypeContext,
    ) -> Optional[TypeHint]:
        """
        Infer type from how symbol is used in code/template.

        Examples:
          - If used in HTML with {{ prop }}: probably boolean/string/Observable
          - If used with await: probably Promise or Observable
          - If used with subscribe(): definitely Observable
        """
        hints = []

        # Analyze template usage
        if context.usage_in_template:
            hint = self._analyze_template_usage(symbol_name, context.usage_in_template)
            if hint:
                hints.append(hint)

        # Analyze code usage
        if context.usage_in_code:
            hint = self._analyze_code_usage(symbol_name, context.usage_in_code)
            if hint:
                hints.append(hint)

        if hints:
            # Return highest confidence
            return max(hints, key=lambda h: h.confidence)

        return None

    def _analyze_template_usage(self, symbol_name: str, usage: str) -> Optional[TypeHint]:
        """Analyze how symbol is used in HTML template."""
        # {{ property }} - likely scalar (boolean, string, number)
        if "{{" in usage and "}}" in usage:
            return TypeHint(
                symbol_name=symbol_name,
                suggested_type="boolean | string | number",
                confidence=0.6,
                method=TypeInferenceMethod.CONTEXT_ANALYSIS,
                reasoning="Used in template interpolation {{ }}, likely scalar",
                alternatives=["boolean", "string", "number", "Observable<any>"],
            )

        # *ngIf property - definitely boolean or Observable<boolean>
        if "*ngif" in usage.lower():
            return TypeHint(
                symbol_name=symbol_name,
                suggested_type="Observable<boolean>",
                confidence=0.8,
                method=TypeInferenceMethod.CONTEXT_ANALYSIS,
                reasoning="Used in *ngIf, likely Observable<boolean>",
                alternatives=["boolean"],
            )

        # (event)="property()" - likely method/function
        if "(" in usage and ")" in usage:
            return TypeHint(
                symbol_name=symbol_name,
                suggested_type="() => void",
                confidence=0.7,
                method=TypeInferenceMethod.CONTEXT_ANALYSIS,
                reasoning="Used as event handler, likely function",
            )

        return None

    def _analyze_code_usage(self, symbol_name: str, usage: str) -> Optional[TypeHint]:
        """Analyze how symbol is used in TypeScript code."""
        # await property - Promise or Observable
        if "await" in usage:
            return TypeHint(
                symbol_name=symbol_name,
                suggested_type="Promise<any>",
                confidence=0.85,
                method=TypeInferenceMethod.CONTEXT_ANALYSIS,
                reasoning="Used with await, likely Promise",
                alternatives=["Observable<any>"],
            )

        # property.subscribe() - definitely Observable
        if ".subscribe(" in usage or ".subscribe " in usage:
            return TypeHint(
                symbol_name=symbol_name,
                suggested_type="Observable<any>",
                confidence=0.95,
                method=TypeInferenceMethod.CONTEXT_ANALYSIS,
                reasoning="Used with .subscribe(), definitely Observable",
                required_imports=["import { Observable } from 'rxjs';"],
            )

        # property.then() - definitely Promise
        if ".then(" in usage:
            return TypeHint(
                symbol_name=symbol_name,
                suggested_type="Promise<any>",
                confidence=0.95,
                method=TypeInferenceMethod.CONTEXT_ANALYSIS,
                reasoning="Used with .then(), definitely Promise",
            )

        # property.length - string or array
        if ".length" in usage:
            return TypeHint(
                symbol_name=symbol_name,
                suggested_type="string | any[]",
                confidence=0.8,
                method=TypeInferenceMethod.CONTEXT_ANALYSIS,
                reasoning="Used with .length, likely string or array",
                alternatives=["string", "any[]"],
            )

        # property === true/false - definitely boolean
        if "=== true" in usage or "=== false" in usage:
            return TypeHint(
                symbol_name=symbol_name,
                suggested_type="boolean",
                confidence=0.95,
                method=TypeInferenceMethod.CONTEXT_ANALYSIS,
                reasoning="Compared with boolean literal",
            )

        return None

    def _infer_from_patterns(
        self,
        symbol_name: str,
        context: TypeContext,
    ) -> Optional[TypeHint]:
        """
        Infer type from naming patterns and conventions.

        Examples:
          - is*, has*, can* prefix: boolean
          - *$ suffix: Observable
          - *List, *Array suffix: array
        """
        # Boolean patterns
        if any(symbol_name.lower().startswith(p) for p in ["is", "has", "can", "should", "need"]):
            return TypeHint(
                symbol_name=symbol_name,
                suggested_type="boolean",
                confidence=0.7,
                method=TypeInferenceMethod.PATTERN_MATCHING,
                reasoning=f"Naming pattern '{symbol_name}' suggests boolean",
            )

        # Observable patterns
        if symbol_name.endswith("$"):
            return TypeHint(
                symbol_name=symbol_name,
                suggested_type="Observable<any>",
                confidence=0.85,
                method=TypeInferenceMethod.PATTERN_MATCHING,
                reasoning=f"Naming convention '{symbol_name}' (ending with $) indicates Observable",
                required_imports=["import { Observable } from 'rxjs';"],
            )

        # Array/List patterns
        if any(symbol_name.lower().endswith(p) for p in ["list", "array", "items", "collection"]):
            return TypeHint(
                symbol_name=symbol_name,
                suggested_type="any[]",
                confidence=0.7,
                method=TypeInferenceMethod.PATTERN_MATCHING,
                reasoning=f"Naming pattern '{symbol_name}' suggests array",
            )

        # Map/Dictionary patterns
        if any(symbol_name.lower().endswith(p) for p in ["map", "dict", "lookup"]):
            return TypeHint(
                symbol_name=symbol_name,
                suggested_type="Record<string, any>",
                confidence=0.7,
                method=TypeInferenceMethod.PATTERN_MATCHING,
                reasoning=f"Naming pattern '{symbol_name}' suggests Map/Record",
            )

        return None

    def suggest_imports_for_type(self, suggested_type: str) -> List[str]:
        """
        Get import statements needed for a type.

        Example:
            suggest_imports_for_type("Observable<User>")
            # Returns: ["import { Observable } from 'rxjs';", "import { User } from './user';"]
        """
        imports = []

        # RxJS types
        if "Observable" in suggested_type:
            imports.append("import { Observable } from 'rxjs';")
        if "Subject" in suggested_type:
            imports.append("import { Subject } from 'rxjs';")
        if "BehaviorSubject" in suggested_type:
            imports.append("import { BehaviorSubject } from 'rxjs';")

        # Angular types
        if "FormBuilder" in suggested_type:
            imports.append("import { FormBuilder } from '@angular/forms';")
        if "HttpClient" in suggested_type:
            imports.append("import { HttpClient } from '@angular/common/http';")

        return imports

    def validate_type_compatibility(
        self,
        generated_type: str,
        expected_type: str,
    ) -> Tuple[bool, str]:
        """
        Check if generated type is compatible with expected type.

        Returns: (is_compatible, message)
        """
        # Exact match
        if generated_type == expected_type:
            return True, "Type matches exactly"

        # any is compatible with anything
        if generated_type == "any" or expected_type == "any":
            return True, "Using 'any' which is compatible with any type"

        # Observable compatible checks
        if "Observable" in expected_type and "Observable" in generated_type:
            return True, "Both are Observable types"

        # Promise compatible checks
        if "Promise" in expected_type and "Promise" in generated_type:
            return True, "Both are Promise types"

        # Array compatible checks
        if ("[]" in expected_type or "Array" in expected_type) and (
            "[]" in generated_type or "Array" in generated_type
        ):
            return True, "Both are array types"

        return False, f"Type mismatch: {generated_type} vs {expected_type}"

    def get_type_suggestions_batch(
        self,
        symbols: List[Dict[str, Any]],
    ) -> List[TypeHint]:
        """
        Infer types for multiple symbols at once.

        Input: List of { symbol_name, context }
        Output: List of TypeHint
        """
        hints = []
        for symbol in symbols:
            hint = self.infer_type(
                symbol.get("symbol_name", ""),
                symbol.get("context", TypeContext(file_path="")),
            )
            hints.append(hint)

        return hints
