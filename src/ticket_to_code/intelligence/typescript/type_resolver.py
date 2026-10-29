"""TypeScript Type Resolver.

Performs transitive type hydration by inspecting method signatures, following
module imports, extracting interface/type AST definitions, and assembling
bounded, authoritative type contracts.
"""

from pathlib import Path
import re
from typing import Dict, List, Optional, Set, Tuple, Union

from .import_resolver import TypeScriptImportResolver
from .type_contract import MethodTypeContract, TypeDefinition, TypeDependencyEdge

BUILTIN_TYPES: Set[str] = {
    "string", "number", "boolean", "any", "void", "null", "undefined",
    "never", "unknown", "object", "symbol", "bigint", "date", "regexp",
    "promise", "observable", "array", "record", "map", "set", "subject",
    "behaviorsubject", "string[]", "number[]", "boolean[]", "any[]",
}

# Regex to locate interface, type alias, or enum declarations
_INTERFACE_HEADER_RE = re.compile(
    r"""(?:export\s+)?(?:interface|type|enum|class)\s+([A-Za-z0-9_$]+)""",
    re.MULTILINE,
)

# Regex to match individual property declarations inside an interface:
# e.g.: email?: FilterStringInput[];
#       projectId: FilterIdInput;
_PROPERTY_RE = re.compile(
    r"""^\s*([A-Za-z0-9_$]+)\s*\??\s*:\s*([^;,\n]+)""",
    re.MULTILINE,
)

# Regex for method definitions in TypeScript class
_METHOD_DEF_RE = re.compile(
    r"""(?:public\s+|private\s+|protected\s+)?([A-Za-z0-9_$]+)\s*\(([^)]*)\)\s*(?::\s*([^{;]+))?""",
    re.MULTILINE,
)


class TypeScriptTypeResolver:
    """Recursively hydrates method parameter types and their dependencies."""

    @staticmethod
    def extract_declaration_block(content: str, type_name: str) -> Optional[Tuple[str, str]]:
        """Extract kind and verbatim text of an interface, type, or enum block.

        Returns:
            (kind, raw_text) or None
        """
        # Match interface or enum with balanced braces
        pattern = re.compile(
            rf"""(?:export\s+)?(interface|enum|class)\s+{re.escape(type_name)}(?:\s+extends\s+[^{{]+)?\s*\{{""",
            re.MULTILINE,
        )
        m = pattern.search(content)
        if m:
            kind = m.group(1)
            start_pos = m.start()
            brace_count = 0
            open_found = False
            end_pos = len(content)
            for idx in range(m.end() - 1, len(content)):
                ch = content[idx]
                if ch == "{":
                    brace_count += 1
                    open_found = True
                elif ch == "}":
                    brace_count -= 1
                    if open_found and brace_count == 0:
                        end_pos = idx + 1
                        break
            return kind, content[start_pos:end_pos].strip()

        # Match type alias: export type Foo = ...;
        type_alias_pat = re.compile(
            rf"""(?:export\s+)?type\s+{re.escape(type_name)}\s*=\s*([^;]+);""",
            re.MULTILINE,
        )
        tm = type_alias_pat.search(content)
        if tm:
            return "type", tm.group(0).strip()

        return None

    @staticmethod
    def parse_properties(declaration_text: str) -> Dict[str, str]:
        """Extract property name to type mapping from an interface block."""
        props: Dict[str, str] = {}
        for m in _PROPERTY_RE.finditer(declaration_text):
            pname = m.group(1)
            ptype = m.group(2).strip()
            # Skip comments or methods
            if pname in ("constructor", "get", "set"):
                continue
            props[pname] = ptype
        return props

    @classmethod
    def extract_referenced_custom_types(cls, type_str: str) -> List[Tuple[str, bool]]:
        """Extract custom type identifiers from a type string, noting if it is an array.

        Example:
            'FilterStringInput[]' -> [('FilterStringInput', True)]
            'FilterIdInput'       -> [('FilterIdInput', False)]
            'string | number'     -> [] (builtins skipped)
        """
        tokens = re.findall(r"""([A-Z][A-Za-z0-9_$]+)(\[\])?""", type_str)
        results: List[Tuple[str, bool]] = []
        for name, arr in tokens:
            if name.lower() not in BUILTIN_TYPES:
                is_array = bool(arr) or "Array<" in type_str
                results.append((name, is_array))
        return results

    @classmethod
    def hydrate_type_definition(
        cls,
        type_name: str,
        context_file: Path,
        workspace_root: Optional[Path] = None,
        is_direct: bool = True,
    ) -> Optional[Tuple[TypeDefinition, Path]]:
        """Locate and extract TypeDefinition for type_name starting from context_file."""
        if not context_file.exists():
            return None

        # 1. Check if type is defined directly in context_file
        try:
            content = context_file.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return None

        decl = cls.extract_declaration_block(content, type_name)
        if decl:
            kind, text = decl
            props = cls.parse_properties(text)
            rel_path = str(context_file)
            if workspace_root:
                try:
                    rel_path = str(context_file.relative_to(workspace_root)).replace("\\", "/")
                except Exception:
                    pass
            return TypeDefinition(
                name=type_name,
                kind=kind,
                source_file=rel_path,
                raw_declaration=text,
                is_direct=is_direct,
                properties=props,
            ), context_file

        # 2. Check if type is imported into context_file
        imports = TypeScriptImportResolver.parse_named_imports(content)
        if type_name in imports:
            target_path = TypeScriptImportResolver.resolve_module_path(
                imports[type_name], context_file, workspace_root=workspace_root
            )
            if target_path and target_path.exists():
                try:
                    target_content = target_path.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    return None
                target_decl = cls.extract_declaration_block(target_content, type_name)
                if target_decl:
                    kind, text = target_decl
                    props = cls.parse_properties(text)
                    rel_path = str(target_path)
                    if workspace_root:
                        try:
                            rel_path = str(target_path.relative_to(workspace_root)).replace("\\", "/")
                        except Exception:
                            pass
                    return TypeDefinition(
                        name=type_name,
                        kind=kind,
                        source_file=rel_path,
                        raw_declaration=text,
                        is_direct=is_direct,
                        properties=props,
                    ), target_path

        # 3. Check same directory for matching model file name (e.g. type_name in kebab-case)
        kebab = re.sub(r'(?<!^)(?=[A-Z])', '-', type_name).lower()
        for cand_name in (f"{kebab}.ts", f"{type_name}.ts", f"{kebab}.model.ts"):
            sibling = context_file.parent / cand_name
            if sibling.exists():
                try:
                    sib_content = sibling.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    continue
                sib_decl = cls.extract_declaration_block(sib_content, type_name)
                if sib_decl:
                    kind, text = sib_decl
                    props = cls.parse_properties(text)
                    rel_path = str(sibling)
                    if workspace_root:
                        try:
                            rel_path = str(sibling.relative_to(workspace_root)).replace("\\", "/")
                        except Exception:
                            pass
                    return TypeDefinition(
                        name=type_name,
                        kind=kind,
                        source_file=rel_path,
                        raw_declaration=text,
                        is_direct=is_direct,
                        properties=props,
                    ), sibling

        return None

    @classmethod
    def build_method_type_contract(
        cls,
        service_file: Union[str, Path],
        method_name: str,
        workspace_root: Optional[Union[str, Path]] = None,
        max_depth: int = 2,
        max_types: int = 6,
    ) -> MethodTypeContract:
        """Hydrate complete direct and transitive type contract for a service method."""
        s_path = Path(str(service_file).replace("\\", "/")).resolve()
        ws_path = Path(str(workspace_root).replace("\\", "/")).resolve() if workspace_root else None

        if not s_path.exists():
            return MethodTypeContract(
                service_class="",
                service_file=str(service_file),
                method_name=method_name,
                method_signature="",
                parameter_name="",
                parameter_type="",
            )

        try:
            service_content = s_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            service_content = ""

        # Find service class name
        cls_match = re.search(r"""(?:export\s+)?class\s+([A-Za-z0-9_$]+)""", service_content)
        service_class = cls_match.group(1) if cls_match else s_path.stem

        # Locate method signature
        method_sig = ""
        param_name = ""
        param_type = ""
        for m in _METHOD_DEF_RE.finditer(service_content):
            if m.group(1) == method_name:
                args_str = m.group(2).strip()
                ret_type = (m.group(3) or "").strip()
                method_sig = f"{method_name}({args_str})"
                if ret_type:
                    method_sig += f": {ret_type}"

                # Extract first parameter
                if args_str:
                    first_arg = args_str.split(",")[0].strip()
                    if ":" in first_arg:
                        param_name, param_type = [x.strip() for x in first_arg.split(":", 1)]
                        # Clean param_type of modifiers
                        param_type = param_type.split("=")[0].strip()
                break

        service_rel = str(s_path)
        if ws_path:
            try:
                service_rel = str(s_path.relative_to(ws_path)).replace("\\", "/")
            except Exception:
                pass

        contract = MethodTypeContract(
            service_class=service_class,
            service_file=service_rel,
            method_name=method_name,
            method_signature=method_sig or f"{method_name}()",
            parameter_name=param_name,
            parameter_type=param_type,
            return_type=ret_type,
        )

        visited_types: Set[str] = set()
        queue: List[Tuple[str, Path, int, bool]] = []

        # 1. Hydrate Direct Parameter Type
        if param_type and param_type.lower() not in BUILTIN_TYPES:
            custom_directs = cls.extract_referenced_custom_types(param_type)
            if not custom_directs:
                custom_directs = [(param_type.strip(), False)]

            for direct_name, _ in custom_directs:
                res = cls.hydrate_type_definition(direct_name, s_path, workspace_root=ws_path, is_direct=True)
                if res:
                    type_def, def_file = res
                    contract.direct_types[direct_name] = type_def
                    visited_types.add(direct_name)
                    queue.append((direct_name, def_file, 1, True))

        # 2. Transitive Hydration (Recursive with bounds)
        while queue and len(contract.direct_types) + len(contract.transitive_types) < max_types:
            curr_name, curr_file, depth, is_parent_direct = queue.pop(0)
            if depth > max_depth:
                continue

            curr_def = contract.direct_types.get(curr_name) or contract.transitive_types.get(curr_name)
            if not curr_def:
                continue

            for field_name, field_type_str in curr_def.properties.items():
                refs = cls.extract_referenced_custom_types(field_type_str)
                for ref_name, is_array in refs:
                    # Record dependency edge
                    contract.dependency_edges.append(
                        TypeDependencyEdge(
                            parent_type=curr_name,
                            field_name=field_name,
                            target_type=ref_name,
                            is_array=is_array,
                        )
                    )

                    if ref_name in visited_types:
                        continue
                    visited_types.add(ref_name)

                    # Hydrate transitive type definition
                    trans_res = cls.hydrate_type_definition(
                        ref_name, curr_file, workspace_root=ws_path, is_direct=False
                    )
                    if trans_res:
                        trans_def, trans_file = trans_res
                        contract.transitive_types[ref_name] = trans_def
                        queue.append((ref_name, trans_file, depth + 1, False))

        # 3. Hydrate Return Type(s) (e.g. Observable<ParticipatingMember[]> -> ParticipatingMember)
        if ret_type:
            ret_customs = cls.extract_referenced_custom_types(ret_type)
            for ret_name, _ in ret_customs:
                if ret_name in visited_types:
                    if ret_name in contract.direct_types:
                        contract.return_types[ret_name] = contract.direct_types[ret_name]
                    elif ret_name in contract.transitive_types:
                        contract.return_types[ret_name] = contract.transitive_types[ret_name]
                    continue
                res = cls.hydrate_type_definition(ret_name, s_path, workspace_root=ws_path, is_direct=True)
                if res:
                    type_def, def_file = res
                    contract.return_types[ret_name] = type_def
                    visited_types.add(ret_name)

        return contract
