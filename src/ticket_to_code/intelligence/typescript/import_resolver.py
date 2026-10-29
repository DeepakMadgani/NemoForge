"""TypeScript Import Resolver.

Resolves relative module imports in TypeScript/Angular files to actual file
paths on disk within the repository.
"""

from pathlib import Path
import re
from typing import Dict, List, Optional, Union

# Regex for matching TypeScript ES6 import declarations:
# e.g.: import { Foo, Bar as B } from './path/to/module';
#       import * as X from '../foo';
#       import Def from './def';
_IMPORT_RE = re.compile(
    r"""import\s+(?:(?:(?:\{([^}]+)\})|(?:\*\s+as\s+([A-Za-z0-9_$]+))|(?:([A-Za-z0-9_$]+)))(?:\s*,\s*(?:\{([^}]+)\}))?)\s+from\s+['"]([^'"]+)['"]""",
    re.MULTILINE,
)

_TS_CANDIDATE_SUFFIXES = [
    ".ts",
    ".tsx",
    ".d.ts",
    "/index.ts",
    "/index.tsx",
    "/index.d.ts",
]


class TypeScriptImportResolver:
    """Resolves imported symbols to their originating .ts files."""

    @staticmethod
    def parse_named_imports(source_content: str) -> Dict[str, str]:
        """Extract mapping from imported symbol name to import path.

        Example:
            import { FilterParticipantMemberInput, ParticipatingMember } from '../../models/participating-members';
            ->
            {
                "FilterParticipantMemberInput": "../../models/participating-members",
                "ParticipatingMember": "../../models/participating-members",
            }
        """
        symbol_to_module: Dict[str, str] = {}
        for m in _IMPORT_RE.finditer(source_content):
            named_part1 = m.group(1)
            star_import = m.group(2)
            default_import = m.group(3)
            named_part2 = m.group(4)
            module_path = m.group(5)

            if star_import:
                symbol_to_module[star_import] = module_path
            if default_import:
                symbol_to_module[default_import] = module_path

            for part in (named_part1, named_part2):
                if not part:
                    continue
                for sym in part.split(","):
                    sym = sym.strip()
                    if not sym:
                        continue
                    # Handle aliasing: "Original as Alias"
                    if " as " in sym:
                        alias = sym.split(" as ")[1].strip()
                        symbol_to_module[alias] = module_path
                    else:
                        symbol_to_module[sym] = module_path

        return symbol_to_module

    @staticmethod
    def resolve_module_path(
        module_specifier: str,
        context_file: Union[str, Path],
        workspace_root: Optional[Union[str, Path]] = None,
    ) -> Optional[Path]:
        """Resolve a module specifier relative to context_file into an absolute Path on disk.

        Handles relative imports (./ and ../) as well as candidate extensions.
        """
        context_path = Path(str(context_file).replace("\\", "/")).resolve()
        if context_path.is_file():
            base_dir = context_path.parent
        else:
            base_dir = context_path

        # Only resolve relative imports or workspace-relative imports
        if module_specifier.startswith("."):
            target_base = (base_dir / module_specifier).resolve()
        elif workspace_root:
            # Check if specifier is relative to workspace_root or app root
            ws = Path(str(workspace_root).replace("\\", "/")).resolve()
            target_base = (ws / module_specifier).resolve()
        else:
            return None

        # Check direct path if extension was explicitly provided
        if target_base.is_file():
            return target_base

        # Try candidate TypeScript extensions
        for suffix in _TS_CANDIDATE_SUFFIXES:
            cand = Path(str(target_base) + suffix)
            if cand.is_file():
                return cand

        # Also check if target_base is a directory containing index.ts
        if target_base.is_dir():
            for suffix in ("/index.ts", "/index.tsx", "/index.d.ts"):
                cand = Path(str(target_base) + suffix)
                if cand.is_file():
                    return cand

        return None
