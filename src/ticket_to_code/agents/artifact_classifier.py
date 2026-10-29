"""
Artifact Classifier

Classifies a target file path into an ArtifactType so GenerationRouter can
pick the right generator. Deterministic, extension-based (no LLM call needed).
"""

from pathlib import Path

from ticket_to_code.models import ArtifactType

_STYLE_EXTS = {".css", ".scss", ".sass", ".less"}
_CONFIG_EXTS = {
    ".json", ".yml", ".yaml", ".toml", ".ini", ".properties", ".env",
    ".xml", ".conf", ".cfg",
}
_SCRIPT_EXTS = {".sh", ".bat", ".ps1", ".bash", ".zsh"}
_SQL_EXTS = {".sql"}
_TEXT_EXTS = {".md", ".txt", ".rst", ".log"}
_SOURCE_CODE_EXTS = {
    ".java", ".ts", ".tsx", ".js", ".jsx", ".py", ".cs", ".go",
    ".html", ".htm",  # Angular/web component templates are source code type-checked by AOT
}


class ArtifactClassifier:
    """Deterministically classifies a file path into an ArtifactType."""

    EXACT_MATCHES = {
        "Dockerfile": ArtifactType.CONFIG,
        "Chart.yaml": ArtifactType.CONFIG,
        "Jenkinsfile": ArtifactType.CONFIG,
        "Makefile": ArtifactType.SCRIPT,
        ".env": ArtifactType.CONFIG,
    }

    EXTENSION_MATCHES = {
        ".java": ArtifactType.SOURCE_CODE,
        ".ts": ArtifactType.SOURCE_CODE,
        ".tsx": ArtifactType.SOURCE_CODE,
        ".js": ArtifactType.SOURCE_CODE,
        ".jsx": ArtifactType.SOURCE_CODE,
        ".py": ArtifactType.SOURCE_CODE,
        ".cs": ArtifactType.SOURCE_CODE,
        ".go": ArtifactType.SOURCE_CODE,
        ".html": ArtifactType.SOURCE_CODE,  # Angular AOT template
        ".htm": ArtifactType.SOURCE_CODE,
        ".scss": ArtifactType.STYLE,
        ".css": ArtifactType.STYLE,
        ".less": ArtifactType.STYLE,
        ".sass": ArtifactType.STYLE,
        ".yml": ArtifactType.CONFIG,
        ".yaml": ArtifactType.CONFIG,
        ".properties": ArtifactType.CONFIG,
        ".json": ArtifactType.CONFIG,
        ".xml": ArtifactType.CONFIG,
        ".toml": ArtifactType.CONFIG,
        ".ini": ArtifactType.CONFIG,
        ".sh": ArtifactType.SCRIPT,
        ".bat": ArtifactType.SCRIPT,
        ".ps1": ArtifactType.SCRIPT,
        ".sql": ArtifactType.SQL,
        ".md": ArtifactType.GENERIC_TEXT,
        ".txt": ArtifactType.GENERIC_TEXT,
        ".log": ArtifactType.GENERIC_TEXT,
    }

    def classify(self, file_path: str) -> ArtifactType:
        path = Path(file_path)
        filename = path.name

        if filename in self.EXACT_MATCHES:
            return self.EXACT_MATCHES[filename]

        ext = path.suffix.lower()
        if ext in self.EXTENSION_MATCHES:
            return self.EXTENSION_MATCHES[ext]

        return ArtifactType.SOURCE_CODE
