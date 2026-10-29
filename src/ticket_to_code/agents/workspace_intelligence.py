"""
WorkspaceIntelligenceAgent — Layer 0 of the Ticket-to-Code pipeline.

Runs ONCE per repository (cached by git SHA). Produces a rich WorkspaceKnowledge
object that every downstream node consumes instead of discovering it themselves.

Key design decisions:
- Deterministic parsing only (no LLM for workspace structure)
- Recursive monorepo discovery (handles CC4E which has Angular + Spring + shell)
- Caches to ~/.aviator/brain/workspace_cache/{hash}.json
- Returns populated values, NEVER "Unknown" if the workspace exists
"""

import json
import os
import re
import subprocess
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field

# Lazy import to avoid circular deps — actual usage in _load_architecture_model
# from ticket_to_code.agents.architecture_model import ArchitectureModel


class ServiceInfo(BaseModel):
    """A discovered service or module within the workspace."""
    name: str
    path: str
    language: str
    framework: str = ""
    build_file: str = ""


class WorkspaceKnowledge(BaseModel):
    """
    Complete structural knowledge of the workspace.
    Populated by WorkspaceIntelligenceAgent and passed to every pipeline node.
    """
    # Top-level classification
    language: str = Field(default="Unknown")
    framework: str = Field(default="Unknown")
    backend: str = Field(default="Unknown")
    package_manager: str = Field(default="Unknown")

    # Commands
    build_command: str = Field(default="Unknown")
    test_command: str = Field(default="Unknown")

    # Roots
    frontend_root: str = Field(default="Unknown")
    backend_root: str = Field(default="Unknown")
    workspace_root: str = Field(default="Unknown")

    # Monorepo services
    services: List[ServiceInfo] = Field(default_factory=list)

    # Key files discovered
    deployment_scripts: List[str] = Field(default_factory=list)
    config_files: List[str] = Field(default_factory=list)
    docs: List[str] = Field(default_factory=list)

    # Module/package lists
    angular_modules: List[str] = Field(default_factory=list)
    spring_services: List[str] = Field(default_factory=list)

    # Architecture summary (for LLM prompts)
    architecture_summary: str = Field(default="")
    
    # Repository Brain JSON path
    repository_brain_path: str = Field(default="")

    # Domain Ownership: Architecture Model (loaded from YAML or auto-generated)
    architecture_model: Optional[Any] = Field(
        default=None,
        description="ArchitectureModel instance for domain ownership gates",
    )

    def to_prompt_block(self) -> str:
        """Returns a compact block for injecting into LLM prompts."""
        lines = [
            "=== WORKSPACE KNOWLEDGE ===",
            f"Language     : {self.language}",
            f"Framework    : {self.framework}",
            f"Backend      : {self.backend}",
            f"Frontend Root: {self.frontend_root}",
            f"Backend Root : {self.backend_root}",
            f"Build Command: {self.build_command}",
            f"Test Command : {self.test_command}",
        ]
        if self.services:
            lines.append(f"Services     : {', '.join(s.name for s in self.services)}")
        if self.deployment_scripts:
            lines.append(f"Deploy Scripts: {', '.join(self.deployment_scripts)}")
        if self.angular_modules:
            lines.append(f"Angular Modules: {', '.join(self.angular_modules[:8])}")
        if self.spring_services:
            lines.append(f"Spring Services: {', '.join(self.spring_services[:5])}")
        if self.architecture_summary:
            lines.append(f"Architecture: {self.architecture_summary}")
        if self.repository_brain_path:
            lines.append(f"Brain Loaded : Yes ({os.path.basename(self.repository_brain_path)})")
        lines.append("=========================")
        return "\n".join(lines)


class WorkspaceIntelligenceAgent:
    """
    Layer 0: Workspace Intelligence — runs ONCE, caches, feeds everything downstream.

    Scans the workspace recursively to discover:
    - Frontend frameworks (Angular, React, Vue, etc.)
    - Backend frameworks (Spring Boot, Django, Express, etc.)
    - Build systems (Gradle, Maven, npm, pip, etc.)
    - Deployment scripts and CI/CD configuration
    - Module structures for component discovery

    Cache invalidation: git SHA (falls back to hash of manifest files).
    """

    # Files that confirm a particular stack
    ANGULAR_SIGNALS = {"angular.json", "@angular/core"}
    SPRING_SIGNALS = {"spring-boot", "spring-web", "spring-data"}
    SCRIPT_EXTENSIONS = {".sh", ".bat", ".ps1"}
    MAX_SEARCH_DEPTH = 5

    def __init__(self, workspace_path: str):
        self.workspace_path = Path(workspace_path)
        self.cache_dir = Path(os.path.expanduser("~/.aviator/brain/workspace_cache"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def analyze(self) -> WorkspaceKnowledge:
        """Return cached WorkspaceKnowledge or scan fresh."""
        workspace_hash = self._workspace_hash()
        cache_file = self.cache_dir / f"{workspace_hash}.json"

        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                knowledge = WorkspaceKnowledge(**data)
                # Only use cache if it's populated
                if knowledge.language != "Unknown":
                    print(f"[WorkspaceIntelligence] ✅ Cache hit: {cache_file.name}")
                    return knowledge
            except Exception:
                pass

        print(f"[WorkspaceIntelligence] 🔍 Scanning workspace: {self.workspace_path}")
        knowledge = self._build_knowledge()

        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(knowledge.model_dump(), f, indent=2)
            print(f"[WorkspaceIntelligence] 💾 Cached to: {cache_file.name}")
        except Exception as e:
            print(f"[WorkspaceIntelligence] ⚠️  Cache write failed: {e}")

        return knowledge

    def invalidate_cache(self):
        """Force cache bust on next call (call after git pull or indexing)."""
        workspace_hash = self._workspace_hash()
        cache_file = self.cache_dir / f"{workspace_hash}.json"
        if cache_file.exists():
            cache_file.unlink()
            print("[WorkspaceIntelligence] 🗑️  Cache invalidated")

    # -------------------------------------------------------------------------
    # Core Build
    # -------------------------------------------------------------------------

    def _build_knowledge(self) -> WorkspaceKnowledge:
        knowledge = WorkspaceKnowledge(workspace_root=str(self.workspace_path))

        # 1. Discover all services (recursive monorepo scan)
        self._discover_services(knowledge)

        # 2. Discover deployment/build scripts
        self._discover_scripts(knowledge)

        # 3. Discover docs
        self._discover_docs(knowledge)

        # 4. Set top-level language/framework from dominant service
        self._set_top_level_classification(knowledge)

        # 5. Discover Angular modules
        if "Angular" in knowledge.framework:
            self._discover_angular_modules(knowledge)

        # 6. Build architecture summary
        knowledge.architecture_summary = self._build_architecture_summary(knowledge)

        # 7. Wire Repository Brain (Phase 5)
        agents_dir = self.workspace_path / ".agents"
        brain_json = agents_dir / "repository_brain.json"
        
        if not brain_json.exists():
            try:
                print("[WorkspaceIntelligence] Auto-generating Repository Brain...")
                from ticket_to_code.brain.generate_repository_brain import generate_repository_brain
                # Try to use IDE knowledge base if standard path exists
                kb_path = Path.home() / ".gemini" / "antigravity-ide" / "knowledge"
                kb_str = str(kb_path) if kb_path.exists() else None
                success = generate_repository_brain(str(self.workspace_path), kb_str)
                if success and brain_json.exists():
                    knowledge.repository_brain_path = str(brain_json)
            except Exception as e:
                print(f"[WorkspaceIntelligence] WARNING: Failed to generate repository brain: {e}")
        else:
            knowledge.repository_brain_path = str(brain_json)

        # 8. Load Architecture Model (Domain Ownership layer)
        knowledge.architecture_model = self._load_architecture_model(knowledge)

        print(f"[WorkspaceIntelligence] ✅ language={knowledge.language} "
              f"framework={knowledge.framework} backend={knowledge.backend} "
              f"services={[s.name for s in knowledge.services]} "
              f"scripts={knowledge.deployment_scripts} "
              f"architecture_model={'loaded' if knowledge.architecture_model else 'none'}")

        return knowledge

    # -------------------------------------------------------------------------
    # Service Discovery
    # -------------------------------------------------------------------------

    def _discover_services(self, knowledge: WorkspaceKnowledge):
        """Walk workspace up to MAX_SEARCH_DEPTH, discover each service."""
        for root, dirs, files in os.walk(self.workspace_path):
            # Compute depth
            rel = Path(root).relative_to(self.workspace_path)
            depth = len(rel.parts)
            if depth > self.MAX_SEARCH_DEPTH:
                dirs.clear()
                continue

            # Skip noise
            dirs[:] = [d for d in dirs if d not in {
                "node_modules", ".git", "__pycache__", ".gradle",
                "target", "dist", "build", ".angular", "coverage"
            }]

            if "package.json" in files:
                svc = self._parse_npm_service(Path(root), files)
                if svc:
                    knowledge.services.append(svc)
                    knowledge.config_files.append(
                        str(Path(root).relative_to(self.workspace_path) / "package.json")
                    )

            if "pom.xml" in files:
                svc = self._parse_maven_service(Path(root), files)
                if svc:
                    knowledge.services.append(svc)
                    knowledge.config_files.append(
                        str(Path(root).relative_to(self.workspace_path) / "pom.xml")
                    )
                    knowledge.spring_services.append(svc.name)

            if "build.gradle" in files or "build.gradle.kts" in files:
                svc = self._parse_gradle_service(Path(root))
                if svc:
                    knowledge.services.append(svc)

    def _parse_npm_service(self, path: Path, files: list) -> Optional[ServiceInfo]:
        pj = path / "package.json"
        try:
            with open(pj, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return None

        deps = {**data.get("dependencies", {}), **data.get("devDependencies", {})}
        scripts = data.get("scripts", {})
        name = data.get("name", path.name)

        language = "TypeScript" if "typescript" in deps else "JavaScript"
        framework = ""
        build_command = ""
        test_command = ""
        frontend_root = ""

        if "@angular/core" in deps:
            framework = "Angular"
            frontend_root = str(path / "src" / "app")
            build_command = f"cd {path.name} && npm run build"
            test_command = f"cd {path.name} && npm run test"
        elif "react" in deps or "react-dom" in deps:
            framework = "React"
            frontend_root = str(path / "src")
            build_command = f"cd {path.name} && npm run build"
        elif "next" in deps:
            framework = "Next.js"
            build_command = f"cd {path.name} && npm run build"
        elif "vue" in deps:
            framework = "Vue"
            build_command = f"cd {path.name} && npm run build"

        # Override with what's in scripts if present
        if "build" in scripts:
            build_command = f"cd {path.name} && npm run build"
        if "test" in scripts:
            test_command = f"cd {path.name} && npm run test"

        rel_path = str(path.relative_to(self.workspace_path))
        return ServiceInfo(
            name=name,
            path=rel_path,
            language=language,
            framework=framework,
            build_file="package.json",
        )

    def _parse_maven_service(self, path: Path, files: list) -> Optional[ServiceInfo]:
        pom = path / "pom.xml"
        try:
            content = pom.read_text(encoding="utf-8")
        except Exception:
            return None

        # Extract artifactId
        m = re.search(r"<artifactId>([^<]+)</artifactId>", content)
        name = m.group(1).strip() if m else path.name

        framework = ""
        if "spring-boot" in content:
            framework = "Spring Boot"
        elif "spring-web" in content:
            framework = "Spring MVC"

        rel_path = str(path.relative_to(self.workspace_path))
        return ServiceInfo(
            name=name,
            path=rel_path,
            language="Java",
            framework=framework,
            build_file="pom.xml",
        )

    def _parse_gradle_service(self, path: Path) -> Optional[ServiceInfo]:
        build_file = path / "build.gradle"
        if not build_file.exists():
            build_file = path / "build.gradle.kts"
        try:
            content = build_file.read_text(encoding="utf-8")
        except Exception:
            return None

        framework = ""
        if "spring-boot" in content:
            framework = "Spring Boot"

        language = "Kotlin" if build_file.suffix == ".kts" else "Java"
        rel_path = str(path.relative_to(self.workspace_path))
        return ServiceInfo(
            name=path.name,
            path=rel_path,
            language=language,
            framework=framework,
            build_file=build_file.name,
        )

    # -------------------------------------------------------------------------
    # Script Discovery
    # -------------------------------------------------------------------------

    def _discover_scripts(self, knowledge: WorkspaceKnowledge):
        """Find all deployment/build shell scripts."""
        deployment_keywords = {"runjob", "deploy", "build", "start", "release", "publish", "entrypoint"}
        for root, dirs, files in os.walk(self.workspace_path):
            rel = Path(root).relative_to(self.workspace_path)
            if len(rel.parts) > 8:  # deep enough for helm/static/ paths
                dirs.clear()
                continue
            dirs[:] = [d for d in dirs if d not in {
                "node_modules", ".git", "__pycache__", "target", "dist", ".angular"
            }]
            for f in files:
                if any(f.endswith(ext) for ext in self.SCRIPT_EXTENSIONS):
                    # Normalize filename: lowercase, strip separators for keyword matching
                    f_norm = f.lower().replace("-", "").replace("_", "").replace(".", "")
                    if any(kw in f_norm for kw in deployment_keywords):
                        rel_path = str(Path(root).relative_to(self.workspace_path) / f)
                        if rel_path not in knowledge.deployment_scripts:
                            knowledge.deployment_scripts.append(rel_path)

    # -------------------------------------------------------------------------
    # Docs
    # -------------------------------------------------------------------------

    def _discover_docs(self, knowledge: WorkspaceKnowledge):
        for doc in ["README.md", "ARCHITECTURE.md", "CONTRIBUTING.md", "DEVELOPMENT.md"]:
            if (self.workspace_path / doc).exists():
                knowledge.docs.append(doc)
        # Also check subdirectories one level deep
        for child in self.workspace_path.iterdir():
            if child.is_dir() and child.name not in {"node_modules", ".git"}:
                if (child / "README.md").exists():
                    knowledge.docs.append(f"{child.name}/README.md")

    # -------------------------------------------------------------------------
    # Classification
    # -------------------------------------------------------------------------

    def _set_top_level_classification(self, knowledge: WorkspaceKnowledge):
        """
        Set top-level language/framework/backend from discovered services.
        For monorepos: language = 'TypeScript + Java', framework = 'Angular + Spring Boot'.
        Frontend root and backend root come from first Angular/Spring service found.
        """
        frontend_svcs = [s for s in knowledge.services if s.framework in ("Angular", "React", "Vue", "Next.js")]
        backend_svcs = [s for s in knowledge.services if s.language == "Java" and s.framework]
        java_svcs = [s for s in knowledge.services if s.language == "Java"]

        # Language
        langs = list({s.language for s in knowledge.services if s.language})
        if len(langs) == 1:
            knowledge.language = langs[0]
        elif langs:
            knowledge.language = " + ".join(sorted(set(langs)))

        # Framework
        frameworks = list({s.framework for s in knowledge.services if s.framework})
        if frameworks:
            knowledge.framework = " + ".join(sorted(set(frameworks)))
        elif knowledge.services:
            knowledge.framework = knowledge.services[0].language

        # Backend
        if backend_svcs:
            knowledge.backend = backend_svcs[0].framework
        elif java_svcs:
            knowledge.backend = "Java"

        # Package manager
        npm_svcs = [s for s in knowledge.services if s.build_file == "package.json"]
        maven_svcs = [s for s in knowledge.services if s.build_file == "pom.xml"]
        if npm_svcs and maven_svcs:
            knowledge.package_manager = "npm + Maven"
        elif npm_svcs:
            knowledge.package_manager = "npm"
        elif maven_svcs:
            knowledge.package_manager = "Maven"

        # Frontend root
        if frontend_svcs:
            knowledge.frontend_root = frontend_svcs[0].path + "/src/app"
            knowledge.build_command = f"cd {frontend_svcs[0].path} && npm run build"
            knowledge.test_command = f"cd {frontend_svcs[0].path} && npm run test"

        # Backend root
        if backend_svcs:
            knowledge.backend_root = backend_svcs[0].path + "/src/main/java"
        elif java_svcs:
            knowledge.backend_root = java_svcs[0].path + "/src/main/java"

    # -------------------------------------------------------------------------
    # Angular Module Discovery
    # -------------------------------------------------------------------------

    def _discover_angular_modules(self, knowledge: WorkspaceKnowledge):
        """Discover Angular module names from *.module.ts files."""
        angular_svc = next(
            (s for s in knowledge.services if s.framework == "Angular"), None
        )
        if not angular_svc:
            return

        angular_root = self.workspace_path / angular_svc.path / "src"
        if not angular_root.exists():
            return

        for path in angular_root.rglob("*.module.ts"):
            if "node_modules" not in str(path):
                # Extract module name from filename
                module_name = path.stem.replace(".module", "")
                if module_name not in knowledge.angular_modules:
                    knowledge.angular_modules.append(module_name)

    # -------------------------------------------------------------------------
    # Architecture Summary
    # -------------------------------------------------------------------------

    def _build_architecture_summary(self, knowledge: WorkspaceKnowledge) -> str:
        parts = []
        if knowledge.services:
            svc_names = ", ".join(f"{s.name}({s.framework or s.language})" for s in knowledge.services)
            parts.append(f"Monorepo with: {svc_names}")
        if knowledge.deployment_scripts:
            parts.append(f"Deployment scripts: {', '.join(knowledge.deployment_scripts)}")
        if knowledge.angular_modules:
            parts.append(f"Angular modules: {', '.join(knowledge.angular_modules[:10])}")
        return ". ".join(parts) if parts else "Standard workspace"

    # -------------------------------------------------------------------------
    # Architecture Model Loader (Domain Ownership)
    # -------------------------------------------------------------------------

    def _load_architecture_model(self, knowledge: WorkspaceKnowledge):
        """Load or auto-generate the ArchitectureModel for domain ownership.

        Search order:
        1. {workspace}/brain/knowledge/architecture_model.yaml  (workspace-level)
        2. {plugin}/config/architecture_model.yaml              (plugin default)
        3. Auto-generate from discovered services               (fallback)

        Returns ArchitectureModel or None on failure.
        """
        try:
            from ticket_to_code.agents.architecture_model import ArchitectureModel
        except ImportError as e:
            print(f"[WorkspaceIntelligence] WARNING: Could not import ArchitectureModel: {e}")
            return None

        # 1. Workspace-level YAML (developer-maintained, highest authority)
        ws_yaml = self.workspace_path / "brain" / "knowledge" / "architecture_model.yaml"
        if ws_yaml.exists():
            print(f"[WorkspaceIntelligence] Loading architecture model from {ws_yaml}")
            return ArchitectureModel.load_from_yaml(str(ws_yaml))

        # 2. Plugin config directory (default for this plugin)
        plugin_yaml = Path(__file__).parent.parent / "config" / "architecture_model.yaml"
        if plugin_yaml.exists():
            print(f"[WorkspaceIntelligence] Loading architecture model from {plugin_yaml}")
            return ArchitectureModel.load_from_yaml(str(plugin_yaml))

        # 3. Auto-generate from discovered services (fallback — soft authority only)
        if knowledge.services:
            print("[WorkspaceIntelligence] Auto-generating architecture model from discovered services")
            service_names = [s.name for s in knowledge.services]
            # Search for any architecture map markdown (project-agnostic)
            _knowledge_dir = self.workspace_path / "brain" / "knowledge"
            _arch_map_path = None
            if _knowledge_dir.exists():
                # Try generic name first, then any *_architecture_map.md
                for _pattern in ["architecture_map.md", "*_architecture_map.md"]:
                    _matches = list(_knowledge_dir.glob(_pattern))
                    if _matches:
                        _arch_map_path = _matches[0]
                        break
            return ArchitectureModel.from_workspace_knowledge(
                service_names,
                str(_arch_map_path) if _arch_map_path else None,
            )

        print("[WorkspaceIntelligence] No architecture model available (no services discovered)")
        return None

    # -------------------------------------------------------------------------
    # Cache Key
    # -------------------------------------------------------------------------

    def _workspace_hash(self) -> str:
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=str(self.workspace_path),
                capture_output=True, text=True, check=True
            )
            sha = result.stdout.strip()
            if sha:
                return f"{self.workspace_path.name}_{sha[:12]}"
        except Exception:
            pass

        hasher = hashlib.md5()
        for manifest in ["package.json", "pom.xml", "angular.json", "tsconfig.json"]:
            for root, dirs, files in os.walk(self.workspace_path):
                dirs[:] = [d for d in dirs if d not in {"node_modules", ".git", "target"}]
                if manifest in files:
                    try:
                        hasher.update((Path(root) / manifest).read_bytes())
                    except Exception:
                        pass
        return f"{self.workspace_path.name}_{hasher.hexdigest()[:12]}"
