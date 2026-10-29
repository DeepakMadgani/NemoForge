import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Set, Optional

class ModuleNode:
    def __init__(self, name: str, path: str):
        self.name = name
        self.path = path
        self.dependencies: Set[str] = set()

class ModuleGraph:
    def __init__(self):
        self.modules: Dict[str, ModuleNode] = {}

    def get_summary_string(self) -> str:
        if not self.modules:
            return "No distinct module boundaries detected."
        lines = ["PROJECT MODULE GRAPH (Microservice Boundaries):"]
        for name, node in self.modules.items():
            deps = ", ".join(node.dependencies) if node.dependencies else "none"
            lines.append(f"  - Module: {name} (path: {node.path})")
            lines.append(f"    Declared Local Dependencies: {deps}")
        return "\n".join(lines)
        
    def get_module_for_file(self, file_path: str) -> Optional[str]:
        target_path = file_path.replace("\\", "/")
        found_module = None
        for name, node in self.modules.items():
            npath = node.path.replace("\\", "/")
            if npath == ".":
                continue
            if target_path.startswith(npath + "/") or target_path == npath:
                if not found_module or len(npath) > len(self.modules[found_module].path):
                    found_module = name
        return found_module

    def can_import(self, source_path: str, target_path: str) -> bool:
        """
        Check if a file in source_path can legitimately import a file in target_path.
        """
        source_module = self.get_module_for_file(source_path)
        target_module = self.get_module_for_file(target_path)
                    
        # If they resolve to the same module, or we couldn't resolve one of them, allow it.
        if source_module == target_module or not source_module or not target_module:
            return True
            
        # They are in different local modules. Check if source depends on target.
        return target_module in self.modules[source_module].dependencies

class DependencyGraphProvider:
    def get_module_boundaries(self, workspace_path: str) -> ModuleGraph:
        raise NotImplementedError

class MavenDependencyGraphProvider(DependencyGraphProvider):
    def get_module_boundaries(self, workspace_path: str) -> ModuleGraph:
        graph = ModuleGraph()
        ws_path = Path(workspace_path).resolve()
        
        # Fast walk to avoid node_modules and target dirs
        pom_files = []
        for root, dirs, files in os.walk(str(ws_path)):
            dirs[:] = [d for d in dirs if d not in {".git", "node_modules", "target", "dist", "build"}]
            if "pom.xml" in files:
                pom_files.append(Path(root) / "pom.xml")
        
        pom_data = {}
        for pom in pom_files:
            try:
                tree = ET.parse(pom)
                root = tree.getroot()
                
                def find_text(element, tag):
                    for child in element:
                        if child.tag.endswith(f'}}' + tag) or child.tag == tag:
                            return child.text
                    return None
                    
                artifact_id = find_text(root, 'artifactId')
                if not artifact_id:
                    continue
                    
                parent_id = None
                for child in root:
                    if child.tag.endswith('}parent') or child.tag == 'parent':
                        parent_id = find_text(child, 'artifactId')
                        
                deps = set()
                for child in root:
                    if child.tag.endswith('}dependencies') or child.tag == 'dependencies':
                        for dep in child:
                            if dep.tag.endswith('}dependency') or dep.tag == 'dependency':
                                dep_artifact = find_text(dep, 'artifactId')
                                if dep_artifact:
                                    deps.add(dep_artifact)
                                    
                rel_path = str(pom.resolve().parent.relative_to(ws_path))
                pom_data[artifact_id] = {
                    "path": rel_path,
                    "parent": parent_id,
                    "dependencies": deps
                }
                
                graph.modules[artifact_id] = ModuleNode(artifact_id, rel_path)
            except Exception:
                continue
                
        # Resolve transitive dependencies from parents
        for artifact_id, node in graph.modules.items():
            data = pom_data[artifact_id]
            deps = set(data["dependencies"])
            
            curr_parent = data["parent"]
            visited = set()
            while curr_parent and curr_parent in pom_data and curr_parent not in visited:
                visited.add(curr_parent)
                deps.update(pom_data[curr_parent]["dependencies"])
                curr_parent = pom_data[curr_parent]["parent"]
                
            local_deps = {d for d in deps if d in pom_data}
            node.dependencies = local_deps
            
        return graph

def get_dependency_graph(workspace_path: str) -> ModuleGraph:
    try:
        has_pom = next(Path(workspace_path).rglob("pom.xml"), None) is not None
        if has_pom:
            return MavenDependencyGraphProvider().get_module_boundaries(workspace_path)
    except Exception:
        pass
    return ModuleGraph()
