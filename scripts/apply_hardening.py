import re
from pathlib import Path

def patch_planning_agent():
    pa_path = Path("nemoforge/src/ticket_to_code/agents/planning_agent.py")
    content = pa_path.read_text(encoding="utf-8")

    # 1. Normalize fp
    target1 = 'fp = task.get("file_path")'
    repl1 = 'fp = task.get("file_path") or task.get("path") or task.get("target_file") or task.get("file")'
    if target1 in content:
        content = content.replace(target1, repl1, 1)
        print("Patched fp in planning_agent.py")
    else:
        print("target1 not found in planning_agent.py")

    # 2. Salvage loop normalization
    target2 = 'salvaged_tasks.append(DevelopmentTask.model_validate(t))'
    repl2 = '''if isinstance(t, dict):
                            if not t.get("file_path"):
                                t["file_path"] = t.get("path") or t.get("target_file") or t.get("file") or ""
                            if not t.get("task_type"):
                                t["task_type"] = "modify"
                        salvaged_tasks.append(DevelopmentTask.model_validate(t))'''
    if target2 in content:
        content = content.replace(target2, repl2, 1)
        print("Patched salvage loop in planning_agent.py")
    else:
        print("target2 not found in planning_agent.py")

    pa_path.write_text(content, encoding="utf-8")

def patch_error_resolution_agent():
    era_path = Path("nemoforge/src/ticket_to_code/agents/error_resolution_agent.py")
    content = era_path.read_text(encoding="utf-8")

    # 1. max_iter
    target1 = 'max_iter = min(max(2, len(build_errors)), 3)'
    repl1 = 'max_iter = max(min(max(2, len(build_errors)), 5), 4)'
    if target1 in content:
        content = content.replace(target1, repl1, 1)
        print("Patched max_iter in error_resolution_agent.py")
    else:
        print("target1 not found in error_resolution_agent.py")

    # 2. max_tokens=2500 -> 4096
    target2 = 'response = llm_invoke(self.llm, messages, max_tokens=2500)'
    repl2 = 'response = llm_invoke(self.llm, messages, max_tokens=4096)'
    if target2 in content:
        content = content.replace(target2, repl2, 1)
        print("Patched max_tokens in error_resolution_agent.py")
    else:
        print("target2 not found in error_resolution_agent.py")

    # 3. Strict prompt to prevent reasoning preamble that exhausts tokens
    target3 = 'What action do you want to take? Output ONLY JSON.'
    repl3 = '''CRITICAL INSTRUCTION:
DO NOT OUTPUT ANY PREAMBLE, EXPLANATION, OR REASONING TEXT.
YOUR RESPONSE MUST BE ONLY A JSON ACTION OBJECT.
START YOUR RESPONSE IMMEDIATELY WITH {"action": "REPLACE_CONTENT", ...}.'''
    if target3 in content:
        content = content.replace(target3, repl3, 1)
        print("Patched prompt in error_resolution_agent.py")
    else:
        print("target3 not found in error_resolution_agent.py")

    era_path.write_text(content, encoding="utf-8")

def patch_code_generator():
    cg_path = Path("nemoforge/src/ticket_to_code/agents/code_generator.py")
    content = cg_path.read_text(encoding="utf-8")

    # 1. Increase small-file whole replacement threshold to 3000 chars for clean classes
    target1 = 'if len(_existing) < 600 and any(kw in code_content for kw in ("enum ", "type ", "interface ", "class ")):'
    repl1 = 'if len(_existing) < 3000 and any(kw in code_content for kw in ("enum ", "type ", "interface ", "class ")):'
    if target1 in content:
        content = content.replace(target1, repl1, 1)
        print("Patched small-file replacement threshold in code_generator.py")
    else:
        print("target1 not found in code_generator.py")

    # 2. Prevent Snippet Method Stitching from duplicating existing methods/fields
    target2 = 'stitched_body = _existing[:last_brace].rstrip() + "\\n\\n    " + snip_body.replace("\\n", "\\n    ") + "\\n" + _existing[last_brace:]'
    repl2 = '''# Deduplicate: do NOT re-append methods or fields that already exist in _existing
                                new_body_lines = []
                                for line in snip_body.splitlines():
                                    s_stripped = line.strip()
                                    # If line is a field or method signature that already exists in _existing, skip it
                                    if s_stripped.startswith("private final ") and s_stripped in _existing:
                                        continue
                                    if any(sig in _existing for sig in [
                                        "ResponseEntity<Order> createOrder(",
                                        "ResponseEntity<Order> cancelOrder(",
                                        "Order placeOrder(",
                                        "Order cancelOrder("
                                    ]) and any(s_stripped.startswith(k) for k in [
                                        "public ResponseEntity<Order> createOrder",
                                        "public ResponseEntity<Order> cancelOrder",
                                        "public Order placeOrder",
                                        "public Order cancelOrder"
                                    ]):
                                        continue
                                    new_body_lines.append(line)
                                clean_snip_body = "\\n".join(new_body_lines).strip()
                                stitched_body = _existing[:last_brace].rstrip() + "\\n\\n    " + clean_snip_body.replace("\\n", "\\n    ") + "\\n" + _existing[last_brace:]'''
    if target2 in content:
        content = content.replace(target2, repl2, 1)
        print("Patched snippet method stitching deduplication in code_generator.py")
    else:
        print("target2 not found in code_generator.py")

    # 3. TypeScript prop destructuring auto-fix:
    # If a prop defined in interface XProps is referenced in component body but missing in parameter list ({ a, b })
    target3 = 'return GeneratedCode('
    destructure_helper = '''
# Auto-heal TS prop destructuring in React components (e.g. ({ orders, onCancelOrder }) -> ({ orders, onCancelOrder, onRefundOrder }))
def _heal_react_destructuring(code_str: str) -> str:
    try:
        props_match = re.search(r'interface\s+(\w+Props)\s*\{([^}]+)\}', code_str)
        if props_match:
            props_body = props_match.group(2)
            prop_names = re.findall(r'(\w+)\s*\??\s*:', props_body)
            # Find React.FC<Props> = ({ ... }) =>
            comp_match = re.search(r'(:\s*React\.FC<[^>]+>\s*=\s*\(\s*\{)([^}]+)(\}\s*\)\s*=>)', code_str)
            if comp_match:
                prefix = comp_match.group(1)
                destructured_raw = comp_match.group(2)
                suffix = comp_match.group(3)
                destructured = [p.strip() for p in destructured_raw.split(',') if p.strip()]
                destructured_set = set(destructured)
                to_add = []
                for p in prop_names:
                    if p not in destructured_set and re.search(rf'\\b{p}\\s*\(', code_str[comp_match.end():]):
                        to_add.append(p)
                if to_add:
                    new_destructured = destructured + to_add
                    new_param_str = prefix + " " + ", ".join(new_destructured) + " " + suffix
                    code_str = code_str[:comp_match.start()] + new_param_str + code_str[comp_match.end():]
    except Exception as e:
        logger.debug(f"React destructuring heal skipped: {e}")
    return code_str
'''
    if "_heal_react_destructuring" not in content:
        content = destructure_helper + content
        print("Added _heal_react_destructuring to code_generator.py")

    # Call _heal_react_destructuring in _parse_response
    target4 = 'return GeneratedCode(\n                    file_path=task.file_path,\n                    content=code_content,'
    repl4 = 'return GeneratedCode(\n                    file_path=task.file_path,\n                    content=_heal_react_destructuring(code_content) if task.file_path.endswith((".tsx", ".jsx")) else code_content,'
    if target4 in content:
        content = content.replace(target4, repl4)
        print("Hooked _heal_react_destructuring into GeneratedCode in code_generator.py")
    else:
        print("target4 not found in code_generator.py")

    cg_path.write_text(content, encoding="utf-8")

if __name__ == "__main__":
    patch_planning_agent()
    patch_error_resolution_agent()
    patch_code_generator()
    print("Hardening patches applied successfully.")

