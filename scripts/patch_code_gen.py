import re
from pathlib import Path

path = Path("nemoforge/src/ticket_to_code/agents/code_generator.py")
content = path.read_text(encoding="utf-8")

func_def = """

def heal_trailing_class_methods(code: str, file_path: str) -> str:
    \"\"\"
    If a Java/C# method was accidentally appended outside the closing brace of the class,
    relocate the closing brace to the end of the file so the method is inside the class.
    \"\"\"
    if not file_path.endswith((".java", ".cs")):
        return code

    if "class " not in code and "record " not in code:
        return code

    lines = code.splitlines()
    depth = 0
    premature_line = -1
    for idx, line in enumerate(lines):
        stripped = line.split("//")[0].strip()
        open_c = stripped.count("{")
        close_c = stripped.count("}")
        depth += open_c - close_c
        if depth == 0 and idx < len(lines) - 2:
            remaining = "\\n".join(lines[idx + 1:]).strip()
            if remaining and any(kw in remaining for kw in ("public ", "private ", "protected ", "@Transactional", "@PostMapping", "@GetMapping", "@Override")):
                premature_line = idx
                break

    if premature_line != -1:
        new_lines = [l for i, l in enumerate(lines) if not (i == premature_line and l.strip() == "}")]
        net_depth = sum(l.split("//")[0].count("{") - l.split("//")[0].count("}") for l in new_lines)
        if net_depth > 0:
            return "\\n".join(new_lines).rstrip() + "\\n" + ("}\\n" * net_depth)
        return "\\n".join(new_lines) + "\\n"

    return code
"""

# Normalize CRLF for robust matching
crlf = "\r\n" in content
clean_content = content.replace("\r\n", "\n")

target1 = '    return code_str\n"""'
if target1 in clean_content:
    clean_content = clean_content.replace(target1, '    return code_str\n' + func_def + '\n"""', 1)
    print("Inserted heal_trailing_class_methods definition.")
else:
    print("Target 1 not found")

target2 = '''    if file_path.endswith((".tsx", ".jsx")):
        result = _heal_react_destructuring(result)'''

replacement2 = '''    if file_path.endswith((".tsx", ".jsx")):
        result = _heal_react_destructuring(result)
    elif file_path.endswith((".java", ".cs")):
        result = heal_trailing_class_methods(result, file_path)'''

if target2 in clean_content:
    clean_content = clean_content.replace(target2, replacement2, 1)
    print("Patched _apply_str_replace_edits.")
else:
    print("Target 2 not found")

target3 = '''                        # ── Immediate in-memory session propagation ──'''
replacement3 = '''                        if task.file_path.endswith((".java", ".cs")):
                            generated.content = heal_trailing_class_methods(generated.content, task.file_path)
                        elif task.file_path.endswith((".tsx", ".jsx")):
                            generated.content = _heal_react_destructuring(generated.content)

                        # ── Immediate in-memory session propagation ──'''

if target3 in clean_content:
    clean_content = clean_content.replace(target3, replacement3, 1)
    print("Patched generate_code session propagation.")
else:
    print("Target 3 not found")

if crlf:
    clean_content = clean_content.replace("\n", "\r\n")

path.write_text(clean_content, encoding="utf-8")
print("Done writing.")

