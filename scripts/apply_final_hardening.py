import re
from pathlib import Path

# 1. Raise effective_max tokens in code_generator.py to 8192
cg_path = Path("nemoforge/src/ticket_to_code/agents/code_generator.py")
cg_text = cg_path.read_text(encoding="utf-8")

target1 = "effective_max = 4096"
repl1 = "effective_max = 8192"
if target1 in cg_text:
    cg_text = cg_text.replace(target1, repl1, 1)
    print("Raised effective_max to 8192 in code_generator.py")
else:
    print("effective_max target not found in code_generator.py")

# 2. Prevent Fallback 2 from writing corrupted prose or AST-broken files on MODIFY tasks
target2 = """            logger.info(f"Extracted code via resilient fallback ({len(extracted_raw_code)} chars)")
            return GeneratedCode(
                file_path=task.file_path,
                content=extracted_raw_code,
                language=task.language,
                change_type=task.task_type,
                documentation="Generated code (resilient fallback)",
                imports=[],
            )"""

repl2 = """            # Strict Gate: For MODIFY tasks, reject unanchored raw text if it has syntax errors, prose leaks, or massive bloat
            if task.task_type.value == "modify" and _existing:
                _has_prose = any(phrase in extracted_raw_code for phrase in [
                    "But wait:", "However,", "Therefore,", "SEARCH for:", "REPLACE with:",
                    "Third, we add", "What about", "We'll need to"
                ])
                if _has_prose or len(extracted_raw_code) > len(_existing) * 2.5:
                    logger.warning(
                        f"Resilient fallback rejected for MODIFY task on {task.file_path}: "
                        f"detected prose leak or excessive length ({len(extracted_raw_code)} chars vs {len(_existing)} orig). "
                        f"Preserving original file."
                    )
                    raise ValueError(
                        f"SEARCH block not found: LLM produced unanchored conversational text instead of SEARCH/REPLACE blocks. "
                        f"You must use <<<<<<< SEARCH / ======= / >>>>>>> REPLACE blocks."
                    )

            logger.info(f"Extracted code via resilient fallback ({len(extracted_raw_code)} chars)")
            return GeneratedCode(
                file_path=task.file_path,
                content=extracted_raw_code,
                language=task.language,
                change_type=task.task_type,
                documentation="Generated code (resilient fallback)",
                imports=[],
            )"""

if target2 in cg_text:
    cg_text = cg_text.replace(target2, repl2, 1)
    print("Added strict prose gate to resilient fallback in code_generator.py")
else:
    print("target2 not found in code_generator.py")

# 3. Add SEARCH for: / REPLACE with: pattern support to code_generator.py
target3 = "                if not _search_matches:\n                    _search_matches = list(re.finditer(\n                        r'(?:TARGET(?:_CONTENT)?|ORIGINAL):"
repl3 = """                if not _search_matches:
                    _search_matches = list(re.finditer(
                        r'(?:SEARCH(?:\s+FOR)?|TARGET(?:_CONTENT)?|ORIGINAL):\s*\r?\n?(.*?)\r?\n?(?:REPLACE(?:\s+WITH)?|REPLACEMENT(?:_CONTENT)?):\s*\r?\n?(.*?)(?=\r?\n(?:SEARCH(?:\s+FOR)?|TARGET(?:_CONTENT)?|ORIGINAL):|\Z)',
                        raw, re.DOTALL | re.IGNORECASE
                    ))
                if not _search_matches:
                    _search_matches = list(re.finditer(
                        r'(?:TARGET(?:_CONTENT)?|ORIGINAL):"""

if target3 in cg_text:
    cg_text = cg_text.replace(target3, repl3, 1)
    print("Added SEARCH for / REPLACE with pattern to code_generator.py")
else:
    print("target3 not found in code_generator.py")

cg_path.write_text(cg_text, encoding="utf-8")

# 4. Enhance conversational prose detection in workflow.py
wf_path = Path("nemoforge/src/ticket_to_code/workflow.py")
wf_text = wf_path.read_text(encoding="utf-8")

target_wf = """        (r"^\s*(?:But\s+note|However,\s*note|Please\s+note|Looking\s+at\s+the|Alternatively|We\s+must\s+place|We\s+should\s+place)\b", "Conversational analysis leak"),
        (r"^\s*(?:Here\s+is\s+the|Below\s+is\s+the|Note\s+that\s+we)\b", "Explanatory prose leak"),"""

repl_wf = """        (r"^\s*(?:But\s+note|However,\s*note|Please\s+note|Looking\s+at\s+the|Alternatively|We\s+must\s+place|We\s+should\s+place)\b", "Conversational analysis leak"),
        (r"^\s*(?:Here\s+is\s+the|Below\s+is\s+the|Note\s+that\s+we)\b", "Explanatory prose leak"),
        (r"^\s*(?:But\s+wait:|However,|Therefore,|SEARCH\s+for:|REPLACE\s+with:|Third,\s+we\s+add|What\s+about\s+an\s+order|We'll\s+need\s+to)\b", "Leaked chain-of-thought analysis"),"""

if target_wf in wf_text:
    wf_text = wf_text.replace(target_wf, repl_wf, 1)
    wf_path.write_text(wf_text, encoding="utf-8")
    print("Enhanced prose patterns in workflow.py")
else:
    print("target_wf not found in workflow.py")

print("All fixes applied.")

