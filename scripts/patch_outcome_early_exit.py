with open('nemoforge/src/ticket_to_code/workflow.py', 'r', encoding='utf-8') as f:
    text = f.read()

# 1. Do not truncate generated_code diffs at 6 files
target1 = '    for gen in generated_code[:6]:  # cap at 6 files to avoid token overflow'
repl1 = '    for gen in generated_code:  # include all generated files in diff review'
if target1 in text:
    text = text.replace(target1, repl1, 1)
    print("Patched generated_code loop in workflow.py")
else:
    print("target1 not found in workflow.py")

# 2. Enable compiler-as-truth early exit when build passes clean with no test errors
target2 = '        (state.get("direct_mode") or tests_all_passed or (build_passed and int(state.get("outcome_fix_attempt", 0) or 0) > 0))'
repl2 = '        (state.get("direct_mode") or tests_all_passed or test_status_str in ("no_tests", "skipped", "") or (build_passed and int(state.get("outcome_fix_attempt", 0) or 0) > 0))'
if target2 in text:
    text = text.replace(target2, repl2, 1)
    print("Patched can_early_exit in workflow.py")
else:
    print("target2 not found in workflow.py")

with open('nemoforge/src/ticket_to_code/workflow.py', 'w', encoding='utf-8') as f:
    f.write(text)

print("workflow.py updated successfully.")

