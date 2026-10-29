with open('nemoforge/src/ticket_to_code/agents/code_generator.py', 'r', encoding='utf-8') as f:
    text = f.read()

target = '    return result\n\n\n# ─── Tier Helper Functions'
replacement = '''    if file_path.endswith((".tsx", ".jsx")):
        result = _heal_react_destructuring(result)

    return result


# ─── Tier Helper Functions'''

if target in text:
    text = text.replace(target, replacement, 1)
    with open('nemoforge/src/ticket_to_code/agents/code_generator.py', 'w', encoding='utf-8') as f:
        f.write(text)
    print("Successfully added _heal_react_destructuring to _apply_str_replace_edits")
else:
    print("Target not found!")

