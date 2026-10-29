with open('nemoforge/src/ticket_to_code/agents/code_generator.py', 'r', encoding='utf-8') as f:
    text = f.read()

target = '- For INSERTING a new method after an existing method: use the closing brace + any trailing lines of the preceding method as SEARCH, include that same closing brace + the new method as REPLACE'
replacement = '''- For INSERTING a new method after an existing method: use the closing brace + any trailing lines of the preceding method as SEARCH, include that same closing brace + the new method as REPLACE
- NEVER re-declare existing fields, class annotations, or existing methods when adding a new method to an existing class. Include ONLY the single NEW method in the REPLACE block.'''

if target in text:
    text = text.replace(target, replacement, 1)
    with open('nemoforge/src/ticket_to_code/agents/code_generator.py', 'w', encoding='utf-8') as f:
        f.write(text)
    print("Added rule to code_generator.py")
else:
    print("Target not found")

