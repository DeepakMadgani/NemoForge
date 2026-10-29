with open('nemoforge/src/ticket_to_code/agents/error_resolution_agent.py', 'r', encoding='utf-8') as f:
    text = f.read()

bad = '''            "CRITICAL INSTRUCTION:
DO NOT OUTPUT ANY PREAMBLE, EXPLANATION, OR REASONING TEXT.
YOUR RESPONSE MUST BE ONLY A JSON ACTION OBJECT.
START YOUR RESPONSE IMMEDIATELY WITH {"action": "REPLACE_CONTENT", ...}."'''

good = '''            "CRITICAL INSTRUCTION:\\n"
            "DO NOT OUTPUT ANY PREAMBLE, EXPLANATION, OR REASONING TEXT.\\n"
            "YOUR RESPONSE MUST BE ONLY A JSON ACTION OBJECT.\\n"
            "START YOUR RESPONSE IMMEDIATELY WITH {\\"action\\": \\"REPLACE_CONTENT\\", ...}.\\n"'''

if bad in text:
    text = text.replace(bad, good, 1)
    with open('nemoforge/src/ticket_to_code/agents/error_resolution_agent.py', 'w', encoding='utf-8') as f:
        f.write(text)
    print("Cleanly replaced in error_resolution_agent.py")
else:
    print("Bad string not found with exact match, searching lines...")
    lines = text.splitlines(keepends=True)
    new_lines = []
    skip = False
    for line in lines:
        if 'CRITICAL INSTRUCTION:' in line:
            new_lines.append('            "CRITICAL INSTRUCTION:\\n"\n')
            new_lines.append('            "DO NOT OUTPUT ANY PREAMBLE, EXPLANATION, OR REASONING TEXT.\\n"\n')
            new_lines.append('            "YOUR RESPONSE MUST BE ONLY A JSON ACTION OBJECT.\\n"\n')
            new_lines.append('            "START YOUR RESPONSE IMMEDIATELY WITH {\\"action\\": \\"REPLACE_CONTENT\\", ...}.\\n"\n')
            skip = True
            continue
        if skip:
            if 'START YOUR RESPONSE IMMEDIATELY' in line:
                skip = False
            continue
        new_lines.append(line)
    with open('nemoforge/src/ticket_to_code/agents/error_resolution_agent.py', 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    print("Replaced line by line.")

