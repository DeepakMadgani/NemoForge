with open('nemoforge/src/ticket_to_code/agents/code_generator.py', 'r', encoding='utf-8') as f:
    text = f.read()

import re

# Remove the broken block
broken_block = text[text.find('                if not _search_matches:\n                    _search_matches = list(re.finditer(\n                        r\'(?:SEARCH'):text.find('                if not _search_matches:\n                    _search_matches = list(re.finditer(\n                        r\'(?:TARGET')]

clean_block = '''                if not _search_matches:
                    _search_matches = list(re.finditer(
                        r'(?:SEARCH(?:\\s+FOR)?|TARGET(?:_CONTENT)?|ORIGINAL):\\s*\\r?\\n?(.*?)\\r?\\n?(?:REPLACE(?:\\s+WITH)?|REPLACEMENT(?:_CONTENT)?):\\s*\\r?\\n?(.*?)(?=\\r?\\n(?:SEARCH(?:\\s+FOR)?|TARGET(?:_CONTENT)?|ORIGINAL):|\\Z)',
                        raw, re.DOTALL | re.IGNORECASE
                    ))
'''

if broken_block:
    text = text.replace(broken_block, clean_block, 1)
    with open('nemoforge/src/ticket_to_code/agents/code_generator.py', 'w', encoding='utf-8') as f:
        f.write(text)
    print("Fixed broken regex block in code_generator.py")
else:
    print("Broken block not found via substring, trying alternative...")

