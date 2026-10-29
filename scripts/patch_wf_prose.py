with open('nemoforge/src/ticket_to_code/workflow.py', 'r', encoding='utf-8') as f:
    text = f.read()

target = '        (r"^\\s*(?:Here\\s+is\\s+the|Below\\s+is\\s+the|Note\\s+that\\s+we)\\b", "Explanatory prose leak"),'
repl = '''        (r"^\\s*(?:Here\\s+is\\s+the|Below\\s+is\\s+the|Note\\s+that\\s+we)\\b", "Explanatory prose leak"),
        (r"^\\s*(?:But\\s+wait:|However,|Therefore,|SEARCH\\s+for:|REPLACE\\s+with:|Third,\\s+we\\s+add|What\\s+about\\s+an\\s+order|We'll\\s+need\\s+to)\\b", "Leaked chain-of-thought analysis"),'''

if target in text:
    text = text.replace(target, repl, 1)
    with open('nemoforge/src/ticket_to_code/workflow.py', 'w', encoding='utf-8') as f:
        f.write(text)
    print("Enhanced prose patterns in workflow.py")
else:
    print("target not found in workflow.py")

