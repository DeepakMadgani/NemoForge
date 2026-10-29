with open(r'C:/Users/dmadgani/.gemini/antigravity/brain/ddeea679-8e9c-4c3c-af4c-63d88148a5bb/.system_generated/tasks/task-4310.log', 'r', encoding='utf-8', errors='ignore') as f:
    text = f.read()

idx = text.rfind("has no attribute 'file_path'")
if idx != -1:
    print(text[max(0, idx-3000):idx])

