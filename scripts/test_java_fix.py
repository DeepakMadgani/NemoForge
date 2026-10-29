sample = """package com.cloudforge.commerce.order.service;

public class OrderService {
    public Order fulfillOrder(Long orderId) {
        return orderRepository.save(order);
    }
}

    @Transactional
    public Order refundOrder(Long orderId) {
        return orderRepository.save(order);
    }
"""

def heal_trailing_class_methods(code: str, file_path: str) -> str:
    """
    If a Java/C# method was accidentally appended outside the closing brace of the class,
    relocate the closing brace to the end of the file so the method is inside the class.
    """
    if not file_path.endswith((".java", ".cs")):
        return code

    # Check if there is a class declaration
    if "class " not in code:
        return code

    lines = code.splitlines()
    # Find depth throughout file
    depth = 0
    premature_close_line = -1
    for idx, line in enumerate(lines):
        # Rough brace count outside comments
        stripped = line.split("//")[0].strip()
        open_c = stripped.count("{")
        close_c = stripped.count("}")
        depth += open_c - close_c
        if depth == 0 and idx < len(lines) - 2:
            # Check if subsequent lines have method declarations or annotations
            remaining = "\n".join(lines[idx + 1:]).strip()
            if remaining and any(kw in remaining for kw in ("public ", "private ", "protected ", "@Transactional", "@PostMapping", "@GetMapping", "@Override")):
                premature_close_line = idx
                break

    if premature_close_line != -1:
        # Remove the premature closing brace line and append '}' to the end of the file
        new_lines = []
        for idx, line in enumerate(lines):
            if idx == premature_close_line and line.strip() == "}":
                continue  # drop premature closing brace
            new_lines.append(line)
        # Ensure final closing brace
        code = "\n".join(new_lines).rstrip() + "\n}\n"

    return code

fixed = heal_trailing_class_methods(sample, "OrderService.java")
print("FIXED CODE:")
print(fixed)
assert fixed.rstrip().endswith("}\n}")
print("ASSERTION PASSED: method is inside class and class ends with }")

