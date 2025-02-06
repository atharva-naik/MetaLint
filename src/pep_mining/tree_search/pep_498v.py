import ast

def detect_pep_498v(code: str, ast_root: ast.AST) -> list | None:
    """
    Detects PEP 498 violations in the provided code string using the given AST root.
    Returns a list of tuples with the start line, end line, and type of violation 
    ('.format()' or '%' usage). Returns None if no violations are found.

    Args:
        code (str): The source code to analyze.
        ast_root (ast.AST): The parsed AST tree of the code.

    Returns:
        list | None: A list of tuples (start_line, end_line, violation_type), 
                     or None if no violations are found.
    """
    violations = []

    for node in ast.walk(ast_root):
        # Detect `.format()` usage
        if isinstance(node, ast.Call):
            if (isinstance(node.func, ast.Attribute) and
                node.func.attr == "format" and
                isinstance(node.func.value, ast.Str)):
                violations.append((node.lineno, node.end_lineno, ".format()"))
        
        # Detect `%` usage
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mod):
            if isinstance(node.left, ast.Str):
                violations.append((node.lineno, node.end_lineno, "%"))

    return violations if violations else None

# main
if __name__ == "__main__":
    code = """
name = "Alice"
greeting = "Hello, {}!".format(name)
result = "Sum: %d" % (2 + 3)
"""

    ast_root = ast.parse(code)
    violations = detect_pep_498v(code, ast_root)
    print(violations)