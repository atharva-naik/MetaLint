import ast

def detect_pep_562(code: str, ast_root: ast.AST) -> list | None:
    pep562_occurrences = []
    code_lines = code.splitlines()

    for node in ast.walk(ast_root):
        # Look for function definitions at the module level (not nested)
        if isinstance(node, ast.FunctionDef) and node.name in {"__getattr__", "__dir__"}:
            # Get the start and end line numbers for the function
            start_line = node.lineno
            end_line = node.end_lineno if hasattr(node, "end_lineno") else start_line

            # Gather code lines for this function
            code_snippet = "\n".join(code_lines[start_line - 1:end_line])

            # Identify whether this is `__getattr__` or `__dir__`
            kind = "getattr" if node.name == "__getattr__" else "dir"
            pep562_occurrences.append((start_line, end_line, code_snippet, kind))

    return pep562_occurrences if pep562_occurrences else None

# Example usage
# main
if __name__ == "__main__":
    code = '''
    def __getattr__(name):
        if name == "dynamic_attr":
            return "This is a dynamic attribute"
        raise AttributeError(f"module '{{__name__}}' has no attribute '{{name}}'")

    def __dir__():
        return ["dynamic_attr", "standard_function"]

    def regular_function():
        return "This is a regular function in the module."
    '''

    # Parse the AST
    ast_root = ast.parse(code)
    print(detect_pep_562(code, ast_root))
