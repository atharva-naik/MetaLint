import ast

def detect_pep_570(code: str, ast_root: ast.AST) -> list | None:
    pep570_occurrences = []
    code_lines = code.splitlines()

    for node in ast.walk(ast_root):
        # Check for both regular and async functions
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Access positional-only arguments within the function's arguments
            if node.args.posonlyargs:
                line_number = node.lineno
                line_content = code_lines[line_number - 1]
                pep570_occurrences.append((line_number, line_content))

    return pep570_occurrences if pep570_occurrences else None

# main
if __name__ == "__main__":
    # Example usage:
    code = '''
def example(a, b, /, c, d):
    return a + b + c + d

def no_positional_only(x, y):
    return x * y

def mixed_args(x, /, y, *, z):
    return x + y + z'''

    # Parse the AST
    ast_root = ast.parse(code)
    print(detect_pep_570(code, ast_root))
