import ast

def detect_pep_498(code: str, ast_root: ast.AST) -> list | None:
    fstring_occurrences = []

    for node in ast.walk(ast_root):
        if isinstance(node, ast.JoinedStr):
            # Extract the line number and the corresponding line of code
            line_number = node.lineno
            line_content = code.splitlines()[line_number - 1]
            fstring_occurrences.append((line_number, line_content))

    return fstring_occurrences if fstring_occurrences else None

# main
if __name__ == "__main__":
    # Example usage:
    code = '''
name = "Alice"
age = 30

greeting = f"Hello, {name}. You are {age} years old."

greeting = "Test {test}".format(test=1)
greeting = "baked"
'''

    # Parse the AST
    ast_root = ast.parse(code)
    print(detect_pep_498(code, ast_root))
