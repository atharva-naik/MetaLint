import ast

def detect_pep_572(code: str, ast_root: ast.Module):
    walrus_occurrences = []
    ast_root = ast.parse(code)
    
    # Traverse the AST tree
    for node in ast.walk(ast_root):
        # Check if we have an assignment expression (walrus operator)
        if isinstance(node, ast.NamedExpr):
            # Get the line number and content of the line
            line_number = node.lineno
            line_content = code.splitlines()[line_number - 1]
            walrus_occurrences.append((line_number, line_content))
    
    return walrus_occurrences if walrus_occurrences else None

# main
if __name__ == "__main__":
    codes = ["""
x = 10
if (y := x + 1) > 10:
    print(y)
"""]
    for code in codes:
        print(detect_pep_572(code))
