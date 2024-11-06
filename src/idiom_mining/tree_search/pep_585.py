import ast

def detect_pep_585(code: str, ast_root: ast.Module):
    issues = []

    # Traverse through all nodes in the syntax tree
    for node in ast.walk(ast_root):
        # Check for type hints using built-in collections directly
        if isinstance(node, ast.AnnAssign):  # Variable annotations
            if isinstance(node.annotation, ast.Subscript) and isinstance(node.annotation.value, ast.Name):
                if node.annotation.value.id in {'list', 'dict', 'set', 'tuple'}:
                    lineno = node.lineno
                    line = code.splitlines()[lineno - 1]
                    issues.append((lineno, line, "error"))

        elif isinstance(node, ast.FunctionDef):  # Function argument annotations
            for arg in node.args.args:
                if arg.annotation and isinstance(arg.annotation, ast.Subscript) and isinstance(arg.annotation.value, ast.Name):
                    if arg.annotation.value.id in {'list', 'dict', 'set', 'tuple'}:
                        lineno = arg.lineno
                        line = code.splitlines()[lineno - 1]
                        issues.append((lineno, line, "error"))

            # Check return type annotation
            if node.returns and isinstance(node.returns, ast.Subscript) and isinstance(node.returns.value, ast.Name):
                if node.returns.value.id in {'list', 'dict', 'set', 'tuple'}:
                    lineno = node.lineno
                    line = code.splitlines()[lineno - 1]
                    issues.append((lineno, line, "error"))

        # Check for imports from `typing`
        elif isinstance(node, ast.ImportFrom) and node.module == 'typing':
            for alias in node.names:
                if alias.name in {'List', 'Dict', 'Set', 'Tuple'}:
                    lineno = node.lineno
                    line = code.splitlines()[lineno - 1]
                    issues.append((lineno, line, "suggestion"))

    # Return the issues or None if no occurrences were found
    return issues if issues else None

# main 
if __name__ == "__main__":
    # Example usage
    code = """
from typing import List, Dict

def process_data(data: list[int]) -> dict[str, int]:
    pass

def transform_data(data: List[int]) -> Dict[str, int]:
    pass
"""
    tree = ast.parse(code)
    print(detect_pep_585(code, tree))