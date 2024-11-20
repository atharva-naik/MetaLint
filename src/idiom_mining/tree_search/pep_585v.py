import ast

def detect_pep_585v(code, ast_root: ast.Module):
    issues = set()  # Use a set to avoid duplicates

    # Traverse through all nodes in the syntax tree
    for node in ast.walk(ast_root):
        # Check for annotations using legacy typing types
        if isinstance(node, ast.AnnAssign):  # Variable annotations
            if isinstance(node.annotation, ast.Subscript) and isinstance(node.annotation.value, ast.Name):
                if node.annotation.value.id in {'List', 'Dict', 'Set', 'Tuple'}:
                    lineno = node.lineno
                    line = code.splitlines()[lineno - 1]
                    issues.add((lineno, line, "violation"))

        elif isinstance(node, ast.FunctionDef):  # Function argument annotations
            for arg in node.args.args:
                if arg.annotation and isinstance(arg.annotation, ast.Subscript) and isinstance(arg.annotation.value, ast.Name):
                    if arg.annotation.value.id in {'List', 'Dict', 'Set', 'Tuple'}:
                        lineno = arg.lineno
                        line = code.splitlines()[lineno - 1]
                        issues.add((lineno, line, "violation"))

            # Check return type annotation
            if node.returns and isinstance(node.returns, ast.Subscript) and isinstance(node.returns.value, ast.Name):
                if node.returns.value.id in {'List', 'Dict', 'Set', 'Tuple'}:
                    lineno = node.lineno
                    line = code.splitlines()[lineno - 1]
                    issues.add((lineno, line, "violation"))
    

    # Return the issues or None if no occureturn sorted(issues) if issues else
    return sorted(issues) if issues else None

# main
if __name__ == "__main__":
    # Example usage
    code = """
from typing import List, Dict

def process_data(data: List[int]) -> Dict[str, int]:
    pass
    
def process_data(data: List[int]):
    pass

def another_function(data: list[int]) -> dict[str, int]:
    pass
    
x : List[int] = []
"""
    ast_root = ast.parse(code)
    print(detect_pep_585v(code, ast_root=ast_root))
