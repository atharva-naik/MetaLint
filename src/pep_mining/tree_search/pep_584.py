import ast

def detect_pep_584(code: str, ast_root: ast.Module):
    union_lines = []
    init_lines = []
    dict_vars = set()

    # First pass: Find dictionary initializations and track variables assigned to dictionaries
    for node in ast.walk(ast_root):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    dict_vars.add(target.id)
                    init_lines.append((node.lineno, code.splitlines()[node.lineno - 1].strip()))

    # Second pass: Find union operations between dictionaries
    for node in ast.walk(ast_root):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
            # Check if both operands are dictionaries or known dictionary variables
            left_is_dict = isinstance(node.left, ast.Dict) or (isinstance(node.left, ast.Name) and node.left.id in dict_vars)
            right_is_dict = isinstance(node.right, ast.Dict) or (isinstance(node.right, ast.Name) and node.right.id in dict_vars)
            
            if left_is_dict and right_is_dict:
                union_lines.append((node.lineno, code.splitlines()[node.lineno - 1].strip()))

    if len(union_lines) == 0: return None
    return {'union_lines': union_lines, 'init_lines': init_lines}

# main
if __name__ == "__main__":
    codes = ["""d1 = {'a': 1}
d2 = {'b': 2}
d3 = d1 | d2""",
"""
d1 = {'a': 1}
d2 = {'b': 2}
d3 = "d1 | d2"
"""]
    for code in codes:
        print(detect_pep_584(code))