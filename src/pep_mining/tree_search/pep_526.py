# detect type annotations (first introduced by PEP 526)
import ast

def detect_pep_526(code: str, ast_root: ast.Module):
    # List to store instances of type annotations found
    annotations = []

    # Traverse each node in the AST
    for node in ast.walk(ast_root):
        # Check if the node is an annotated assignment (i.e., has a type annotation)
        if isinstance(node, ast.AnnAssign):
            # Extract the line number and line of code
            line_num = node.lineno
            line = code.splitlines()[line_num - 1].strip()
            annotations.append((line_num, line))

    # Return the list of annotations or None if empty
    return annotations if annotations else None

# main
if __name__ == "__main__":
    codes = ["""primes: List[int] = []

captain: str  # Note: no initial value!

class Starship:
    stats: Dict[str, int] = {}
"""]
    for code in codes:
        tree = ast.parse(code)
        print(detect_pep_526(code, tree))

# 201k blocks and 36210 files out of 500k files (7.24% files)