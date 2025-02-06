# detect occurrences of Github Issue #118216 about relative imports of the future module.

import ast

def detect_gh_118216(code, ast_root):
    relative_future_imports = []

    for node in ast.walk(ast_root):
        # Check if node is an ImportFrom (relative import)
        if isinstance(node, ast.ImportFrom) and node.module == "__future__" and node.level > 0:
            # Append a tuple with line number and code line
            relative_future_imports.append((node.lineno, code.splitlines()[node.lineno - 1]))

    return relative_future_imports if relative_future_imports else None

# main
if __name__ == "__main__":
    codes = ["""from .__future__ import barry_as_FLUFL
from __future__ import print_function"""]
    for code in codes:
        tree = ast.parse(code)
        print(detect_gh_118216(code, tree))