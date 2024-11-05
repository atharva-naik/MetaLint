import ast

def detect_pep_589(code: str, ast_root: ast.Module):
    class TypedDictVisitor(ast.NodeVisitor):
        def __init__(self):
            self.occurrences = []

        def visit_ImportFrom(self, node):
            # Check if TypedDict is imported
            if node.module == 'typing' and any(alias.name == 'TypedDict' for alias in node.names):
                for alias in node.names:
                    if alias.name == 'TypedDict':
                        self.occurrences.append((
                            node.lineno,
                            node.lineno,
                            ast.get_source_segment(code, node),
                            "typed dict import"
                        ))
            self.generic_visit(node)

        def visit_ClassDef(self, node):
            # Check if TypedDict is in the bases of the class definition
            for base in node.bases:
                if isinstance(base, ast.Name) and base.id == 'TypedDict':
                    self.occurrences.append((
                        node.lineno,
                        node.end_lineno,
                        ast.get_source_segment(code, node),
                        "typed dict definition"
                    ))
            self.generic_visit(node)

        def visit_Assign(self, node):
            # Check for the alternate syntax of TypedDict definition
            for target in node.targets:
                if isinstance(target, ast.Name) and isinstance(node.value, ast.Call):
                    if isinstance(node.value.func, ast.Name) and node.value.func.id == 'TypedDict':
                        self.occurrences.append((
                            node.lineno,
                            node.end_lineno,
                            ast.get_source_segment(code, node),
                            "typed dict alternate syntax"
                        ))
            self.generic_visit(node)

    visitor = TypedDictVisitor()
    visitor.visit(ast_root)

    return visitor.occurrences if visitor.occurrences else None

# main
if __name__ == "__main__":
    # Example usage
    codes = ["""
from typing import TypedDict

class Movie(TypedDict):
    title: str
    year: int
    rating: float

def func():
    pass
    ""","""from typing import TypedDict, List

class Movie(TypedDict):
    title: str
    year: int
    rating: float

def func():
    pass""",
"""class A(TypedDict, total=False):
    x: int

class B(TypedDict):
    x: int

def f(a: A) -> None:
    del a['x']

b: B = {'x': 0}
f(b)  # Type check error: 'B' not compatible with 'A'
b['x'] + 1  # Runtime KeyError: 'x'""","""
Movie = TypedDict('Movie', {'name': str, 'year': int})""",
"""
Movie = TypedDict('Movie',
                  {'name': str, 'year': int},
                  total=False)
"""]
    for code in codes:
        tree = ast.parse(code)
        result = detect_pep_589(code, tree)
        print(result)  # Output: [(3, 'class Movie(TypedDict):\n    title: str\n    year: int\n    rating: float')]
