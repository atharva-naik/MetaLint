import ast

def detect_pep_519(code: str, ast_root: ast.AST):
    class PEP519Visitor(ast.NodeVisitor):
        def __init__(self):
            self.occurrences = []

        def visit_ClassDef(self, node):
            # Check if the class defines a __fspath__ method
            for body_item in node.body:
                if isinstance(body_item, ast.FunctionDef) and body_item.name == '__fspath__':
                    self.occurrences.append((
                        node.lineno,
                        node.end_lineno,
                        "os.PathLike",
                        ast.get_source_segment(code, node)
                    ))
            self.generic_visit(node)

        def visit_Call(self, node):
            # Check if the call is to os.fspath
            if (
                isinstance(node.func, ast.Attribute) and
                node.func.attr == 'fspath' and
                isinstance(node.func.value, ast.Name) and
                node.func.value.id == 'os'
            ):
                self.occurrences.append((
                    node.lineno,
                    node.end_lineno,
                    "os.fspath",
                    ast.get_source_segment(code, node)
                ))
            self.generic_visit(node)

    visitor = PEP519Visitor()
    visitor.visit(ast_root)
    return visitor.occurrences if visitor.occurrences else None

# main
if __name__ == "__main__":
    # Example usage
    code_example = """
import os

class MyPath:
    def __init__(self):
        pass

    def __fspath__(self):
        return "/home/user/file.txt"

path = MyPath()
result = os.fspath(path)
"""
    # Parse the code to an AST
    ast_root = ast.parse(code_example)

    # Find occurrences of PEP 519 features
    occurrences = detect_pep_519(code_example, ast_root)
    print(occurrences)
