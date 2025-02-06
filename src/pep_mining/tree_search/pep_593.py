import ast

def detect_pep_593(code, ast_root):
    results = []

    class AnnotatedVisitor(ast.NodeVisitor):
        def visit_ImportFrom(self, node):
            # Check for 'from typing import Annotated' or similar forms
            if node.module == 'typing':
                for alias in node.names:
                    if alias.name == 'Annotated':
                        results.append((node.lineno, node.end_lineno, "import", ast.get_source_segment(code, node)))

        def visit_FunctionDef(self, node):
            # Check function arguments and return annotations for 'Annotated'
            for arg in node.args.args:
                if arg.annotation and isinstance(arg.annotation, ast.Subscript):
                    if getattr(arg.annotation.value, 'id', None) == 'Annotated':
                        results.append((arg.annotation.lineno, arg.annotation.end_lineno, "type annotation", ast.get_source_segment(code, arg.annotation)))
            # Check return type annotation
            if node.returns and isinstance(node.returns, ast.Subscript):
                if getattr(node.returns.value, 'id', None) == 'Annotated':
                    results.append((node.returns.lineno, node.returns.end_lineno, "type annotation", ast.get_source_segment(code, node.returns)))

            self.generic_visit(node)

        def visit_Assign(self, node):
            # Check assignments with 'Annotated' type hints
            if isinstance(node.value, ast.Subscript) and isinstance(node.targets[0], ast.Name):
                if getattr(node.value.value, 'id', None) == 'Annotated':
                    results.append((node.lineno, node.end_lineno, "type annotation", ast.get_source_segment(code, node)))
            self.generic_visit(node)

    AnnotatedVisitor().visit(ast_root)
    return results

# main
if __name__ == "__main__":
    # Example usage
    code_example = """
from typing import Annotated

def process_user(age: Annotated[int, "User's age in years"]) -> None:
    pass

age: Annotated[int, "age in years"] = 25
    """
    tree = ast.parse(code_example)
    for op in detect_pep_593(code_example, tree):
        print(op)
