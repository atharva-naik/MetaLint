import ast

def detect_pep_495(code: str, ast_root: ast.Module):
    class FoldUsageVisitor(ast.NodeVisitor):
        def __init__(self):
            self.occurrences = []

        def visit_Attribute(self, node):
            # Detect usage of `fold` as an attribute
            if isinstance(node.value, ast.Name) and node.attr == "fold":
                self.occurrences.append((
                    node.lineno,
                    ast.get_source_segment(code, node)
                ))
            self.generic_visit(node)

        def visit_Call(self, node):
            # Detect `fold` being used as a keyword argument in function calls
            if any(kw.arg == "fold" for kw in node.keywords):
                self.occurrences.append((
                    node.lineno,
                    ast.get_source_segment(code, node)
                ))
            self.generic_visit(node)

    visitor = FoldUsageVisitor()
    visitor.visit(ast_root)
    return visitor.occurrences if visitor.occurrences else None

# main
if __name__ == "__main__":
    # Example usage
    code_example = """
from datetime import datetime, timezone, timedelta

dt = datetime(2024, 11, 3, 2, 30, tzinfo=timezone(timedelta(hours=-5)))
dt.fold = 1  # Setting fold attribute manually

# Using fold as a keyword argument in replace()
dt2 = dt.replace(fold=1)

# fold used in a custom function
process_datetime(dt, fold=0)
"""
    tree = ast.parse(code_example)
    result = detect_pep_495(code_example, tree)
    print(result)