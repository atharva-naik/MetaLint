import ast

def detect_pep_468(code: str, ast_root: ast.Module):
    class KwargsUsageVisitor(ast.NodeVisitor):
        def __init__(self):
            self.occurrences = []

        def visit_Call(self, node):
            # Check if any argument in the function call is a `**kwargs`
            # print(node.keywords, ast.unparse(node.keywords))
            if any(isinstance(kw.value, ast.Name) and kw.arg == None for kw in node.keywords):
                # print(node.keywords[0].value.id)
                # for kw in node.keywords:
                #     if isinstance(kw.value, ast.Name):
                #         print(kw.arg, kw.value.id)
                self.occurrences.append((
                    node.lineno,
                    ast.get_source_segment(code, node)
                ))
            self.generic_visit(node)

    visitor = KwargsUsageVisitor()
    visitor.visit(ast_root)

    return visitor.occurrences if visitor.occurrences else None

# main
if __name__ == "__main__":
    # Example usage
    code_example = """
def greet(name, age=None):
    print(f"Hello {name}, age {age}")

kwargs = {"name": "Alice", "age": 30}
greet(**kwargs)
greet(age="hello")

another_kwargs = {"name": "Bob"}
greet("John", **another_kwargs)

greet("John", age="test", **another_kwargs)
greet("John", age=test)
"""
    ast_root = ast.parse(code_example)
    result = detect_pep_468(code_example, ast_root)
    print(result)
