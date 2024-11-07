import ast

def detect_pep_487(code: str, ast_root: ast.AST) -> list | None:
    pep487_occurrences = []

    for node in ast.walk(ast_root):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in {"__init_subclass__", "__set_name__"}:
            # Extract start and end line numbers, method type, and code block
            start_line = node.lineno
            end_line = node.end_lineno
            method_type = node.name
            code_block = "\n".join(code.splitlines()[start_line - 1 : end_line])

            pep487_occurrences.append((start_line, end_line, method_type, code_block))

    return pep487_occurrences if pep487_occurrences else None

# main
if __name__ == "__main__":
    # Example usage:
    code = '''
class Base:
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        print(f"Initializing subclass: {cls.__name__}")

class Descriptor:
    def __set_name__(self, owner, name):
        self.name = name

class MyClass(Base):
    pass
    '''

    # Parse the AST
    ast_root = ast.parse(code)
    print(detect_pep_487(code, ast_root))
