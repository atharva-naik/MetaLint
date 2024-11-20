# postponed evaluation of type annotations.
import ast

def detect_pep_563(code: str, ast_root: ast.AST) -> list | None:
    pep563_occurrences = []
    code_lines = code.splitlines()

    for node in ast.walk(ast_root):
        # Detect `from __future__ import annotations`
        if isinstance(node, ast.ImportFrom):
            if node.module == "__future__" and any(alias.name == "annotations" for alias in node.names):
                line_number = node.lineno
                line_content = code_lines[line_number - 1]
                pep563_occurrences.append((line_number, line_content, "import"))
        
        # Detect quoted type annotations in function arguments and variables
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Check for quoted type annotations in function arguments
            for arg in node.args.args + node.args.kwonlyargs + ([node.args.vararg] if node.args.vararg else []) + ([node.args.kwarg] if node.args.kwarg else []):
                if arg.annotation and isinstance(arg.annotation, ast.Constant) and arg.annotation.value is not None:
                    line_number = arg.lineno
                    line_content = code_lines[line_number - 1]
                    pep563_occurrences.append((line_number, line_content, "quoted annotation"))
                    # print(ast.unparse(arg.annotation))
            # Check for quoted return annotation
            if node.returns and isinstance(node.returns, ast.Constant) and node.returns.value is not None:
                line_number = node.returns.lineno
                line_content = code_lines[line_number - 1]
                pep563_occurrences.append((line_number, line_content, "quoted annotation"))
                # print(node.returns.value)
        
        # Detect quoted type annotations in variable assignments
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.annotation, ast.Constant):
                line_number = node.lineno
                line_content = code_lines[line_number - 1]
                pep563_occurrences.append((line_number, line_content, "quoted annotation"))

    return pep563_occurrences if pep563_occurrences else None

# main
if __name__ == "__main__":
    # Example usage
    code = '''
from __future__ import annotations

class Node:
    def __init__(self, next: 'Node'):  # Quoted annotation
        self.next = next

    def test_is_uvm_tensor(self, sizes: List[int], uvm_op) -> None:
        pass
        
    def test_is_uvm_tensor(self, sizes: None, uvm_op):
        pass

    def test(self, next: Node):  # Quoted annotation
        self.next = next

value: 'int' = 5  # Quoted annotation
value: int = 5  # Quoted annotation
'''

    # Parse the AST
    ast_root = ast.parse(code)
    for op in detect_pep_563(code, ast_root):
        print(op)
