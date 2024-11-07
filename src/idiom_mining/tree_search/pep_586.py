import ast

def detect_pep_586(code: str, ast_root: ast.AST) -> list | None:
    occurrences = []
    code_lines = code.splitlines()

    for node in ast.walk(ast_root):
        # Detect import of Literal
        if isinstance(node, ast.ImportFrom) and node.module == "typing":
            for alias in node.names:
                if alias.name == "Literal":
                    line_number = node.lineno
                    line_content = code_lines[line_number - 1]
                    occurrences.append((line_number, line_content, "import"))

        # Detect type annotations using Literal
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.AnnAssign)):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Check function argument annotations
                for arg in node.args.args:
                    if isinstance(arg.annotation, ast.Subscript) and getattr(arg.annotation.value, 'id', None) == "Literal":
                        line_number = arg.lineno
                        line_content = code_lines[line_number - 1]
                        occurrences.append((line_number, line_content, "type annotation"))
            elif isinstance(node, ast.AnnAssign):
                # Check variable annotations
                if isinstance(node.annotation, ast.Subscript) and getattr(node.annotation.value, 'id', None) == "Literal":
                    line_number = node.lineno
                    line_content = code_lines[line_number - 1]
                    occurrences.append((line_number, line_content, "type annotation"))

    return occurrences if occurrences else None

# main
if __name__ == "__main__":
    # Example usage:
    code = '''from typing import Literal

def get_status(status: Literal["open", "closed", "pending"]) -> str:
    return "Status: " + status

mode: Literal[1, 2, 3] = 1'''

    # Parse the AST
    ast_root = ast.parse(code)
    print(detect_pep_586(code, ast_root))
