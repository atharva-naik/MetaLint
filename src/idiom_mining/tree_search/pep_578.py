import ast

def detect_pep_578(code: str, ast_root: ast.AST) -> list | None:
    pep578_occurrences = []
    code_lines = code.splitlines()

    for node in ast.walk(ast_root):
        # Detect function calls to sys.audit and sys.addaudithook
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            # Check if the call is to sys.audit or sys.addaudithook
            if node.func.attr in {"audit", "addaudithook"} and isinstance(node.func.value, ast.Name) and node.func.value.id == "sys":
                line_number = node.lineno
                line_content = code_lines[line_number - 1]
                pep578_occurrences.append((line_number, line_content))

    return pep578_occurrences if pep578_occurrences else None

# main
if __name__ == "__main__":
    # Example usage:
    code = '''
import sys

def example():
    sys.audit("event_name", "arg1", "arg2")
    sys.addaudithook(lambda event, args: print(f"Audit event: {event}, Arguments: {args}"))

sys.audit("another_event", "arg3")
    '''

    # Parse the AST
    ast_root = ast.parse(code)
    print(detect_pep_578(code, ast_root))
