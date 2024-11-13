# NEED TO REINSPECT THIS.
import ast

def detect_pep_709(code: str, ast_root: ast.AST) -> list | None:
    pep709_occurrences = []
    code_lines = code.splitlines()
    tracing_set = False  # Track if tracing/profiling has been set

    # Collect function definitions to check if they raise exceptions
    function_definitions = {node.name: node for node in ast.walk(ast_root) if isinstance(node, ast.FunctionDef)}

    for node in ast.walk(ast_root):
        # 1. Detect `locals()` calls inside comprehensions
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp)):
            for inner_node in ast.walk(node):
                if isinstance(inner_node, ast.Call) and isinstance(inner_node.func, ast.Name) and inner_node.func.id == 'locals':
                    line_number = inner_node.lineno
                    pep709_occurrences.append(
                        (line_number, line_number, code_lines[line_number - 1], "locals() includes outer variables")
                    )

        # 2. Detect comprehensions that call functions with exception-raising code
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp)):
            for inner_node in ast.walk(node):
                if isinstance(inner_node, ast.Call) and isinstance(inner_node.func, ast.Name):
                    called_function = inner_node.func.id
                    # Check if the function is defined in the code and raises an exception
                    if called_function in function_definitions:
                        func_node = function_definitions[called_function]
                        for func_body_node in ast.walk(func_node):
                            if isinstance(func_body_node, ast.Raise):
                                line_number = inner_node.lineno
                                pep709_occurrences.append(
                                    (line_number, line_number, code_lines[line_number - 1], "no dedicated frame in traceback")
                                )
                                break

        # 3. Detect sys.settrace or sys.setprofile calls before comprehensions
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute):
            if node.value.func.attr in ['settrace', 'setprofile']:
                line_number = node.lineno
                pep709_occurrences.append(
                    (line_number, line_number, code_lines[line_number - 1], "tracing/profiling setup")
                )
                tracing_set = True  # Mark that tracing/profiling is set

        # Capture comprehensions in traced functions
        if tracing_set and isinstance(node, ast.FunctionDef):
            contains_comprehension = any(isinstance(child, (ast.ListComp, ast.SetComp, ast.DictComp)) for child in ast.walk(node))
            if contains_comprehension:
                line_number = node.lineno
                pep709_occurrences.append(
                    (line_number, line_number, code_lines[line_number - 1], "tracing/profiling changes")
                )

        # 4. Detect warnings with stacklevel argument
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == 'warn':
                for kw in node.keywords:
                    if kw.arg == 'stacklevel':
                        line_number = node.lineno
                        pep709_occurrences.append(
                            (line_number, line_number, code_lines[line_number - 1], "stacklevel warnings")
                        )

    return pep709_occurrences if pep709_occurrences else None

# main
if __name__ == "__main__":
    # Example usage
    code = '''
def f(lst):
    return [locals() for x in lst]

def g():
    raise RuntimeError("boom")

def h():
    sys.settrace(lambda *args: None)
    return [g() for x in [1]]

import sys
sys.setprofile(lambda *args: None)

def j():
    return [i for i in range(10)]
'''

    ast_root = ast.parse(code)
    print(detect_pep_709(code, ast_root))