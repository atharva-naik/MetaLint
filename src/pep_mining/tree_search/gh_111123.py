import ast

def detect_gh_111123(code: str, ast_root: ast.AST):
    result = []

    # Helper function to extract line numbers of a node
    def get_line_numbers(node):
        if hasattr(node, 'lineno') and hasattr(node, 'end_lineno'):
            return node.lineno, node.end_lineno
        return None, None

    # Helper function to extract all names used within a node
    def get_used_names(node):
        if isinstance(node, list):
            names = set()
            for subnode in node:
                for n in ast.walk(subnode):
                    if isinstance(n, ast.Name): names.add(n.id) 
            return names
        else:
            return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}

    for node in ast.walk(ast_root):
        # Only focus on Try nodes
        if isinstance(node, ast.Try):
            # Track if we have a global declaration in except block
            globals_declared_in_except = set()
            for handler in node.handlers:
                # Only process except blocks
                if handler.type is None or isinstance(handler.type, ast.Name):
                    for stmt in handler.body:
                        # Check for global statements
                        if isinstance(stmt, ast.Global):
                            globals_declared_in_except.update(stmt.names)

            if globals_declared_in_except:
                # Get all names used in try and else blocks
                names_in_try_block = get_used_names(node.body)
                names_in_else_block = get_used_names(node.orelse)
                # print(names_in_else_block)
                # print(names_in_try_block)

                # Analyze each global declaration
                for global_var in globals_declared_in_except:
                    start_line, end_line = get_line_numbers(node)
                    if start_line and end_line:
                        if global_var in names_in_try_block or global_var in names_in_else_block:
                            result.append((start_line, end_line, ast.get_source_segment(code, node), "error"))
                        else:
                            result.append((start_line, end_line, ast.get_source_segment(code, node), "warning"))

    return result if result else None

# main
if __name__ == "__main__":
    codes = ["""a=5

def f():
    try:
        pass
    except:
        global a
    else:
        print(a)
""","""a=5

def f():
    try:
        print(a)
    except:
        global a""",
"""a=5

def f():
    try:
        a = 1
        print(a)
    except:
        global a""",
"""a=5

def f():
    try:
        pass
    except:
        global a
    else:
        a = 1
        print(a)""","""a=5

def f():
    try:
        pass
    except:
        global a
    else:
        pass""","""a=5

def f():
    try:
        pass
    except:
        global a"""]
    for code in codes:
        tree = ast.parse(code)
        print(detect_gh_111123(code, tree))