import ast

def detect_pep_567(code: str, ast_root: ast.AST):
    occurrences = []
    
    # Walk through all nodes in the AST
    for node in ast.walk(ast_root):
        
        # Check for imports of contextvars or asyncio
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in {"contextvars", "asyncio"}:
                    start_line = node.lineno
                    end_line = node.end_lineno
                    import_type = "import"
                    import_code = code.splitlines()[start_line - 1:end_line]
                    occurrences.append((start_line, end_line, import_type, "\n".join(import_code)))
        
        elif isinstance(node, ast.ImportFrom):
            if node.module in {"contextvars", "asyncio"}:
                start_line = node.lineno
                end_line = node.end_lineno
                import_type = "import"
                import_code = code.splitlines()[start_line - 1:end_line]
                occurrences.append((start_line, end_line, import_type, "\n".join(import_code)))

        # Detect ContextVar usage
        elif isinstance(node, ast.Assign):
            if any(isinstance(target, ast.Name) for target in node.targets):
                if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id == "ContextVar":
                    start_line = node.lineno
                    end_line = node.end_lineno
                    var_type = "context variable"
                    var_code = code.splitlines()[start_line - 1:end_line]
                    occurrences.append((start_line, end_line, var_type, "\n".join(var_code)))

    return occurrences if occurrences else None

# main
if __name__ == "__main__":
    codes = ["""import contextvars
from contextvars import ContextVar

var = ContextVar('example')""","""
import contextvars

user = contextvars.ContextVar('user', default="Anonymous")

def set_user(name):
    user.set(name)

def get_user():
    return user.get()

set_user("Lti")
print(get_user())  """]
    for code in codes:
        tree = ast.parse(code)
        print(detect_pep_567(code, tree))