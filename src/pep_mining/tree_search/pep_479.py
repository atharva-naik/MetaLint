import ast

def detect_pep_479(code: str, ast_root: ast.AST):
    matches = []
    code_lines = code.splitlines()

    # Helper function to check if a function contains a yield statement
    def is_generator_function(node):
        for child in ast.walk(node):
            if isinstance(child, ast.Yield) or isinstance(child, ast.YieldFrom):
                return True
        return False

    # Traverse the AST and look for raise statements within generators
    for node in ast.walk(ast_root):
        # Check if it's a function or async function definition
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and is_generator_function(node):
            func_start = node.lineno
            func_end = node.end_lineno
            func_code = "\n".join(code_lines[func_start - 1:func_end])

            # Traverse function body to find raises of StopIteration
            for subnode in ast.walk(node):
                if isinstance(subnode, ast.Raise) and isinstance(subnode.exc, ast.Name) and subnode.exc.id == "StopIteration":
                    line_number = subnode.lineno
                    line_content = code_lines[line_number - 1].strip()
                    matches.append((line_number, line_content, func_start, func_end, func_code))

    return matches if matches else None

# main
if __name__ == "__main__":
    codes = ["""
def example():
    yield 1
    raise StopIteration

def non_generator():
    raise StopIteration
""","""
def generator():
    yield 1
    raise StopIteration 

gen = generator()
print(next(gen)) 
print(next(gen))  # Raises RuntimeError in Python 3.7+
"""]
    for code in codes:
        ast_root = ast.parse(code)
        print(detect_pep_479(code, ast_root))
