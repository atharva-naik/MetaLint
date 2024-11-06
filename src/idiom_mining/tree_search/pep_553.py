import ast

def detect_pep_553(code, ast_root):
    # Parse the code into an AST
    breakpoints = []

    # Walk through each node in the AST
    for node in ast.walk(ast_root):
        # Check for function calls
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            # Check if the function is 'breakpoint'
            if node.func.id == "breakpoint":
                # Extract line number and the code line
                line_number = node.lineno
                line_content = code.splitlines()[line_number - 1].strip()
                breakpoints.append((line_number, line_content))

    return breakpoints if breakpoints else None

# main
if __name__ == "__main__":
    codes = ["""
def example():
    x = 5
    breakpoint()  # Entering the debugging mode here
    y = x + 10
    return y

example()
""",
"""def divide(a, b):
    breakpoint()  # Debugging step
    return a / b

divide(10, 2)"""]
    for code in codes:
        tree = ast.parse(code)
        print(detect_pep_553(code, tree))