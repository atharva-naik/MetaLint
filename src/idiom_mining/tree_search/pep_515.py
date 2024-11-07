import ast
import re

def detect_pep_515(code: str, ast_root: ast.AST) -> list | None:
    occurrences = []
    # underscore_pattern = re.compile(r'\d+_\d+')  # Pattern to detect underscore in numeric literals
    codelines = code.splitlines()
    for node in ast.walk(ast_root):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float, complex)):
            # Extract the line number and the line of code
            line_number = node.lineno
            line_content = codelines[line_number - 1]
            constant_symbolic_representation = line_content[node.col_offset : node.end_col_offset]

            # Check if the line contains an underscore in a numeric literal
            if '_' in constant_symbolic_representation:
                occurrences.append((line_number, line_content))

    return occurrences if occurrences else None

# main
if __name__ == "__main__":
    # Example usage:
    code = '''
price = 1_000_000
discount = 50
hex_value = 0x_FF_FF
not_a_number = "1_000_000"
'''
    # Parse the AST
    ast_root = ast.parse(code)
    print(detect_pep_515(code, ast_root))