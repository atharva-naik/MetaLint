# do code-to-code matching based on tree vector representations

import os
import ast
import json
from tqdm import tqdm

from src.datautils import load_stack_dump

def find_try_block_with_global_in_except_and_usage_in_else(code):
    tree = ast.parse(code)

    for node in ast.walk(tree):
        # Check for Try block with both except and else blocks
        if isinstance(node, ast.Try) and node.handlers and node.orelse:
            global_vars = set()
            try_line = node.lineno

            # Check each except handler for global declarations
            for handler in node.handlers:
                if isinstance(handler, ast.ExceptHandler):
                    for stmt in handler.body:
                        if isinstance(stmt, ast.Global):
                            global_vars.update(stmt.names)

            # Check if any of the global variables are used in the else block
            for stmt in node.orelse:
                for subnode in ast.walk(stmt):
                    if isinstance(subnode, ast.Name) and subnode.id in global_vars:
                        return try_line  # Return the line number of the try block if pattern is detected

    return None  # Return None if pattern is not found

def find_try_block_with_global_in_except(code):
    tree = ast.parse(code)

    for node in ast.walk(tree):
        # Check for Try block with both except and else blocks
        if isinstance(node, ast.Try) and node.handlers and node.orelse:
            try_line = node.lineno

            # Check each except handler for global declarations
            for handler in node.handlers:
                if isinstance(handler, ast.ExceptHandler):
                    for stmt in handler.body:
                        if isinstance(stmt, ast.Global):
                            return try_line  # Return the line number of the try block if pattern is detected

    return None  # Return None if pattern is not found

# main
if __name__ == "__main__":
    # Example usage
#     codes = ["""try:
#     pass
# except:
#     global x, y
#     x = 5
# else:
#     print(x)
# ""","""try:
#     pass
# except:
#     x = 5
# else:
#     print(x)""",
# """a=5

# def f():
#     try:
#         pass
#     except:
#         global a
#     else:
#         print(a)"""]
    stack_data = load_stack_dump("./data/STACK-V2")
    codes = [file['content'] for file in stack_data]
    syntax_errors = 0
    for code in tqdm(codes):
        try: 
            # try_line = find_try_block_with_global_in_except_and_usage_in_else(code)
            try_line = find_try_block_with_global_in_except(code)
        except SyntaxError: 
            syntax_errors += 1
            continue
        if try_line:
            print("Pattern found. 'Try' block starts at line:", try_line)
        else: pass
            # print("Pattern not found.")
    print(syntax_errors)