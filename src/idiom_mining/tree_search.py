# do code-to-code matching based on tree vector representations

import os
import ast
import json
import warnings
from tqdm import tqdm

from src.datautils import load_stack_dump

# Suppress only SyntaxWarnings
warnings.filterwarnings("ignore", category=SyntaxWarning)

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

def find_try_except_else_block(code):
    tree = ast.parse(code)

    for node in ast.walk(tree):
        # Check for Try block with both except and else blocks
        if isinstance(node, ast.Try) and node.handlers and node.orelse:
            return node.lineno  # Return the line number of the try block if pattern is detected

    return None  # Return None if pattern is not found

def find_try_except_else_with_global_in_try(code):
    tree = ast.parse(code)
    results = []

    for node in ast.walk(tree):
        # Check for Try block with both except and else blocks
        if isinstance(node, ast.Try) and node.handlers and node.orelse:
            try_line = node.lineno
            global_vars = []

            # Check for global declarations in the try block
            for stmt in node.body:
                if isinstance(stmt, ast.Global):
                    global_vars.extend(stmt.names)  # Collect names of global variables

            if global_vars:
                results.append({
                    "try_line": try_line,
                    "global_vars": global_vars
                })

    return results if results else None

def find_try_except_else_with_global_in_else(code):
    tree = ast.parse(code)
    results = []

    for node in ast.walk(tree):
        # Check for Try block with both except and else blocks
        if isinstance(node, ast.Try) and node.handlers and node.orelse:
            try_line = node.lineno
            global_vars = []

            # Check for global declarations in the else block
            for stmt in node.orelse:
                if isinstance(stmt, ast.Global):
                    global_vars.extend(stmt.names)  # Collect names of global variables

            if global_vars:
                results.append({
                    "try_line": try_line,
                    "global_vars": global_vars
                })

    return results if results else None

def extract_try_except_else_blocks(code):
    tree = ast.parse(code)
    blocks = []

    for node in ast.walk(tree):
        # Check for Try block with both except and else blocks
        if isinstance(node, ast.Try) and node.handlers and node.orelse:
            # Get start and end line numbers of the try-except-else block
            start_line = node.lineno
            end_line = max(
                max(handler.end_lineno for handler in node.handlers if handler.end_lineno),
                node.orelse[-1].end_lineno if node.orelse[-1].end_lineno else start_line
            )
            # Extract the block of code from the original code string
            block_code = "\n".join(code.splitlines()[start_line - 1:end_line])
            blocks.append((start_line, end_line, block_code))

    return blocks

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
    blob_ids = [file['blob_id'] for file in stack_data]
    syntax_errors = 0
    matched_blocks = []
    for code, blob_id in tqdm(zip(codes, blob_ids)):
        try: 
            global_in_except_usage_in_else = find_try_block_with_global_in_except_and_usage_in_else(code)
            global_in_except = find_try_block_with_global_in_except(code)
            try_except_else_block_line = find_try_except_else_block(code)
            global_in_try = find_try_except_else_with_global_in_try(code)
            global_in_else = find_try_except_else_with_global_in_else(code)
        except SyntaxError: 
            syntax_errors += 1
            continue
        if try_except_else_block_line:
            try_except_else_block = extract_try_except_else_blocks(code)
            # print("Pattern found. 'Try' block starts at line:", try_except_else_block_line)
            # print(try_except_else_block)
            matched_blocks.append({
                "blob_id": blob_id,
                "matched_blocks": try_except_else_block, 
                "flags": {
                    "try_except_else_found": try_except_else_block_line is not None,
                    "global_in_except": global_in_except is not None,
                    "global_in_except_usage_in_else": global_in_except_usage_in_else is not None,
                    "global_in_else": global_in_else,
                    "global_in_try": global_in_try,
                }
            })
        else: pass
            # print("Pattern not found.")
    print(syntax_errors)
    with open("data/pattern_mining/tree_patterns/gh-111123.json", "w") as f:
        json.dump(matched_blocks, f, indent=4)