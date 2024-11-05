import ast

def detect_pep_530(code: str, ast_root: ast.AST):
    async_comprehensions = []
    code_lines = code.splitlines()  # Split the code into individual lines for easy access

    # Walk through all nodes in the AST
    for node in ast.walk(ast_root):
        # Check for comprehensions (not generators) with an async 'for'
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            # Confirm it's an async comprehension by checking 'AsyncFor'
            for generator in node.generators:
                if isinstance(generator, ast.comprehension) and generator.is_async:
                    line_number = node.lineno
                    code_line = code_lines[line_number - 1].strip()  # Get the code line and strip whitespace
                    async_comprehensions.append((line_number, code_line))
                    break  # No need to check further within this comprehension

    return async_comprehensions if async_comprehensions else None

# main
if __name__ == "__main__":
    # Example usage
    codes = ['''async def async_gen():
    for i in range(3):
        await asyncio.sleep(1)
        yield i  # async generator, should be ignored

async def main():
    result = [i async for i in async_gen()]  # async comprehension
    print(result)
             
result = [i async for i in aiter() if i % 2]
''']
    for code in codes:
        # Parse the code into an AST
        ast_root = ast.parse(code)
        # Find async comprehensions
        result = detect_pep_530(code, ast_root)
        print("Async comprehensions found at lines:", result)
