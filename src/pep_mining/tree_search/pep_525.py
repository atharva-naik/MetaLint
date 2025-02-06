# pep 525 - async FORs and async Generators (AsyncFunctionDef with yield statements).
import ast

def detect_pep_525(code, ast_root):
    matches = []
    
    # Helper function to check if an async function is a generator
    def is_async_generator(node):
        for subnode in ast.walk(node):
            if isinstance(subnode, ast.Yield):
                return True
        return False

    for node in ast.walk(ast_root):
        # Detect async for statements
        if isinstance(node, ast.AsyncFor):
            start_line = node.lineno
            end_line = node.end_lineno
            async_for_code = "\n".join(code.splitlines()[start_line - 1:end_line]).strip()
            matches.append((start_line, end_line, "async for", async_for_code))
        
        # Detect async functions that are generators (contain yield)
        elif isinstance(node, ast.AsyncFunctionDef) and is_async_generator(node):
            start_line = node.lineno
            end_line = node.end_lineno
            async_gen_code = "\n".join(code.splitlines()[start_line - 1:end_line]).strip()
            matches.append((start_line, end_line, "async generator", async_gen_code))
    
    return matches if matches else None

# main
if __name__ == "__main__":
    codes = ["""import asyncio

async def async_generator():
    for i in range(3):
        yield i
        await asyncio.sleep(1)

async def main():
    async for number in async_generator():
        print(number)

asyncio.run(main())
"""]
    for code in codes:
        tree = ast.parse(code)
        op = detect_pep_525(code, tree)
        print(op)