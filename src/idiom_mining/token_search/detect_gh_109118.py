import os 
import re
import ast
import json
import tokenize

def detect_generic_class_declarations_with_line_numbers(code: str):
    # Compile the pattern once
    pattern = re.compile(r'class\s+\w+\s*\[\s*[^]]*\s*\]:')

    # Use a generator for lazy evaluation
    return (
        (line_number, line.strip())
        for line_number, line in enumerate(code.splitlines(), start=1)
        if pattern.search(line)
    )

# main
if __name__ == "__main__":
    # Test example
    code = """
    class C[T]:
        pass

    class D:
        pass

    class E[T, K]:
        pass
    """

    # Detect patterns
    print(detect_generic_class_declarations_with_line_numbers(code))