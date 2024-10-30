import os 
import re
import ast
import json
import tokenize

# pattern to match generic class declaration:
generic_class_pattern = re.compile(r'class\s+\w+\s*\[\s*[^]]*\s*\]:')

def detect_generic_class_declarations_with_line_numbers(code: str):
    # use a generator for lazy evaluation
    return (
        (line_number, line.strip())
        for line_number, line in enumerate(code.splitlines(), start=1)
        if generic_class_pattern.search(line)
    )

def detect_gh_109118_and_gh_11816(code: str):
    return []

# main
if __name__ == "__main__":
    # test examples
    codes = ["""
    class C[T]:
        pass

    class D:
        pass

    class E[T, K]:
        pass
    """]
    for code in codes:
        print(list(detect_generic_class_declarations_with_line_numbers(code)))
        print(list(detect_gh_109118_and_gh_118160(code)))