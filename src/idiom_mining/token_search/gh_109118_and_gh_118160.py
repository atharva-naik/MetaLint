import os 
import re
import ast
import json
import tokenize

# pattern to match generic class declaration:
generic_class_pattern = re.compile(r'class\s+\w+\s*\[\s*[^]]*\s*\]:')

def detect_gh_109118_and_gh_118160(code: str):
# def detect_generic_class_declarations_with_line_numbers(code: str):
    # use a generator for lazy evaluation
    candidates = list(
        (line_number, line.strip())
        for line_number, line in enumerate(code.splitlines(), start=1)
        if generic_class_pattern.search(line)
    )
    filt_cands = []
    for lineno, line in candidates:
        if line.strip().startswith("class") or " class " in line or "class[" in line or "class(" in line:
            filt_cands.append((lineno, line))

    return filt_cands

# def detect_gh_109118_and_gh_118160(code: str):
#     """
#     Patterns for detecting:
#     - Class declaration with generics
#     - Lambda expression
#     - List comprehension
#     """
#     class_pattern = re.compile(r'class\s+\w+\s*\[\s*[^]]*\s*\]:')
#     lambda_pattern = re.compile(r'lambda\s*:\s*\w+')
#     list_comprehension_pattern = re.compile(r'\[\s*\w+\s+for\s+.*?\]')

#     results = []
#     lines = code.splitlines()

#     for line_number, line in enumerate(lines, start=1):
#         # Check for a generic class declaration
#         if class_pattern.search(line):
#             class_scope_lines = []
#             class_scope_lines.append((line_number, line.strip()))
#             inner_line_number = line_number

#             # Capture subsequent lines within the class scope
#             while inner_line_number + 1 < len(lines):
#                 inner_line_number += 1
#                 inner_line = lines[inner_line_number].strip()
                
#                 # Check for lambda or list comprehension within class scope
#                 if lambda_pattern.search(inner_line) or list_comprehension_pattern.search(inner_line):
#                     class_scope_lines.append((inner_line_number + 1, inner_line))
#                 elif inner_line == '':  # End if an empty line (or indentation change) is detected
#                     break

#             results.extend(class_scope_lines)

#     return results

# def detect_gh_109118_and_gh_118160(code: str):
#     """
#     Patterns for detecting:
#     - Class declaration with generics
#     - Lambda expressions within assignments
#     - List comprehensions within assignments or function arguments
#     """

#     class_pattern = re.compile(r'class\s+\w+\s*\[\s*[^]]*\s*\]:')
#     lambda_pattern = re.compile(r'\w+\s*=\s*lambda\s*:\s*\w+')
#     list_comprehension_pattern = re.compile(r'\w+\s*=\s*\[.*?\s+for\s+.*?\]')
#     function_arg_pattern = re.compile(r'\(.*?\[.*?\s+for\s+.*?\].*?\)')

#     results = []
#     lines = code.splitlines()

#     for line_number, line in enumerate(lines, start=1):
#         # Check for a generic class declaration
#         if class_pattern.search(line):
#             class_scope_lines = [(line_number, line.strip())]
#             inner_line_number = line_number

#             # Capture subsequent lines within the class scope
#             while inner_line_number + 1 < len(lines):
#                 inner_line_number += 1
#                 inner_line = lines[inner_line_number].strip()
                
#                 # Check for lambda, list comprehension, or function argument pattern within class scope
#                 if (
#                     lambda_pattern.search(inner_line) or 
#                     list_comprehension_pattern.search(inner_line) or 
#                     function_arg_pattern.search(inner_line)
#                 ):
#                     class_scope_lines.append((inner_line_number + 1, inner_line))
#                 elif inner_line == '':  # Stop at an empty line or out of indentation
#                     break

#             results.extend(class_scope_lines)

#     return results


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
    """,
"""
class C[T]:
    type Alias = lambda: T

class D:
    pass

class E[T]:
    T = "class"
    class Inner[U](make_base([T for _ in (1,)]), make_base(T)):
        pass
"""]
    for code in codes:
        # print(list(detect_generic_class_declarations_with_line_numbers(code)))
        print(list(detect_gh_109118_and_gh_118160(code)))