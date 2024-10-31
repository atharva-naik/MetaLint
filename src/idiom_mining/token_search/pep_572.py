# PEP 572 walrus operator.

def detect_walrus_simple(code):
    walrus_lines = []
    for i, line in enumerate(code.splitlines(), start=1):
        if ":=" in line:
            walrus_lines.append([i, line])
    return walrus_lines if walrus_lines else None

import re
import tokenize
from io import StringIO

# def detect_pep_572(code):
#     walrus_lines = set()  # Use a set to avoid duplicate line numbers
#     tokens = tokenize.generate_tokens(StringIO(code).readline)

#     try:
#         for token in tokens:
#             if token.type == tokenize.OP and token.string == ":=":
#                 walrus_lines.add(token.start[0])
#     except tokenize.TokenError as e:
#         # print("tokenize.TokenError:", e)
#         walrus_lines = detect_walrus_simple(code)
#     except (TabError, IndentationError) as e:
#         # print("TabError/IndentationError:", e)
#         walrus_lines = detect_walrus_simple(code)

#     return sorted(walrus_lines) if walrus_lines else None

def detect_pep_572(code):
    # Regex pattern to match the walrus operator outside of strings
    pattern = re.compile(r'(?<!["\'`])\b:=\b(?!["\'`])')
    walrus_lines = []

    for i, line in enumerate(code.splitlines(), start=1):
        # Remove strings to ignore walrus operators inside them
        line_without_strings = re.sub(r'(["\']{3}.*?["\']{3}|["\'].*?["\'])', '', line)
        if pattern.search(line_without_strings):
            if len(line) < 500:
                walrus_lines.append([i, line])
    
    return walrus_lines if walrus_lines else None

# main
if __name__ == "__main__":
    codes = ["""if (n := len(items)) > 10:
    print(f"List has {n} items")""",
"""
a = 5
if (n := 10) > a:
    print("Walrus detected!")
for i in range(5):
    if (x := i * 2) > 4:
        print(x)
""",
"""
a = 5
b = 10
if b > a:
    print("No walrus here!")
""",
"""
a = 5
if (n := 10) > a:
    print("Walrus detected!")
text = ":="
comment = ''' This line contains ':=' inside a string '''
if (x := 15) > 10:
    print(x)
"""]
    for code in codes:
        walrus_lines = detect_pep_572(code)
        if walrus_lines: print(walrus_lines)
        else: pass