# PEP 572 walrus operator.

def detect_pep_572(code):
    walrus_lines = []
    for i, line in enumerate(code.splitlines(), start=1):
        if ":=" in line:
            walrus_lines.append(i)
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
"""]
    for code in codes:
        walrus_lines = detect_pep_572(code)
        if walrus_lines: print(walrus_lines)
        else: pass