# detect occurrences of dataclasses imports or dataclass definitions made using the decorator.

import ast

def detect_pep_557(source_code, ast_root):
    results = []
    tree = ast.parse(source_code)
    
    for node in ast.walk(ast_root):
        # Detect dataclass import statements
        if isinstance(node, ast.ImportFrom) and node.module == 'dataclasses':
            start_line = node.lineno
            end_line = node.end_lineno or start_line
            code = source_code.splitlines()[start_line - 1:end_line]
            results.append((start_line, end_line, 'dataclasses import', '\n'.join(code)))
        
        # Detect class definitions with @dataclass decorator
        elif isinstance(node, ast.ClassDef):
            for decorator in node.decorator_list:
                if isinstance(decorator, ast.Name) and decorator.id == 'dataclass':
                    start_line = node.lineno
                    end_line = node.end_lineno or start_line
                    code = source_code.splitlines()[start_line - 1:end_line]
                    results.append((start_line, end_line, 'dataclass definition', '\n'.join(code)))
                    break
    
    return results

# main
if __name__ == "__main__":
    codes = ["""from dataclasses import dataclass, field

@dataclass
class Person:
    name: str
    age: int

@dataclass
class Car:
    make: str
    model: str
""", """from dataclasses import dataclass

@dataclass
class Point:
    x: float
    y: float
    z: float = 0.0

p = Point(1.5, 2.5)
print(p)"""]
    for code in codes:
        print(detect_pep_557(code))