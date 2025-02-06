import os
import ast
import sys
import json
import torch
import warnings
from transformers import pipeline
from src.idiom_mining.semantic_search.llm_search import PEP_VIOLATION_META_INSTRUCTION

# Suppress only SyntaxWarnings
warnings.filterwarnings("ignore", category=SyntaxWarning)

# def pre_filter():
PEP_557_EXAMPLES = {
    "Example 1: Basic Data Storage": {
        "before": '''class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return f"Point(x={self.x}, y={self.y})"

p = Point(1, 2)
print(p)
''',
        "after": '''from dataclasses import dataclass

@dataclass
class Point:
    x: int
    y: int

p = Point(1, 2)
print(p)
'''
    },
    "Example 2: Adding Default Values": {
        "before": '''class Rectangle:
    def __init__(self, width, height=1):
        self.width = width
        self.height = height

    def __repr__(self):
        return f"Rectangle(width={self.width}, height={self.height})"

r = Rectangle(5)
print(r)
''',
        "after": '''from dataclasses import dataclass

@dataclass
class Rectangle:
    width: int
    height: int = 1

r = Rectangle(5)
print(r)
'''
    },
    "Example 3: Comparison Support": {
        "before": '''class Card:
    def __init__(self, rank, suit):
        self.rank = rank
        self.suit = suit

    def __eq__(self, other):
        if isinstance(other, Card):
            return self.rank == other.rank and self.suit == other.suit
        return False

c1 = Card('A', 'hearts')
c2 = Card('A', 'hearts')
print(c1 == c2)
''',
        "after": '''from dataclasses import dataclass

@dataclass
class Card:
    rank: str
    suit: str

c1 = Card('A', 'hearts')
c2 = Card('A', 'hearts')
print(c1 == c2)
'''
    },
    "Example 4: Mutable vs Immutable Data": {
        "before": '''class Position:
    def __init__(self, x, y):
        self.x = x
        self.y = y

p = Position(1, 2)
p.x = 3  # Mutable by default
''',
        "after": '''from dataclasses import dataclass

@dataclass(frozen=True)
class Position:
    x: int
    y: int

p = Position(1, 2)
# p.x = 3  # Raises error because Position is immutable
'''
    },
    "Example 5: Default Factory for Mutable Fields": {
        "before": '''class Inventory:
    def __init__(self):
        self.items = []

    def add_item(self, item):
        self.items.append(item)

inventory = Inventory()
inventory.add_item('apple')
print(inventory.items)
''',
        "after": '''from dataclasses import dataclass, field

@dataclass
class Inventory:
    items: list = field(default_factory=list)

    def add_item(self, item):
        self.items.append(item)

inventory = Inventory()
inventory.add_item('apple')
print(inventory.items)
'''
    },
    "Example 6: Post-Initialization Logic": {
        "before": '''class Employee:
    def __init__(self, name, salary):
        self.name = name
        self.salary = salary
        self.tax = self.salary * 0.2

e = Employee("Alice", 50000)
print(e.tax)
''',
        "after": '''from dataclasses import dataclass, field

@dataclass
class Employee:
    name: str
    salary: float
    tax: float = field(init=False)

    def __post_init__(self):
        self.tax = self.salary * 0.2

e = Employee("Alice", 50000)
print(e.tax)
'''
    },
    "Example 7: Combining with Type Hints and Validation": {
        "before": '''class Product:
    def __init__(self, name, price):
        if price < 0:
            raise ValueError("Price cannot be negative")
        self.name = name
        self.price = price

p = Product("Widget", 10)
''',
        "after": '''from dataclasses import dataclass

@dataclass
class Product:
    name: str
    price: float

    def __post_init__(self):
        if self.price < 0:
            raise ValueError("Price cannot be negative")

p = Product("Widget", 10)
'''
    }
}

PEP_557_DESCRIPTION = '''You are given a description of PEP 557. This PEP describes an addition to the standard library called Data Classes. Although they use a very different mechanism, Data Classes can be thought of as “mutable namedtuples with defaults”. Because Data Classes use normal class definition syntax, you are free to use inheritance, metaclasses, docstrings, user-defined methods, class factories, and other Python class features.

A class decorator is provided which inspects a class definition for variables with type annotations as defined in PEP 526, “Syntax for Variable Annotations”. In this document, such variables are called fields. Using these fields, the decorator adds generated method definitions to the class to support instance initialization, a repr, comparison methods, and optionally other methods as described in the Specification section. Such a class is called a Data Class, but there’s really nothing special about the class: the decorator adds generated methods to the class and returns the same class it was given.'''

def detect_special_methods(source_code: str):
    """Detect classes with __repr__ or __eq__ methods in a Python file."""
    tree = ast.parse(source_code)
    
    classes_with_methods = []

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            methods = {
                method.name
                for method in node.body
                if isinstance(method, ast.FunctionDef) and method.name in {'__repr__', '__eq__'}
            }
            if methods:
                start_lineno = node.lineno
                end_lineno = max(
                    child.end_lineno for child in ast.walk(node) if hasattr(child, 'end_lineno')
                )
                class_code_block = "\n".join(source_code.splitlines()[start_lineno-1:end_lineno])
                classes_with_methods.append((start_lineno, end_lineno, class_code_block))

    return classes_with_methods

def build_prompt(file: str=''):
    return PEP_VIOLATION_META_INSTRUCTION + "\n\n" + PEP_557_DESCRIPTION + "\n\n" + "\n\n".join([k+"\n\n"+"Before:\n"+v['before']+"\nAfter:\n"+v['after'] for k,v in PEP_557_EXAMPLES.items()])+"""
    
Now look at the code below and report code locations that can be refactored according to the PEP and say whether it can be refactored according to the PEP ( (YES) or (NO) )

File:
{file}
    
Answer: (""".format(file=file)

# main
if __name__ == "__main__":
    from src.datautils import load_stack_dump, strip_comments_and_docstrings

    # Magicoder PEP relevance classification.
    generator = pipeline(
        model="ise-uiuc/Magicoder-S-DS-6.7B",
        task="text-generation",
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
    )
    write_path: str = "data/pattern_mining/semantic_patterns/llm_patterns/pep_557.jsonl"
    if os.path.exists(write_path):
        resp = input("overwrite? (y/N)").lower().strip()
        if resp not in ["y", "yes"]: exit()
    open(write_path, "w")

    stack_data = load_stack_dump("./data/STACK-V2")
    
    for file in stack_data:
        content = strip_comments_and_docstrings(file['content'])
        try: classes_with_methods = detect_special_methods(content)
        except SyntaxError: continue
        if len(classes_with_methods) > 0:
            blob_id = file['blob_id']
            patterns = []
            for start, end, code in classes_with_methods:
                prompt = build_prompt(file=code)
                # print(classes_with_methods)
                # print(prompt)
                result = generator(prompt, max_new_tokens=10, num_return_sequences=1, temperature=0.0)
                
                op = result[0]["generated_text"].replace(prompt, "").strip().split("\n")[0].strip()
                print(op)
                if "YES" in op:
                    patterns.append((start, end, code))
            with open(write_path, "a") as f:
                f.write(json.dumps({"blob_id": blob_id, "patterns": patterns})+"\n")
                    
