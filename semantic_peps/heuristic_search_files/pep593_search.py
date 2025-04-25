import ast
import re
import os
import json
import tokenize
from io import StringIO
from typing import List, Tuple

METADATA_PATTERNS = [
    (r"\b(?:must be|should be|has to be|is|are)?\s*(positive|negative|non-negative|non-zero)", "range"),
    (r"\b(?:minimum|min|at least)\s+\d+", "min"),
    (r"\b(?:maximum|max|at most)\s+\d+", "max"),
    (r"\b(optional|may be none|null|can be none)", "nullable"),
    (r"\bchoices?:?\s*\[.*?\]", "enum"),
    (r"\blength\s+(\d+)\s+(to|-)\s+(\d+)", "length_range"),
    (r"\b(regex|pattern)\b", "regex"),
    (r"\bexample\b.*", "example"),
    (r"\bunit\b:?\s*(seconds|kg|cm|ms)", "unit"),
    (r"\bdeprecated\b", "deprecated"),
]

def extract_comments_and_docstrings(code: str) -> List[Tuple[int, str]]:
    code = code.replace("\r\n", "\n").replace("\r", "\n")
    
    lines = code.splitlines()
    comments = []

    for tok_type, tok_str, (srow, scol), _, _ in tokenize.generate_tokens(StringIO(code).readline):
        if tok_type == tokenize.COMMENT:
            comments.append((srow, tok_str.strip("# ").strip()))

    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            docstring = ast.get_docstring(node)
            if docstring:
                lineno = node.lineno
                comments.append((lineno, docstring.strip()))
        elif isinstance(node, ast.Module):
            docstring = ast.get_docstring(node)
            if docstring:
                comments.append((1, docstring.strip()))
    return comments

def find_metadata_matches(comments: List[Tuple[int, str]]) -> List[Tuple[int, str]]:
    matches = []
    for lineno, text in comments:
        for pattern, meta in METADATA_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                matches.append((lineno, meta))
    return matches

def match_metadata_to_nodes(code: str) -> List[str]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []

    comments = extract_comments_and_docstrings(code)
    metadata_hints = find_metadata_matches(comments)

    suggestions = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            fn_line = node.lineno
            fn_name = node.name
            args = node.args.args
            for arg in args:
                if arg.annotation:
                    arg_name = arg.arg
                    arg_line = arg.lineno
                    for hint_line, meta in metadata_hints:
                        if abs(hint_line - arg_line) <= 2:
                            suggestions.append(f"`{arg_name}` in `{fn_name}`: Annotated[{ast.unparse(arg.annotation)}, {meta}=...]")
            if node.returns:
                for hint_line, meta in metadata_hints:
                    if abs(hint_line - fn_line) <= 2:
                        suggestions.append(f"return of `{fn_name}`: Annotated[{ast.unparse(node.returns)}, {meta}=...]")

        if isinstance(node, ast.AnnAssign) and hasattr(node, 'annotation'):
            attr_name = node.target.id if isinstance(node.target, ast.Name) else None
            attr_line = node.lineno
            for hint_line, meta in metadata_hints:
                if abs(hint_line - attr_line) <= 2:
                    suggestions.append(f"attribute `{attr_name}`: Annotated[{ast.unparse(node.annotation)}, {meta}=...]")
    return suggestions

def get_avenues(code: str) -> List[str]:
    return match_metadata_to_nodes(code)

def process_jsonl_for_pep593_avenues(input_dir: str, output_file: str):
    with open(output_file, 'w', encoding='utf-8') as out_f:
        for filename in os.listdir(input_dir):
            if filename.endswith(".jsonl"):
                filepath = os.path.join(input_dir, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        try:
                            data = json.loads(line)
                            content = data.get("content", "").replace("\\", "\\\\")  # Escape backslashes correctly
                            avenues = get_avenues(content)
                            if avenues:
                                data["avenues"] = avenues
                                out_f.write(json.dumps(data) + '\n')
                        except json.JSONDecodeError:
                            print(f"Skipping malformed line {line_num} in {filename}")

input_directory = "./STACK_v2"
output_jsonl = "pep_593_darsh_violations.jsonl"

if __name__ == "__main__":
    process_jsonl_for_pep593_avenues(input_dir=input_directory, output_file=output_jsonl)
