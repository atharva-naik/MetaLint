import ast
import os
import json
import warnings

def extract_assignments(tree):
    """Map variable names to expressions from top-level assignments."""
    assignments = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            var_name = node.targets[0].id
            assignments[var_name] = node.value
    return assignments

def find_pep614_violations(code: str):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=SyntaxWarning)
            tree = ast.parse(code)

        assignments = extract_assignments(tree)
        violations = []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                for deco in node.decorator_list:
                    if isinstance(deco, ast.Name):
                        assigned_expr = assignments.get(deco.id)
                        if isinstance(assigned_expr, (ast.Subscript, ast.Attribute, ast.Call)):
                            violations.append({
                                "function": node.name,
                                "decorator": deco.id,
                                "expr_type": type(assigned_expr).__name__
                            })
        return violations

    except (SyntaxError, RecursionError):
        return []

def process_jsonl_for_pep614_violations(input_dir: str, output_file: str):
    with open(output_file, 'w', encoding='utf-8') as out_f:
        for filename in os.listdir(input_dir):
            if filename.endswith(".jsonl"):
                filepath = os.path.join(input_dir, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        try:
                            data = json.loads(line)
                            code = data.get("content", "")
                            violations = find_pep614_violations(code)
                            if violations:
                                data["pep614_violations"] = violations
                                out_f.write(json.dumps(data) + '\n')
                        except json.JSONDecodeError:
                            print(f"[warn] Skipping malformed line {line_num} in {filename}")

if __name__ == "__main__":
    input_directory = "./STACK_v2"
    output_jsonl = "pep_614_darsh_violations.jsonl"
    process_jsonl_for_pep614_violations(input_directory, output_jsonl)
