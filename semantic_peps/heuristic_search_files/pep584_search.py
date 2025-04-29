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

def find_pep584_opportunities(code: str):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=SyntaxWarning)
        tree = ast.parse(code)

        assignments = extract_assignments(tree)
        opportunities = []

        for node in ast.walk(tree):

            if isinstance(node, ast.Dict) and any(isinstance(k, ast.Starred) for k in node.keys if k is not None):
                opportunities.append({
                    "type": "dict_unpacking",
                    "lineno": node.lineno
                })

            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr == "update" and len(node.args) == 1:
                    opportunities.append({
                        "type": "dict_update",
                        "target": ast.unparse(node.func.value) if hasattr(ast, 'unparse') else "<dict>",
                        "lineno": node.lineno
                    })

            if isinstance(node, ast.For):
                if isinstance(node.target, ast.Tuple) and len(node.target.elts) == 2:
                    if isinstance(node.iter, ast.Call) and isinstance(node.iter.func, ast.Attribute):
                        if node.iter.func.attr == "items":
                            for inner in ast.walk(node):
                                if isinstance(inner, ast.Subscript) and isinstance(inner.ctx, ast.Store):
                                    opportunities.append({
                                        "type": "manual_items_loop",
                                        "lineno": node.lineno
                                    })

        return opportunities

    except (SyntaxError, RecursionError):
        return []

def process_jsonl_for_pep584_opportunities(input_dir: str, output_file: str):
    with open(output_file, 'w', encoding='utf-8') as out_f:
        for filename in os.listdir(input_dir):
            if filename.endswith(".jsonl"):
                filepath = os.path.join(input_dir, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        try:
                            data = json.loads(line)
                            code = data.get("content", "")
                            opportunities = find_pep584_opportunities(code)
                            if opportunities:
                                data["pep584_opportunities"] = opportunities
                                out_f.write(json.dumps(data) + '\n')
                        except json.JSONDecodeError:
                            print(f"[warn] Skipping malformed line {line_num} in {filename}")

if __name__ == "__main__":
    input_directory = "./STACK_v2"
    output_jsonl = "pep_584_violations.jsonl"
    process_jsonl_for_pep584_opportunities(input_directory, output_jsonl)