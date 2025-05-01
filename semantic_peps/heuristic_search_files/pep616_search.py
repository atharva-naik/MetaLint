import ast
import os
import json
import warnings

def find_pep616_opportunities(code: str):
    """Find places where removeprefix/removesuffix could simplify startswith/endswith + slicing patterns."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=SyntaxWarning)
        tree = ast.parse(code)

        opportunities = []

        for node in ast.walk(tree):
            if isinstance(node, ast.If):
                if isinstance(node.test, ast.Call):
                    func = node.test.func
                    if isinstance(func, ast.Attribute) and func.attr in ("startswith", "endswith"):
                        if len(node.body) == 1:
                            stmt = node.body[0]
                            if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
                                target = stmt.targets[0]
                                value = stmt.value
                                if (isinstance(target, ast.Name) and 
                                    isinstance(value, ast.Subscript) and
                                    isinstance(value.value, ast.Name) and
                                    target.id == value.value.id and
                                    isinstance(value.slice, ast.Slice)):
                                    
                                    slice_ = value.slice
                                    if func.attr == "startswith" and slice_.lower and not slice_.upper:
                                        opportunities.append({
                                            "line": node.lineno,
                                            "type": "removeprefix",
                                            "variable": target.id
                                        })
                                    elif func.attr == "endswith" and slice_.upper and not slice_.lower:
                                        opportunities.append({
                                            "line": node.lineno,
                                            "type": "removesuffix",
                                            "variable": target.id
                                        })
        return opportunities

    except (SyntaxError, RecursionError):
        return []

def process_jsonl_for_pep616_opportunities(input_dir: str, output_file: str):
    with open(output_file, 'w', encoding='utf-8') as out_f:
        for filename in os.listdir(input_dir):
            if filename.endswith(".jsonl"):
                filepath = os.path.join(input_dir, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        try:
                            data = json.loads(line)
                            code = data.get("content", "")
                            opportunities = find_pep616_opportunities(code)
                            if opportunities:
                                data["pep616_opportunities"] = opportunities
                                out_f.write(json.dumps(data) + '\n')
                        except json.JSONDecodeError:
                            print(f"[warn] Skipping malformed line {line_num} in {filename}")

if __name__ == "__main__":
    input_directory = "./STACK_v2"
    output_jsonl = "pep_616_violations.jsonl"
    process_jsonl_for_pep616_opportunities(input_directory, output_jsonl)
