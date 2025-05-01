import ast
import os
import json
import warnings

def find_pep584_opportunities(code: str):
    """Find and suggest places where dict union (|, |=) could be used."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=SyntaxWarning)
        tree = ast.parse(code)

        opportunities = []

        for node in ast.walk(tree):

            if isinstance(node, ast.Dict):
                if any(isinstance(key, ast.Starred) for key in node.keys if key is not None):

                    sources = []
                    for key, value in zip(node.keys, node.values):
                        if key is None and isinstance(value, ast.Starred):
                            src = ast.unparse(value.value) if hasattr(ast, 'unparse') else "<expr>"
                            sources.append(src)
                    if sources:
                        suggestion = f" {' | '.join(sources)} "
                        opportunities.append({
                            "pattern": "dict_unpacking",
                            "lineno": node.lineno,
                            "suggestion": suggestion.strip()
                        })


            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr == "update" and len(node.args) == 1:
                    target = ast.unparse(node.func.value) if hasattr(ast, 'unparse') else "<dict>"
                    arg = ast.unparse(node.args[0]) if hasattr(ast, 'unparse') else "<arg>"
                    suggestion = f"{target} |= {arg}"
                    opportunities.append({
                        "pattern": "dict_update",
                        "lineno": node.lineno,
                        "suggestion": suggestion.strip()
                    })


            elif isinstance(node, ast.For):
                if isinstance(node.target, ast.Tuple) and len(node.target.elts) == 2:

                    if isinstance(node.iter, ast.Call) and isinstance(node.iter.func, ast.Attribute):
                        if node.iter.func.attr == "items":
                            source_dict = ast.unparse(node.iter.func.value) if hasattr(ast, 'unparse') else "<dict>"
                            dest_dict = None

                            for inner_node in ast.walk(node):
                                if isinstance(inner_node, ast.Subscript) and isinstance(inner_node.ctx, ast.Store):
                                    dest_dict = ast.unparse(inner_node.value) if hasattr(ast, 'unparse') else "<dict>"
                            if dest_dict:
                                suggestion = f"{dest_dict} |= {source_dict}"
                                opportunities.append({
                                    "pattern": "manual_items_loop",
                                    "lineno": node.lineno,
                                    "suggestion": suggestion.strip()
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
    process_jsonl_for_pep584_opportunities(input_dir=input_directory, output_file=output_jsonl)
