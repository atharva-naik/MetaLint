import warnings
from tree_search import *
from tree_search import __detectors__ as tree_detectors

# Suppress only SyntaxWarnings
warnings.filterwarnings("ignore", category=SyntaxWarning)

# main
if __name__ == "__main__":
    import sys

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

    from tqdm import tqdm
    from src.datautils import load_stack_dump, strip_comments_and_docstrings
    
    stack_data = load_stack_dump("./data/STACK-V2")
    codes = [file['content'] for file in stack_data]
    blob_ids = [file['blob_id'] for file in stack_data]

    open("data/pattern_mining/tree_patterns/all_patterns_streaming.jsonl", 'w')
    detected_patterns = []
    for code, blob_id in tqdm(zip(codes, blob_ids)):
        file_patterns = {}
        code_without_comments_and_docstrings = strip_comments_and_docstrings(code)

        # syntax/AST parsing specific handling for tree-searching.
        try: 
            ast_root = ast.parse(code_without_comments_and_docstrings) # get the top level AST root/module objecct.
            for detector in tree_detectors:
                pattern_name = detector.replace('detect_', '')
                try: 
                    result = eval(f"{detector}(code_without_comments_and_docstrings, ast_root)")
                except RecursionError as e:
                    result = None
                if result: file_patterns[pattern_name] = result

        except (SyntaxError, IndentationError) as e:
            # print(e)
            continue

        if len(file_patterns) > 0:
            detected_patterns.append({
                "blob_id": blob_id,
                "patterns": file_patterns
            })

            with open("data/pattern_mining/tree_patterns/all_patterns_streaming.jsonl", "a") as f:
                f.write(json.dumps({"blob_id": blob_id, "patterns": file_patterns})+"\n")
    