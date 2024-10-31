from token_search import *
from token_search import __detectors__

# main
if __name__ == "__main__":
    import sys

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

    from tqdm import tqdm
    from src.datautils import load_stack_dump, strip_comments_and_docstrings
    
    stack_data = load_stack_dump("./data/STACK-V2")
    codes = [file['content'] for file in stack_data]
    blob_ids = [file['blob_id'] for file in stack_data]

    detected_patterns = []
    for code, blob_id in tqdm(zip(codes, blob_ids)):
        file_patterns = {}
        for detector in __detectors__:
            pattern_name = detector.replace('detect_', '')
            code_without_comments_and_docstrings = strip_comments_and_docstrings(code)
            result = eval(f"{detector}(code_without_comments_and_docstrings)")
            if result: file_patterns[pattern_name] = result
        if len(file_patterns) > 0:
            detected_patterns.append({
                "blob_id": blob_id,
                "patterns": file_patterns
            })
    # print(detected_patterns)
    with open("data/pattern_mining/token_patterns/all_patterns.json", "w") as f:
        json.dump(detected_patterns, f, indent=4)
    