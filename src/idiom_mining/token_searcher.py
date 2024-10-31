from token_search import *
from token_search import __detectors__

# main
if __name__ == "__main__":
    import sys

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

    from tqdm import tqdm
    from src.datautils import load_stack_dump
    
    stack_data = load_stack_dump("./data/STACK-V2")
    codes = [file['content'] for file in stack_data]
    blob_ids = [file['blob_id'] for file in stack_data]

    for code, blob_id in tqdm(zip(codes, blob_ids)):
        for detector in __detectors__:
            result = eval(f"{detector}(code)")
            print(result)
            break