import os
import sys
import json
import pathlib
from tqdm import tqdm
from sklearn.metrics import cohen_kappa_score

module_path = str(pathlib.Path(os.path.abspath(__file__)).parent.parent)
# print(module_path)
sys.path.append(module_path)

from src.datautils import load_stack_dump

if __name__ == "__main__":
    stack_data = load_stack_dump("data/STACK-V2", as_dict=True)
    inter_annot_agreement_dir = "data/pep_benchmark/inter-annotator_annotations"
    annot_dir = "data/pep_benchmark/annotations"
    annotator_1 = []
    annotator_2 = []

    for file in tqdm(os.listdir(inter_annot_agreement_dir)):
        annot1_file = os.path.join(inter_annot_agreement_dir, file)
        annot2_file = os.path.join(annot_dir, file)
        if not os.path.exists(annot2_file): continue # skip missing PEP annots by annotator 2 or 3.
        # print(file)

        try: 
            annot1 = json.load(open(annot1_file))
        except json.decoder.JSONDecodeError:
            if os.path.getsize(annot1_file) == 0: continue
            else: 
                print(f"issue with: {annot1_file}")
                exit()
        
        try: 
            annot2 = json.load(open(annot2_file))
        except json.decoder.JSONDecodeError:
            if os.path.getsize(annot2_file) == 0: continue
            else: 
                print(f"issue with: {annot2_file}")
                exit()
        
        blob_id = annot1["blob_id"].split('_')[0].strip()
        stack_file = stack_data[blob_id]['content']
        linenos = len(stack_file.split("\n"))
        annotator_1_inst = [0 for _ in range(linenos)]
        annotator_2_inst = [0 for _ in range(linenos)]

        for violation in annot1["violations"]:
            for lineno in range(violation['start_lineno'], violation["end_lineno"]+1):
                annotator_1_inst[lineno-1] = 1
        annotator_1.extend(annotator_1_inst)

        for violation in annot2["violations"]:
            for lineno in range(violation['start_lineno'], violation["end_lineno"]+1):
                annotator_2_inst[lineno-1] = 1
        annotator_2.extend(annotator_2_inst)


    assert len(annotator_1) == len(annotator_2)
    kappa = cohen_kappa_score(annotator_1, annotator_2)
    print(kappa)