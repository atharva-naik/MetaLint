# metrics for the meta linting task.
# Coarse Evaluation - Idiom Detection (this basically casts the task as a multi-label classification task: is a given idiom violated/present in a code file.)
# change from v3: this script is specifically hard coded to evaluate on all idioms/meta-tasks.

import os
import sys
import json
import pathlib
import numpy as np
from tqdm import tqdm
from fuzzywuzzy import fuzz
from collections import defaultdict
from sklearn.metrics import precision_score, recall_score

module_path = str(pathlib.Path(os.path.abspath(__file__)).parent.parent.parent.parent)
sys.path.append(module_path)

from src.datautils import read_jsonl

ALL_META_TASKS = [
    # Pyflakes
    ["F401", "F402", "F403", "F404", "F405"],
    ["F406", "F407", "F501", "F502", "F503"],
    ["F504", "F505", "F506", "F507", "F508"],
    ["F509", "F521", "F522", "F523", "F524"],
    ["F525", "F541", "F601", "F602", "F621"],
    ["F622", "F631", "F632", "F633", "F634"],
    ["F701", "F702", "F704", "F706", "F707"],
    ["F722", "F811", "F821", "F822"],
    ["F823", "F841", "F842", "F901"],
    # pygrep-hooks
    ["PGH003","PGH004","PGH005"],
    # pydoclint
    ["DOC201", "DOC202", "DOC402"],
    ["DOC403", "DOC501", "DOC502"],
    # perflint
    ["PERF101", "PERF102", "PERF203"],
    ["PERF401", "PERF402", "PERF403"],
    # pep8-naming
    ["N801", "N802", "N803", "N804"],
    ["N805", "N806", "N807", "N811"],
    ["N812", "N813", "N814", "N815"],
    ["N816", "N817", "N818", "N999"],
    # pycodestyle (E501 - line too long is skipped)
    ["E101", "E111", "E112", "E113", "E114"],
    ["E115", "E116", "E117", "E201", "E202"],
    ["E203", "E204", "E211", "E221", "E222"],
    ["E223", "E224", "E225", "E226", "E227"],
    ["E228", "E231", "E241", "E242", "E251"],
    ["E252", "E261", "E262", "E265", "E266"],
    ["E271", "E272", "E273", "E274", "E275"],
    ["E301", "E302", "E303", "E304", "E305"],
    ["E306", "E401", "E402", "E502"],
    ["E701", "E702", "E703", "E711", "E713"],
    ["E714", "E721", "E722", "E731"],
    ["E741", "E742", "E743", "E902"],
    ["W291", "W292", "W293"],
    ["W391", "W505", "W605"],
    # pathlib + flynt (FLY)
    ["PTH100", "PTH101", "PTH102", "PTH103", "PTH104"],
    ["PTH105", "PTH106", "PTH107", "PTH108", "PTH109"],
    ["PTH110", "PTH111", "PTH112", "PTH113", "PTH114"],
    ["PTH115", "PTH116", "PTH117", "PTH118", "PTH119"],
    ["PTH120", "PTH121", "PTH122", "PTH123", "PTH124"],
    ["PTH201", "PTH202", "PTH203", "PTH204", "PTH205"],
    ["PTH206", "PTH207", "PTH208", "PTH210", 'FLY002'],
    # Misc.
    ["ERA001", "C901", "I001", "I002", "BLE001"],
    # Numpy
    ["NPY001", "NPY002", "NPY003", "NPY201"],
    # Pandas
    ["PD002", "PD003", "PD004", 'PD007', 'PD008'],
    ["PD009", "PD010", "PD011", 'PD012'],
    ['PD013', "PD015", "PD101", 'PD901'],
    # pydocstyle
    ["D100", "D101", "D102", "D103", "D104"],
    ["D105", "D106", "D107", "D200", "D201"],
    ["D202", "D203", "D204", "D205", "D206"],
    ["D207", "D208", "D209", "D210", "D211"],
    ["D212", "D213", "D214", "D215", "D300"],
    ["D301", "D400", "D401", "D402", "D403"],
    ["D404", "D405", "D406", "D407", "D408"],
    ["D409", "D410", "D411", "D412"],
    ["D413", "D414", "D415", "D416"],
    ["D417", "D418", 'D419'],
    # pylint 
    # (PLC)
    ["PLC0105", "PLC0131", "PLC0132", "PLC0205", "PLC0206"],
    ["PLC0208", "PLC0414", "PLC0415", "PLC1802", "PLC1901"],
    ["PLC2401", "PLC2403", "PLC2701", "PLC2801", "PLC3002"],
    # (PLE)
    ["PLE0100", "PLE0101", "PLE0115", "PLE0116", "PLE0117"],
    ["PLE0118", "PLE0237", "PLE0241", "PLE0302", "PLE0303"],
    ["PLE0304", "PLE0305", "PLE0307", "PLE0308", "PLE0309"],
    ["PLE0604", "PLE0605", "PLE0643", "PLE1132"],
    ["PLE1141", "PLE1142", "PLE1205", "PLE1206", "PLE1300"],
    ["PLE1307", "PLE1310", "PLE1507", "PLE1519", "PLE1520"],
    ["PLE2502", "PLE2512", "PLE2513", "PLE2514"],
    ["PLE0704", "PLE1700", "PLE2515", "PLE4703"],
    # (PLR)
    ["PLR0911", "PLR0912", "PLR0913", "PLR0914", "PLR0915"],
    ["PLR0916", "PLR0917", "PLR1702", "PLR1704", "PLR1711"],
    ["PLR1714", "PLR1716", "PLR1722", "PLR1730"],
    ["PLR1733", "PLR1736", "PLR2004", "PLR2044"],
    ["PLR5501", "PLR6104", "PLR6201", "PLR6301"],
    # (PLW)
    ["PLW0108", "PLW0120", "PLW0127", "PLW0128", "PLW0129"],
    ["PLW0131", "PLW0133", "PLW0177", "PLW0211", "PLW0244"],
    ["PLW0245", "PLW0406", "PLW0602", "PLW0603", "PLW0604"],
    ["PLW0642", "PLW0711", "PLW1501", "PLW1507", "PLW1508"],
    ["PLW1509", "PLW1510", "PLW1514", "PLW1641"], 
    ["PLW2101", "PLW2901", "PLW3201", "PLW3301"],
    # pyupgrade (UP)
    ["UP001", "UP003", "UP004", "UP005", "UP006"],
    ["UP007", "UP008", "UP009", "UP010", "UP011"],
    ["UP012", "UP013", "UP014", "UP015", "UP017"],
    ["UP018", "UP019", "UP020", "UP021", "UP022"],
    ["UP023", "UP024", "UP025", "UP026", "UP028"],
    ["UP029", "UP030", "UP031", "UP032", "UP033"],
    ["UP034", "UP035", "UP036", "UP037", "UP039"],
    ["UP040", "UP041", "UP042", "UP043", "UP044"],
    ["UP045", "UP046", "UP047", "UP049"],
    # refurb (FURB)
    ["FURB101", "FURB103", "FURB105", "FURB110", "FURB113"],
    ["FURB116", "FURB118", "FURB122", "FURB129", "FURB131"],
    ["FURB132", "FURB136", "FURB140", "FURB142", "FURB145"],
    ["FURB148", "FURB152", "FURB154", "FURB156", "FURB157"],
    ["FURB161", "FURB162", "FURB163", "FURB164"],
    ["FURB166", "FURB167", "FURB168", "FURB169"],
    ["FURB171", "FURB177", "FURB180", "FURB181"],
    ["FURB187", "FURB188", "FURB189", "FURB192"],
    # ruff specifc rules (RUF)
    ["RUF001", "RUF002", "RUF003", "RUF005", "RUF006"],
    ["RUF007", "RUF008", "RUF009", "RUF010", "RUF012"],
    ["RUF013", "RUF015", "RUF016", "RUF017", "RUF018"],
    ["RUF019", "RUF020", "RUF021", "RUF022", "RUF023"],
    ["RUF024", "RUF026", "RUF027", "RUF028", "RUF029"],
    ["RUF030", "RUF031", "RUF032", "RUF033", "RUF034"],
    ["RUF036", "RUF037", "RUF038", "RUF039", "RUF040"],
    ["RUF041", "RUF043", "RUF045", "RUF046", "RUF047"],
    ["RUF048", "RUF049", "RUF051", "RUF052", "RUF053"],
    ["RUF054", "RUF055", "RUF056", "RUF057", "RUF058"],
    ["RUF059", "RUF100", "RUF101", "RUF102", "RUF200"],
    # tryceratops (TRY)
    ["TRY002", "TRY003", "TRY004", "TRY201", "TRY203"],
    ["TRY300", "TRY301", "TRY400", "TRY401"],
    # flake8-unused-arguments (ARG)
    ["ARG001", "ARG002", "ARG003", "ARG004", "ARG005"],
    # flake8-type-checking (TC)
    ["TC001", "TC002", "TC003", "TC004", "TC005"],
    ["TC006", "TC007", "TC008", "TC010"],
    # flake8-todos (TD)
    ["TD001", "TD002", "TD003", "TD004"],
    ["TD005", "TD006", "TD007"],
    # flake8-tidy-imports (TID)
    ["TID251", "TID252", "TID253"],
    # flake8-slots (SLOT)
    ["SLOT000", "SLOT001", "SLOT002"],
    # flake8-simplify (SIM) + flake8-self (SLF) + flake8-raise (RSE)
    ["SIM101", "SIM102", "SIM103", "SIM105"],
    ["SIM107", "SIM108", "SIM109", "SIM110", "SIM112"],
    ["SIM113", "SIM114", "SIM115", "SIM116", "SIM117"],
    ["SIM118", "SIM201", "SIM202", "SIM208", "SIM210"],
    ["SIM211", "SIM212", "SIM220", "SIM221", "SIM222"],
    ["SIM223", "SIM300", "SIM401", "SLF001"],
    ["SIM905", "SIM910", "SIM911", "RSE102"],
    # flake8-return (RET)
    ["RET501", "RET502", "RET503", "RET504"],
    ["RET505", "RET506", "RET507", "RET508"],
    # flake8-quotes (Q)
    ["Q001", "Q002", "Q003", "Q004"],
    # flake8-pytest-style (PT)
    ["PT001", "PT002", "PT003", "PT006", "PT007"],
    ["PT008", "PT009", "PT010", "PT011", "PT012"],
    ["PT013", "PT014", "PT015", "PT016", "PT017"],
    ["PT018", "PT019", "PT020", "PT021", "PT022"],
    ["PT023", "PT024", "PT025", "PT026", "PT027"],
    ["PT028", "PT029", "PT030", "PT031"],
    # flake8-pyi (PYI)
    ["PYI001", "PYI002", "PYI003", "PYI004", "PYI005"],
    ["PYI006", "PYI007", "PYI008", "PYI009", "PYI010"],
    ["PYI011", "PYI012", "PYI013", "PYI014", "PYI015"],
    ["PYI016", "PYI017", "PYI018", "PYI019", "PYI020"],
    ["PYI021", "PYI024", "PYI025", "PYI026", "PYI029"],
    ["PYI030", "PYI032", "PYI033", "PYI034", "PYI035"],
    ["PYI036", "PYI041", "PYI042", "PYI043", "PYI044"],
    ["PYI045", "PYI046", "PYI047", "PYI048", "PYI049"],
    ["PYI050", "PYI051", "PYI052", "PYI053", "PYI054"],
    ["PYI055", "PYI056", "PYI057", "PYI058", "PYI059"],
    ["PYI061", "PYI062", "PYI063", "PYI064", "PYI066"],
    # flake8-print (T20) + flake8-no-pep420 (INP)
    ["T201", "T203", "INP001"],
    # flake8-pie (PIE)
    ["PIE790", "PIE794", "PIE796", "PIE800"],
    ["PIE804", "PIE807", "PIE808", "PIE810"],
    # flake8-logging-format (G)
    ["G001", "G002", "G003", "G004"],
    ["G010", "G101", "G201", "G202"],
    # flake8-logging (LOG)
    ["LOG001", "LOG002", "LOG004", "LOG007"],
    ["LOG009", "LOG014", "LOG015"],
    # flake8-import-conventions (ICN)
    ["ICN001", "ICN002", "ICN003"],
    # flake8-implicit-str-concat (ISC)
    ["ISC001", "ISC002", "ISC003"],
    # flake8-gettext (INT) + flake8-future-annotations (FA)
    ["INT001", "INT002", "INT003", "FA100", "FA102"],
    # flake8-fixme (FIX)
    ["FIX001", "FIX002", "FIX003", "FIX004"],
    # flake8-executable (EXE)
    ["EXE001", "EXE002", "EXE003", "EXE004", "EXE005"],
    # flake8-errmsg (EM) + flake8-debugger (T10) + flake8-copyright (CPY)
    ["EM101", "EM102", "EM103", "T100", "CPY001"],
    # flake8-django (DJ)
    ["DJ001", "DJ003", "DJ006", "DJ007"],
    ["DJ008", "DJ012", "DJ013"],
    # flake8-datetimez (DTZ)
    ["DTZ001", "DTZ002", "DTZ003", "DTZ004", "DTZ005"],
    ["DTZ006", "DTZ007", "DTZ011", "DTZ012", "DTZ901"],
    # airflow (AIR) + FastAPI (FAST)
    ["AIR001", "AIR002", "AIR301", "AIR302", "AIR311"], 
    ["AIR312", "FAST001", "FAST002", "FAST003"],
    # flake8-2020 (YTT)
    ["YTT101", "YTT102", "YTT103", "YTT201", "YTT202"],
    ["YTT203", "YTT204", "YTT301", "YTT302", "YTT303"],
    # flake8-annotations (ANN)
    ["ANN001", "ANN002", "ANN003", "ANN201", "ANN202"],
    ["ANN204", "ANN205", "ANN206", "ANN401"],
    # flake8-async (ASYNC)
    ["ASYNC100", "ASYNC105", "ASYNC109", "ASYNC110"],
    ["ASYNC115", "ASYNC116", "ASYNC210", "ASYNC220"], 
    ["ASYNC221", "ASYNC222", "ASYNC230", "ASYNC251"],
    # flake8-bandit (S) + flake8-boolean-trap (FBT)
    ["S101", "S102", "S103", "S104", "S105"], 
    ["S106", "S107", "S108", "S110", "S112"], 
    ["S113", "S201", "S202", "S301", "S302"], 
    ["S303", "S304", "S305", "S306", "S307"],
    ["S308", "S310", "S311", "S312", "S313"],
    ["S314", "S315", "S316", "S317", "S318"],
    ["S319", "S321", "S323", "S324", "S401"],
    ["S402", "S403", "S404", "S405", "S406"],
    ["S407", "S408", "S409", "S411", "S412"],
    ["S413", "S415", "S501", "S502", "S503"],
    ["S504", "S505", "S506", "S507", "S508"],
    ["S509", "S601", "S602", "S603", "S604"],
    ["S605", "S606", "S607", "S608", "S609"],
    ["S610", "S611", "S612", "S701", "S702"], 
    ["S704", "FBT001", "FBT002", "FBT003"],
    # flake8-bugbear (B)
    ["B002", "B003", "B004", "B005", "B006"],
    ["B007", "B008", "B009", "B010", "B011"],
    ["B012", "B013", "B014", "B015", "B016"],
    ["B017", "B018", "B019", "B020", "B021"],
    ["B022", "B023", "B024", "B025", "B026"],
    ["B027", "B028", "B029", "B030", "B031"],
    ["B032", "B033", "B034", "B035", "B039"],
    ["B901", "B903", "B904"], 
    ["B905", "B909", "B911"],
    # flake8-builtins (A)
    ["A001", "A002", "A003"],
    ["A004", "A005", "A006"],
    # flake8-commas (COM)
    ["COM812", "COM818", "COM819"],
    # flake8-comprehensions (C4)
    ["C400", "C401", "C402", "C403", "C404"],
    ["C405", "C406", "C408", "C409", "C410"],
    ["C411", "C413", "C414", "C415", "C416"],
    ["C417", "C418", "C419", "C420"],
]
TEST_SET_IDIOMS = []
for meta_task_idioms in ALL_META_TASKS:
    TEST_SET_IDIOMS.extend(meta_task_idioms)

IDIOMS_ABSENT_FROM_TEST_SET = set()

def load_linter_results(text):
    results = []
    idiom_code = None
    for line in text.split("\n"):
        line = line.strip()
        if line == "": continue
        elif line.startswith("**Idiom") and line.endswith("Violations:**"):
            idiom_code = line.removesuffix("Violations:**").removeprefix("**Idiom").strip()
        elif line == "NO VIOLATIONS FOUND": continue
        else:
            try: 
                result = json.loads(line)
                result["code"] = idiom_code
                results.append(result)
            except Exception as e: pass
                # print(e)
                # print(f"{e}: {line}")
    return results

def compute_f_score(p, r):
    if p + r == 0: return 0
    return 2*p*r/(p+r)

def compute_line_level_metric(data):
    p_line, r_line = [], []

    for index,rec in tqdm(enumerate(data)):
        model_resp = load_linter_results(rec["model_response"])
        gt = load_linter_results(rec["ground_truth"])

        idiom_wise_pred_lines = defaultdict(lambda: set())
        idiom_wise_gt_lines = defaultdict(lambda: set())
        for i,model_violation in enumerate(model_resp):
            try: 
                idiom_wise_pred_lines[model_violation['code']].add(int(model_violation['line'].split()[0].strip().removesuffix(":")))    
            except AttributeError: pass   
            except ValueError: pass # print(model_violation['line'])
            except KeyError: pass # print(model_violation.keys())
            except IndexError: print("\x1b[31;1mline:", model_violation['line'], "\x1b[0m")
        for i,gt_violation in enumerate(gt):
            idiom_wise_gt_lines[gt_violation['code']].add(int(gt_violation['line'][:4].strip()))
        for idiom_code in idiom_wise_gt_lines.keys():
            overlap = len(idiom_wise_pred_lines[idiom_code].intersection(idiom_wise_gt_lines[idiom_code]))
            try: p_line_inst_idiom = overlap/len(idiom_wise_pred_lines[idiom_code])
            except ZeroDivisionError: p_line_inst_idiom = 0
            r_line_inst_idiom = overlap/len(idiom_wise_gt_lines[idiom_code])
            p_line.append(p_line_inst_idiom)
            r_line.append(r_line_inst_idiom)

    # average over instances and idioms
    p_line = np.mean(p_line).item()
    r_line = np.mean(r_line).item()
    f_line = compute_f_score(p_line, r_line)

    return {"P": p_line, "R": r_line, "F": f_line}

def compute_overall_metric(data, match_code: bool=True):
    p_line, r_line, p_span, r_span = [], [], [], []

    for index,rec in tqdm(enumerate(data)):
        model_resp = load_linter_results(rec["model_response"])
        gt = load_linter_results(rec["ground_truth"])

        # edge cases.
        if len(model_resp) == 0 and len(gt) == 0: 
            # we skip cases where both are "NO VIOLATION" for the metric (but not for the reward).
            # the reason we skip these for the metric is because they tend to overinflate the scores, since "NO VIOLATION" is pretty common.
            continue
        elif len(model_resp) == 0 or len(gt) == 0: 
            p_line.append(0)
            r_line.append(0)
            p_span.append(0)
            r_span.append(0)
            continue

        span_scores = np.zeros((len(model_resp), len(gt)))
        line_scores = np.zeros((len(model_resp), len(gt)))
        for i,model_violation in enumerate(model_resp):
            for j,gt_violation in enumerate(gt):
                if match_code:
                    # try:
                    span_scores[i][j] = int(model_violation["code"] == gt_violation["code"] and model_violation.get("span","") == gt_violation["span"])
                    line_scores[i][j] = int(model_violation["code"] == gt_violation["code"] and model_violation.get("line","") == gt_violation["line"])
                    # except KeyError as e:
                    #     print(e)
                    #     print(model_violation)
                    #     exit()
                else:
                    # try:
                    span_scores[i][j] = int(model_violation.get("span","") == gt_violation["span"])
                    line_scores[i][j] = int(model_violation.get("line","") == gt_violation["line"])
                    # except KeyError as e:
                    #     print(e)
                    #     print(model_violation)
                    #     exit()
        p_line.append((line_scores.sum(1)>=1).sum().item()/len(model_resp))
        r_line.append((line_scores.sum(0)>=1).sum().item()/len(gt))
        p_span.append((span_scores.sum(1)>=1).sum().item()/len(model_resp))
        r_span.append((span_scores.sum(0)>=1).sum().item()/len(gt))
    
    p_line = np.mean(p_line).item()
    r_line = np.mean(r_line).item()
    f_line = compute_f_score(p_line, r_line)
    p_span = np.mean(p_span).item()
    r_span = np.mean(r_span).item()
    f_span = compute_f_score(p_span, r_span)

    return {"line": {"P": p_line, "R": r_line, "F": f_line}, "span": {"P": p_span, "R": r_span, "F": f_span}}

def compute_meta_task_conf_mat(preds, test_data):
    meta_task_instr_follow_rate = len(preds)
    for pred_rec, test_rec in zip(preds, test_data):
        meta_task_idiom_codes = test_rec["id"].split("_")[0].strip().split("-")
        model_resp = load_linter_results(pred_rec["model_response"])
        pred_idiom_not_in_prompt = False
        for violation in model_resp:
            if violation.get("code","") not in meta_task_idiom_codes: 
                pred_idiom_not_in_prompt = True
        if pred_idiom_not_in_prompt:
            meta_task_instr_follow_rate -= 1
    meta_task_instr_follow_rate /= len(preds)

    return meta_task_instr_follow_rate

def compute_idiom_wise_pr(data):
    global IDIOMS_ABSENT_FROM_TEST_SET
    idiom_binary_presence_pred = {idiom_code: [0 for _ in range(len(data))] for idiom_code in TEST_SET_IDIOMS}
    idiom_binary_presence_gt = {idiom_code: [0 for _ in range(len(data))] for idiom_code in TEST_SET_IDIOMS}
    idiom_precisions, idiom_recalls = {}, {}

    for index,rec in enumerate(data):
        model_resp = load_linter_results(rec["model_response"])
        gt = load_linter_results(rec["ground_truth"])
        for violation in model_resp:
            if "code" not in violation: continue
            idiom_code = violation["code"]
            if idiom_code in TEST_SET_IDIOMS:
                idiom_binary_presence_pred[idiom_code][index] = 1

        # tools_seen = set()    
        for violation in gt:
            if "code" not in violation: continue
            idiom_code = violation["code"]
            idiom_binary_presence_gt[idiom_code][index] = 1

    for idiom_code in TEST_SET_IDIOMS:
        idiom_precisions[idiom_code] = precision_score(idiom_binary_presence_gt[idiom_code], idiom_binary_presence_pred[idiom_code], zero_division=0) # NaN output for undefined recall (no GT present for several idioms).
        if sum(idiom_binary_presence_gt[idiom_code]) == 0: 
            IDIOMS_ABSENT_FROM_TEST_SET.add(idiom_code)
            # print(idiom_code)
        idiom_recalls[idiom_code] = recall_score(idiom_binary_presence_gt[idiom_code], idiom_binary_presence_pred[idiom_code], zero_division=0) # NaN output for undefined recall (no GT present for several idioms).

    return idiom_precisions, idiom_recalls

def compute_aggregate_metrics(idiom_precisions, idiom_recalls):
    print("Overall Detection Metrics:")
    print(f"{len(IDIOMS_ABSENT_FROM_TEST_SET)}/{len(TEST_SET_IDIOMS)} idioms missing from test set.")
    P = np.mean([v for k,v in idiom_precisions.items() if k not in IDIOMS_ABSENT_FROM_TEST_SET]) # this is the only change from our prior evaluation code.
    R = np.mean([v for k,v in idiom_recalls.items() if k not in IDIOMS_ABSENT_FROM_TEST_SET])
    F = compute_f_score(P, R)
    print(f"P: {P:.4f} R: {R:.4f} F: {F:.4f}")

# main
if __name__ == "__main__":
    steps = sys.argv[1]
    test_preds = read_jsonl(f"data/meta_linting_preds_vllm/qwen2.5coder_3b_instruct_sft_preds_{steps}_all_idioms_subtask_cot_star.jsonl")
    test_data = json.load(open("data/ruff_meta_linting/all_idioms/test.json"))
    
    idiom_precisions, idiom_recalls = compute_idiom_wise_pr(test_preds)
    compute_aggregate_metrics(idiom_precisions, idiom_recalls)
    meta_task_instr_follow_rate = compute_meta_task_conf_mat(preds=test_preds, test_data=test_data)
    print(f"\x1b[34;1minstruction follow rate: {meta_task_instr_follow_rate:.4f}\x1b[0m")

    print("\n\x1b[32;1mOverall Metric (Detection+Violation)\x1b[0m")
    overall_det_loc_metric = compute_overall_metric(test_preds)
    for k,v in overall_det_loc_metric["span"].items():
        print(f"span: {k}={v:.4f}")

    overall_det_loc_metric = compute_line_level_metric(test_preds)
    for k,v in overall_det_loc_metric.items():
        print(f"line: {k}={v:.4f}")