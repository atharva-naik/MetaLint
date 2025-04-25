import os
import json
import re

def is_pep506_violation(code: str) -> bool:
    """
    Detects true PEP 506 violations: using random or numpy.random to generate secrets.
    """

    # Must use random or np.random
    if not re.search(r'\b(random|np\.random)\b', code):
        return False

    # Must NOT use secrets module
    if re.search(r'\bimport secrets\b|\bsecrets\.', code):
        return False

    # Safe patterns that are never security-related
    safe_patterns = [
        r'\brandom\.shuffle\b',
        r'\brandom\.seed\b',
        r'\brandom\.setstate\b',
        r'\brandom\.getstate\b',
        r'\bnp\.random\.shuffle\b',
        r'\bnp\.random\.seed\b'
    ]
    for pattern in safe_patterns:
        if re.search(pattern, code):
            return False

    # Sensitive context keywords
    sensitive_keywords = ['password', 'token', 'secret', 'auth', 'otp', 'key']

    # Check if any sensitive variable is being assigned from random
    assignment_pattern = re.compile(
        r'(?i)(%s)\s*=\s*.*?(random|np\.random)\.' % '|'.join(sensitive_keywords)
    )

    return bool(assignment_pattern.search(code))

def process_jsonl_for_random_secrets_violations(input_dir, output_file):
    with open(output_file, 'w', encoding='utf-8') as out_f:
        for filename in os.listdir(input_dir):
            if filename.endswith(".jsonl"):
                filepath = os.path.join(input_dir, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        try:
                            data = json.loads(line)
                            code = data.get("content", "")
                            if is_pep506_violation(code):
                                out_f.write(json.dumps(data) + '\n')
                        except json.JSONDecodeError:
                            print(f"Skipping malformed line {line_num} in {filename}")

# Usage
input_directory = "/home/dkundego/stack_v2/data"
output_jsonl = "pep_506_second22_violations.jsonl"

process_jsonl_for_random_secrets_violations(input_directory, output_jsonl)