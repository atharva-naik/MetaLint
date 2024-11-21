import google.generativeai as genai
import json
import os
import time

genai.configure(api_key="AIzaSyDTjWWdzszUosXTZatVrSQ5H61skDuq_zM")

import json

output_data = []
with open('/home/dkundego/OracleProject/data/semantic_search/pep506_password_import_random.jsonl', 'r') as file:
    for line in file:
        code_entry = json.loads(line)
        code_content = code_entry.get("content", "")
        file_path = code_entry.get("path", "unknown")

        prompt = f"""Analyze the following Python code to determine if it violates PEP 506 by using the `random` module for security-sensitive purposes instead of the `secrets` module:

        {code_content}

        PEP 506 introduced the secrets module to Python, which provides cryptographically strong random number generation for managing sensitive data like passwords and tokens. The secrets module or os.urandom() should be used instead. Pseudo-random generators in the random module should NOT be used for security purposes.

        Does this code violate PEP 506? Return answer as "yes" or "no". Do not return anything else.
        """

        try:
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(prompt)
            print(response.text)
            if response and hasattr(response, 'text') and response.text.strip().lower() == "yes":
                output_data.append({
                    "file_path": file_path,
                    "violates": "yes",
                    "code_content": code_content
                })

        except Exception as e:
            print(f"Failed to process file {file_path}: {e}")

        time.sleep(2)

with open('/home/dkundego/OracleProject/data/semantic_search/pep506_analysis_results.json', 'w') as output_file:
    json.dump(output_data, output_file, indent=4)
