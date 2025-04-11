import json

def sort_violations(violations):
    def sort_key(v):
        loc = v["location"]
        edit_loc = v["fix"]["edits"][0]["location"]
        return (loc["row"], loc["column"], edit_loc["row"], edit_loc["column"])
    return sorted(violations, key=sort_key)

def pos_to_index(lines, pos):
    row, col = pos["row"], pos["column"]
    offset = sum(len(lines[i]) for i in range(row - 1)) + (col - 1)
    return offset

def index_to_pos(text, index):
    row = text.count("\n", 0, index) + 1
    last_newline = text.rfind("\n", 0, index)
    col = index + 1 if last_newline == -1 else index - last_newline
    return {"row": row, "column": col}

def apply_fix(content, edits):
    lines = content.splitlines(keepends=True)
    sorted_edits = sorted(edits, key=lambda e: (e["location"]["row"], e["location"]["column"]))
    offset = 0
    new_content = content
    reverse_edits = []
    
    for edit in sorted_edits:
        start = pos_to_index(new_content.splitlines(keepends=True), edit["location"]) + offset
        end = pos_to_index(new_content.splitlines(keepends=True), edit["end_location"]) + offset
        
        original_text = new_content[start:end]
        replacement_text = edit["content"]
        
        new_content = new_content[:start] + replacement_text + new_content[end:]
        delta = len(replacement_text) - (end - start)
        offset += delta
        
        new_start = start
        new_end = start + len(replacement_text)
        new_loc = index_to_pos(new_content, new_start)
        new_end_loc = index_to_pos(new_content, new_end)
        
        reverse_edits.append({
            "location": new_loc,
            "end_location": new_end_loc,
            "content": original_text
        })
    return new_content, reverse_edits

def create_reverse_fixes(content, violations):
    new_content = content
    reverse_violations = []
    
    for v in violations:
        v_copy = v.copy()
        fix = v_copy.get("fix", {}).copy()
        edits = fix.get("edits", [])
        new_content, reverse_edits = apply_fix(new_content, edits)
        fix["edits"] = reverse_edits
        v_copy["fix"] = fix
        reverse_violations.append(v_copy)
    return new_content, reverse_violations

def main():
    original_json_content = {
    "content": "\ufeff\nCreated on 2017年9月27日\n\n@author: Mi\n"
               "from setuptools import setup\n\n"
               "setup(\n"
               "    name='proxy',\n"
               "    py_modules=['proxy','regeditor'],\n"
               "    install_requires=['docopt'],\n"
               "    entry_points={\n"
               "        'console_scripts': ['proxy=proxy:cli']\n"
               "    }\n"
               ")"
}
    linter_output_json = r'''{
  "blob_id": "f1488942efb065a9ddd13a48146c5eb151ed920f",
  "violations": [
    {
      "cell": null,
      "code": "Q002",
      "end_location": {
        "column": 4,
        "row": 5
      },
      "filename": "/home/arnaik/OracleProject/Ps31I57K4ACyD2Wo",
      "fix": {
        "applicability": "safe",
        "edits": [
          {
            "content": "\"\"\"\\nCreated on 2017\\u5e749\\u670827\\u65e5\\n\\n@author: Mi\\n\"\"\"",
            "end_location": {
              "column": 4,
              "row": 5
            },
            "location": {
              "column": 1,
              "row": 1
            }
          }
        ],
        "message": "Replace single quotes docstring with double quotes"
      },
      "location": {
        "column": 1,
        "row": 1
      },
      "message": "Single quote docstring found but double quotes preferred",
      "noqa_row": 5,
      "url": "https://docs.astral.sh/ruff/rules/bad-quotes-docstring"
    },
    {
      "cell": null,
      "code": "D212",
      "end_location": {
        "column": 4,
        "row": 5
      },
      "filename": "/home/arnaik/OracleProject/Ps31I57K4ACyD2Wo",
      "fix": {
        "applicability": "safe",
        "edits": [
          {
            "content": "",
            "end_location": {
              "column": 1,
              "row": 2
            },
            "location": {
              "column": 4,
              "row": 1
            }
          }
        ],
        "message": "Remove whitespace after opening quotes"
      },
      "location": {
        "column": 1,
        "row": 1
      },
      "message": "Multi-line docstring summary should start at the first line",
      "noqa_row": 5,
      "url": "https://docs.astral.sh/ruff/rules/multi-line-summary-first-line"
    },
    {
      "cell": null,
      "code": "D300",
      "end_location": {
        "column": 4,
        "row": 5
      },
      "filename": "/home/arnaik/OracleProject/Ps31I57K4ACyD2Wo",
      "fix": {
        "applicability": "safe",
        "edits": [
          {
            "content": "\"\"\"\\nCreated on 2017\\u5e749\\u670827\\u65e5\\n\\n@author: Mi\\n\"\"\"",
            "end_location": {
              "column": 4,
              "row": 5
            },
            "location": {
              "column": 1,
              "row": 1
            }
          }
        ],
        "message": "Convert to triple double quotes"
      },
      "location": {
        "column": 1,
        "row": 1
      },
      "message": "Use triple double quotes `\"\"\"`",
      "noqa_row": 5,
      "url": "https://docs.astral.sh/ruff/rules/triple-single-quotes"
    },
    {
      "cell": null,
      "code": "D400",
      "end_location": {
        "column": 4,
        "row": 5
      },
      "filename": "/home/arnaik/OracleProject/Ps31I57K4ACyD2Wo",
      "fix": {
        "applicability": "unsafe",
        "edits": [
          {
            "content": ".",
            "end_location": {
              "column": 22,
              "row": 2
            },
            "location": {
              "column": 22,
              "row": 2
            }
          }
        ],
        "message": "Add period"
      },
      "location": {
        "column": 1,
        "row": 1
      },
      "message": "First line should end with a period",
      "noqa_row": 5,
      "url": "https://docs.astral.sh/ruff/rules/missing-trailing-period"
    },
    {
      "cell": null,
      "code": "D415",
      "end_location": {
        "column": 4,
        "row": 5
      },
      "filename": "/home/arnaik/OracleProject/Ps31I57K4ACyD2Wo",
      "fix": {
        "applicability": "unsafe",
        "edits": [
          {
            "content": ".",
            "end_location": {
              "column": 22,
              "row": 2
            },
            "location": {
              "column": 22,
              "row": 2
            }
          }
        ],
        "message": "Add closing punctuation"
      },
      "location": {
        "column": 1,
        "row": 1
      },
      "message": "First line should end with a period, question mark, or exclamation point",
      "noqa_row": 5,
      "url": "https://docs.astral.sh/ruff/rules/missing-terminal-punctuation"
    },
    {
      "cell": null,
      "code": "COM812",
      "end_location": {
        "column": 47,
        "row": 13
      },
      "filename": "/home/arnaik/OracleProject/Ps31I57K4ACyD2Wo",
      "fix": {
        "applicability": "safe",
        "edits": [
          {
            "content": "],",
            "end_location": {
              "column": 47,
              "row": 13
            },
            "location": {
              "column": 46,
              "row": 13
            }
          }
        ],
        "message": "Add trailing comma"
      },
      "location": {
        "column": 47,
        "row": 13
      },
      "message": "Trailing comma missing",
      "noqa_row": 13,
      "url": "https://docs.astral.sh/ruff/rules/missing-trailing-comma"
    },
    {
      "cell": null,
      "code": "COM812",
      "end_location": {
        "column": 6,
        "row": 14
      },
      "filename": "/home/arnaik/OracleProject/Ps31I57K4ACyD2Wo",
      "fix": {
        "applicability": "safe",
        "edits": [
          {
            "content": "},",
            "end_location": {
              "column": 6,
              "row": 14
            },
            "location": {
              "column": 5,
              "row": 14
            }
          }
        ],
        "message": "Add trailing comma"
      },
      "location": {
        "column": 6,
        "row": 14
      },
      "message": "Trailing comma missing",
      "noqa_row": 14,
      "url": "https://docs.astral.sh/ruff/rules/missing-trailing-comma"
    }
  ]
}'''

    content_obj = json.loads(original_json_content)
    linter_obj = json.loads(linter_output_json)
    content = content_obj["content"]
    violations = linter_obj["violations"]

    sorted_violations = sort_violations(violations)
    new_code, reverse_violations = create_reverse_fixes(content, sorted_violations)
    
    result = {
        "new_code": new_code,
        "reverse_fixes": {
            "blob_id": linter_obj["blob_id"],
            "violations": reverse_violations
        }
    }
    
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
