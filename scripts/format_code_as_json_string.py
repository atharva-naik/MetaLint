#!/usr/bin/env python3
"""
py_to_json_string.py

Usage examples:
  python py_to_json_string.py script.py
  python py_to_json_string.py -             # read from stdin
  python py_to_json_string.py script.py --no-quotes --no-ensure-ascii

Output is a JSON string (the result of json.dumps(file_contents)).
"""
import argparse
import json
import sys
from pathlib import Path

def read_file(path: str, encoding: str):
    if path == "-":
        # read from stdin
        return sys.stdin.read()
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"file not found: {path}")
    # use surrogateescape so we don't fail on odd bytes but still preserve content
    return p.read_text(encoding=encoding, errors="surrogateescape")

def main():
    ap = argparse.ArgumentParser(
        description="Print a json.dumps(...) string of a Python file (escapes newlines/tabs/etc.)"
    )
    ap.add_argument("filename", help="path to file to read, or '-' to read from stdin")
    ap.add_argument(
        "--encoding", "-e", default="utf-8",
        help="file encoding (default: utf-8)"
    )
    ap.add_argument(
        "--no-quotes", action="store_true",
        help="print the JSON string without the surrounding double quotes"
    )
    ap.add_argument(
        "--no-ensure-ascii", action="store_true",
        help="allow non-ASCII characters to appear unescaped (passes ensure_ascii=False to json.dumps)"
    )
    args = ap.parse_args()

    try:
        content = read_file(args.filename, args.encoding)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(2)

    ensure_ascii = not args.no_ensure_ascii
    dumped = json.dumps(content, ensure_ascii=ensure_ascii)

    if args.no_quotes:
        # remove the outer JSON quotes; the inner escapes stay intact
        if len(dumped) >= 2 and dumped[0] == '"' and dumped[-1] == '"':
            dumped = dumped[1:-1]
    print(dumped)

if __name__ == "__main__":
    main()