# code for finding token/regex based patterns.

import os
import re
import ast
import json
import pathlib
import tokenize
from token_search.pep_572 import *
from token_search.gh_109118_and_gh_118160 import *

# Get the directory where the script is located
script_directory = os.path.dirname(os.path.abspath(__file__))

# List all files in the directory
file_stems = [pathlib.Path(f).stem for f in os.listdir(script_directory) if os.path.isfile(os.path.join(script_directory, f))]

__detectors__ = [
    "detect_"+file for file in file_stems if file != "__init__"
]

# print(__detectors__)