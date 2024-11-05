# code for detecting syntax tree based idioms.

import os
import re
import ast
import json
import pathlib
import tokenize
from tree_search.pep_525 import *
from tree_search.pep_526 import *
from tree_search.pep_557 import *
from tree_search.pep_572 import *
from tree_search.pep_584 import *
from tree_search.pep_616 import *
from tree_search.pep_634 import *

# Get the directory where the script is located
script_directory = os.path.dirname(os.path.abspath(__file__))

# List all files in the directory
file_stems = [pathlib.Path(f).stem for f in os.listdir(script_directory) if os.path.isfile(os.path.join(script_directory, f))]

__detectors__ = [
    "detect_"+file for file in file_stems if file != "__init__"
]