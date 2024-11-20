# code for detecting syntax tree based idioms.

import os
import re
import ast
import json
import pathlib
import tokenize

# PEP and PEP violation matchers
from tree_search.pep_468 import *
from tree_search.pep_479 import *
from tree_search.pep_487 import *
from tree_search.pep_495 import *
from tree_search.pep_498 import *
from tree_search.pep_498v import *

from tree_search.pep_515 import *
from tree_search.pep_519 import *
from tree_search.pep_525 import *
from tree_search.pep_526 import *
from tree_search.pep_530 import *

from tree_search.pep_553 import *
from tree_search.pep_557 import *
from tree_search.pep_562 import *
from tree_search.pep_563 import *
from tree_search.pep_567 import *

from tree_search.pep_570 import *
from tree_search.pep_572 import *
from tree_search.pep_578 import *
from tree_search.pep_584 import *
from tree_search.pep_585 import *

from tree_search.pep_585v import *
from tree_search.pep_586 import *
from tree_search.pep_589 import *
from tree_search.pep_593 import *
from tree_search.pep_616 import *

from tree_search.pep_634 import *
from tree_search.pep_655 import *
from tree_search.pep_709 import *

# Github issue matchers.
from tree_search.gh_111123 import *
from tree_search.gh_118216 import *

# Get the directory where the script is located
script_directory = os.path.dirname(os.path.abspath(__file__))

# List all files in the directory
file_stems = [pathlib.Path(f).stem for f in os.listdir(script_directory) if os.path.isfile(os.path.join(script_directory, f))]

__detectors__ = [
    "detect_"+file for file in file_stems if file != "__init__"
]