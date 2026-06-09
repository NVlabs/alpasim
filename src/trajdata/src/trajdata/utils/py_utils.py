# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

import hashlib
import json
from typing import Dict, List, Set, Tuple, Union


def hash_dict(o: Union[Dict, List, Tuple, Set]) -> str:
    """
    Makes a hash from a dictionary, list, tuple or set to any level, that contains
    only other hashable types (including any lists, tuples, sets, and
    dictionaries).
    """
    string_rep: str = json.dumps(o)
    return hashlib.sha1(str.encode(string_rep)).hexdigest()
