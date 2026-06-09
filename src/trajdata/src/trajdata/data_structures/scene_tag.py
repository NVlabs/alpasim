# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

import re
from typing import Set, Tuple


class SceneTag:
    def __init__(self, tag_tuple: Tuple[str, ...]) -> None:
        self._tag_tuple: Set[str] = set(tag_tuple)

    def contains(self, query: Set[str]) -> bool:
        return query.issubset(self._tag_tuple)

    def matches_any(self, regex: re.Pattern) -> bool:
        return any(regex.search(x) is not None for x in self._tag_tuple)

    def __contains__(self, item) -> bool:
        return item in self._tag_tuple

    def __repr__(self) -> str:
        return "-".join(self._tag_tuple)
