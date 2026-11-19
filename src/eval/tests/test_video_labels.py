# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Text labels the default video layout draws on the camera pane."""

from eval.video import REASONING_MAX_LINES, REASONING_WRAP_CHARS, reasoning_label


def test_reasoning_label_is_absent_without_text() -> None:
    assert reasoning_label(None) is None
    assert reasoning_label("") is None
    assert reasoning_label("   \n ") is None


def test_reasoning_label_wraps_and_collapses_whitespace() -> None:
    text = (
        "The pedestrian  is crossing.\nSlow down and\tyield before the crosswalk. " * 3
    )
    label = reasoning_label(text)
    assert label is not None
    lines = label.split("\n")
    assert lines[0].startswith("CoT: The pedestrian is crossing. Slow down and yield")
    assert all(len(line) <= REASONING_WRAP_CHARS + len("CoT: ") for line in lines)
    assert "\t" not in label


def test_reasoning_label_truncates_a_long_chain_of_thought() -> None:
    label = reasoning_label("word " * 2000)
    assert label is not None
    lines = label.split("\n")
    assert len(lines) == REASONING_MAX_LINES
    assert lines[-1].endswith("…")
