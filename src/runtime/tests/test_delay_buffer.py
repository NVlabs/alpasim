# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

import pytest
from alpasim_runtime.delay_buffer import DelayBuffer


def test_delay_buffer():
    delay_buffer = DelayBuffer(1)

    OBJ1 = "obj1"
    OBJ2 = "obj2"
    OBJ3 = "obj3"
    delay_buffer.add(OBJ1, 1)
    assert delay_buffer.at(0) == OBJ1  # Note: primed using the first object
    assert delay_buffer.at(1) == OBJ1

    delay_buffer.add(OBJ2, 2)
    delay_buffer.add(OBJ3, 3)
    assert delay_buffer.at(2) == OBJ1
    assert delay_buffer.at(3) == OBJ2
    assert delay_buffer.at(4) == OBJ3


def test_delay_buffer_item_at():
    delay_buffer = DelayBuffer(1)

    OBJ1 = "obj1"
    OBJ2 = "obj2"
    OBJ3 = "obj3"
    delay_buffer.add(OBJ1, 1)
    assert delay_buffer.item_at(1) == (1, OBJ1)

    delay_buffer.add(OBJ2, 2)
    delay_buffer.add(OBJ3, 3)
    assert delay_buffer.item_at(2) == (1, OBJ1)
    assert delay_buffer.item_at(3) == (2, OBJ2)
    assert delay_buffer.item_at(4) == (3, OBJ3)


def test_delay_buffer_item_at_empty():
    delay_buffer = DelayBuffer(1)
    assert delay_buffer.item_at(5) == (None, None)


def test_delay_buffer_raise_on_out_of_order():
    delay_buffer = DelayBuffer(1)

    OBJ1 = "obj1"
    OBJ2 = "obj2"
    delay_buffer.add(OBJ1, 1)
    with pytest.raises(ValueError):
        delay_buffer.add(OBJ2, 0)
    assert delay_buffer.at(1) == OBJ1
    assert delay_buffer.at(2) == OBJ1
