# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 NVIDIA Corporation

from unittest.mock import Mock

import pytest
from alpasim_runtime.clip_cache_monitor import ClipCacheMonitor


def test_clip_cache_monitor_init():
    """Test that ClipCacheMonitor initializes with empty dictionaries."""
    monitor = ClipCacheMonitor()
    assert monitor._address_to_clip_ids_in_use == {}
    assert monitor._max_unique_clip_count_per_address == {}


def test_clip_cache_monitor_update_increment_nominal():
    """Test incrementing usage for all nominal cases: new service/clip,
    existing service/new clip, existing service/existing clip."""
    monitor = ClipCacheMonitor()
    service1 = Mock()
    service1.address = "service1:8000"

    service2 = Mock()
    service2.address = "service2:8000"

    # Case 1: New service and new clip_id
    monitor.increment(service1, "clip1")
    assert monitor._address_to_clip_ids_in_use == {service1.address: {"clip1": 1}}
    assert monitor._max_unique_clip_count_per_address == {service1.address: 1}

    # Case 2: Existing service with a new clip_id
    monitor.increment(service1, "clip2")
    assert monitor._address_to_clip_ids_in_use == {
        service1.address: {"clip1": 1, "clip2": 1}
    }
    assert monitor._max_unique_clip_count_per_address == {service1.address: 2}

    # Case 3: Existing service and existing clip_id (increment count)
    monitor.increment(service1, "clip1")
    assert monitor._address_to_clip_ids_in_use == {
        service1.address: {"clip1": 2, "clip2": 1}
    }
    assert monitor._max_unique_clip_count_per_address == {service1.address: 2}

    # Case 4: New service
    for i in range(3):
        monitor.increment(service2, "clipA")
    assert monitor._address_to_clip_ids_in_use == {
        service1.address: {"clip1": 2, "clip2": 1},
        service2.address: {"clipA": 3},
    }

    # Decrement back to original state, check that max count remains
    monitor.decrement(service1, "clip1")
    monitor.decrement(service1, "clip1")
    monitor.decrement(service1, "clip2")
    assert monitor._max_unique_clip_count_per_address == {
        service1.address: 2,
        service2.address: 1,
    }


def test_clip_cache_monitor_update_decrement_non_existent_service_raises_exception():
    """Test that decrementing for a non-existent service raises and exception."""
    monitor = ClipCacheMonitor()
    service = Mock()
    service.address = "service1:8000"

    with pytest.raises(ValueError):
        monitor.decrement(service, "clip1")
    assert monitor._address_to_clip_ids_in_use == {}


def test_clip_cache_monitor_update_decrement_non_existent_clip_returns_failure():
    """Test that decrementing for a non-existent clip_id returns failure."""
    monitor = ClipCacheMonitor()
    service = Mock()
    service.address = "service1:8000"

    monitor.increment(service, "clip1")
    with pytest.raises(ValueError):
        monitor.decrement(service, "clip2")
    assert monitor._address_to_clip_ids_in_use == {"service1:8000": {"clip1": 1}}
