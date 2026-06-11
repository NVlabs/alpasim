# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

from unittest.mock import Mock

import pytest
from alpasim_runtime.scene_cache_monitor import SceneCacheMonitor


def test_scene_cache_monitor_init():
    """Test that SceneCacheMonitor initializes with empty dictionaries."""
    monitor = SceneCacheMonitor()
    assert monitor._address_to_scene_ids_in_use == {}
    assert monitor._max_unique_scene_count_per_address == {}


def test_scene_cache_monitor_update_increment_nominal():
    """Test incrementing usage for all nominal cases: new service/scene,
    existing service/new scene, existing service/existing scene."""
    monitor = SceneCacheMonitor()
    service1 = Mock()
    service1.address = "service1:8000"

    service2 = Mock()
    service2.address = "service2:8000"

    # Case 1: New service and new scene_id
    monitor.increment(service1, "scene1")
    assert monitor._address_to_scene_ids_in_use == {service1.address: {"scene1": 1}}
    assert monitor._max_unique_scene_count_per_address == {service1.address: 1}

    # Case 2: Existing service with a new scene_id
    monitor.increment(service1, "scene2")
    assert monitor._address_to_scene_ids_in_use == {
        service1.address: {"scene1": 1, "scene2": 1}
    }
    assert monitor._max_unique_scene_count_per_address == {service1.address: 2}

    # Case 3: Existing service and existing scene_id (increment count)
    monitor.increment(service1, "scene1")
    assert monitor._address_to_scene_ids_in_use == {
        service1.address: {"scene1": 2, "scene2": 1}
    }
    assert monitor._max_unique_scene_count_per_address == {service1.address: 2}

    # Case 4: New service
    for i in range(3):
        monitor.increment(service2, "sceneA")
    assert monitor._address_to_scene_ids_in_use == {
        service1.address: {"scene1": 2, "scene2": 1},
        service2.address: {"sceneA": 3},
    }

    # Decrement back to original state, check that max count remains
    monitor.decrement(service1, "scene1")
    monitor.decrement(service1, "scene1")
    monitor.decrement(service1, "scene2")
    assert monitor._max_unique_scene_count_per_address == {
        service1.address: 2,
        service2.address: 1,
    }


def test_scene_cache_monitor_update_decrement_non_existent_service_raises_exception():
    """Test that decrementing for a non-existent service raises and exception."""
    monitor = SceneCacheMonitor()
    service = Mock()
    service.address = "service1:8000"

    with pytest.raises(ValueError):
        monitor.decrement(service, "scene1")
    assert monitor._address_to_scene_ids_in_use == {}


def test_scene_cache_monitor_update_decrement_non_existent_scene_returns_failure():
    """Test that decrementing for a non-existent scene_id returns failure."""
    monitor = SceneCacheMonitor()
    service = Mock()
    service.address = "service1:8000"

    monitor.increment(service, "scene1")
    with pytest.raises(ValueError):
        monitor.decrement(service, "scene2")
    assert monitor._address_to_scene_ids_in_use == {"service1:8000": {"scene1": 1}}
