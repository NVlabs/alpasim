# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

import logging
from collections import defaultdict

from alpasim_runtime.services.service_base import ServiceBase

logger = logging.getLogger(__name__)


class ClipCacheMonitor:
    def __init__(self, service_cache_size: int | None = None) -> None:
        self._address_to_clip_ids_in_use: dict[str, dict[str, int]] = defaultdict(
            lambda: defaultdict(lambda: 0)
        )
        self._max_unique_clip_count_per_address: dict[str, int] = defaultdict(lambda: 0)
        self._service_cache_size = service_cache_size
        self._warned_about_cache_size = False

    def increment(self, service: ServiceBase, clip_id: str) -> None:
        """
        Increment method for clip cache usage tracking.

        Args:
            service: The service instance
            clip_id: The clip ID to track
        """
        self._address_to_clip_ids_in_use[service.address][clip_id] += 1
        self._update_usage()

    def decrement(self, service: ServiceBase, clip_id: str) -> None:
        """
        Decrement method for clip cache usage tracking.

        Args:
            service: The service instance
            clip_id: The clip ID to track
        """
        if service.address not in self._address_to_clip_ids_in_use:
            raise ValueError(
                f"Attempted to decrement clip ID usage for service at {service.address} "
                f"but no clip IDs are currently tracked for this service."
            )
        elif clip_id not in self._address_to_clip_ids_in_use[service.address]:
            raise ValueError(
                f"Attempted to decrement clip ID usage for service at {service.address} "
                f"but clip ID {clip_id} is not currently tracked for this service."
            )
        self._address_to_clip_ids_in_use[service.address][clip_id] -= 1
        if self._address_to_clip_ids_in_use[service.address][clip_id] <= 0:
            del self._address_to_clip_ids_in_use[service.address][clip_id]

    def _update_usage(self) -> None:
        for address, clip_id_usage in self._address_to_clip_ids_in_use.items():
            unique_clip_count = len(clip_id_usage)
            self._max_unique_clip_count_per_address[address] = max(
                self._max_unique_clip_count_per_address[address],
                unique_clip_count,
            )
            if (
                not self._warned_about_cache_size
                and self._service_cache_size is not None
            ):
                if unique_clip_count >= self._service_cache_size:
                    logger.warning(
                        f"Service at {address} is using {unique_clip_count} unique clip IDs, "
                        f"which meets/exceeds the configured cache size of {self._service_cache_size}. "
                        f"Increasing the service cache size to avoid massive performance issues."
                    )
                    self._warned_about_cache_size = True

    def __del__(self):
        for (
            address,
            max_unique_clip_count,
        ) in self._max_unique_clip_count_per_address.items():
            logger.info(
                f"Service at {address} had a maximum of {max_unique_clip_count} unique clip IDs in use concurrently."
            )
