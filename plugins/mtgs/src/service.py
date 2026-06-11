# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""MTGS renderer gRPC service client.

Inherits from SensorsimService since the MTGS server implements the
standard SensorsimService gRPC interface. Only overrides plugin discovery
hooks (from_config, get_config_schema) and optionally session initialization.
"""

from __future__ import annotations

import logging
from typing import Any

from alpasim_mtgs.config import MTGSConfig
from alpasim_runtime.camera_catalog import CameraCatalog
from alpasim_runtime.services.sensorsim_service import SensorsimService
from alpasim_runtime.services.service_base import SessionInfo
from alpasim_runtime.services.session_configs import RendererSessionConfig

logger = logging.getLogger(__name__)


class MTGSService(SensorsimService):
    """gRPC client for the externally-connected MTGS renderer.

    MTGS implements the SensorsimService protocol, so this subclass
    inherits all render methods (render, aggregated_render,
    get_available_cameras, etc.) from SensorsimService and only
    customizes plugin discovery hooks and session initialization.
    """

    def __init__(
        self,
        address: str,
        skip: bool,
        camera_catalog: CameraCatalog,
        config: MTGSConfig,
    ):
        super().__init__(address=address, skip=skip, camera_catalog=camera_catalog)
        self._mtgs_config = config

    @classmethod
    def from_config(
        cls,
        raw_config: dict[str, Any],
        address: str,
        skip: bool = False,
        *,
        camera_catalog: CameraCatalog | None = None,
    ) -> MTGSService:
        """Factory used by the core worker's plugin discovery."""
        config = MTGSConfig(**raw_config) if raw_config else MTGSConfig()
        return cls(
            address=address,
            skip=skip,
            camera_catalog=camera_catalog,
            config=config,
        )

    @classmethod
    def get_config_schema(cls) -> type:
        """Return the plugin's typed config schema for wizard validation."""
        return MTGSConfig

    async def _initialize_session(self, session_info: SessionInfo) -> None:
        """Initialize MTGS session.

        Performs camera discovery (same as sensorsim) but skips the warmup
        render if configured to do so -- the MTGS server loads scenes lazily
        on the first render call and handles its own warmup internally.
        """
        cfg = session_info.session_config
        if not isinstance(cfg, RendererSessionConfig):
            return

        scene_id = cfg.data_source.scene_id
        sensorsim_cameras = await self.get_available_cameras(scene_id)
        await self._camera_catalog.merge_local_and_sensorsim_cameras(
            scene_id, sensorsim_cameras
        )

        if not self._mtgs_config.skip_warmup and not self.skip:
            logger.info(
                "MTGS client warmup requested, but MTGS loads scenes lazily on "
                "the server; skipping client-side warmup."
            )
