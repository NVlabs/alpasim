# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Traffic event for the split policy pipeline.

Runs the traffic simulation service. Physics constraints on traffic objects
are handled by the subsequent PhysicsEvent(TRAFFIC).

Traffic events derive their own timing from self.timestamp_us rather than
reading ctx.target_time_us, enabling traffic to run at a different cadence
than the driver pipeline.
"""

from alpasim_runtime.events.base import EventPriority, EventQueue, RecurringEvent
from alpasim_runtime.events.state import RolloutState, ServiceBundle


class TrafficEvent(RecurringEvent):
    """Run traffic simulation, writing response to StepContext."""

    priority: int = EventPriority.TRAFFIC

    def __init__(
        self,
        timestamp_us: int,
        control_timestep_us: int,
        services: ServiceBundle,
    ):
        super().__init__(timestamp_us=timestamp_us)
        self.interval_us = control_timestep_us
        self.services = services

    async def run(self, state: RolloutState, queue: EventQueue) -> None:
        ctx = state.step_context
        assert ctx is not None, "StepContext missing — driver cycle did not start"
        assert ctx.corrected_ego_trajectory is not None, "ego physics did not run"

        target_time_us = self.timestamp_us + self.interval_us

        # Interpolate ego pose at this traffic step's target time
        ego_pose_local_to_rig = ctx.corrected_ego_trajectory.interpolate_pose(
            target_time_us
        )
        ego_aabb_pose = (
            ego_pose_local_to_rig @ state.unbound.transform_ego_coords_ds_to_aabb
        )
        ctx.traffic_response = await self.services.trafficsim.simulate_traffic(
            ego_aabb_pose_future=ego_aabb_pose,
            future_us=target_time_us,
        )
