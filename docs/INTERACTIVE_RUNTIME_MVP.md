# Interactive Runtime MVP

This document scopes the first interactive runtime iteration for Alpasim.
The goal is not to replace batch rollout execution. The goal is to add a
camera-first debugging control plane that supports stable single-step
simulation and current-state inspection for a frontend.

## Goals

- Support frontend-driven single-step simulation.
- Support viewing current camera outputs from the stepped state.
- Reuse the existing event-based runtime loop and rollout services.
- Keep the existing batch `v0.RuntimeService` unchanged.

## Non-goals

- Replacing the existing batch rollout API.
- Reworking sensorsim or NRE service contracts in the first iteration.
- Adding lidar/radar streaming in the first iteration.
- Full replay productization.
- Full online eval streaming.

## Existing runtime cut points

The current runtime already contains the key semantic boundary needed for
single-step simulation:

- `EventBasedRollout` owns the event queue and main loop.
- `StepEvent` is the commit boundary for simulation state.
- `RolloutState` is the authoritative mutable world state.
- `GroupedRenderEvent` already assembles multi-camera renders within a control
  step.

This means the MVP should not introduce a second scheduler. It should expose
controlled execution over the existing event queue until the next committed
`StepEvent`.

## External API shape

The frontend-facing API lives in
`src/grpc/alpasim_grpc/v1/interactive_runtime.proto`.

First iteration RPCs:

- `CreateSession`
- `StartSession`
- `PauseSession`
- `ResumeSession`
- `StepSession`
- `GetSessionState`
- `ListSensors`
- `GetFrame`
- `StreamSession`

The key IDs are intentionally split:

- `interactive_session_id`: frontend-visible session ID
- `rollout_uuid`: underlying runtime rollout/logging identity

This avoids colliding with existing rollout-scoped service sessions and ASL
logging semantics.

## Runtime-side components

### `InteractiveSessionManager`

Suggested file:
`src/runtime/alpasim_runtime/interactive/session_manager.py`

Responsibilities:

- Own the registry of active interactive sessions.
- Create and destroy sessions.
- Route RPCs by `interactive_session_id`.
- Enforce per-session locking so `StepSession` and `ResumeSession` do not race.
- Hold session metadata needed by the servicer:
  - scene ID
  - status
  - latest snapshot
  - retained frame refs
  - event subscribers

Suggested interface:

```python
class InteractiveSessionManager:
    async def create_session(...) -> InteractiveSessionHandle: ...
    async def get_session(session_id: str) -> InteractiveSessionHandle: ...
    async def close_session(session_id: str) -> None: ...
```

### `InteractiveSessionRunner`

Suggested file:
`src/runtime/alpasim_runtime/interactive/session_runner.py`

Responsibilities:

- Build the same service objects as a normal rollout.
- Build `UnboundRollout`.
- Initialize service rollout sessions.
- Construct `RolloutState` and `EventQueue`.
- Expose controlled execution:
  - `start_background()`
  - `pause()`
  - `resume()`
  - `step(num_steps=1)`
- Stop execution exactly after a committed `StepEvent`.

Important point:

This runner should reuse `EventBasedRollout` setup logic instead of duplicating
 runtime initialization ad hoc. The cleanest refactor is to split
`EventBasedRollout.run()` into:

- `initialize()`
- `run_until_step_commit()`
- `run_until_complete()`
- `shutdown()`

The batch path can continue to call `run_until_complete()`.
The interactive path can repeatedly call `run_until_step_commit()`.

### `InteractiveSnapshotStore`

Suggested file:
`src/runtime/alpasim_runtime/interactive/snapshot.py`

Responsibilities:

- Build immutable post-commit snapshots from `RolloutState`.
- Assign monotonically increasing `tick_id`.
- Convert runtime state into frontend-safe summary structures.

First iteration snapshot contents:

- `interactive_session_id`
- `tick_id`
- `sim_time_us`
- ego pose + dynamics
- actor poses
- frame refs for camera outputs produced for the committed tick

Source of truth remains `RolloutState`.
The snapshot store is only a projection layer.

### `InteractiveFrameStore`

Suggested file:
`src/runtime/alpasim_runtime/interactive/frame_store.py`

Responsibilities:

- Retain a bounded in-memory window of rendered camera frames.
- Index by `(tick_id, sensor_id)`.
- Return raw bytes and metadata for `GetFrame`.
- Evict old frames when `max_retained_ticks` is exceeded.

The snapshot should only carry `FrameRef`.
Raw image bytes should stay out of `SessionState` and `StreamSession`.

### `InteractiveEventBus`

Suggested file:
`src/runtime/alpasim_runtime/interactive/event_bus.py`

Responsibilities:

- Broadcast state/snapshot updates to `StreamSession` subscribers.
- Decouple runtime execution from gRPC streaming backpressure.

The MVP can keep this simple:

- one `asyncio.Queue` per subscriber
- best-effort fanout
- drop/close slow subscribers if necessary

## Minimal refactor inside `EventBasedRollout`

Suggested refactor target:
`src/runtime/alpasim_runtime/event_loop.py`

Current `run()` mixes four phases:

1. setup
2. service session initialization
3. main event loop
4. teardown/eval/finalization

To support stepping cleanly, split it into reusable methods:

```python
async def initialize(self) -> None: ...
async def step_until_commit(self) -> SessionStepResult: ...
async def run_to_completion(self) -> Optional[ScenarioEvalResult]: ...
async def finalize(self) -> Optional[ScenarioEvalResult]: ...
```

Recommended behavior of `step_until_commit()`:

- Pop events sequentially from the existing `EventQueue`.
- Run them normally.
- Stop only after one `StepEvent` has completed and committed state.
- Collect frame metadata generated since the previous commit.
- Build and return a `SessionStepResult`.

This preserves current event ordering and current multi-clock behavior.

## How to detect "one step committed"

Do not infer this indirectly from timestamps.
Make it explicit.

Suggested approach:

- Add a lightweight callback hook or result object around `StepEvent.run()`.
- After `StepEvent` commits state, emit a `step_committed` signal with:
  - committed simulation timestamp
  - committed camera frame refs for this step

This is more reliable than inspecting queue contents after every event.

## Frame capture integration

Suggested integration point:
`src/runtime/alpasim_runtime/events/camera.py`

`GroupedRenderEvent` already receives all camera images for a control window.
The interactive path should additionally forward those images into the
`InteractiveFrameStore`.

Recommended approach:

- keep existing driver submission behavior unchanged
- add an optional frame sink callback
- when images are rendered, send `(camera_logical_id, start_ts, end_ts, bytes)`
  to the sink

This avoids mixing frontend retention concerns into the driver service.

## Session state model

Suggested runtime-side model:

```python
@dataclass
class InteractiveSessionState:
    interactive_session_id: str
    rollout_uuid: str
    scene_id: str
    status: SessionStatus
    current_tick_id: int
    current_sim_time_us: int
    latest_snapshot: SessionSnapshot | None
    error: str | None
```

The gRPC `SessionState` should be derived from this internal state.

## gRPC servicer integration

Suggested new file:
`src/runtime/alpasim_runtime/daemon/interactive_servicer.py`

Responsibilities:

- Map v1 RPCs to `InteractiveSessionManager`.
- Translate domain errors into gRPC status codes.
- Manage `StreamSession` subscription lifecycle.

Suggested app integration:

- keep `RuntimeDaemonServicer` untouched for batch mode
- additionally register `InteractiveRuntimeServiceServicer` in
  `src/runtime/alpasim_runtime/daemon/app.py`

This keeps one daemon process exposing both APIs.

## Why not use the worker pool for the MVP

The existing daemon worker model is optimized for stateless rollout jobs:

- parent expands jobs
- scheduler assigns endpoints
- worker runs one rollout to completion
- worker returns `JobResult`

Interactive sessions do not fit this model well because they require:

- long-lived in-memory state
- per-session control operations
- frame retention
- streaming subscriptions

So the MVP should run interactive sessions in-process in the daemon, not in the
existing worker pool.

This keeps the batch path unchanged and avoids forcing sessionful behavior into
job IPC designed for fire-and-forget rollouts.

## Online eval for the MVP

Do not refactor the evaluation stack in phase 1.

Recommended MVP behavior:

- Keep current `RuntimeEvaluator` behavior for batch rollouts unchanged.
- Expose only simple online state derived from committed snapshots:
  - current ego speed
  - current tick/time
  - actor count

If needed later, add a lightweight online metric publisher that subscribes at
the same commit boundary as snapshots.

## File-level implementation order

1. Refactor `EventBasedRollout` for reusable initialization and step execution.
2. Add runtime-side interactive models:
   - `interactive/session_manager.py`
   - `interactive/session_runner.py`
   - `interactive/snapshot.py`
   - `interactive/frame_store.py`
3. Add gRPC servicer:
   - `daemon/interactive_servicer.py`
4. Register the new servicer in `daemon/app.py`.
5. Compile the new v1 protobufs.
6. Add targeted tests.

## Tests to add

Suggested test files:

- `src/runtime/tests/test_interactive_session_manager.py`
- `src/runtime/tests/test_interactive_step_runner.py`
- `src/runtime/tests/test_interactive_frame_store.py`
- `src/runtime/tests/test_interactive_servicer.py`

Critical test cases:

- `StepSession(num_steps=1)` advances exactly one committed step.
- Two consecutive `StepSession` calls return increasing `tick_id`.
- `GetFrame` returns the frame associated with the requested committed tick.
- `PauseSession` prevents background stepping.
- `ResumeSession` continues stepping until paused/completed.
- Interactive sessions do not interfere with batch `simulate`.

## Immediate next coding task

The best next implementation step is:

1. Refactor `EventBasedRollout` so the event loop can stop after a committed
   `StepEvent`.
2. Add a small `SessionStepResult` model carrying:
   - `tick_id`
   - `sim_time_us`
   - `frame_refs`
3. Keep everything else mocked or in-memory until that seam is working.

Once that seam exists, the gRPC layer becomes straightforward.
