# Adapting NAVSIM Models for AlpaSim Inference

This guide covers only the NuPlan/MTGS E2E track. The model receives
`1920x1080` RGB images from `CAM_L0`, `CAM_F0`, and `CAM_R0` and returns a
trajectory in the vehicle rig frame. It does not cover training, PAI camera
adaptation, lidar, maps, or extra history that is unavailable at inference.
Run the commands below from the repository root.

## Adaptation Boundary

| Source contract | AlpaSim inference contract |
| --- | --- |
| NAVSIM/nuPlan/Hydra/Lightning/training | Keep only forward-required local network modules and use local imports. |
| Scene feature builder | Build timestamped per-session gRPC tensors from cached observations. |
| Training/loss/metrics agent | Expose a narrow `Policy.predict_batch` boundary for preprocessing, inference, and output validation. |
| Permissive/external weights | Validate local hashes, set `pretrained=False`, and load with `strict=True`. |
| Rig-relative output | Anchor the rig-relative trajectory in the AlpaSim local frame, attach absolute timestamps, and cache the plan. |
| Development/download environment | Package the checkpoint, vocabulary, code, and runtime in an offline container whose root filesystem is read-only. |

## Adaptation Workflow

### 1. Choose the Closest Complete Sample

Start from the complete sample nearest to the model contract. Use
[TransFuser](sample_submission_simscale_navsim_transfuser/) for image
regression, [DiffusionDrive](sample_submission_simscale_navsim_diffusiondrive/)
for diffusion, and [GTRS ResNet](sample_submission_simscale_navsim_gtrs_dense/)
or [GTRS VoV](sample_submission_simscale_navsim_gtrs_dense_vov_reward/) for
vocabulary-based trajectories.

### 2. Record the Training Contract

Record the original sensor configuration, feature builder, model configuration,
and checkpoint. Capture camera order, crop, normalization, status fields,
coordinate frame, number of points (`N`), point interval, and every external
asset. Do not substitute preprocessing that only appears equivalent.

### 3. Extract the Inference-Only Network

Copy only configuration, backbone, model, and modules reached by `forward`.
Remove datasets, targets, losses, optimizers, and callbacks. Set every encoder
to `pretrained=False`, preserve checkpoint module and parameter names, and
retain the upstream `LICENSE`; record copied files and modifications in
`NOTICE.md`.

### 4. Keep a Narrow Policy Boundary

The public policy contract is exactly:

```text
predict_batch(list[InferenceInput]) -> list[Prediction]
InferenceInput = three cameras + command one-hot(4) + velocity_xy(2) + acceleration_xy(2)
Prediction.trajectory = finite float array, shape (N, 3), rig-frame (x, y, yaw)
```

Normalize checkpoint keys and prefixes before calling
`load_state_dict(..., strict=True)`. Report ready only after warmup. TransFuser
and DiffusionDrive return `(8, 3)` at `0.5 s`; GTRS returns `(40, 3)` at
`0.1 s`. Keep `N` and the interval consistent across the policy, trajectory,
and tests.

### 5. Reuse the AlpaSim Adapter

When the input contract is unchanged, retain the existing `driver.py`,
`batch_worker.py`, `preprocessing.py`, `navigation.py`, and `trajectory.py`
responsibilities: RPC, timestamp synchronization, the 8-value status vector,
microbatching, rig-to-local conversion, 10 Hz output, cached plans, and a
straight-line fallback. Other input or time-axis contracts require explicit
adapter and test changes; a policy-only change is insufficient.

Synchronize each request against the latest pose at or before the Drive timestamp.
For model dynamics, use only its same-timestamp RPC `DynamicState`, which provides
rig-frame velocity and acceleration. Use pose history only when that state is missing,
and increment `dynamic_state_fallback` for each such fallback. This contract requires
official runtime `4d352ba` or newer.

### 6. Build an Offline Submission Image

Update the package, image and version names, environment variables, pinned
requirements, checkpoint and vocabulary hashes, `prepare_assets.sh`, Docker
allowlist, and probe. Runtime must not download assets or depend on host
mounts, the full NAVSIM tree, or `reference/SimScale`.

## Model-Specific Requirements

| Model | Required handling |
| --- | --- |
| TransFuser | Use FP16 autocast and preserve the checkpoint's three-camera preprocessing and 8-point, 0.5 s output contract. |
| DiffusionDrive | Vendor a minimal DDIM implementation. Derive reproducible noise from the session seed and inference index. |
| GTRS | Use the independent `navsim_16384.npy` asset by default (16,384 candidates). Keep scoring and inference in FP32. The release profile is `nc_dac_ep`, `EP=3`, `top K=64`, and `speed weight=3`. Retain the NAVHARD 8,192 vocabulary only as a rollback option. |

## Minimum Validation

Use the copied sample's README to prepare assets and dependencies, then run
the following from the repository root (replace `SAMPLE` with the new sample
directory):

```bash
SAMPLE=e2e_challenge/sample_submission_simscale_navsim_transfuser
uv run --extra grpc --extra transfuser pytest "$SAMPLE/tests" -q
bash "$SAMPLE/scripts/build_image.sh"
ALPASIM_DRIVER_DETACH=1 "$SAMPLE/run_local_container.sh"
PYTHONPATH="$SAMPLE" uv run --extra grpc python \
  "$SAMPLE/scripts/probe_container.py" --address 127.0.0.1:6789 --timeout 180
```

Finish a NuPlan/MTGS closed-loop run. A completed probe or rollout alone is
not sufficient: logs must contain `<model>_inference > 0` and
`inference_error = 0`. Inspect `cached_plan` and `straight_fallback`, and
confirm that the entire run did not use fallback trajectories.
