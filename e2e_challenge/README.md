# AlpaSim E2E Challenge

## Overview

1. Register a team in the [public Hugging Face Space](https://huggingface.co/spaces/nvidia/AlpasimE2EClosedLoopChallenge2026)
and wait for approval.
1. Build a Docker image that serves the AlpaSim driver gRPC API. Start with the [starter kit](starter_kit/README.md) for a minimal example or the [VAVAM sample submission](sample_submission_vavam/README.md) for a richer model-backed example, then customize and test locally.
1. Push the image to your team's ECR repository and use the challenge CLI to submit the image URI for evaluation.
1. Wait for evaluation and check status or leaderboard results.

## Competition Resources

- [Starter kit](starter_kit/README.md): build and locally test a minimal driver container
- [VAVAM sample submission](sample_submission_vavam/README.md): build and locally test a VAVAM-backed driver container
- [Challenge CLI](competitor_cli/README.md): authenticate, log in to ECR, submit images, check status, view the leaderboard

## Tracks

The competition has two tracks: the Physical AI AV Track and the nuPlan Track. For both
tracks, the image contract for submissions is the same across tracks: contestants submit only
a driver container that serves the AlpaSim driver gRPC API. Depending upon the requested
submission type, the evaluator starts the appropriate simulator stack and connects to the
submitted driver image. Use `pai` for
the Physical AI AV track and `nuplan` for the nuPlan track. Submission limits
are shared across tracks because both tracks use the same managed evaluator
capacity.

### Physical AI (PAI) AV

The Physical AI AV track uses an internal set of NuRec-compatible scenes similar to those
available in the public NuRec dataset.

### NuPlan / MTGS

The NuPlan track uses managed nuPlan scenes and MTGS rendering in the evaluation environment.


## Submission Image Requirements and Constraints

The image is expected to:

- implement `egodriver.EgodriverService` from `src/grpc/alpasim_grpc/v0/egodriver.proto`
- listen on the configured gRPC host and port
- support multiple concurrent calls on multiple instances (replicas) of the same image
- target 0.1 seconds or less of model work per `Drive` call

Each replica receives `ALPASIM_DRIVER_HOST`, `ALPASIM_DRIVER_PORT`,
`ALPASIM_CONTESTANT_REPLICA_INDEX`, and `ALPASIM_CONTESTANT_REPLICAS`. GPU
access is provided during official evaluation.

The 0.1 second target corresponds to a 10 Hz driving loop: a policy should be
able to produce each driving decision within one control tick. Official
evaluation enforces this with a per-track throughput budget. Submission status
reports the observed throughput wall time and the applicable limit.

Both tracks use the `ec2` preset (see `src/wizard/configs/{e2e_challenge,e2e_challenge_nuplan}`).
The EC2 configs start 16 replicas of the submitted image across GPUs 4-7 with 2 concurrent rollouts per replica.

Local smoke tests use `+e2e_challenge=dev` and a 1-GPU topology.

Some additional constraints of the environment:

- image size limit: 40 GiB
- each driver instance must use no more than 16 GiB of VRAM
- outbound network access is blocked
- the root filesystem is read-only
- writable scratch space is limited to `/tmp` (2 GiB) and `/run` (64 MiB)
- no host volumes, Docker socket, scene data, or cloud credentials are exposed

## Submission Instructions

See the [Challenge CLI README](competitor_cli/README.md) for authentication, ECR upload, submission,
status, and leaderboard commands.
