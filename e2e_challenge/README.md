# AlpaSim E2E Challenge

## Overview

1. Register a team in the [public Hugging Face Space](https://huggingface.co/spaces/nvidia/AlpasimE2EClosedLoopChallenge2026)
and wait for approval.
1. Build a Docker image that serves the AlpaSim driver gRPC API. Start with the [starter kit](starter_kit/README.md) for a minimal example, then customize and test locally.
1. Push the image to your team's ECR repository and use the challenge CLI to submit the image URI for evaluation.
1. Wait for evaluation and check status or leaderboard results.

## Competition Resources

- [Starter kit](starter_kit/README.md): build and locally test a minimal driver container
- [Challenge CLI](competitor_cli/README.md): authenticate, log in to ECR, submit images, check status, view the leaderboard

## Tracks

The submitted image contract is the same across tracks: contestants submit only
a driver container that serves the AlpaSim driver gRPC API. The evaluator starts
the simulator stack and connects it to the submitted driver image.

### NuPlan / MTGS Track

The NuPlan track uses managed nuPlan scenes and MTGS rendering in the official
evaluation environment. Contestants submit the same driver container used by the
other tracks: it must implement `egodriver.EgodriverService` and respond to
route/drive requests using the inputs provided over gRPC, including rendered
camera images.

Contestant images should not package or depend on direct access to nuPlan data,
trajdata caches, navtest configs, or MTGS assets. Those resources are managed by
the trusted evaluator and are not mounted into contestant containers.

## Submission Image Requirements and Constraints

The image is expected to:

- implement `egodriver.EgodriverService` from `src/grpc/alpasim_grpc/v0/egodriver.proto`
- listen on the configured gRPC host and port
- support multiple concurrent calls on multiple instances (replicas) of the same image
- keep average `Drive` RPC handling time at or below 0.1 seconds

Each replica receives `ALPASIM_DRIVER_HOST`, `ALPASIM_DRIVER_PORT`,
`ALPASIM_CONTESTANT_REPLICA_INDEX`, and `ALPASIM_CONTESTANT_REPLICAS`. GPU
access is provided during official evaluation.

The default public challenge evaluation currently uses the `+e2e_challenge=ec2`
preset, which selects `topology=8gpu_36rollouts`. The backend starts 12 replicas
of the submitted image across GPUs 4-7 with 3 concurrent rollouts per replica.
Local smoke tests use `+e2e_challenge=dev` and a 1-GPU topology.

Some additional constraints of the environment:

- image size limit: 40 GiB
- outbound network access is blocked
- the root filesystem is read-only
- writable scratch space is limited to `/tmp` (2 GiB) and `/run` (64 MiB)
- no host volumes, Docker socket, scene data, or cloud credentials are exposed

## Submission Instructions

See the [Challenge CLI README](competitor_cli/README.md) for authentication, ECR upload, submission,
status, and leaderboard commands.
