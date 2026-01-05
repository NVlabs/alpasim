# Alpasim data
This document is meant to describe data handling in Alpasim to help test and data engineers build the test cases they want and troubleshoot issues.

## asl files
The output of simulation in alpasim are `asl` files (it stands for AlpaSim Log). These are a
size-delimited protobuf stream with a custom schema defined
[here](/src/grpc/alpasim_grpc/v0/logging.proto). Each rollout will create its own `asl` file with
three types of messages:
* A metadata header (see `RolloutMetadata`) aiming to help with reproducibility and book keeping
* Actor poses (see `ActorPoses`) messages which inform about the location of all actors (including
`'EGO'`) in global coordinate space
* Microservice requests and responses (see `*_request`/`*_return` messages) which enable
reproducing behavior of a given service in replay mode without starting up the entire simulator

> :green_book: `RolloutCameraImage` requests  allow for assembling an `.mp4` video out of an `.asl`
log.

> :warning: The simulation header doesn't specify the `usdz` file uuid.

### Reading asl logs
`alpasim-grpc` provides [async_read_pb_log](//src/utils/alpasim_utils/logs.py) for reading `asl`
logs as a stream of messages.
An example usage to print the first 20 messages in a log (since `async_read_pb_log` is an async
function it needs to be executed from a jupyter notebook or submitted to an async runtime loop):

```python
from alpasim_grpc.utils.logs import async_read_pb_log

i = 0
async for log_entry in async_read_pb_log("<path_to_log>.asl"):
    print(log_entry)
    i += 1
    if i == 20:
        break
```

results in

```
rollout_metadata {
  session_metadata {
    session_uuid: "a5823758-a782-11ef-aa43-0242c0a89003"
    scene_id: "clipgt-3055a5c9-53e8-4e20-b41a-19c0f917b081"
    batch_size: 1
    n_sim_steps: 120
    start_timestamp_us: 1689697803493732
    control_timestep_us: 99000
  }
  actor_definitions {
  }
  force_gt_duration: 1700000
  version_ids {
    runtime_version {
      version_id: "0.3.0"
      git_hash: "83bf78502c43dabac683d68b3712cdca17f6a810+dirty"
      grpc_api_version {
        minor: 24
      }
    }
    egodriver_version {
      version_id: "0.0.0"
      git_hash: "mock"
      grpc_api_version {
        minor: 23
...
    image_bytes: "\377\330\377\340\000\020JFIF\000\001..."
  }
}

Output is truncated.
```
