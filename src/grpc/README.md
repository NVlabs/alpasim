# Alpasim core library
This folder contains gRPC definitions for interoperation between alpasim components and certain common utility functions.

Refer to [CONTRIBUTING.md](../../CONTRIBUTING.md#coordinate-systems) for the coordinate frame conventions shared across the runtime and gRPC APIs.

## gRPC APIs
The APIs defined in this repository are used by the alpasim components,
including in additional repositories:
1. [Neural rendering engine](https://www.nvidia.com/en-us/glossary/3d-reconstruction/)
2. Traffic model (coming soon)
3. [Physics model](/src/physics)
4. [Driver](/src/driver)


### Building and installing

Protobuf definitions compiled into python are not packaged and must be compiled 
first with running
```python
uv run compile-protos
``` 
from this folder. This does not happen automatically when installing the 
`gRPC` package!
This command also re-compiles them after you changed the definitions.

You can also clean them with
```python
uv run clean-protos
```


### Usage
Primarily this repository contains protobufs specifying the microservice APIs. 
After installing usage is as follows:

```python
import grpc

from alpasim_grpc.v0.sensorsim_pb2 import RenderRequest, RenderReturn
from alpasim_grpc.v0.sensorsim_pb2_grpc import SensorsimServiceStub

with grpc.insecure_channel('host:port') as channel:
    service = SensorsimServiceStub(channel)
    render_request = RenderRequest(
        scene_id="scene_id",
        # ...
    )
    response: RenderReturn = service.render(render_request)
```
