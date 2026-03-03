# GRPC tutorial
This is meant as a tutorial for implementing gRPC (google remote procedure call) endpoints for the alpamayo simulator.

## Preliminaries
gRPC combines 
1. A language-agnostic binary serialization/deserialization format (`protobuf`)
2. A language-agnostic way to describe services (without implementation)
3. A bunch of packages for different languages which can generate code to serialize/deserialize the messages and provide implementation stubs (think `C` headers) for serving/consuming the APIs.

The format for describing messages and services is `.proto`. The package for serving/consuming `gRPC` is called `grpcio` in pip and imported as `import grpc`. The package for autogenerating code is called `grpcio-tools` on pip and is imported as `grpc_tools` in Python. You do not need the latter in your code.

## The role of this project
This module contains the `.proto` files defining all network interfaces in alpasim and code
(`compile_protos.py`) which will use `grpcio-tools` to build these into Python "headers" (empty
base classes). All you should need to do is `uv run compile-protos` and include this package as
editable.

## Generated files
For a given `<some_name>.proto` file the compiler will create 3 files: `<some_name>_pb2.py`, `<some_name>_pb2_grpc.py`, and `<some_name>_pb2.pyi`.

The `*_pb2.py` files contain message (struct) definitions for the serialization format - think `dataclass` but completely unreadable. `*_pb2.pyi` provides type hints for those, making your IDE actually helpful. When building your service, you will receive these structs as inputs and return them as outputs.

The `*_pb2_grpc.py` files contain the "headers" for the service itself and unfortunately comes without `.pyi` hints. This file will contain 3 objects of interest. On the example of a `runtime.proto` file defining the following service
```proto
service RuntimeService {
    rpc simulate (SimulationRequest) returns (SimulationReturn);
}
```
the generated `runtime_pb2_grpc.py` will contain
1. `class RuntimeServiceStub`
2. `class RuntimeServiceServicer`
3. `def add_RuntimeServiceServicer_to_server`

Number 1. is for **clients**, 2. and 3. are for the **server**.

### Directory structure
Unfortunately, the bare gRPC codegen doesn't produce valid Python packages - without `__init__.py` and relative imports you can't just output the generated files to an arbitrary location in your codebase and import them from there. This is the reason for this package, which has a well defined root, contains "hand-made" `__init__.py` and after installing allows imports like `from alpasim_grpc.v0.your_service_pb2_grpc import your_thing`.

## Implementing
<<<<<<< HEAD
For more information on implementation, see the [gRPC official docs](https://grpc.io/docs/languages/python/basics/).
=======
For implementing both sides (client and server) we'll be importing `grpc`.

### Server
To implement your endpoint, we take two steps. First, we subclass the `RuntimeServiceServicer` base class (bullet 2. in [generated files](generated-files)) and provide an actual implementation for its `simulate` method. It will have the signature of
`def simulate(self, request: SimulationRequest, context: grpc.ServicerContext) -> SimulationReturn`. The request and return types depend on your `.proto` file, the context is always `grpc.ServicerContext` and allows for setting various metadata (such as error codes). You can put arbitrary logic inside your methods and add new python-only methods (such as `__init__`).

[Here](https://gitlab-master.nvidia.com/Toronto_DL_Lab/nre/-/blob/8a471fa8c4f1f1e7917abb36e2ddeefd711c94a5/nre/grpc/serve.py#L499) is an example how this looks for `NRE`, including some error handling.

The second step is to add your `Servicer` object to a server object. To do this, we'll use the `add_RuntimeServiceServicer_to_server` function (bullet 3. in [generated files](generated-files)). Finally there is some boiler plate to set the address on which our server will work and start the loop. The whole code should look like

```python
import grpc
from alpasim_grpc.v0.runtime_pb2 import SimulationRequest, SimulationReturn, ... # whatever other types you need to assemble your return
from alpasim_grpc.v0.runtime_pb2_grpc import RuntimeServiceServicer, add_RuntimeServiceServicer_to_server

class RuntimeService(RuntimeServiceServicer):
    def __init__(self, ...):
        ...

    def simulate(self, request: SimulationRequest, context: grpc.ServicerContext) -> SimulationReturn:
        ...

if __name__ == '__main__':
    # parse cmdline args. we want to always provide --host and --port to be able to deploy containers easily
    args = parse_cmdline_args()

    # create your service instance
    service = RuntimeService(args)

    # start up the server. The executor doesn't really matter
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    
    # set the server address
    server.add_insecure_port(f"{args.host}:{args.port}")

    # add your service to the server
    add_RuntimeServiceServicer_to_server(service, server)

    # start the server
    server.start()
    server.wait_for_termination()
```

An example on the case of `NRE` is [here](https://gitlab-master.nvidia.com/Toronto_DL_Lab/nre/-/merge_requests/433/diffs#4ee808ddc1ad05b14e8263300149f0dee7353154_0_540).

### Client
For now the runtime will be the only consumer in alpasim (so you don't need to write client code), but I am providing this info for completeness and to help you debug and test your servers.

For creating a client we use the `RuntimeServiceStub` (bullet 1. in [generated files](generated-files)) like so:
```python
import grpc
from alpasim_grpc.v0.runtime_pb2_grpc import RuntimeServiceStub
from alpasim_grpc.v0.runtime_pb2 import SimulationRequest, SimulationReturn

with grpc.insecure_channel(f"{args.host}:{args.port}") as channel:
    # create the conterpart to the server
    service = SensorsimServiceStub(channel)

    # create a request
    request = SimulationRequest(...)

    # call the function
    response: SimulationReturn = service.simulate(request)
```

An example client for the NRE endpoint can be found [here](https://gitlab-master.nvidia.com/mtyszkiewicz/nre-gradio-client/-/blob/main/client.py?ref_type=heads) (note the `gradio` stuff is there to expose a nice GUI and irrelevant to actually consuming the endpoint).

### Testing
There is a package called `pytest-grpc` which provides a `pytest` plugin to test your code. It defines a couple of fixtures and leaves the user to implement `grpc_add_to_server`, `grpc_servicer` and `grpc_stub_cls` after which they can use `grpc_stub` to test their code.

See [their page](https://pypi.org/project/pytest-grpc/) for a generic example and my `NRE` test implementation (and an extra example how to use the client) [here](https://gitlab-master.nvidia.com/Toronto_DL_Lab/nre/-/merge_requests/433/diffs#9baf0473c41d0eb32ec53d3bf45b997291ae4987_95_116).

_Unverified_: According to the documentation, by default the package uses an actual server (on `localhost`) to run the tests, so errors on the server side will not give nice stack traces. You can run pytest with `--grpc-fake-server` to get local stack traces instead.

## Known issues
### Missing generated modules
If after installation you see errors like
```
>>> from alpasim_grpc.v0.sensorsim_pb2 import RenderRequest, RenderReturn
ModuleNotFoundError: No module named 'alpasim_grpc.v0.sensorsim_pb2'
```
this means the proto stubs were not compiled. Run `uv run compile-protos` from `src/grpc` to regenerate them.

### Updating the repository
After updating (`git pull`) you may need to recompile protos: `cd src/grpc && uv run compile-protos`.
>>>>>>> 025d93ba (Reimplement perf-critical parts of `alpasim_utils` in Rust)
