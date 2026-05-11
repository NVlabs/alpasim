# Run Interactive GPU1 Local

This note records the local interactive simulation commands for the generated
`run_interactive_gpu1_local` run directory.

Run the three command groups below from three separate terminals.

## Terminal 1: Microservices

Starts the driver, controller, physics, and sensorsim services. The generated
compose file pins GPU-backed services to GPU `1`.

```bash
cd /media/a8001/BigSSD/gjc/alpasim/run_interactive_gpu1_local
docker compose -f docker-compose.yaml up driver-0 controller-0 physics-0 sensorsim-0
```

Expected service ports:

- driver: `localhost:6000`
- sensorsim: `localhost:6001`
- physics: `localhost:6002`
- controller: `localhost:6003`

## Terminal 2: Runtime Server

Starts the interactive runtime gRPC server on `127.0.0.1:50051`.

```bash
cd /media/a8001/BigSSD/gjc/alpasim
source setup_local_env.sh
cd src/runtime

env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy \
  NO_PROXY=127.0.0.1,localhost,::1 \
  no_proxy=127.0.0.1,localhost,::1 \
  uv run python -m alpasim_runtime.simulate \
    --serve \
    --listen-address 127.0.0.1:50051 \
    --usdz-glob="/media/a8001/BigSSD/gjc/alpasim/data/nre-artifacts/scenesets/48ef968906b9f603e44089c7d235a66c/**/*.usdz" \
    --user-config=/media/a8001/BigSSD/gjc/alpasim/run_interactive_gpu1_local/generated-user-config-0.yaml \
    --network-config=/media/a8001/BigSSD/gjc/alpasim/run_interactive_gpu1_local/generated-network-config.yaml \
    --log-dir=/media/a8001/BigSSD/gjc/alpasim/run_interactive_gpu1_local \
    --eval-config=/media/a8001/BigSSD/gjc/alpasim/run_interactive_gpu1_local/eval-config.yaml \
    --log-level=INFO
```

## Terminal 3: Web Debugger

Starts the browser UI on `127.0.0.1:8080` and connects it to the runtime server.

```bash
cd /media/a8001/BigSSD/gjc/alpasim
source setup_local_env.sh
cd src/runtime

env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy \
  NO_PROXY=127.0.0.1,localhost,::1 \
  no_proxy=127.0.0.1,localhost,::1 \
  uv run python -m alpasim_runtime.web_debugger.server \
    --host 127.0.0.1 \
    --port 8080 \
    --runtime-address 127.0.0.1:50051 \
    --user-config /media/a8001/BigSSD/gjc/alpasim/run_interactive_gpu1_local/generated-user-config-0.yaml \
    --usdz-glob "/media/a8001/BigSSD/gjc/alpasim/data/nre-artifacts/scenesets/48ef968906b9f603e44089c7d235a66c/**/*.usdz"
```

Open the UI:

```text
http://127.0.0.1:8080
```

## VPN And Proxy Notes

If the UI or runtime reports an error like:

```text
failed to connect to all addresses; last error: UNAVAILABLE: ipv4:127.0.0.1:7890: Socket closed
```

then local gRPC traffic is being routed through a VPN/proxy. The commands above
clear proxy environment variables and set `NO_PROXY` for loopback addresses.

If the browser still fails, add `127.0.0.1` and `localhost` to the VPN/proxy
extension's bypass/direct list.

## Stop

Stop the runtime and web debugger with `Ctrl-C`.

Stop the microservices from the compose terminal with `Ctrl-C`, then clean up
containers if needed:

```bash
cd /media/a8001/BigSSD/gjc/alpasim/run_interactive_gpu1_local
docker compose -f docker-compose.yaml down
```
