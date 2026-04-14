```
cd /media/kemove/BigSSD/gjc/alpasim/src/runtime

uv run python -m alpasim_runtime.simulate \
  --serve \
  --listen-address 127.0.0.1:50051 \
  --usdz-glob=/media/kemove/BigSSD/gjc/alpasim/data/nre-artifacts/all-usdzs/**/*.usdz \
  --user-config=/media/kemove/BigSSD/gjc/alpasim/run_interactive_gpu1/generated-user-config-0.yaml \
  --network-config=/media/kemove/BigSSD/gjc/alpasim/run_interactive_gpu1/generated-network-config.yaml \
  --log-dir=/media/kemove/BigSSD/gjc/alpasim/run_interactive_gpu1 \
  --eval-config=/media/kemove/BigSSD/gjc/alpasim/run_interactive_gpu1/eval-config.yaml \
  --log-level=INFO
```

```
cd /media/kemove/BigSSD/gjc/alpasim/src/runtime

uv run python -m alpasim_runtime.web_debugger.server \
  --host 127.0.0.1 \
  --port 8080 \
  --runtime-address 127.0.0.1:50051 \
  --user-config /media/kemove/BigSSD/gjc/alpasim/run_interactive_gpu1/generated-user-config-0.yaml \
  --usdz-glob "/media/kemove/BigSSD/gjc/alpasim/data/nre-artifacts/all-usdzs/**/*.usdz"

```

cd /media/kemove/BigSSD/gjc/alpasim/run_interactive_gpu1
docker compose -f docker-compose.yaml --profile sim up driver-0 controller-0 physics-0 sensorsim-0
