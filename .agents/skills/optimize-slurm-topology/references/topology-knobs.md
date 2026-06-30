# Topology Knobs

## Stable Window

Evaluate each candidate once it has enough completed full 20s rollouts for a
stable comparison: In the beginning, all rollouts start simultaneoulsy and the
data isn't representative. Monitor
`alpasim:simulation_seconds_per_rollout:rate5m` only for early and recent
behavior. Use `alpasim:simulation_seconds_per_rollout:avg` as the primary
comparison metric once it stops materially drifting across multiple
observations, typically after 30-45 minutes of rollout production. Do not infer
stability from elapsed time alone.

## The Topology Configuration

During optimizations, only change the configuration listed here. Do not change
other simulation parameters (such as frequencies, driver configuration, etc..)

### Capacity

AlpaSim simulates multiple rollouts in parallel. The exact number is giving by
its overall capacity, which is the minimum capacity across all services. The
capacity of each service is determined by the number of instances of that
service (which is the number of entries in the `gpus` list field), and the
number of concurrent rollouts each instance can handle. The number of instances
is determined by the number of containers of that service, multiplied by the
number of replicas per container.

In short:

```text
capacity = nr_containers * replicas_per_container * n_concurrent_rollouts
```

See, for example, the following configuration:

```yaml
defines:
  nre_cache_size: 17 # renderer.n_concurrent_rollouts + 1
  physics_cache_size: 6

eval:
  num_processes: 32

services:
  renderer:
    environments:
      - HOME=/tmp
      - XDG_CACHE_HOME=/tmp/.cache
      - OMP_NUM_THREADS=1
      - PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.7
    replicas_per_container: 1
    gpus: [0, 1, 2, 3]

  driver:
    replicas_per_container: 8
    gpus: [4, 5, 6, 7]

  physics:
    replicas_per_container: 4
    gpus: [4, 5, 6, 7]

  controller:
    replicas_per_container: 16
    gpus: null

  trafficsim:
    replicas_per_container: 1
    gpus: [0, 1]

runtime:
  nr_workers: 8
  endpoints:
    # Total capacity per service = nr_gpus x replicas_per_container x n_concurrent_rollouts
    # REND: 4 x 1  x 16 = 64
    # DRIV: 4 x 8  x 2  = 64
    # PHYS: 4 x 4  x 4  = 64
    # CONT: 1 (CPU) x 16 x 4 = 64
    renderer:
      n_concurrent_rollouts: 16
    driver:
      n_concurrent_rollouts: 2
      skip: false
    physics:
      n_concurrent_rollouts: 4
      skip: false
    controller:
      n_concurrent_rollouts: 4
    trafficsim:
      skip: true
```

Here, `renderer` has 4 containers (`gpus: [0, 1, 2, 3]`) with 1 replica each and
can handle 16 concurrent rollouts per container, giving it a total capacity of
64. The `driver` service has 4 containers (gpus: [4, 5, 6, 7]) with 8 replicas
each and can handle 2 concurrent rollouts per container, also giving it a total
capacity of 64. The `physics` service has the same GPU allocation as `driver`,
but with 4 replicas and can handle 4 concurrent rollouts per container, again
giving it a total capacity of 64. The `controller` service is CPU-bound with a
single container and can handle 16 replicas with 4 concurrent rollouts, also
resulting in a total capacity of 64. The `trafficsim` service is skipped in this
configuration.

Generally, it makes sense if the capacity of each service is approximately
equal, as otherwise it can lead to wasted resources. For example, if one service
has a much higher capacity than others, the additional capacity will not be
utilized.

If `skip: true` is set for a service, it will not be used in the rollout and can
be completely ignored. Do not change this value yourself, but use the default
configuration you got in the beginning.

### Runtime Workers

Rollouts are managed by the runtime, which is a CPU intensive process. To avoid
the runtime being the bottleneck, the load is spread across multiple runtime
workers, configured by `runtime.nr_workers`. The total number of rollouts (i.e.
the max capacity) is spread evenly across the runtime workers, so each worker
handles `capacity / nr_workers` rollouts. The runtime workers are also
responsible for managing the service queues and RPCs, so if the runtime is
saturated, it can lead to increased queue times and reduced throughput.
This shows up as a low, i.e. <60%, `alpasim:event_loop_idle_fraction:ratio` metric.)

### Cache sizes

There are two important caches that can be tuned: the NRE cache and the physics cache:
* `defines.nre_cache_size`  for NRE
* `defines.physics_cache_size` for physics

These values are important because:
* Loading a scene into the NRE cache takes about 10s, so if the cache is too
  small, it can dominate rollout time.
* Choosing a cache size that is too large can lead to GPU memory exhaustion,
  which can cause rollouts to fail.

So these values, especially for NRE, have to be precisely tuned to the minimum
required size. A good starting point for both services is
`(n_concurrent_rollouts + 1)`.

## Decision Rules

* IMPORTANT: The capacity of each service should approximately match as otherwise it's
  wasted.
* For NRE-backed renderer deployments, multiple replicas per container are not
  supported. Use repeated GPU entries when more than one renderer container
  should share a GPU, for example `gpus: [0, 0, 1, 1]`.

### Finding the bottleneck service

Use `alpasim:rpc_queue_depth_at_start_latest:max` for bottleneck decisions and
`alpasim:rpc_queue_depth_at_start_latest:min` to detect worker starvation. High
max for one service means this service is likely the bottleneck and the current
capacity should be spread out across more processes. Persistent low min with
high max indicates uneven load across runtime workers or across time; low min and max suggests
insufficient offered load or an upstream bottleneck.

If you find cyclic patterns that are detrimental, this cannot be changed by the config alone - surface this to the user.

These metrics are derived from an exact gauge.

* Adding more containers (to the same or other GPUs) or increasing
  `replicas_per_container` (if the service supports it)
* Decreasing `n_concurrent_rollouts` s.t. the overall capacity stays
  approximately the same.

### Determining the overall capacity

#### When should overall capacity be reduced?

If `alpasim:rpc_queue_depth_at_start_latest:max` is high for multiple services
(typically both renderer and driver), the overall capacity is likely
unnecessarily high. Reducing it can reduce GPU memory pressure by reducing the
number of concurrent rollouts (which allows smaller cache sizes) or the number
of instances of each service.

#### When should the overall capacity be incrased?

We want the minimum overall capacity that allows all services to run at full speed without their latest queue depth being empty.

* If queues of services are empty because the rollouts are piling up and waiting at another service, the solution is not to increase the overall capacity, but to increase the capacity of the bottleneck service.
* If, however, there is no clear bottleneck and
  `alpasim:rpc_queue_depth_at_start_latest:min` shows some workers repeatedly
  near empty (i.e. below 3-5 waiting rollouts), the overall capacity should be
  increased to keep all workers utilized. Check that max is also low first;
  low min with high max indicates imbalance rather than insufficient capacity.

Changing the overall capacity means changing the capacity of all services simultaneously.

### GPU utilization and memory

Broadly, the level of GPU utilization tells us how optimized the topology is.
On the other hand, GPU memory is our main limiting factor.

For every stable run, record utilization min/mean/max, memory used
min/mean/max, total memory, peak memory pressure, and minimum memory headroom
for each GPU. Include the services placed on each GPU so resource pressure can
be attributed to topology changes.

We can improve GPU utilization by adding more instances of bottlenecking services to that GPU.
If we hit the limit of 90% GPU memory, we need to reduce the number of instances of services on that GPU or reduce the cache size of those services (which can be done by reducing overall capacity or by spreading the rollouts across more instances).

Low queue depth with low GPU utilization and ample memory headroom indicates
over-capacity. High queue depth with low GPU utilization and memory headroom
suggests adding or splitting service instances. High utilization with high
queue depth indicates compute pressure; high memory pressure is a hard limit
even when utilization is low.

### CPU utilization

The CPU can also be a limiting factor for many services, especially controller and driver.
If this is the case, we can try to spread the number of rollouts across more instances of that service, but this might cost additional GPU usage.
If CPU is the bottleneck, this should also be surfaced explicitly to the user, as they might want to optimize the code as well.

### Runtime idleness

Runtime idle should usually stay high enough to accept service results. If it is
too low, i.e. lower than 60%, and CPUs have headroom, increase
`runtime.nr_workers`. Do not increase `runtime.nr_workers` without bound. Aim
for roughly 70-80% runtime idle; if idle is already above 80%, look for service
queue/RPC bottlenecks first.
