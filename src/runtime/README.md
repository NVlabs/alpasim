# Alpasim runtime

Refer to [CONTRIBUTING.md](../../CONTRIBUTING.md#coordinate-systems) for the coordinate frame
conventions used by the runtime services.

## Configuration

### Zero delay mode

The simulator has multiple "clocks" ticking under the hood, triggering events for each camera,
egopose information (GPS) and policy decision-making. In general a policy decision may be requested
"out of sync" with the input information, requiring the policy to extrapolate to the current state
(pose) in order to make decisions for the future. This is challenging and may be desirable to turn
off to simplify debugging. The example config below explains how to do that. The flag
`scenarios[i].assert_zero_decision_delay` enables an assertion to warn the user if they have
misconfigured the remaining parameters.

```yaml
scenarios:
  - # ...
    egopose_interval_us: 99_000 # a multiple of camera's `frame_interval_us`
    control_timestep_us: 99_000 # a multiple of camera's `frame_interval_us`

    time_start_offset_us: 297_000 # a multiple of `control_timestep_us`

    assert_zero_decision_delay: true # adds an assertion to error out if something is misconfigured

    cameras:
      - # ...
        frame_interval_us: 33_000
        shutter_duration_us: 15_000
        first_frame_offset_us: -15_000 # negative `shutter_duration_us`
```
