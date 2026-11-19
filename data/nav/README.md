# Navigation route indices

Turn-by-turn navigation for public scenes, as GPS-free route indices per clip, read by
the runtime's recorded-navigation lookup. Select them with `nav_instruction=v5_anonymized`.

The route index files are stored in Git LFS. Fetch them before a run:

```bash
git lfs pull --include="data/nav/**"
```

## Contents

| Directory | Navigation | Scenes covered |
| --- | --- | --- |
| `v5_anonymized/` | V5, anonymized | the 875 scenes of `public_2601_eval_filtered` |

Scenes without a route index fail as invalid scenes, so a run with this navigation is
limited to the scenes listed above.

## Format

Each directory holds one or more `route_indices*.json` files mapping bare clip IDs
(without `clipgt-`) to route indices; `v5_anonymized/route_indices.json` holds all of its clips. A clip
listed in more than one file must have the same route index in each. One clip's entry:

```json
{
  "timestamps_us": [18242904000, 18243904000],
  "cumulative_distances_m": [0.0, 12.4],
  "instructions": [
    {"cumulative_distance_m": 182.4, "maneuver": "TURN_LEFT", "instruction_text": "Turn left onto Street"}
  ],
  "active_maneuver_lookback_m": 10.0,
  "max_instruction_distance_m": 250.0,
  "roundabout_phase_lookback_m": 20.0
}
```

- **`timestamps_us`:** the route's 1 Hz samples inside the clip, on the recording clock
  of the scene's USDZ (`rig_trajectories.json`), not time since the clip started.
- **`cumulative_distances_m`:** distance along the route at those samples, starting at
  0 at the clip's first sample.
- **`instructions`:** the maneuvers the clip can report, in route order.
  - The window runs from 110 m before the clip to 350 m past its end: the reporting
    horizon plus 100 m on each side.
  - The next maneuver after the window is also included, so that "Continue straight" is
    reported as beyond the horizon.
  - Text is the maneuver's base text, without the distance.
- **Lookup settings:** how long a maneuver stays active, how far ahead it is announced,
  and how far back the roundabout check looks.

The runtime anchors these distances to the scene's recorded trajectory. Each step, it
sends the next maneuver and the distance to it from the simulated car's position,
including past the end of the recording.

## Content and privacy

- **No GPS:** no latitude, longitude, speed, session ID, or time outside the clip.
- **Times and distances:** the scene's USDZ already contains the recording times, and
  the distances follow its recorded trajectory.
- **Instruction text:** fixed wording, e.g. `Turn left` or `At roundabout, take exit 2`,
  optionally followed by `onto` and a generic road type, e.g. Street, Avenue or Road.
  - Street names and road references are not included.
  - Every text comes from this closed vocabulary.
- **Size:** 1,438 maneuvers over the 875 clips, using 85 distinct texts.
