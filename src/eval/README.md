## Evaluation

This module is a refactored version of the `KPI` service. It
* Reads in ASL logs
* Computes metrics (see [`src/eval/scorers/__init__.py`](src/eval/scorers/__init__.py) for list of
  implemented "Scorers")
* And generates a video

### Configuration

See [schema.py](src/eval/schema.py).

## Writing your own metric scorer

A key motivation for this module was to make writing new scorers fast and easy.
To do so, we:
* Rely heavily on dataclasses for storing the information parsed from ASL. The
  information is organized hierarchically, with the root being `EvaluationResultContainer` in
  [`data.py`](src/eval/data.py).
* We don't use indexing by index, but always by timestamp_us, to reduce
  off-by-one errors.
* We rely on the `Trajectory` class from AlpaSim, which allows indexing into
  trajectories by timestamp. We expand this class to `RenderableTrajectory` in
  [`data.py`](src/eval/data.py) which also contains the bounding box and knows how
  to render itself onto a video frame.
* Lastly, we also rely heavily on the `shapely` library, to abstract away complex
  geometric computations such as `distance`, `contains`, `project`,
  `intersects`, etc... The `RenderableTrajectory` class has helper methods to
  convert itself to shapely objects.
* We also have a `ShapelyMap` class, which is primarily used for fast video
  rendering of maps. For computing map-based metrics, it's probably easiest to use the
  `trajdata.vec_map` directly, which is also stored in
  `EvaluationResultContainer.sim_result` and allows querying for current lanes, etc..

### Running locally

This part of the codebase is managed by `uv`.

```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Recommended workflow:

1. First run the wizard normally (after installing it with `uv tool install -e
   src/wizard`) and generating ASL files.
```bash
alpasim_wizard wizard.log_dir=<log_dir> +deploy=local
```
2. Execute this from `src/eval`:
```bash
uv run alpasim-eval  \
  --asl_search_glob=<log_dir>/asl/clipgt-d8cbf4ca-b7ff-44bd-a5be-260f736a02fe/15f2c488-10ad-11f0-b123-0242c0a84004/\*\*/\*.asl \
  --config_path=<log_dir>eval-config.yaml \
  --output_dir=<log_dir>/eval \
  --trajdata_cache_dir=<path_to_alpasim_repo>/data/trafficsim/unified_data_cache \
  --usdz_glob="<path_to_alpasim_repo>/data/nre-artifacts/all-usdzs/**/*.usdz"
```

The environment is shared with that of the main project and is automatically managed by `uv`.

### Overview over the codebase, e.g. for writing

Main components of the codebase:
* [`data.py`](src/eval/data.py) contains most datastructures. Start exploring
  from `EvaluationResultContainer`
* Parsing ASL logs is done in `load_simulation_results()` in [`main.py`](src/eval/main.py)
* Scorers are implemented in the folder [`scorers`](src/eval/scorers/). If you
  add a new scorer, don't forget to add it to the list in
  [`scorers.__init__.py`](src/eval/scorers/__init__.py)
* Scorers produce metrics per timestamp per rollout. These results are
  aggregated in
  [`eval_aggregation.py`](src/eval/aggregation/eval_aggregation.py). As long as
  you conform to the existing datastructure, you probably won't need to touch this.
* Lastly, video generation is done in [`video.py`](src/eval/video.py)
