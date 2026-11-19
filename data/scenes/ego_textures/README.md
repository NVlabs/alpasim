# Ego textures

This directory maps scene clips to shared carline texture banks in
`data/nre-artifacts/ego-hoods`.

Each carline has one canonical image per camera in that bank. Direct rendering
and camera reprojection use the same front-wide image; the matching source
camera calibration is recorded in `data/ego-hood-transforms/public_carline.yaml`.
Carline bank names are opaque IDs such as `carline_002`. The default renderer mount uses
these bundled banks independently of the deployment's scene cache.

Use the carline texture preset with any public scene suite. It re-renders the
front-wide hood of the carlines listed in `data/ego-hood-transforms/public_carline.yaml`
into each scene's own camera; other carlines use their bank unchanged:

```bash
uv run alpasim_wizard deploy=local topology=1gpu driver=alpamayo1_5 \
  scenes.suite_ids=[public_2604] +ego_textures=public_carline_transformed \
  wizard.log_dir=./run_public_carline
```

The re-rendering uses the camera's rotation and intrinsics; translation is ignored
as it does not improve the fit.

Every ClipGT clip referenced by the public catalogs and suites is covered,
including 25.05, 26.01, 26.04, and the curated splits. Front-wide and
rear-left/right images are bundled in the repository.
