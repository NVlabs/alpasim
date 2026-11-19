# Alpamayo 2 Edge in AlpaSim

These notes cover how [Alpamayo 2 Edge](https://huggingface.co/nvidia/Alpamayo2-Edge)
behaves in AlpaSim as integrated in this repository.

## Reproducing the model-card run

`+exp/presets=alpamayo2_edge_public_2601` runs the model card's closed-loop protocol:
875 `public_2601` scenes, three rollouts each, and six trajectory candidates per call.

Its navigation is the model card's recorded turn-by-turn text, anonymized: generic road
types instead of street names, and no GPS. The model card's per-scene ego hood textures
are not public yet, so the preset uses the public carline hood textures, re-rendered
into each scene's camera.

## Results

All runs use the full protocol (875 scenes × 3 rollouts).

| Setup | Ego hoods | Navigation | km per at-fault incident | Distance travelled (m) |
| --- | --- | --- | ---: | ---: |
| Model card | per-scene | recorded | 15.15 ± 0.04 | 190.4 |
| AlpaSim | per-scene | recorded | 14.85 ± 1.63 | 190.9 |
| AlpaSim | calibrated carline | recorded | 13.00 ± 0.61 | 187.9 |
| AlpaSim (this preset) | calibrated carline | anonymized | 12.99 ± 0.63 | 187.8 |

- With the model card's inputs, AlpaSim reproduces the model card.
- The carline hood textures account for the remaining gap. They lower the at-fault
  distance by about 2 km and the distance travelled by about 3 m. Collision,
  wrong-lane and progress rates match across all setups.
- Anonymizing the navigation does not change the results.
- At-fault incidents are rare: about 33–38 per run. Run-to-run noise is therefore
  around ±6 incidents, or about ±2 km. Distance travelled is the steadier signal.
