# Changelog
This document lists major updates which change UX and require adaptation.
It should be sorted by date (more recent on top) and link to MRs which introduce the changes.

## AR1 (1.5.25)
AR1 support has been added to Alpasim. Users can now run AR1 as described in the tutorial.

## Update to OSS tutorial definitions / Driver Abstraction (12.17.25)
The tutorials have been updated to reflect the changed scene storage model. Manual downloadds are
no longer required, although users must provide the `HF_TOKEN` env var to support automatic
downloads.

Additionally, and abstraction has been made in the driver code in the hopes that this will make it
easier to add new driver types in the future. As a result, the location of driver code has moved
to `data/drivers`.

## Update to Local USDZ support (12.12.25)
Local directory support was recently dropped in one of the larger refactorings. This has been
restored with a slightly different interface. Now, for users to run Alpasim with local USDZ files,
they can use the `scenes.local_usdz_dir` configuration parameter. For example:
``` bash
# to run all scenes in the local_usdz_dir directory:
alpasim_wizard +deploy=local wizard.log_dir=<output_dir> scenes.local_usdz_dir=<abs or rel path to directory> scenes.test_suite_id=local
# to run a subset  of the scenes:
alpasim_wizard +deploy=local wizard.log_dir=<output_dir> scenes.local_usdz_dir=<abs or rel path to directory> scenes.scene_ids=[<your scene ids>]
```
