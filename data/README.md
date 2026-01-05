# Alpasim Data

This directory contains subdirectories which we will need to mount in the various docker containers
to run alpasim.

1. `nre-artifacts`
   - Contains the `.usdz` files which are scene reconstructions for neural rendering with NRE
   - Also used by the runtime to obtain ground truth vehicle trajectories and physics for access to
     scene meshes
2. `trafficsim` (To be released at a later date)
   - Contains scene road maps and configuration for the traffic simulator
