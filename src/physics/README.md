# AlpaSim Physics

This project contains the code for the Physics micro-service of the AlpaSim project, which uses a
mesh of the environment to constrain the motion of simulated agents to the ground surface. It does
not handle collisions or vehicle dynamics.

## Environment Setup

`uv` is used to manage the development environment.

## Running the Sim Service

Run `uv run physics_server <args>` to start a server for the micro-service. Use `--help` to see
available options.
