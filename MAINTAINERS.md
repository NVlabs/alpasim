# Maintainers Guide

This document provides an introduction to project management and an overview of processes for the Alpasim project.

[TOC]

## Overview

### Assumptions

Alpasim is meant to be an open-source, best-effort research project. As such, the processes for
maintaining the project are designed to allow for rapid development/iteration while maintaining
a reasonable level of quality.

### Constraints

The project is maintained by a small team with limited time and resources. Therefore,
the processes are designed to minimize overhead and maximize efficiency.

## Software Development Process

### Branching Strategy

- **`main`** - Primary branch, requires fast-forward merges only
- **Feature branches** - Created from `main` for new work
- **Branch naming** - Use descriptive names (e.g., `dev/<username>/<feature-name>`)


### Rebasing

This repository requires **fast-forward merges only** to maintain a linear commit history. If your branch falls behind `main`, you must rebase:

**Note:** Force pushes are expected and necessary when rebasing due to the linear history requirement.

### Merge Request Process

Merge/Pull requests are required for all changes to the codebase. Templates are provided to ensure
consistency and completeness.

## Release Process

Versioning is automated by the CICD pipeline to ensure that versions remain uniquely identified.
There is no plan to develop process for qualification/certification of releases at this time.
Updates will be rolled out continuously via the `main` branch.

## Roles and Responsibilities

**Program Manager/PIC**: Max Igl

**Primary Researcher**: Max Igl

**Owner**: Max Igl

**PLC Security PIC**: Michael Watson
