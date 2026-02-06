# ASL to RoadCast Converter

A standalone tool for converting Alpasim Simulation Log (ASL) files to the RoadCast log format
(RCLog) used by internal AV tooling.

## Prerequisites

This package requires the `maglev.av` package, which is hosted on an internal PyPI server. You must
authenticate before installing dependencies.

## Setup

1. **Authenticate with internal PyPI** (required for `maglev.av`):

   ```bash
   cd src/tools/asl_to_roadcast
   ./buildauth login
   ```

1. **Install dependencies**:

   ```bash
   uv sync
   ```

## Usage

### Command Line

```bash
uv run asl-to-roadcast -i <input.asl> -o <output_dir>
```

### Options

| Flag                | Description                                                        |
| ------------------- | ------------------------------------------------------------------ |
| `-i, --input_file`  | Path to input ASL file (required)                                  |
| `-o, --output_path` | Path to output folder for RCLog files (required)                   |
| `-u, --usdz_glob`   | Path to search for USDZ files (optional, only needed for map data) |
| `-s, --scene_id`    | Scene to load from USDZ if multiple are found (optional)           |
| `-v, --verbose`     | Enable verbose logging                                             |

### Example

```bash
# Basic conversion (vehicle config and trajectory extracted from ASL file)
uv run asl-to-roadcast -i /path/to/simulation.asl -o /path/to/output

# With USDZ artifact for map/lane graph data
uv run asl-to-roadcast -i /path/to/simulation.asl -o /path/to/output -u "/path/to/artifacts/**/*.usdz"
```

### Output

The tool creates a timestamped subdirectory in the output path containing:

- `roadcast_debug.log` - The converted RCLog file

A `latest` symlink is also created pointing to the most recent conversion output.

## Running Tests

```bash
uv run pytest -v
```

## Using DDB to View RCLogs

First, one should follow the instructions to
[set up DDB](https://maglev.nvda.ai/docs/components/debugging/drivedebugger/using/workstation/index.md)
on your workstation. Once done, ddb can be launched as:

```bash
# single log view:
ddb replay 0.rclog --open-ui

# multiple log view (note that more than two logs can be specified by separating with colons):
ddb replay . --open-ui --rc-log=0.rclog:1.rclog
```
