#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 NVIDIA Corporation

# Override account and partition on the command line:
#   sbatch --account=<account> --partition=<partition> resume_slurm_job.sh <JOB_DIR>
#SBATCH --time 03:59:00
#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --exclusive
#SBATCH --job-name resume-alpasim
#SBATCH --output=./runs/slurm_output/%j.log

# USAGE:
# For single jobs:
#   sbatch -A <allocation> --job-name=<job-name> resume_slurm_job.sh <ORIGINAL_JOB_DIR> [<hydra_overrides_for_resume>]
#
# For array jobs:
#   sbatch -A <allocation> --array=0-<n> --job-name=<job-name> resume_slurm_job.sh <ORIGINAL_ARRAY_JOB_PARENT_DIR> [<hydra_overrides_for_resume>]

# Where <allocation> is your project allocation and `<job-name>` should probably match the original job-name, but doesn't have to, if you want to e.g. upload metrics under a new name.
# For array jobs, <n> is the number of jobs that was used to submit the original run.
# <ORIGINAL_JOB_DIR> or <ORIGINAL_ARRAY_JOB_PARENT_DIR> is the directory of the original run,
# relative to the directory where this script is called.
# hydra overrides are only required if you desire to change any config from the original run.

# Uncomment to log every line
# set -x

# --- Input Validation ---
if [ -z "$1" ]; then
    echo "Usage for single job: $0 <ORIGINAL_JOB_DIR> [<hydra_overrides>]"
    echo "Usage for array job: $0 <ORIGINAL_ARRAY_JOB_PARENT_DIR> [<hydra_overrides>]"
    echo "Note: For array jobs, must be submitted with --array option"
    exit 1
fi

ORIGINAL_DIR="$1"
shift 1 # Remove the first arg, leaving only optional Hydra overrides in "$@"

# --- Configuration ---
# Find parent directory, no matter where the script is called from
SCRIPT_PATH=$(scontrol show job "${SLURM_JOB_ID}${SLURM_ARRAY_TASK_ID:+_$SLURM_ARRAY_TASK_ID}" | awk -F= '/Command=/{print $2}')
SCRIPT_DIR=$(readlink -f "$(dirname "$SCRIPT_PATH")")
REPO_ROOT_DIR=$(readlink -f "${SCRIPT_DIR}/../../..")

# Handle relative paths
if [[ "$ORIGINAL_DIR" != /* ]]; then
    ORIGINAL_DIR="${SCRIPT_DIR}/${ORIGINAL_DIR}"
fi
# Add closing slash if not present
if [[ "$ORIGINAL_DIR" != */ ]]; then
    ORIGINAL_DIR="${ORIGINAL_DIR}/"
fi

# Determine if this is an array job
if [ -n "$SLURM_ARRAY_TASK_ID" ]; then
    # --- ARRAY JOB MODE ---
    echo "Running in array job mode with task ID: ${SLURM_ARRAY_TASK_ID}"

    if [ ! -d "$ORIGINAL_DIR" ]; then
        echo "ERROR: Original parent log directory not found: ${ORIGINAL_DIR}"
        exit 1
    fi

    # Find the specific log directory for *this* task ID within the original parent directory.
    # It looks for a directory whose name matches the pattern <JOB_ID>_<TASK_ID>_<TIMESTAMP>.
    TARGET_LOG_DIR_BASENAME=$(ls -1 "$ORIGINAL_DIR" | grep -E "^[0-9]+_${SLURM_ARRAY_TASK_ID}_.*$")

    # Check if exactly one directory was found
    MATCH_COUNT=$(echo "$TARGET_LOG_DIR_BASENAME" | wc -l)

    if [ "$MATCH_COUNT" -ne 1 ]; then
        echo "ERROR: Expected exactly one log directory for task ${SLURM_ARRAY_TASK_ID} in ${ORIGINAL_DIR}, but found ${MATCH_COUNT}."
        ls -1 "$ORIGINAL_DIR" | grep -E "^[0-9]+_${SLURM_ARRAY_TASK_ID}_.*$" # Show matches/non-matches
        exit 1
    fi

    TARGET_LOG_DIR="${ORIGINAL_DIR}${TARGET_LOG_DIR_BASENAME}"

    LOG_ID="${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

else
    # --- SINGLE JOB MODE ---
    echo "Running in single job mode"

    # Construct the job directory path
    TARGET_LOG_DIR="${ORIGINAL_DIR}"

    LOG_ID="${SLURM_JOB_ID}"
fi

if [ ! -d "$TARGET_LOG_DIR" ]; then
    echo "ERROR: Target log directory not found: ${TARGET_LOG_DIR}"
    exit 1
fi

echo "Resuming job: Target log directory is ${TARGET_LOG_DIR}"

ORIGINAL_WIZARD_CONFIG="${TARGET_LOG_DIR}/wizard-config-loadable.yaml"
if [ ! -f "$ORIGINAL_WIZARD_CONFIG" ]; then
    echo "ERROR: Original wizard config not found: ${ORIGINAL_WIZARD_CONFIG}"
    exit 1
fi

# --- Execution ---
# Create a dedicated log directory for resume logs within the target directory
RESUME_LOG_SUBDIR="${TARGET_LOG_DIR}/resume-logs"
mkdir -p "${RESUME_LOG_SUBDIR}"

RESUME_LOG_FILE="${RESUME_LOG_SUBDIR}/slurm-${LOG_ID}.log"
# Copy any early output to our new log file.
cat "${SCRIPT_DIR}/runs/slurm_output/${SLURM_JOB_ID}.log" > "${RESUME_LOG_FILE}" 2>/dev/null
# Redirect all output for this resume task to a log file inside the original job's directory
exec > >(tee -a "${RESUME_LOG_FILE}") 2>&1

echo "--- Starting Resume Job ---"
echo "Timestamp: $(date)"
echo "Current Job ID: ${SLURM_JOB_ID}"
if [ -n "$SLURM_ARRAY_TASK_ID" ]; then
    echo "Current Task ID: ${SLURM_ARRAY_TASK_ID}"
    echo "Original Array Job Parent Directory: ${ORIGINAL_DIR}"
else
    echo "Original Job Directory: ${ORIGINAL_DIR}"
fi
echo "Target Log Directory: ${TARGET_LOG_DIR}"
echo "Original Wizard Config: ${ORIGINAL_WIZARD_CONFIG}"
echo "Resume Log File: ${RESUME_LOG_FILE}"
echo "Optional Overrides: $@"
echo "---------------------------"

# Install new dependencies if required.
uv tool upgrade alpasim_wizard

# Call the wizard using the original config
alpasim_wizard \
    --config-path "$TARGET_LOG_DIR" \
    --config-name wizard-config-loadable \
    --config-dir "$REPO_ROOT_DIR/src/wizard/configs" \
    runtime.enable_autoresume=true \
    wizard.slurm_job_id="$SLURM_JOB_ID" \
    wizard.log_dir="$TARGET_LOG_DIR" \
    "$@" # Pass any additional overrides specific to the resume
