#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

# Unified SLURM submit script. All arguments are forwarded to the wizard as
# Hydra overrides.
#
# Usage:
#   sbatch [sbatch_opts] submit.sh deploy=<target> topology=<layout> driver=<model> [hydra_overrides...]
#
# Examples:
#   sbatch submit.sh deploy=ord topology=8gpu_64rollouts driver=vavam
#   sbatch submit.sh deploy=ord topology=8gpu_12rollouts driver=alpamayo1 controller=ndas trafficsim=internal
#   sbatch --account=wlew --partition=gtc_demo --gpus=4 submit.sh deploy=ipp5 topology=1gpu driver=alpamayo1

#SBATCH --account av_alpamayo_sim
#SBATCH --partition polar,polar3,polar4,grizzly
#SBATCH --time 03:59:00
#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --exclusive
#SBATCH --job-name alpasim
#SBATCH --output=./runs/slurm_output/%j.log

# Detect if running on slurm node
if [ -z "$SLURM_JOB_ID" ]; then
    echo "This script should run on SLURM. Example: sbatch submit.sh deploy=ord topology=8gpu_64rollouts driver=vavam"
    exit 1
fi

if [[ -z "$SUBMITTER" ]]; then
    SUBMITTER="$(whoami)"
fi

if [[ -z "$DESCRIPTION" ]]; then
    DESCRIPTION="unspecified"
fi

# Verify required config groups are present in arguments
ALL_ARGS="$*"
MISSING=()
[[ "$ALL_ARGS" =~ deploy=[^\ ]+ ]]   || MISSING+=("deploy=<target>    (e.g., ord)")
[[ "$ALL_ARGS" =~ topology=[^\ ]+ ]] || MISSING+=("topology=<layout>  (e.g., 8gpu_64rollouts, 8gpu_12rollouts)")
[[ "$ALL_ARGS" =~ driver=[^\ ]+ ]]   || MISSING+=("driver=<model>     (e.g., vavam, alpamayo1, alpamayo1_5)")
if [ ${#MISSING[@]} -gt 0 ]; then
    echo "Error: missing required config groups:"
    for m in "${MISSING[@]}"; do echo "  $m"; done
    echo ""
    echo "Example: sbatch submit.sh deploy=ord topology=8gpu_64rollouts driver=vavam"
    exit 1
fi

# All script arguments are forwarded as Hydra overrides
HYDRA_ARGS=("$@")

# Find parent directory, no matter where the script is called from
#
# Note (Qi): On SLURM, SLURM_JOB_ID is uniquely defined for every job, including jobs in an array.
# However, for array jobs, it is possible for SLURM_JOB_ID to equal SLURM_ARRAY_JOB_ID for one of the
# jobs. This can cause issues with scontrol because scontrol may return multiple job entries corresponding
# to the entire array. To resolve this issue, array launches and regular launches can be distinguished
# by checking SLURM_ARRAY_JOB_ID. For array jobs, use SLURM_ARRAY_JOB_ID_SLURM_ARRAY_TASK_ID instead of
# SLURM_JOB_ID when interacting with scontrol.
if [[ -z $SLURM_ARRAY_JOB_ID ]]; then
    UNIQUE_JOB_ID="${SLURM_JOB_ID}"
else
    UNIQUE_JOB_ID="${SLURM_ARRAY_JOB_ID}${SLURM_ARRAY_TASK_ID:+_$SLURM_ARRAY_TASK_ID}"
fi
SCRIPT_PATH=$(scontrol show job "${UNIQUE_JOB_ID}" | awk -F= '/Command=/{print $2}')

SCRIPT_DIR=$(readlink -f "$(dirname $SCRIPT_PATH)")
REPO_ROOT_DIR=$(readlink -f "${SCRIPT_DIR}/../../..")

# If LOGDIR is not specified, we generate a logdir in the folder where this script lives. If
# a relative LOGDIR is specified, we assume the user wants to set the LOGDIR relative to where
# the script is submitted.
if [[ -z "$LOGDIR" ]]; then
    if [ -z "${SLURM_ARRAY_JOB_ID}" ]; then
        # Non array job
        LOGDIR=$SCRIPT_DIR/runs/${SLURM_JOB_ID}_${SLURM_JOB_NAME}_$(date +%Y_%m_%d__%H_%M_%S)
        ARRAY_JOB_DIR=$LOGDIR
    else
        # Array job, we use a hierarchical logdir to group all array jobs
        ARRAY_JOB_DIR=$SCRIPT_DIR/runs/${SLURM_ARRAY_JOB_ID}_${SLURM_JOB_NAME}
        LOGDIR=$ARRAY_JOB_DIR/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}_$(date +%Y_%m_%d__%H_%M_%S)
    fi
else
    if [[ "$LOGDIR" != /* ]]; then
        LOGDIR=$(readlink -f "$SLURM_SUBMIT_DIR/$LOGDIR")
    fi
fi

# Create txt-logs directory if it doesn't exist
mkdir -p ${LOGDIR}/txt-logs

# Want the Slurm logs in LOGDIR, but don't know LOGDIR until after job started.
# Copy any early output to our new log file.
cat ./runs/slurm_output/${SLURM_JOB_ID}.log > ${LOGDIR}/txt-logs/slurm.log 2>/dev/null

# Redirect all future output to both terminal and the log file
exec > >(tee -a "${LOGDIR}/txt-logs/slurm.log") 2>&1

# Create resume.sh script
ORIG_SUBMIT_CMD=$(sacct -j ${SLURM_JOB_ID} -o submitline -P | head -n 2 | sed '1d')
ORIG_ACCOUNT=${SLURM_JOB_ACCOUNT}

if [[ -z "${ORIG_ACCOUNT}" ]]; then
    ORIG_ACCOUNT=$(scontrol show job "${UNIQUE_JOB_ID}" | awk -F= '/Account=/{print $2}' | awk '{print $1}')
fi

if [ ! -f "${ARRAY_JOB_DIR}/resume.sh" ] && [[ -z "${SLURM_ARRAY_TASK_ID}" || "${SLURM_ARRAY_TASK_ID}" == "${SLURM_ARRAY_TASK_MIN}" ]]; then
    cat > ${ARRAY_JOB_DIR}/resume.sh <<RESUME_EOF
#!/bin/bash
# Resume script — re-submits with the same SLURM options and Hydra overrides
${ORIG_SUBMIT_CMD} "\$@"
RESUME_EOF
    chmod +x ${ARRAY_JOB_DIR}/resume.sh
fi

# Create reeval.sh script
if [ ! -f "${ARRAY_JOB_DIR}/reeval.sh" ] && [[ -z "${SLURM_ARRAY_TASK_ID}" || "${SLURM_ARRAY_TASK_ID}" == "${SLURM_ARRAY_TASK_MIN}" ]]; then
    cat > ${ARRAY_JOB_DIR}/reeval.sh <<REEVAL_EOF
#!/bin/bash
# Re-evaluation script — recomputes eval + aggregation on this job folder
uv run --project ${REPO_ROOT_DIR}/src/eval --python 3.12 alpasim-reeval "${ARRAY_JOB_DIR}" --slurm --account "${ORIG_ACCOUNT}" "\$@"
REEVAL_EOF
    chmod +x ${ARRAY_JOB_DIR}/reeval.sh
fi

uv run --project ${REPO_ROOT_DIR}/src/wizard --python 3.12 \
    alpasim_wizard \
    wizard.log_dir=$LOGDIR \
    wizard.array_job_dir=$ARRAY_JOB_DIR \
    wizard.latest_symlink=true \
    wizard.submitter="$SUBMITTER" \
    wizard.description="$DESCRIPTION" \
    "${HYDRA_ARGS[@]}"
