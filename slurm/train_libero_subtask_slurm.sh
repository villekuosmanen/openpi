#!/bin/bash
#SBATCH --job-name=pi05_subtask_hybrid
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=24
#SBATCH --mem=0G
#SBATCH --exclusive
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --requeue

set -e

module purge
module load brics/apptainer-multi-node

# Paths (edit these for your cluster)
home_dir="/home/u6cr/pravsels.u6cr"
scratch_dir="/scratch/u6cr/pravsels.u6cr"
repo_dir="${home_dir}/openpi"
data_dir="${scratch_dir}/openpi"
container="${data_dir}/container/openpi_arm64.sif"
HF_CACHE="${scratch_dir}/huggingface_cache"
WANDB_DIR="${data_dir}"
WANDB_CACHE_DIR="${scratch_dir}/.cache/wandb"
WANDB_CONFIG_DIR="${scratch_dir}/.config/wandb"
XDG_CACHE_HOME="${scratch_dir}/.cache"
XDG_CONFIG_HOME="${scratch_dir}/.config"

# Training config
CONFIG_NAME="libero_pi05_subtask_hybrid"
EXP_NAME="hybrid_1"

CHECKPOINT_DIR="${data_dir}/checkpoints/${CONFIG_NAME}/${EXP_NAME}"
ASSETS_DIR="${CHECKPOINT_DIR}/assets"
FAST_TOKENIZER_DIR="${data_dir}/weights/fast"

if [ -z "${ASSETS_DIR}" ]; then
    echo "ERROR: ASSETS_DIR is empty; refusing to run."
    exit 1
fi
if [[ "${ASSETS_DIR}" != */assets ]]; then
    echo "ERROR: ASSETS_DIR must end with /assets (got: ${ASSETS_DIR})"
    exit 1
fi
if [ ! -d "${FAST_TOKENIZER_DIR}" ]; then
    echo "ERROR: FAST tokenizer not found at ${FAST_TOKENIZER_DIR}"
    echo "Populate it first, e.g. with uv run python scripts/download_fast_tokenizer.py and sync to scratch weights/."
    exit 1
fi

mkdir -p "${HF_CACHE}" "${WANDB_DIR}" "${WANDB_CACHE_DIR}" "${WANDB_CONFIG_DIR}" "${XDG_CACHE_HOME}" "${XDG_CONFIG_HOME}" "${data_dir}/checkpoints" "${data_dir}/assets" "${data_dir}/weights" "${data_dir}/.venv" "${ASSETS_DIR}"

start_time="$(date -Is --utc)"
echo "===================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Started (UTC): ${start_time}"
echo "===================================="

# Training commands
COMPUTE_NORM_STATS_CMD="uv run scripts/compute_norm_stats_per_timestep.py --config-name=${CONFIG_NAME} --assets-dir=${ASSETS_DIR}"
VALID_INDICES_PATH="${ASSETS_DIR}/valid_indices.txt"
NORM_STATS_PATH="${ASSETS_DIR}/norm_stats.json"
PER_TIMESTEP_STATS_PATH="${ASSETS_DIR}/norm_stats_actions_per_timestep.json"
TRAIN_CMD="uv run scripts/train.py ${CONFIG_NAME} --exp-name=${EXP_NAME} --assets-dir=${ASSETS_DIR} --resume"

EXPORT_VARS="export PYTHONUNBUFFERED=1"
EXPORT_VARS="${EXPORT_VARS} && export WANDB_MODE=offline"
EXPORT_VARS="${EXPORT_VARS} && export WANDB_DIR=${WANDB_DIR}"
EXPORT_VARS="${EXPORT_VARS} && export WANDB_CACHE_DIR=${WANDB_CACHE_DIR}"
EXPORT_VARS="${EXPORT_VARS} && export WANDB_CONFIG_DIR=${WANDB_CONFIG_DIR}"
EXPORT_VARS="${EXPORT_VARS} && export XDG_CACHE_HOME=${XDG_CACHE_HOME}"
EXPORT_VARS="${EXPORT_VARS} && export XDG_CONFIG_HOME=${XDG_CONFIG_HOME}"
EXPORT_VARS="${EXPORT_VARS} && export WANDB_ENTITY=pravsels"
EXPORT_VARS="${EXPORT_VARS} && export OPENPI_DATA_HOME=${data_dir}"
EXPORT_VARS="${EXPORT_VARS} && export UV_PROJECT_ENVIRONMENT=${data_dir}/.venv"

# This dataset currently lacks the metadata needed for filtered valid-index generation,
# so train on the full dataset by ensuring no stale filter file is present.
if [ -f "${VALID_INDICES_PATH}" ]; then
    echo "Removing stale valid-indices file to train on full dataset: ${VALID_INDICES_PATH}"
    rm -f "${VALID_INDICES_PATH}"
else
    echo "Skipping valid-index precompute; training will use the full dataset."
fi

if [ -f "${NORM_STATS_PATH}" ] && [ -f "${PER_TIMESTEP_STATS_PATH}" ]; then
    echo "Skipping normalization precompute (found stats files)."
else
    echo "Running normalization precompute..."
    echo "Command: ${COMPUTE_NORM_STATS_CMD}"
    echo ""
    PRECOMPUTE_CMD="${PRECOMPUTE_CMD}${COMPUTE_NORM_STATS_CMD} && "
fi

echo "Running training command..."
echo "Command: ${TRAIN_CMD}"
echo ""

set +e
srun --ntasks=1 --gpus-per-task=4 --cpu-bind=cores \
apptainer exec --nv \
    --pwd "${repo_dir}" \
    --bind "${scratch_dir}:${scratch_dir}" \
    --bind "${data_dir}/assets:${repo_dir}/assets" \
    --bind "${data_dir}/weights:${repo_dir}/weights" \
    --bind "${data_dir}/checkpoints:${repo_dir}/checkpoints" \
    --bind "${HF_CACHE}:/root/.cache/huggingface" \
    --env "HF_HOME=/root/.cache/huggingface" \
    "${container}" \
    bash -c "${EXPORT_VARS} && ${PRECOMPUTE_CMD}${TRAIN_CMD}"
EXIT_CODE=$?
set -e

end_time="$(date -Is --utc)"

echo ""
echo "===================================="
echo "Started (UTC):  ${start_time}"
echo "Finished (UTC): ${end_time}"
echo "Exit Code: ${EXIT_CODE}"
echo "===================================="

if [ ${EXIT_CODE} -ne 0 ]; then
    echo ""
    echo "ERROR: Training failed with exit code ${EXIT_CODE}"
    echo "Check slurm-${SLURM_JOB_ID}.err for detailed error messages"
    echo "Checkpoint location: ${CHECKPOINT_DIR}"
    exit ${EXIT_CODE}
fi

exit 0
