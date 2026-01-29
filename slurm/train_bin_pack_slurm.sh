#!/bin/bash
#SBATCH --job-name=pi05_bin_pack_delta
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --requeue

set -e

module purge
module load brics/apptainer-multi-node

# Paths (edit these for your cluster)
home_dir="/home/u5dm/pravsels.u5dm"
scratch_dir="/scratch/u5dm/pravsels.u5dm"
repo_dir="${home_dir}/openpi"
data_dir="${scratch_dir}/openpi"
container="${data_dir}/container/openpi_arm64.sif"
HF_CACHE="${scratch_dir}/huggingface_cache"

# Training config
CONFIG_NAME="pi05_bin_pack_coffee_capsules_delta"
EXP_NAME="${1:-${CONFIG_NAME}}"

CHECKPOINT_DIR="${data_dir}/checkpoints/${CONFIG_NAME}/${EXP_NAME}"

mkdir -p "${HF_CACHE}" "${data_dir}/checkpoints" "${data_dir}/assets" "${data_dir}/weights" "${data_dir}/.venv"

start_time="$(date -Is --utc)"
echo "===================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Started (UTC): ${start_time}"
echo "===================================="

# Training command
TRAIN_CMD="uv run scripts/train.py ${CONFIG_NAME} --exp-name=${EXP_NAME} --resume"

EXPORT_VARS="export PYTHONUNBUFFERED=1"
EXPORT_VARS="${EXPORT_VARS} && export WANDB_MODE=offline"
EXPORT_VARS="${EXPORT_VARS} && export WANDB_ENTITY=pravsels"
EXPORT_VARS="${EXPORT_VARS} && export OPENPI_DATA_HOME=${data_dir}"
EXPORT_VARS="${EXPORT_VARS} && export UV_PROJECT_ENVIRONMENT=${data_dir}/.venv"
EXPORT_VARS="${EXPORT_VARS} && export CUDA_VISIBLE_DEVICES=0,1,2"

echo "Running training command..."
echo "Command: ${TRAIN_CMD}"
echo ""

set +e
srun --ntasks=1 --gpus-per-task=3 --cpu-bind=cores \
apptainer exec --nv \
    --pwd "${repo_dir}" \
    --bind "${scratch_dir}:${scratch_dir}" \
    --bind "${data_dir}/assets:${repo_dir}/assets" \
    --bind "${data_dir}/weights:${repo_dir}/weights" \
    --bind "${data_dir}/checkpoints:${repo_dir}/checkpoints" \
    --bind "${HF_CACHE}:/root/.cache/huggingface" \
    --env "HF_HOME=/root/.cache/huggingface" \
    "${container}" \
    bash -c "${EXPORT_VARS} && ${TRAIN_CMD}"
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

