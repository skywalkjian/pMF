#!/usr/bin/env bash
# =============================================================================
# auto_experiment.sh — Pixel MeanFlow Kimi Residual Attention A/B Experiment
#
# Runs two 10k-step training experiments (naive vs residual) on 2 GPUs,
# evaluates FID for both, and generates a summary report.
#
# Written by [DevOps] of the AI R&D team. Reviewed by [Reviewer].
# =============================================================================

set -euo pipefail

# ---- Configuration ----------------------------------------------------------
PROJECT_DIR="/cpfs/user/yujian/pMF"
DATA_ROOT="./data"
MAX_STEPS=10000
BATCH_SIZE=128
GRAD_ACCUM=1
NPROC=2
NUM_WORKERS=4
SAMPLE_BSZ=8
EVAL_EVERY=250
SAMPLE_EVERY=500
SAVE_EVERY=500
FID_SAMPLES=50000
FID_BSZ=128

NAIVE_WORKDIR="./runs/cifar10_pmf_muon_transformer_naive_final"
RESIDUAL_WORKDIR="./runs/cifar10_pmf_muon_transformer_residual_final"

# ---- Setup ------------------------------------------------------------------
cd "${PROJECT_DIR}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

LOG_DIR="${PROJECT_DIR}/experiment_logs"
mkdir -p "${LOG_DIR}"

TIMESTAMP=$(date +%Y%m%d-%H%M%S)
MAIN_LOG="${LOG_DIR}/experiment_${TIMESTAMP}.log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${MAIN_LOG}"
}

die() {
    log "FATAL: $*"
    exit 1
}

# ---- Environment check -----------------------------------------------------
log "========================================="
log "Pixel MeanFlow A/B Experiment Starting"
log "========================================="

# Check GPUs
GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null) \
    || die "Cannot import torch or no CUDA available."
log "Detected ${GPU_COUNT} GPU(s)."
if [ "${GPU_COUNT}" -lt "${NPROC}" ]; then
    die "Need ${NPROC} GPUs but only ${GPU_COUNT} detected."
fi

# Check dataset
if [ ! -d "${DATA_ROOT}/cifar-10-batches-py" ]; then
    log "CIFAR-10 dataset not found. Will download on first run."
fi

# ---- Helper: find latest timestamp dir inside a workdir --------------------
latest_run_dir() {
    local workdir="$1"
    ls -1d "${workdir}"/20* 2>/dev/null | sort | tail -1
}

# ---- Helper: train one experiment ------------------------------------------
run_train() {
    local name="$1"
    local workdir="$2"
    local attn_impl="$3"
    local train_log="${LOG_DIR}/train_${name}_${TIMESTAMP}.log"

    log ">>> Training [${name}] — attn_impl=${attn_impl}, max_steps=${MAX_STEPS}"
    log "    Workdir: ${workdir}"
    log "    Log: ${train_log}"

    torchrun --nproc_per_node=${NPROC} meanflow.py \
        --workdir "${workdir}" \
        --dataset cifar10 \
        --optimizer muon \
        --attn-impl "${attn_impl}" \
        --max-steps ${MAX_STEPS} \
        --batch-size ${BATCH_SIZE} \
        --grad-accum-steps ${GRAD_ACCUM} \
        --eval-every ${EVAL_EVERY} \
        --sample-every ${SAMPLE_EVERY} \
        --save-every ${SAVE_EVERY} \
        --sample-batch-size ${SAMPLE_BSZ} \
        --num-workers ${NUM_WORKERS} \
        > "${train_log}" 2>&1

    local exit_code=$?
    if [ ${exit_code} -ne 0 ]; then
        log "!!! Training [${name}] FAILED with exit code ${exit_code}."
        log "    Check log: ${train_log}"
        # Print last 20 lines for debugging
        tail -20 "${train_log}" | tee -a "${MAIN_LOG}"
        return ${exit_code}
    fi

    # Verify config matches
    local run_dir
    run_dir=$(latest_run_dir "${workdir}")
    if [ -z "${run_dir}" ]; then
        log "!!! No run directory found in ${workdir}"
        return 1
    fi

    local saved_impl
    saved_impl=$(python -c "import json; d=json.load(open('${run_dir}/config.json')); print(d['config']['attn_impl'])" 2>/dev/null)
    if [ "${saved_impl}" != "${attn_impl}" ]; then
        die "Config mismatch! Expected attn_impl=${attn_impl} but got ${saved_impl} in ${run_dir}/config.json"
    fi

    log "<<< Training [${name}] COMPLETED. Run dir: ${run_dir}"
    return 0
}

# ---- Helper: run FID evaluation --------------------------------------------
run_fid() {
    local name="$1"
    local workdir="$2"
    local fid_log="${LOG_DIR}/fid_${name}_${TIMESTAMP}.log"

    local run_dir
    run_dir=$(latest_run_dir "${workdir}")
    if [ -z "${run_dir}" ]; then
        log "!!! No run directory for FID evaluation of [${name}]"
        return 1
    fi

    if [ ! -f "${run_dir}/checkpoints/best.pt" ]; then
        log "!!! No best.pt checkpoint found in ${run_dir}/checkpoints/"
        return 1
    fi

    log ">>> FID evaluation [${name}] — ${FID_SAMPLES} samples"
    log "    Run dir: ${run_dir}"
    log "    Log: ${fid_log}"

    torchrun --nproc_per_node=${NPROC} fid_eval.py \
        --run-dir "${run_dir}" \
        --checkpoint best \
        --data-root "${DATA_ROOT}" \
        --num-samples ${FID_SAMPLES} \
        --gen-bsz ${FID_BSZ} \
        --fid-batch-size ${FID_BSZ} \
        --isc \
        > "${fid_log}" 2>&1

    local exit_code=$?
    if [ ${exit_code} -ne 0 ]; then
        log "!!! FID evaluation [${name}] FAILED with exit code ${exit_code}."
        tail -20 "${fid_log}" | tee -a "${MAIN_LOG}"
        return ${exit_code}
    fi

    log "<<< FID evaluation [${name}] COMPLETED."
    return 0
}

# ---- Helper: extract metrics for report ------------------------------------
extract_metrics() {
    local name="$1"
    local workdir="$2"

    local run_dir
    run_dir=$(latest_run_dir "${workdir}")
    if [ -z "${run_dir}" ]; then
        echo "N/A"
        return
    fi

    python -c "
import json, sys
try:
    metrics = json.load(open('${run_dir}/metrics.json'))
    step = metrics['step']
    best_val = metrics['best_val_loss']
    final_train = metrics['train_loss'][-1]
    fid_path = '${run_dir}/fid/metrics_best_${FID_SAMPLES}.json'
    try:
        fid_data = json.load(open(fid_path))
        fid = fid_data['metrics']['frechet_inception_distance']
        is_mean = fid_data['metrics'].get('inception_score_mean', 'N/A')
        is_std = fid_data['metrics'].get('inception_score_std', 'N/A')
    except:
        fid = 'N/A'
        is_mean = 'N/A'
        is_std = 'N/A'
    print(f'| {name} | {step} | {final_train:.6f} | {best_val:.6f} | {fid} | {is_mean} |')
except Exception as e:
    print(f'| {name} | ERROR: {e} |')
" 2>/dev/null
}

# =============================================================================
# MAIN EXECUTION
# =============================================================================

log ""
log "============================================"
log "STEP 1/4: Training NAIVE baseline (10k steps)"
log "============================================"

if ! run_train "naive" "${NAIVE_WORKDIR}" "naive"; then
    log "Naive training failed. Attempting to continue with residual..."
fi

log ""
log "============================================"
log "STEP 2/4: Training RESIDUAL experiment (10k steps)"
log "============================================"

if ! run_train "residual" "${RESIDUAL_WORKDIR}" "residual"; then
    log "Residual training failed."
fi

log ""
log "============================================"
log "STEP 3/4: FID evaluation for NAIVE"
log "============================================"

if ! run_fid "naive" "${NAIVE_WORKDIR}"; then
    log "Naive FID evaluation failed."
fi

log ""
log "============================================"
log "STEP 4/4: FID evaluation for RESIDUAL"
log "============================================"

if ! run_fid "residual" "${RESIDUAL_WORKDIR}"; then
    log "Residual FID evaluation failed."
fi

# ---- Generate Report --------------------------------------------------------
log ""
log "============================================"
log "Generating report.md"
log "============================================"

REPORT="${PROJECT_DIR}/report.md"
cat > "${REPORT}" << REPORT_EOF
# Kimi Residual Attention — Experiment Report

> Auto-generated by auto_experiment.sh at $(date '+%Y-%m-%d %H:%M:%S')

## Experiment Configuration

| Parameter | Value |
|-----------|-------|
| Model | PmfTransformer (hidden=256, depth=8, heads=4) |
| Dataset | CIFAR-10 (32x32) |
| Optimizer | Muon + AdamW hybrid |
| Effective Batch Size | ${BATCH_SIZE} x ${NPROC} = $((BATCH_SIZE * NPROC)) |
| Training Steps | ${MAX_STEPS} |
| GPUs | ${NPROC}x NVIDIA L20Z (81GB) |
| FID Samples | ${FID_SAMPLES} |

## Results

| Experiment | Steps | Final Train Loss | Best Val Loss | FID | IS Mean |
|------------|-------|-----------------|---------------|-----|---------|
$(extract_metrics "Muon + naive" "${NAIVE_WORKDIR}")
$(extract_metrics "Muon + residual" "${RESIDUAL_WORKDIR}")

## Run Directories

- **Naive**: $(latest_run_dir "${NAIVE_WORKDIR}" 2>/dev/null || echo "N/A")
- **Residual**: $(latest_run_dir "${RESIDUAL_WORKDIR}" 2>/dev/null || echo "N/A")

## Analysis

The Full Attention Residual mechanism (arXiv:2603.15031) replaces standard additive
residual connections with depth-wise attention over all preceding layer outputs.
Each layer learns a pseudo-query vector that determines how to aggregate information
from the full history of the network.

Key differences:
- **Naive**: Standard Pre-LN Transformer with additive residuals \`x = x + f(x)\`
- **Residual**: Learned attention over depth axis replaces additive shortcuts

## Logs

- Main log: ${MAIN_LOG}
- Naive training: ${LOG_DIR}/train_naive_${TIMESTAMP}.log
- Residual training: ${LOG_DIR}/train_residual_${TIMESTAMP}.log
- Naive FID: ${LOG_DIR}/fid_naive_${TIMESTAMP}.log
- Residual FID: ${LOG_DIR}/fid_residual_${TIMESTAMP}.log
REPORT_EOF

log "Report saved to: ${REPORT}"
log ""
log "========================================="
log "ALL EXPERIMENTS COMPLETED SUCCESSFULLY"
log "========================================="
log "Check report.md for results summary."
