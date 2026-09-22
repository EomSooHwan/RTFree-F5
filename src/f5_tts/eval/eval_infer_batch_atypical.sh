#!/bin/bash
set -e
export PYTHONWARNINGS="ignore::UserWarning,ignore::FutureWarning"

# =========================================================================
# Configuration
# =========================================================================

MODEL_NAME="LibriTTS_100_360_500"
CONFIG_NAME="F5TTS_v1_Base"
SEEDS=(0)
GPUS="[0,1,2,3]"
OFFLINE_MODE=false

# Checkpoint
CKPT_PATH="/data2/esyoon_hdd/soohwan/interspeech26/F5-TTS/ckpts/F5-TTS/F5TTS_v1_Base/model_1250000.safetensors"

# ---- Datasets ----
TASKS=("l2arctic")

SAP_DATA_ROOT="/data2/esyoon_hdd/soohwan/interspeech26/SAP/SpeechAccessibility_Research_Release"
L2ARCTIC_DATA_ROOT="/data2/esyoon_hdd/soohwan/interspeech26/L2_ARCTIC"

# ---- Modes to evaluate ----
# oracle and asr are F5-TTS baselines; reffree is our method
MODES=("oracle")

# ---- Optional: limit pairs per speaker ----
MAX_PAIRS="--max_pairs_per_speaker 100"  # e.g. MAX_PAIRS="--max_pairs_per_speaker 5"

# =========================================================================
# Derived settings
# =========================================================================

if [ $OFFLINE_MODE = true ]; then
    LOCAL="--local"
else
    LOCAL=""
fi

CONFIG_ARGS=""
if [ -n "$CONFIG_NAME" ] && [ "$CONFIG_NAME" != "$MODEL_NAME" ]; then
    CONFIG_ARGS="--config ${CONFIG_NAME}"
fi

if [ -n "$CKPT_PATH" ]; then
    ckpt_basename=$(basename "$CKPT_PATH")
    CKPT_LABEL="${ckpt_basename%.*}"
    CKPT_ARGS="--ckpt_path ${CKPT_PATH}"
else
    echo "ERROR: CKPT_PATH must be set."
    exit 1
fi

# Parse arguments
INFER_ONLY=false
EVAL_ONLY=false
ORIG_BASELINE=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --infer-only)   INFER_ONLY=true;   shift ;;
        --eval-only)    EVAL_ONLY=true;    shift ;;
        --orig-baseline)     ORIG_BASELINE=true; shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
done

# =========================================================================
# Step 0 (optional): Measure ASR WER on original audio
# Run once with --orig-wer to get baseline numbers for the paper.
# =========================================================================

if [ "$ORIG_BASELINE" = true ]; then
    echo "======== Computing original audio baselines (WER + UTMOS)..."
    for task in "${TASKS[@]}"; do
        case $task in
            sap_dev|sap_train)
                SPLIT="Dev"
                [ "$task" = "sap_train" ] && SPLIT="Train"
                python src/f5_tts/eval/eval_atypical.py -e orig_wer \
                    -t "$task" \
                    -d "$SAP_DATA_ROOT" \
                    -m "${SAP_DATA_ROOT}/manifest/${SPLIT}.csv" \
                    -n "$GPUS"
                python src/f5_tts/eval/eval_atypical.py -e orig_mos \
                    -t "$task" \
                    -d "$SAP_DATA_ROOT" \
                    -m "${SAP_DATA_ROOT}/manifest/${SPLIT}.csv" \
                    -n "$GPUS"
                ;;
            l2arctic)
                python src/f5_tts/eval/eval_atypical.py -e orig_wer \
                    -t l2arctic \
                    -d "$L2ARCTIC_DATA_ROOT" \
                    -n "$GPUS"
                python src/f5_tts/eval/eval_atypical.py -e orig_mos \
                    -t l2arctic \
                    -d "$L2ARCTIC_DATA_ROOT" \
                    -n "$GPUS"
                ;;
        esac
    done
    echo "======== Done. Original baselines saved to dataset directories."
    exit 0
fi

# =========================================================================
# Helper functions
# =========================================================================

get_data_args() {
    local task=$1
    case $task in
        sap_dev|sap_train)
            echo "--sap_data_root ${SAP_DATA_ROOT}" ;;
        l2arctic)
            echo "--l2arctic_data_root ${L2ARCTIC_DATA_ROOT}" ;;
    esac
}

get_eval_data_args() {
    local task=$1
    case $task in
        sap_dev)
            echo "-d ${SAP_DATA_ROOT} -m ${SAP_DATA_ROOT}/manifest/Dev.csv" ;;
        sap_train)
            echo "-d ${SAP_DATA_ROOT} -m ${SAP_DATA_ROOT}/manifest/Train.csv" ;;
        l2arctic)
            echo "-d ${L2ARCTIC_DATA_ROOT}" ;;
    esac
}

execute_eval_tasks() {
    # Clear accelerate/NCCL env vars so eval runs as single-process
    unset MASTER_ADDR MASTER_PORT RANK WORLD_SIZE LOCAL_RANK
    unset ACCELERATE_MIXED_PRECISION ACCELERATE_USE_FSDP
    
    local gen_wav_dir=$1
    local task=$2

    # Skip if all eval results already exist
    if [ -f "${gen_wav_dir}/eval_wer_results.json" ] && \
       [ -f "${gen_wav_dir}/eval_sim_results.json" ] && \
       [ -f "${gen_wav_dir}/_utmos_result.txt" ]; then
        echo ">>>>>>>> Skipping eval (already complete): ${gen_wav_dir}"
        return
    fi

    local eval_data_args=$(get_eval_data_args "$task")

    echo ">>>>>>>> Eval WER: ${gen_wav_dir}"
    python src/f5_tts/eval/eval_atypical.py -e wer \
        -g "$gen_wav_dir" -t "$task" $eval_data_args -n "$GPUS" $MAX_PAIRS

    echo ">>>>>>>> Eval SIM: ${gen_wav_dir}"
    python src/f5_tts/eval/eval_atypical.py -e sim \
        -g "$gen_wav_dir" -t "$task" -n "$GPUS"

    echo ">>>>>>>> Eval UTMOS: ${gen_wav_dir}"
    python src/f5_tts/eval/eval_utmos.py --audio_dir "$gen_wav_dir"
}

# =========================================================================
# Main loop: mode x task x seed
# =========================================================================

echo "======== Starting atypical speech evaluation pipeline"
echo "======== Checkpoint: ${CKPT_PATH} (label: ${CKPT_LABEL})"
echo "======== Tasks: ${TASKS[*]}"
echo "======== Modes: ${MODES[*]}"
echo "======== Seeds: ${SEEDS[*]}"
echo

for mode in "${MODES[@]}"; do
    echo "======== Mode: ${mode}"

    MODE_ARGS=""
    case $mode in
        reffree) MODE_ARGS="--reffree --speech_encoder microsoft/wavlm-large" ;;
        oracle)  MODE_ARGS="--ref_text_mode oracle" ;;
        asr)     MODE_ARGS="--ref_text_mode asr" ;;
    esac

    for task in "${TASKS[@]}"; do
        echo "-------- Task: ${task}"
        DATA_ARGS=$(get_data_args "$task")

        for seed in "${SEEDS[@]}"; do
            echo ">>>>>>>> Seed: ${seed}"

            gen_wav_dir="results/${MODEL_NAME}_${CKPT_LABEL}/${task}_${mode}/seed${seed}_euler_nfe32_vocos_ss-1_cfg2.0_speed1.0"

            # ---- Inference ----
            if [ "$EVAL_ONLY" = false ]; then
                echo ">>>>>>>> Inference: mode=${mode} task=${task} seed=${seed}"
                accelerate launch src/f5_tts/eval/eval_infer_batch_atypical.py \
                    -s ${seed} -n "${MODEL_NAME}" -t "${task}" \
                    ${CKPT_ARGS} ${CONFIG_ARGS} \
                    ${DATA_ARGS} ${MODE_ARGS} ${MAX_PAIRS} $LOCAL
            fi

            # ---- Evaluation (background) ----
            if [ "$INFER_ONLY" = false ]; then
                if [ -d "$gen_wav_dir" ]; then
                    execute_eval_tasks "$gen_wav_dir" "$task" &
                else
                    echo "WARNING: output dir not found: ${gen_wav_dir}"
                fi
            fi
        done

        # Wait for eval of this task before proceeding
        if [ "$INFER_ONLY" = false ]; then
            wait
        fi
    done
    echo "======== Completed mode: ${mode}"
    echo
done

echo "======== All tasks completed!"
echo

# =========================================================================
# Summary table
# =========================================================================

if [ "$INFER_ONLY" = false ]; then
    echo "======== Results Summary"
    echo "========================================================"
    printf "%-12s %-12s %-6s %8s %8s %8s\n" "Task" "Mode" "Seed" "WER%" "SIM" "UTMOS"
    echo "--------------------------------------------------------"

    for task in "${TASKS[@]}"; do
        for mode in "${MODES[@]}"; do
            for seed in "${SEEDS[@]}"; do
                dir="results/${MODEL_NAME}_${CKPT_LABEL}/${task}_${mode}/seed${seed}_euler_nfe32_vocos_ss-1_cfg2.0_speed1.0"

                wer_val="-"; sim_val="-"; utmos_val="-"

                if [ -f "${dir}/_eval_wer_results.json" ]; then
                    wer_val=$(python3 -c "
import json
d=json.load(open('${dir}/_eval_wer_results.json'))
print(f\"{d['summary']['corpus_wer']*100:.2f}\")
" 2>/dev/null || echo "-")
                fi
                if [ -f "${dir}/eval_sim_results.json" ]; then
                    sim_val=$(python3 -c "
import json
d=json.load(open('${dir}/eval_sim_results.json'))
print(f\"{d['summary']['mean_sim']:.4f}\")
" 2>/dev/null || echo "-")
                fi
                if [ -f "${dir}/utmos_result.txt" ]; then
                    utmos_val=$(grep -oP '[\d.]+' "${dir}/utmos_result.txt" | tail -1 || echo "-")
                fi

                printf "%-12s %-12s %-6s %8s %8s %8s\n" "$task" "$mode" "$seed" "$wer_val" "$sim_val" "$utmos_val"
            done
        done
    done
    echo "========================================================"
fi