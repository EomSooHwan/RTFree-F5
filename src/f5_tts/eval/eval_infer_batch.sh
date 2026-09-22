#!/bin/bash
set -e
export PYTHONWARNINGS="ignore::UserWarning,ignore::FutureWarning"

# Configuration parameters
MODEL_NAME="LibriTTS_100_360_500"
CONFIG_NAME="F5TTS_v1_Base"  # Config to load (can differ from MODEL_NAME)
SEEDS=(0 1 2)
TASKS=("seedtts_test_en" "ls_pc_test_clean")
# TASKS=("sap_dev")
LS_TEST_CLEAN_PATH="data/LibriSpeech/test-clean"
GPUS="[0,1,2,3]"
OFFLINE_MODE=false
REFFREE=false

# Reference text modes to evaluate
# "reffree" uses speech encoder; "oracle" and "asr" are F5-TTS baselines
if [ "$REFFREE" = true ]; then
    MODES=("reffree")
else
    MODES=("asr")
fi
# Uncomment to run all three for comparison:
# MODES=("reffree" "oracle" "asr")

# Checkpoint specification: use ONE of these two approaches
# Approach 1: Explicit checkpoint path (takes priority)
CKPT_PATH="/data2/esyoon_hdd/soohwan/interspeech26/F5-TTS/ckpts/F5-TTS/F5TTS_v1_Base/model_1250000.safetensors"
# Approach 2: Step numbers (used only if CKPT_PATH is empty)
CKPTSTEPS=(300000)

# Parse arguments
if [ $OFFLINE_MODE = true ]; then
    LOCAL="--local"
else
    LOCAL=""
fi
INFER_ONLY=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --infer-only)
            INFER_ONLY=true
            shift
            ;;
        *)
            echo "======== Unknown parameter: $1"
            exit 1
            ;;
    esac
done

echo "======== Starting F5-TTS batch evaluation task..."
if [ "$INFER_ONLY" = true ]; then
    echo "======== Mode: Execute infer tasks only"
else
    echo "======== Mode: Execute full pipeline (infer + eval)"
fi

# Function: Execute eval tasks
execute_eval_tasks() {
    local ckpt_label=$1
    local seed=$2
    local task_name=$3
    local mode=$4
    
    local gen_wav_dir="results/${MODEL_NAME}_${ckpt_label}/${task_name}_${mode}/seed${seed}_euler_nfe32_vocos_ss-1_cfg2.0_speed1.0"

    echo ">>>>>>>> Starting eval task: ckpt=${ckpt_label}, seed=${seed}, task=${task_name}, mode=${mode}"
    
    case $task_name in
        "seedtts_test_zh")
            python src/f5_tts/eval/eval_seedtts_testset.py -e wer -l zh -g "$gen_wav_dir" -n "$GPUS" $LOCAL
            python src/f5_tts/eval/eval_seedtts_testset.py -e sim -l zh -g "$gen_wav_dir" -n "$GPUS" $LOCAL
            python src/f5_tts/eval/eval_utmos.py --audio_dir "$gen_wav_dir"
            ;;
        "seedtts_test_en")
            python src/f5_tts/eval/eval_seedtts_testset.py -e wer -l en -g "$gen_wav_dir" -n "$GPUS" $LOCAL
            python src/f5_tts/eval/eval_seedtts_testset.py -e sim -l en -g "$gen_wav_dir" -n "$GPUS" $LOCAL
            python src/f5_tts/eval/eval_utmos.py --audio_dir "$gen_wav_dir"
            ;;
        "ls_pc_test_clean")
            python src/f5_tts/eval/eval_librispeech_test_clean.py -e wer -g "$gen_wav_dir" -n "$GPUS" -p "$LS_TEST_CLEAN_PATH" $LOCAL
            python src/f5_tts/eval/eval_librispeech_test_clean.py -e sim -g "$gen_wav_dir" -n "$GPUS" -p "$LS_TEST_CLEAN_PATH" $LOCAL
            python src/f5_tts/eval/eval_utmos.py --audio_dir "$gen_wav_dir"
            ;;
    esac
    
    echo ">>>>>>>> Completed eval task: ckpt=${ckpt_label}, seed=${seed}, task=${task_name}"
}

# Build the list of (ckpt_label, ckpt_args) pairs to iterate over
# Each entry: "label|args" where args are the flags to pass to the python script
declare -a CKPT_ENTRIES
if [ -n "$CKPT_PATH" ]; then
    # Explicit path mode: derive label from filename (e.g. "model_last" from "ckpts/.../model_last.pt")
    ckpt_basename=$(basename "$CKPT_PATH")
    ckpt_label="${ckpt_basename%.*}"  # strip extension
    CKPT_ENTRIES=("${ckpt_label}|--ckpt_path ${CKPT_PATH}")
    echo "======== Using explicit checkpoint: ${CKPT_PATH} (label: ${ckpt_label})"
else
    for step in "${CKPTSTEPS[@]}"; do
        CKPT_ENTRIES+=("${step}|-c ${step}")
    done
    echo "======== Using checkpoint steps: ${CKPTSTEPS[*]}"
fi

# Build config args
CONFIG_ARGS=""
if [ -n "$CONFIG_NAME" ] && [ "$CONFIG_NAME" != "$MODEL_NAME" ]; then
    CONFIG_ARGS="--config ${CONFIG_NAME}"
fi

# Main execution loop
for entry in "${CKPT_ENTRIES[@]}"; do
    ckpt_label="${entry%%|*}"
    ckpt_args="${entry##*|}"
    
    echo "======== Processing checkpoint: ${ckpt_label}"
    
    for seed in "${SEEDS[@]}"; do
        echo "-------- Processing seed: ${seed}"
        
        # Store eval task PIDs for current seed (if not infer-only mode)
        if [ "$INFER_ONLY" = false ]; then
            declare -a eval_pids
        fi
        
        # Execute each infer task sequentially
        for task in "${TASKS[@]}"; do
            echo ">>>>>>>> Executing infer task: accelerate launch src/f5_tts/eval/eval_infer_batch.py -s ${seed} -n \"${MODEL_NAME}\" -t \"${task}\" ${ckpt_args} ${CONFIG_ARGS} $LOCAL"

            for mode in "${MODES[@]}"; do   
                MODE_ARGS=""
                case $mode in
                    reffree) MODE_ARGS="--reffree --speech_encoder microsoft/wavlm-large" ;;
                    oracle)  MODE_ARGS="--ref_text_mode oracle" ;;
                    asr)     MODE_ARGS="--ref_text_mode asr" ;;
                esac

                echo ">>>>>>>> Mode: ${mode}"
                accelerate launch src/f5_tts/eval/eval_infer_batch.py \
                    -s ${seed} -n "${MODEL_NAME}" -t "${task}" \
                    ${ckpt_args} ${CONFIG_ARGS} \
                    -p "${LS_TEST_CLEAN_PATH}" $LOCAL \
                    ${MODE_ARGS}
            done
            
            # If not infer-only mode, launch corresponding eval task
            if [ "$INFER_ONLY" = false ]; then
                for mode in "${MODES[@]}"; do
                    execute_eval_tasks "$ckpt_label" $seed $task $mode &
                    eval_pids+=($!)
                done
            fi
        done
        
        # If not infer-only mode, wait for all eval tasks of current seed to complete
        if [ "$INFER_ONLY" = false ]; then
            echo ">>>>>>>> All infer tasks for seed ${seed} completed, waiting for corresponding eval tasks to finish..."
            
            for pid in "${eval_pids[@]}"; do
                wait $pid
            done
            
            unset eval_pids  # Clean up array
        fi
        echo "-------- All eval tasks for seed ${seed} completed"
    done
    
    echo "======== Completed checkpoint: ${ckpt_label}"
    echo
done

echo "======== All tasks completed!"