#!/bin/bash
# Batch inference + objective evaluation, run from the repo root:
#   bash src/f5_tts/eval/eval_infer_batch.sh [--infer-only] [--eval-only]
# Every setting below can be overridden from the environment, e.g.
#   MODEL_NAME=F5TTS_v1_Base MODES="oracle asr" TASKS="ls_pc_test_clean" bash src/f5_tts/eval/eval_infer_batch.sh
set -e
export PYTHONWARNINGS="ignore::UserWarning,ignore::FutureWarning"

# ---- model: config name (src/f5_tts/configs/<MODEL_NAME>.yaml) and checkpoint (path, or step under ckpts/<MODEL_NAME>/)
MODEL_NAME="${MODEL_NAME:-RTFree_F5}"
CKPT_PATH="${CKPT_PATH:-ckpts/RTFree_F5/model_last.pt}"
CKPTSTEP="${CKPTSTEP:-1250000}"

# ---- what to run
# modes: rtfree (RTFree_F5 models, no reference transcript) | oracle | asr (F5-TTS baselines, reference transcript source)
MODES="${MODES:-rtfree}"
TASKS="${TASKS:-ls_pc_test_clean seedtts_test_en sap_dev l2arctic}"
SEEDS="${SEEDS:-0 1 2}"
GPUS="${GPUS:-[0,1,2,3]}"
OFFLINE_MODE="${OFFLINE_MODE:-false}"

# ---- data
LS_TEST_CLEAN_PATH="${LS_TEST_CLEAN_PATH:-data/LibriSpeech/test-clean}"
SAP_DATA_ROOT="${SAP_DATA_ROOT:-data/SpeechAccessibility_Research_Release}"
L2ARCTIC_DATA_ROOT="${L2ARCTIC_DATA_ROOT:-data/L2-ARCTIC}"
MAX_PAIRS_PER_SPEAKER="${MAX_PAIRS_PER_SPEAKER:-}"  # atypical testsets, empty = all pairs
WAVLM_SV_CKPT="${WAVLM_SV_CKPT:-../checkpoints/UniSpeech/wavlm_large_finetune.pth}"

# ---------------------------------------------------------------------------------------------------------------

INFER_ONLY=false
EVAL_ONLY=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --infer-only) INFER_ONLY=true; shift ;;
        --eval-only)  EVAL_ONLY=true; shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
done

[ "$OFFLINE_MODE" = true ] && LOCAL="--local" || LOCAL=""
if [ -n "$CKPT_PATH" ]; then
    CKPT_ARGS="--ckpt_path ${CKPT_PATH}"; CKPT_LABEL=$(basename "${CKPT_PATH%.*}")
else
    CKPT_ARGS="-c ${CKPTSTEP}"; CKPT_LABEL="${CKPTSTEP}"
fi
[ -n "$MAX_PAIRS_PER_SPEAKER" ] && PAIR_ARGS="--max_pairs_per_speaker ${MAX_PAIRS_PER_SPEAKER}" || PAIR_ARGS=""
DATA_ARGS="-p ${LS_TEST_CLEAN_PATH} --sap_data_root ${SAP_DATA_ROOT} --l2arctic_data_root ${L2ARCTIC_DATA_ROOT}"

gen_wav_dir() {  # task mode seed
    local suffix=""
    case $1 in sap_*|l2arctic) suffix="_gt-dur" ;; esac
    echo "results/${MODEL_NAME}_${CKPT_LABEL}/$1_$2/seed$3_euler_nfe32_vocos_ss-1_cfg2.0_speed1.0${suffix}"
}

run_eval() {  # task gen_wav_dir
    local task=$1 dir=$2
    echo ">>>>>>>> Evaluating ${dir}"
    case $task in
        seedtts_test_zh|seedtts_test_en)
            local lang="${task##*_}"
            python src/f5_tts/eval/eval_seedtts_testset.py -e wer -l $lang -g "$dir" -n "$GPUS" $LOCAL
            python src/f5_tts/eval/eval_seedtts_testset.py -e sim -l $lang -g "$dir" -n "$GPUS" $LOCAL
            ;;
        ls_pc_test_clean)
            python src/f5_tts/eval/eval_librispeech_test_clean.py -e wer -g "$dir" -p "$LS_TEST_CLEAN_PATH" -n "$GPUS" $LOCAL
            python src/f5_tts/eval/eval_librispeech_test_clean.py -e sim -g "$dir" -p "$LS_TEST_CLEAN_PATH" -n "$GPUS" $LOCAL
            ;;
        sap_*|l2arctic)
            python src/f5_tts/eval/eval_atypical.py -e wer -t $task -g "$dir" -n "$GPUS" $LOCAL
            python src/f5_tts/eval/eval_atypical.py -e sim -t $task -g "$dir" -n "$GPUS" --wavlm_ckpt "$WAVLM_SV_CKPT"
            ;;
    esac
    python src/f5_tts/eval/eval_utmos.py --audio_dir "$dir"
}

for mode in $MODES; do
    [ "$mode" = rtfree ] && MODE_ARGS="" || MODE_ARGS="--ref_text_mode ${mode}"
    for task in $TASKS; do
        for seed in $SEEDS; do
            dir=$(gen_wav_dir $task $mode $seed)
            if [ "$EVAL_ONLY" = false ]; then
                echo ">>>>>>>> Inference: model=${MODEL_NAME} mode=${mode} task=${task} seed=${seed}"
                accelerate launch src/f5_tts/eval/eval_infer_batch.py \
                    -s $seed -n "$MODEL_NAME" -t "$task" $CKPT_ARGS $MODE_ARGS $DATA_ARGS $PAIR_ARGS $LOCAL
            fi
            if [ "$INFER_ONLY" = false ]; then
                run_eval $task "$dir"
            fi
        done
    done
done

echo "All done."
