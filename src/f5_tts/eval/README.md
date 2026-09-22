
# Evaluation

Install packages for evaluation:

```bash
pip install -e .[eval]
```

## Test sets

Typical speakers (as in F5-TTS):

1. *LibriSpeech-PC test-clean*, 4-10 s cross-sentence subset: `data/librispeech_pc_test_clean_cross_sentence.lst`,
   audio from [OpenSLR](http://www.openslr.org/12/) placed at `data/LibriSpeech/test-clean`.
2. *Seed-TTS test-en*: download from [seed-tts-eval](https://github.com/BytedanceSpeech/seed-tts-eval) to `data/seedtts_testset`.

Atypical speakers (RTFree-F5), evaluated cross-utterance: the reference is one utterance of a speaker, the target
text and duration come from the next utterance of the same speaker.

3. *SAP* (Speech Accessibility Project, dysarthric speech), `sap_dev` / `sap_train`: the release directory with
   `manifest/{Dev,Train}.csv` (columns `id, speaker, audio_filepath, duration, norm_text_without_disfluency, etiology`).
4. *L2-ARCTIC* (non-native English), `l2arctic`: `<root>/<SPEAKER>/{wav/*.wav, transcript/*.txt}`.

## Batch inference

```bash
accelerate config  # once

# RTFree-F5 (no reference transcript)
accelerate launch src/f5_tts/eval/eval_infer_batch.py -n RTFree_F5 --ckpt_path ckpts/RTFree_F5/model_last.pt \
    -t ls_pc_test_clean -p data/LibriSpeech/test-clean -s 0
accelerate launch src/f5_tts/eval/eval_infer_batch.py -n RTFree_F5 --ckpt_path ckpts/RTFree_F5/model_last.pt \
    -t sap_dev --sap_data_root <SAP_ROOT> -s 0

# F5-TTS baselines: reference transcript from the dataset (oracle) or from Whisper large-v3 (asr)
accelerate launch src/f5_tts/eval/eval_infer_batch.py -n F5TTS_v1_Base -t sap_dev --sap_data_root <SAP_ROOT> --ref_text_mode oracle
accelerate launch src/f5_tts/eval/eval_infer_batch.py -n F5TTS_v1_Base -t sap_dev --sap_data_root <SAP_ROOT> --ref_text_mode asr
```

Generated wavs go to `results/<model>_<ckpt>/<testset>_<rtfree|oracle|asr>/seed<seed>_...`. For the atypical test
sets the target duration is the duration of the target utterance (`_gt-dur`), and a `_pair_metadata.json` with the
speaker, group (etiology / L1) and target text of every pair is saved next to the wavs.

`eval_infer_batch.sh` runs inference and evaluation over models, modes, test sets and seeds; settings are
environment variables (see the top of the script):

```bash
MODEL_NAME=RTFree_F5 CKPT_PATH=ckpts/RTFree_F5/model_last.pt MODES=rtfree bash src/f5_tts/eval/eval_infer_batch.sh
MODEL_NAME=F5TTS_v1_Base CKPT_PATH="" MODES="oracle asr" bash src/f5_tts/eval/eval_infer_batch.sh
```

## Objective evaluation

Evaluation models: [Whisper large-v3](https://huggingface.co/openai/whisper-large-v3) for WER (downloaded
automatically, [Paraformer-zh](https://huggingface.co/funasr/paraformer-zh) for Chinese),
[WavLM-large speaker verification](https://drive.google.com/file/d/1-aE1NfzpRCLxA4GUxX9ITI3F9LlbtEGP/view) for SIM
(download to `../checkpoints/UniSpeech/wavlm_large_finetune.pth`), [UTMOS](https://github.com/tarepan/SpeechMOS) for
naturalness.

```bash
# LibriSpeech-PC / Seed-TTS, as in F5-TTS (WER averaged over utterances)
python src/f5_tts/eval/eval_librispeech_test_clean.py -e wer -g <GEN_WAV_DIR> -p data/LibriSpeech/test-clean -n 4
python src/f5_tts/eval/eval_librispeech_test_clean.py -e sim -g <GEN_WAV_DIR> -p data/LibriSpeech/test-clean -n 4
python src/f5_tts/eval/eval_seedtts_testset.py -e wer -l en -g <GEN_WAV_DIR> -n 4

# SAP / L2-ARCTIC, with per-etiology / per-L1 breakdown (corpus-level WER)
python src/f5_tts/eval/eval_atypical.py -e wer -t sap_dev -g <GEN_WAV_DIR> -n 4
python src/f5_tts/eval/eval_atypical.py -e sim -t sap_dev -g <GEN_WAV_DIR> -n 4
# the same metrics on the original recordings (the "Original" rows of the paper)
python src/f5_tts/eval/eval_atypical.py -e orig_wer -t sap_dev -d <SAP_ROOT> -n 4
python src/f5_tts/eval/eval_atypical.py -e orig_sim -t sap_dev -d <SAP_ROOT> -n 4 --max_pairs_per_speaker 10
python src/f5_tts/eval/eval_atypical.py -e orig_mos -t sap_dev -d <SAP_ROOT> -n 4

# UTMOS
python src/f5_tts/eval/eval_utmos.py --audio_dir <GEN_WAV_DIR>
```

Results are written to `_<task>_results.json[l]` in `<GEN_WAV_DIR>` (or in the dataset root for `orig_*`).
