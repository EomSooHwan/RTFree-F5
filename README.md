# RTFree-F5: Transcript-Free Flow-Matching TTS via Speech Feature Conditioning

[![arXiv](https://img.shields.io/badge/arXiv-2606.20266-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2606.20266)
[![python](https://img.shields.io/badge/Python-3.10+-brightgreen)](https://github.com/EomSooHwan/RTFree-F5)

Official code for **"Transcript-Free Flow-Matching Text-to-Speech via Speech Feature Conditioning"** (Interspeech 2026).

Zero-shot TTS models such as [F5-TTS](https://github.com/SWivid/F5-TTS) need a transcript of the reference audio at
inference time, usually from an ASR system. This makes them brittle exactly where zero-shot TTS is most useful:
accented and dysarthric speakers. RTFree-F5 (**R**eference-**T**ranscript-**Free** F5) replaces the reference transcript
with continuous self-supervised speech features:

- a frozen **WavLM-Large** encodes the reference audio;
- a lightweight **MLP projector** (0.8M parameters) maps the features into F5-TTS's text-conditioning space;
- the DiT backbone then sees `[projected speech features ; target-text features]` instead of
  `[reference-text features ; target-text features]`, so the pretrained F5-TTS checkpoint is reused as is.

Training uses cross-utterance pairs of the same speaker in two stages: (1) projector only, with F5-TTS frozen;
(2) projector + DiT backbone, with the text encoder and WavLM frozen.
On dysarthric speech (SAP), WER drops from 24.6% (original recordings) to 10.4%, below the F5-TTS baseline given the
ground-truth reference transcript (20.7%), while naturalness improves and results on standard benchmarks stay competitive.

This repository is a fork of [F5-TTS](https://github.com/SWivid/F5-TTS); everything specific to RTFree-F5 is marked
`RTFree-F5` in the code. Main additions:

| File | What |
|---|---|
| `src/f5_tts/model/speech_encoder.py` | frozen WavLM encoder, MLP projector, frame-rate alignment |
| `src/f5_tts/model/cfm.py`, `model/backbones/dit.py` | speech-feature conditioning hooks, two-stage freezing |
| `src/f5_tts/model/dataset.py` | `CrossUtteranceDataset` (same-speaker reference/target pairs) |
| `src/f5_tts/model/trainer.py`, `train/finetune_cli.py` | stage-wise training, projector learning rate |
| `src/f5_tts/configs/RTFree_F5.yaml` | model config |
| `src/f5_tts/eval/` | baselines with oracle / ASR reference transcripts, SAP and L2-ARCTIC evaluation |

## Installation

```bash
conda create -n rtfree python=3.11 && conda activate rtfree
conda install ffmpeg
# install torch / torchaudio for your CUDA version first, e.g.
pip install torch==2.8.0+cu128 torchaudio==2.8.0+cu128 --extra-index-url https://download.pytorch.org/whl/cu128

git clone https://github.com/EomSooHwan/RTFree-F5.git
cd RTFree-F5
pip install -e .        # add .[eval] for the evaluation tools
```

## Checkpoints

| Model | Description | Download |
|---|---|---|
| RTFree-F5 (Stage 2) | main model of the paper, trained on LibriTTS | *coming soon* |
| RTFree-F5 (Stage 1) | projector only | *coming soon* |

The pretrained F5-TTS v1 Base checkpoint and WavLM-Large are downloaded automatically from Hugging Face.

## Inference

No reference transcript is needed:

```bash
f5-tts_infer-cli --model RTFree_F5 --ckpt_file ckpts/RTFree_F5/model_last.pt \
    --ref_audio "path/to/reference.wav" \
    --gen_text "The text you want to synthesize in the reference speaker's voice."
```

The output duration is estimated from the reference speaking rate. Without a transcript this uses a fixed
prior (about 15 characters per second); pass `--ref_text` to estimate it from the actual transcript, or
`--fix_duration <seconds>` to set the total duration directly. Other options (NFE, CFG strength, sway sampling,
speed, vocoder) are the same as in F5-TTS, see `f5-tts_infer-cli --help`.

Python API:

```python
from f5_tts.api import F5TTS

tts = F5TTS(model="RTFree_F5", ckpt_file="ckpts/RTFree_F5/model_last.pt")
wav, sr, spec = tts.infer(ref_file="reference.wav", ref_text="", gen_text="Hello world.", file_wave="out.wav")
```

## Training

Training fine-tunes the pretrained F5-TTS v1 Base checkpoint on LibriTTS (train-clean-100/360, train-other-500).

1. Prepare LibriTTS (fill in `dataset_dir` in the script; it also stores the speaker id of every utterance):

   ```bash
   python src/f5_tts/train/datasets/prepare_libritts.py
   # the fine-tuned model must keep the vocabulary of the pretrained checkpoint
   cp data/Emilia_ZH_EN_pinyin/vocab.txt data/LibriTTS_100_360_500_pinyin/vocab.txt
   ```

2. Stage 1, cross-modal alignment (projector only, F5-TTS frozen; ~1-2 days on 4 A100s):

   ```bash
   accelerate launch src/f5_tts/train/finetune_cli.py --exp_name RTFree_F5 --stage 1 --finetune \
       --dataset_name LibriTTS_100_360_500 --run_name rtfree_stage1 \
       --epochs 10 --learning_rate 1e-5 --lr_projector 5e-5 --logger wandb
   ```

3. Stage 2, joint fine-tuning (projector + DiT backbone; ~2-3 days on 4 A100s), initialized from stage 1:

   ```bash
   accelerate launch src/f5_tts/train/finetune_cli.py --exp_name RTFree_F5 --stage 2 --finetune \
       --pretrain ckpts/LibriTTS_100_360_500/rtfree_stage1/model_last.pt \
       --dataset_name LibriTTS_100_360_500 --run_name rtfree_stage2 \
       --epochs 20 --learning_rate 1e-5 --lr_projector 5e-5 --logger wandb
   ```

`--batch_size_per_gpu` (frames per GPU, counted on the reference utterance), `--num_warmup_updates` and the other
options follow F5-TTS; run `accelerate config` first for multi-GPU / mixed precision. Checkpoints land in
`ckpts/<dataset_name>/<run_name>/`; re-launching the same command resumes from `model_last.pt`.
The frozen WavLM weights are not stored in checkpoints.

## Evaluation

See [`src/f5_tts/eval`](src/f5_tts/eval) for the objective evaluation (WER / SIM / UTMOS) on LibriSpeech-PC,
Seed-TTS test-en, SAP (dysarthric) and L2-ARCTIC (non-native), including the F5-TTS baselines with oracle and
Whisper-transcribed reference text.

## Citation

```bibtex
@inproceedings{eom2026rtfree,
  title     = {Transcript-Free Flow-Matching Text-to-Speech via Speech Feature Conditioning},
  author    = {Eom, SooHwan and Yoon, Hee Suk and Yoon, Eunseop and Hasegawa-Johnson, Mark and Yoo, Chang D.},
  booktitle = {Interspeech},
  year      = {2026}
}
```

## Acknowledgements

Built on [F5-TTS](https://github.com/SWivid/F5-TTS) (Chen et al., 2024); please also cite it if you use this code.
Speech features from [WavLM](https://github.com/microsoft/unilm/tree/master/wavlm), vocoder from
[Vocos](https://github.com/gemelo-ai/vocos).

## License

Code is released under the MIT License (see [LICENSE](LICENSE)). The pretrained F5-TTS weights are CC-BY-NC licensed
because of their training data; the same applies to checkpoints derived from them.
