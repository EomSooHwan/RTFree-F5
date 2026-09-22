"""
Batch inference for atypical speech datasets: SAP (dysarthric) and L2-ARCTIC (accented).

Supports three reference text modes:
  - oracle:  ground truth transcript of the reference audio
  - asr:     Whisper-transcribed reference audio (simulates real-world)
  - (none):  used with --reffree, speech encoder replaces text conditioning

Usage:
  # RefFree mode (our method)
  accelerate launch eval_infer_batch_atypical.py \
      -t sap_dev --reffree --ckpt_path <path> --config F5TTS_v1_Base

  # Oracle transcript baseline
  accelerate launch eval_infer_batch_atypical.py \
      -t sap_dev --ref_text_mode oracle --ckpt_path <path> --config F5TTS_v1_Base

  # ASR transcript baseline
  accelerate launch eval_infer_batch_atypical.py \
      -t sap_dev --ref_text_mode asr --ckpt_path <path> --config F5TTS_v1_Base
"""

import os
import sys
import csv
import json
import string

sys.path.append(os.getcwd())

import argparse
import time
from collections import defaultdict
from importlib.resources import files

import torch
import torchaudio
from accelerate import Accelerator
from hydra.utils import get_class
from omegaconf import OmegaConf
from tqdm import tqdm

from f5_tts.eval.utils_eval import get_inference_prompt
from f5_tts.infer.utils_infer import load_checkpoint, load_vocoder
from f5_tts.model import CFM
from f5_tts.model.utils import get_tokenizer


accelerator = Accelerator()
device = f"cuda:{accelerator.process_index}"

use_ema = True
target_rms = 0.1

rel_path = str(files("f5_tts").joinpath("../../"))


# =========================================================================
# L2-ARCTIC speaker-to-L1 mapping (24 speakers, 6 L1 backgrounds)
# =========================================================================

L2ARCTIC_SPEAKER_L1 = {
    "ABA": "Arabic",    "SKA": "Arabic",    "YBAA": "Arabic",  "ZHAA": "Arabic",
    "BWC": "Mandarin",  "LXC": "Mandarin",  "NCC": "Mandarin", "TXHC": "Mandarin",
    "ASI": "Hindi",     "RRBI": "Hindi",    "SVBI": "Hindi",   "TNI": "Hindi",
    "HJK": "Korean",    "HKK": "Korean",    "YDCK": "Korean",  "YKWK": "Korean",
    "EBVS": "Spanish",  "ERMS": "Spanish",  "MBMPS": "Spanish","NJS": "Spanish",
    "HQTV": "Vietnamese","PNV": "Vietnamese","THV": "Vietnamese","TLV": "Vietnamese",
}


# =========================================================================
# Dataset loaders — return metainfo as (utt, prompt_text, prompt_wav, gt_text, gt_wav)
# =========================================================================

def get_sap_metainfo(manifest_csv, sap_data_root, max_pairs_per_speaker=2):
    """
    Build cross-utterance pairs from SAP manifest.

    Task: Given reference audio from a dysarthric speaker (utt_A),
    generate speech with the content of a different utterance (utt_B)
    from the same speaker. Tests voice cloning without needing ASR
    on dysarthric speech.

    Returns:
        metainfo: list of (pair_id, ref_text, ref_audio, gt_text, gt_audio)
        metadata: dict mapping pair_id -> {speaker, etiology}
    """
    spk2utts = defaultdict(list)
    with open(manifest_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            dur = float(row["duration"])
            if dur < 0.3 or dur > 30:
                continue
            text = row["norm_text_without_disfluency"].strip()
            if not text:
                continue
            spk2utts[row["speaker"]].append({
                "id": row["id"],
                "audio": os.path.join(sap_data_root, row["audio_filepath"]),
                "text": text,
                "speaker": row["speaker"],
                "etiology": row.get("etiology", "unknown"),
            })

    metainfo = []
    metadata = {}
    for spk, utts in spk2utts.items():
        if len(utts) < 2:
            continue
        count = 0
        for i, ref_utt in enumerate(utts):
            if max_pairs_per_speaker is not None and count >= max_pairs_per_speaker:
                break
            tgt_utt = utts[(i + 1) % len(utts)]
            
            pair_id = f"{ref_utt['id']}__to__{tgt_utt['id']}"
            metainfo.append((
                pair_id,                    # index 0: utt_id
                ref_utt["text"],            # index 1: prompt_text (ref transcript)
                ref_utt["audio"],           # index 2: prompt_wav  (ref audio)
                " " + tgt_utt["text"],      # index 3: gt_text     (target transcript, leading space)
                tgt_utt["audio"],           # index 4: gt_wav      (target audio, for duration)
            ))
            metadata[pair_id] = {
                "speaker": spk,
                "etiology": ref_utt["etiology"],
                "ref_audio": ref_utt["audio"],
            }
            count += 1

    if accelerator.is_main_process:
        n_spk = sum(1 for u in spk2utts.values() if len(u) >= 2)
        print(f"SAP: {len(metainfo)} cross-utterance pairs from {n_spk} speakers")
    return metainfo, metadata


def get_l2arctic_metainfo(data_root, max_pairs_per_speaker=5):
    """
    Build cross-utterance pairs from L2-ARCTIC.

    Task: Given reference audio from a non-native English speaker (utt_A),
    generate speech with the content of a different utterance (utt_B) from
    the same speaker. Tests voice cloning on accented speech where ASR may
    produce inaccurate transcripts due to pronunciation deviations.

    Directory structure:
        data_root/
        ├── SPEAKER_ID/
        │   ├── wav/arctic_XXXX.wav
        │   └── transcript/arctic_XXXX.txt

    Returns:
        metainfo: list of (pair_id, ref_text, ref_audio, gt_text, gt_audio)
        metadata: dict mapping pair_id -> {speaker, l1}
    """
    metainfo = []
    metadata = {}

    speaker_dirs = sorted([
        d for d in os.listdir(data_root)
        if os.path.isdir(os.path.join(data_root, d))
        and os.path.isdir(os.path.join(data_root, d, "wav"))
    ])

    for spk in speaker_dirs:
        wav_dir = os.path.join(data_root, spk, "wav")
        txt_dir = os.path.join(data_root, spk, "transcript")

        if not os.path.isdir(txt_dir):
            continue

        # Collect utterances that have both wav and transcript
        utts = []
        wav_files = sorted([f for f in os.listdir(wav_dir) if f.endswith(".wav")])
        for wav_file in wav_files:
            utt_id = wav_file.replace(".wav", "")
            txt_path = os.path.join(txt_dir, utt_id + ".txt")
            if not os.path.exists(txt_path):
                continue
            with open(txt_path) as f:
                text = f.read().strip()
            if not text:
                continue

            wav_path = os.path.join(wav_dir, wav_file)
            # Quick duration filter: skip very short files
            info = torchaudio.info(wav_path)
            dur = info.num_frames / info.sample_rate
            if dur < 0.5 or dur > 30:
                continue

            utts.append({
                "id": utt_id,
                "audio": wav_path,
                "text": text,
            })

        if len(utts) < 2:
            continue

        l1 = L2ARCTIC_SPEAKER_L1.get(spk, "Unknown")

        count = 0
        for i, ref_utt in enumerate(utts):
            if max_pairs_per_speaker is not None and count >= max_pairs_per_speaker:
                break
            tgt_utt = utts[(i + 1) % len(utts)]
            pair_id = f"{spk}_{ref_utt['id']}__to__{tgt_utt['id']}"
            metainfo.append((
                pair_id,                    # index 0: utt_id
                ref_utt["text"],            # index 1: prompt_text
                ref_utt["audio"],           # index 2: prompt_wav
                " " + tgt_utt["text"],      # index 3: gt_text (leading space)
                tgt_utt["audio"],           # index 4: gt_wav
            ))
            metadata[pair_id] = {
                "speaker": spk,
                "l1": l1,
                "ref_audio": ref_utt["audio"],
            }
            count += 1

    if accelerator.is_main_process:
        print(f"L2-ARCTIC: {len(metainfo)} cross-utterance pairs from {len(speaker_dirs)} speakers")
    return metainfo, metadata


# =========================================================================
# ASR transcript precomputation
# =========================================================================

def precompute_asr_transcripts(metainfo, cache_path):
    """
    Run Whisper on all reference audios and cache the transcripts.
    Only runs on main process; other processes wait and load cache.

    Args:
        metainfo: list of (utt, prompt_text, prompt_wav, gt_text, gt_wav)
        cache_path: path to save/load JSON cache

    Returns:
        dict mapping ref_audio_path -> ASR transcript string
    """
    if os.path.exists(cache_path):
        if accelerator.is_main_process:
            print(f"Loading ASR transcript cache from {cache_path}")
        with open(cache_path) as f:
            return json.load(f)

    if accelerator.is_main_process:
        from transformers import WhisperProcessor, WhisperForConditionalGeneration

        print(f"Precomputing ASR transcripts for {len(metainfo)} reference audios...")
        processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
        whisper_model = WhisperForConditionalGeneration.from_pretrained(
            "openai/whisper-large-v3", torch_dtype=torch.float16
        ).cuda().eval()

        # Deduplicate: multiple pairs may share the same reference audio
        ref_audios = list(set(item[2] for item in metainfo))
        transcripts = {}

        for audio_path in tqdm(ref_audios, desc="ASR transcription"):
            audio, sr = torchaudio.load(audio_path)
            if audio.shape[0] > 1:
                audio = audio.mean(0, keepdim=True)
            if sr != 16000:
                audio = torchaudio.functional.resample(audio, sr, 16000)
            audio_np = audio.squeeze(0).numpy()

            input_features = processor(
                audio_np, sampling_rate=16000, return_tensors="pt"
            ).input_features.to(device="cuda", dtype=torch.float16)

            with torch.no_grad():
                predicted_ids = whisper_model.generate(
                    input_features, language="en", task="transcribe"
                )
            hypo = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            transcripts[audio_path] = hypo.strip()

        # Save cache
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(transcripts, f, indent=2, ensure_ascii=False)
        print(f"Saved ASR transcript cache to {cache_path} ({len(transcripts)} entries)")

        # Free GPU memory
        del whisper_model, processor
        torch.cuda.empty_cache()

    accelerator.wait_for_everyone()

    # All processes load cache
    with open(cache_path) as f:
        return json.load(f)


def apply_asr_transcripts(metainfo, asr_cache):
    """
    Replace prompt_text (index 1) with ASR transcripts from cache.
    Returns new metainfo list.
    """
    new_metainfo = []
    for utt, prompt_text, prompt_wav, gt_text, gt_wav in metainfo:
        asr_text = asr_cache.get(prompt_wav, prompt_text)
        new_metainfo.append((utt, asr_text, prompt_wav, gt_text, gt_wav))
    return new_metainfo


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description="Batch inference for atypical speech (SAP / L2-ARCTIC)")

    # Experiment
    parser.add_argument("-s", "--seed", default=0, type=int)
    parser.add_argument("-n", "--expname", default="LibriTTS_100_360_500")
    parser.add_argument("-t", "--testset", required=True,
                        choices=["sap_dev", "sap_train", "l2arctic"],
                        help="Which dataset/split to evaluate")

    # Model
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("-c", "--ckptstep", default=None, type=int)
    parser.add_argument("--config", type=str, default=None,
                        help="Config name (e.g. F5TTS_v1_Base). Defaults to --expname")

    # Inference params
    parser.add_argument("-nfe", "--nfestep", default=32, type=int)
    parser.add_argument("-o", "--odemethod", default="euler")
    parser.add_argument("-ss", "--swaysampling", default=-1, type=float)
    parser.add_argument("--cfg_strength", default=2.0, type=float)
    parser.add_argument("--speed", default=1.0, type=float)

    # RefFree mode
    parser.add_argument("--reffree", action="store_true")
    parser.add_argument("--speech_encoder", type=str, default="microsoft/wavlm-large")

    # Reference text mode (for non-reffree baselines)
    parser.add_argument("--ref_text_mode", type=str, default="oracle",
                        choices=["oracle", "asr"],
                        help="How to obtain reference text: oracle=ground truth, asr=Whisper")

    # Dataset paths
    parser.add_argument("--sap_data_root", type=str,
                        default=f"{rel_path}/data/SpeechAccessibility_Research_Release")
    parser.add_argument("--sap_manifest", type=str, default=None)
    parser.add_argument("--l2arctic_data_root", type=str,
                        default=f"{rel_path}/data/L2-ARCTIC")

    # Pairing
    parser.add_argument("--max_pairs_per_speaker", type=int, default=None,
                        help="Limit cross-utterance pairs per speaker. None=all pairs")

    # Misc
    parser.add_argument("--local", action="store_true", help="Use local vocoder checkpoint")
    parser.add_argument("--use_truth_duration", action="store_true", default=True,
                        help="Use target audio duration (default True for atypical speech)")
    parser.add_argument("--no_truth_duration", action="store_true",
                        help="Disable truth duration, estimate from text lengths instead")

    args = parser.parse_args()

    if args.no_truth_duration:
        args.use_truth_duration = False

    seed = args.seed
    exp_name = args.expname
    testset = args.testset
    nfe_step = args.nfestep
    ode_method = args.odemethod
    sway_sampling_coef = args.swaysampling
    cfg_strength = args.cfg_strength
    speed = args.speed
    no_ref_audio = False
    infer_batch_size = 1

    # ---- Load model config ----
    config_name = args.config if args.config else exp_name
    model_cfg = OmegaConf.load(str(files("f5_tts").joinpath(f"configs/{config_name}.yaml")))
    model_cls = get_class(f"f5_tts.model.{model_cfg.model.backbone}")
    model_arc = model_cfg.model.arch
    dataset_name = model_cfg.datasets.name
    tokenizer = model_cfg.model.tokenizer

    mel_spec_type = model_cfg.model.mel_spec.mel_spec_type
    target_sample_rate = model_cfg.model.mel_spec.target_sample_rate
    n_mel_channels = model_cfg.model.mel_spec.n_mel_channels
    hop_length = model_cfg.model.mel_spec.hop_length
    win_length = model_cfg.model.mel_spec.win_length
    n_fft = model_cfg.model.mel_spec.n_fft

    # ---- Load dataset metainfo ----
    if testset == "sap_dev":
        manifest = args.sap_manifest or os.path.join(args.sap_data_root, "manifest", "Dev.csv")
        metainfo, metadata = get_sap_metainfo(manifest, args.sap_data_root,
                                               max_pairs_per_speaker=args.max_pairs_per_speaker)
    elif testset == "sap_train":
        manifest = args.sap_manifest or os.path.join(args.sap_data_root, "manifest", "Train.csv")
        metainfo, metadata = get_sap_metainfo(manifest, args.sap_data_root,
                                               max_pairs_per_speaker=args.max_pairs_per_speaker)
    elif testset == "l2arctic":
        metainfo, metadata = get_l2arctic_metainfo(args.l2arctic_data_root,
                                                    max_pairs_per_speaker=args.max_pairs_per_speaker)

    # ---- Determine effective ref_text_mode label for output dir ----
    if args.reffree:
        mode_label = "reffree"
    else:
        mode_label = args.ref_text_mode  # "oracle" or "asr"

    # ---- ASR transcript precomputation ----
    if args.ref_text_mode == "asr" and not args.reffree:
        if testset.startswith("sap"):
            cache_dir = args.sap_data_root
        else:
            cache_dir = args.l2arctic_data_root
        cache_path = os.path.join(cache_dir, f"asr_cache_{testset}.json")
        asr_cache = precompute_asr_transcripts(metainfo, cache_path)
        metainfo = apply_asr_transcripts(metainfo, asr_cache)

    # ---- Build metainfo_dict for reffree audio loading ----
    metainfo_dict = {}
    for item in metainfo:
        utt = item[0]
        metainfo_dict[utt] = {
            "ref_audio": item[2],   # prompt_wav
            "gen_text": item[3],    # gt_text
        }

    # ---- Resolve checkpoint ----
    if args.ckpt_path:
        ckpt_path = args.ckpt_path
        ckpt_basename = os.path.splitext(os.path.basename(ckpt_path))[0]
        ckpt_step_label = ckpt_basename
    else:
        if args.ckptstep is None:
            raise ValueError("Either --ckpt_path or --ckptstep must be provided.")
        ckpt_step_label = str(args.ckptstep)
        ckpt_prefix = rel_path + f"/ckpts/{exp_name}/model_{args.ckptstep}"
        for ext in [".pt", ".safetensors"]:
            if os.path.exists(ckpt_prefix + ext):
                ckpt_path = ckpt_prefix + ext
                break
        else:
            ckpt_prefix = rel_path + f"/{model_cfg.ckpts.save_dir}/model_{args.ckptstep}"
            for ext in [".pt", ".safetensors"]:
                if os.path.exists(ckpt_prefix + ext):
                    ckpt_path = ckpt_prefix + ext
                    break
            else:
                raise ValueError("Checkpoint not found.")

    # ---- Output directory ----
    output_dir = (
        f"{rel_path}/results/{exp_name}_{ckpt_step_label}/{testset}_{mode_label}/"
        f"seed{seed}_{ode_method}_nfe{nfe_step}_{mel_spec_type}"
        f"{f'_ss{sway_sampling_coef}' if sway_sampling_coef else ''}"
        f"_cfg{cfg_strength}_speed{speed}"
    )

    # ---- Save metadata alongside outputs for eval script ----
    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        meta_path = os.path.join(output_dir, "_pair_metadata.json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

    # ---- Build inference prompts ----
    prompts_all = get_inference_prompt(
        metainfo,
        speed=speed,
        tokenizer=tokenizer,
        target_sample_rate=target_sample_rate,
        n_mel_channels=n_mel_channels,
        hop_length=hop_length,
        mel_spec_type=mel_spec_type,
        target_rms=target_rms,
        use_truth_duration=args.use_truth_duration,
        infer_batch_size=infer_batch_size,
    )

    # ---- Load vocoder ----
    local = args.local
    if mel_spec_type == "vocos":
        vocoder_local_path = "../checkpoints/charactr/vocos-mel-24khz"
    elif mel_spec_type == "bigvgan":
        vocoder_local_path = "../checkpoints/bigvgan_v2_24khz_100band_256x"
    vocoder = load_vocoder(vocoder_name=mel_spec_type, is_local=local, local_path=vocoder_local_path)

    # ---- Load model ----
    vocab_char_map, vocab_size = get_tokenizer(dataset_name, tokenizer)
    model = CFM(
        transformer=model_cls(**model_arc, text_num_embeds=vocab_size, mel_dim=n_mel_channels),
        mel_spec_kwargs=dict(
            n_fft=n_fft, hop_length=hop_length, win_length=win_length,
            n_mel_channels=n_mel_channels, target_sample_rate=target_sample_rate,
            mel_spec_type=mel_spec_type,
        ),
        odeint_kwargs=dict(method=ode_method),
        vocab_char_map=vocab_char_map,
        speech_encoder_name=args.speech_encoder if args.reffree else None,
    ).to(device)

    dtype = torch.float32 if mel_spec_type == "bigvgan" else None
    model = load_checkpoint(model, ckpt_path, device, dtype=dtype, use_ema=use_ema)

    # ---- Inference loop ----
    accelerator.wait_for_everyone()
    start = time.time()

    with accelerator.split_between_processes(prompts_all) as prompts:
        for prompt in tqdm(prompts, disable=not accelerator.is_local_main_process):
            utts, ref_rms_list, ref_mels, ref_mel_lens, total_mel_lens, final_text_list = prompt

            # In RefFree mode: override text to gen_text only (no ref_text prefix)
            if args.reffree:
                gen_only_texts = []
                for utt in utts:
                    gen_only_texts.append(metainfo_dict[utt]["gen_text"])
                # Apply tokenization if needed
                if tokenizer == "pinyin":
                    from f5_tts.model.utils import convert_char_to_pinyin
                    gen_only_texts = convert_char_to_pinyin(gen_only_texts, polyphone=True)
                final_text_list = gen_only_texts

            # Skip already generated
            all_exist = all(os.path.exists(f"{output_dir}/{utt}.wav") for utt in utts)
            if all_exist:
                continue

            ref_mels = ref_mels.to(device)
            ref_mel_lens = torch.tensor(ref_mel_lens, dtype=torch.long).to(device)
            total_mel_lens = torch.tensor(total_mel_lens, dtype=torch.long).to(device)

            # Load reference audio for speech encoder (RefFree mode)
            ref_audio_tensor = None
            ref_audio_sample_lens = None
            if args.reffree:
                ref_audio_list = []
                for utt in utts:
                    ref_audio_path = metainfo_dict[utt]["ref_audio"]
                    ref_audio, sr = torchaudio.load(ref_audio_path)
                    if ref_audio.shape[0] > 1:
                        ref_audio = ref_audio.mean(0, keepdim=True)
                    if sr != target_sample_rate:
                        ref_audio = torchaudio.functional.resample(ref_audio, sr, target_sample_rate)
                    ref_audio_list.append(ref_audio.squeeze(0))

                ref_audio_sample_lens = torch.tensor(
                    [a.shape[0] for a in ref_audio_list], dtype=torch.long
                ).to(device)
                max_len = ref_audio_sample_lens.amax().item()
                ref_audio_tensor = torch.stack([
                    torch.nn.functional.pad(a, (0, max_len - a.shape[0]))
                    for a in ref_audio_list
                ]).to(device)

            # Generate
            with torch.inference_mode():
                generated, _ = model.sample(
                    cond=ref_mels,
                    text=final_text_list,
                    duration=total_mel_lens,
                    lens=ref_mel_lens,
                    steps=nfe_step,
                    cfg_strength=cfg_strength,
                    sway_sampling_coef=sway_sampling_coef,
                    no_ref_audio=no_ref_audio,
                    seed=seed,
                    ref_audio=ref_audio_tensor,
                    ref_audio_lens=ref_audio_sample_lens,
                )

                for i, gen in enumerate(generated):
                    gen = gen[ref_mel_lens[i]:total_mel_lens[i], :].unsqueeze(0)
                    gen_mel_spec = gen.permute(0, 2, 1).to(torch.float32)
                    if mel_spec_type == "vocos":
                        generated_wave = vocoder.decode(gen_mel_spec).cpu()
                    elif mel_spec_type == "bigvgan":
                        generated_wave = vocoder(gen_mel_spec).squeeze(0).cpu()

                    if ref_rms_list[i] < target_rms:
                        generated_wave = generated_wave * ref_rms_list[i] / target_rms
                    torchaudio.save(f"{output_dir}/{utts[i]}.wav", generated_wave, target_sample_rate)

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        elapsed = time.time() - start
        n_files = len([f for f in os.listdir(output_dir) if f.endswith(".wav")])
        print(f"Done batch inference: {n_files} files in {elapsed/60:.2f} min → {output_dir}")


if __name__ == "__main__":
    main()