"""
Evaluate generated atypical speech: SAP (dysarthric) and L2-ARCTIC (accented).

Metrics:
  - WER / CER: intelligibility (via Whisper, same model as F5-TTS eval)
  - SIM: speaker similarity (via ECAPA-TDNN + WavLM, same model as F5-TTS eval)
  - UTMOS: naturalness (delegates to existing eval_utmos.py)

Reports per-group breakdowns:
  - SAP: by etiology (ALS, Cerebral Palsy, etc.)
  - L2-ARCTIC: by L1 (Arabic, Mandarin, Hindi, Korean, Spanish, Vietnamese)

Usage:
  # WER + CER on generated audio
  python eval_atypical.py -e wer -g <gen_wav_dir> -n "[0,1,2,3]"

  # Speaker similarity
  python eval_atypical.py -e sim -g <gen_wav_dir> -n "[0,1,2,3]"

  # WER of original audio (baseline: how bad is ASR on this population?)
  python eval_atypical.py -e orig_wer -t sap_dev -d <sap_data_root> -m <manifest> -n "[0]"
  python eval_atypical.py -e orig_wer -t l2arctic -d <l2arctic_root> -n "[0]"
"""

import argparse
import csv
import json
import os
import string
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from tqdm import tqdm

from f5_tts.eval.ecapa_tdnn import ECAPA_TDNN_SMALL


# =========================================================================
# L2-ARCTIC speaker-to-L1 mapping
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
# Metadata loading
# =========================================================================

def load_pair_metadata(gen_wav_dir):
    """
    Load pair metadata saved by eval_infer_batch_atypical.py.
    Returns dict: pair_id -> {speaker, etiology/l1, ref_audio, ...}
    """
    meta_path = os.path.join(gen_wav_dir, "_pair_metadata.json")
    if not os.path.exists(meta_path):
        print(f"WARNING: No metadata file found at {meta_path}")
        print("  Per-group breakdowns will not be available.")
        return {}
    with open(meta_path) as f:
        return json.load(f)


def get_group_key(pair_meta):
    """Get the group label for a pair (etiology for SAP, L1 for L2-ARCTIC)."""
    if "etiology" in pair_meta:
        return pair_meta["etiology"]
    elif "l1" in pair_meta:
        return pair_meta["l1"]
    return "unknown"


def get_group_type(metadata):
    """Determine whether this is SAP (etiology) or L2-ARCTIC (L1)."""
    for v in metadata.values():
        if "etiology" in v:
            return "etiology"
        elif "l1" in v:
            return "L1"
    return "group"


# =========================================================================
# Test set builders (format: list of (gen_wav, ref_wav, gt_text))
# =========================================================================

def build_test_set(gen_wav_dir, metadata):
    """
    Build test set from generated wavs + metadata.
    Returns:
        test_set: list of (gen_wav_path, ref_wav_path, gt_text)
        pair_ids: list of pair_id (parallel with test_set)
    """
    test_set = []
    pair_ids = []

    gen_files = sorted([f for f in os.listdir(gen_wav_dir)
                        if f.endswith(".wav") and not f.startswith("_")])

    for fname in gen_files:
        pair_id = fname.replace(".wav", "")
        if pair_id not in metadata:
            continue

        gen_path = os.path.join(gen_wav_dir, fname)
        ref_path = metadata[pair_id]["ref_audio"]

        if not os.path.exists(ref_path):
            print(f"  Warning: ref audio missing: {ref_path}")
            continue

        # gt_text is not stored in metadata; we only need it for WER
        # It will be loaded separately in the WER function
        test_set.append((gen_path, ref_path, ""))
        pair_ids.append(pair_id)

    return test_set, pair_ids


def build_test_set_with_text(gen_wav_dir, metadata, text_source):
    """
    Build test set with ground truth text for WER evaluation.

    text_source: dict mapping pair_id -> gt_text
    """
    test_set = []
    pair_ids = []

    gen_files = sorted([f for f in os.listdir(gen_wav_dir)
                        if f.endswith(".wav") and not f.startswith("_")])

    for fname in gen_files:
        pair_id = fname.replace(".wav", "")
        if pair_id not in metadata or pair_id not in text_source:
            continue

        gen_path = os.path.join(gen_wav_dir, fname)
        ref_path = metadata[pair_id]["ref_audio"]
        gt_text = text_source[pair_id]

        test_set.append((gen_path, ref_path, gt_text))
        pair_ids.append(pair_id)

    return test_set, pair_ids


# =========================================================================
# Ground truth text loaders
# =========================================================================

def load_sap_gt_texts(manifest_csv, sap_data_root, max_pairs_per_speaker=None):
    """Reconstruct pair_id -> gt_text mapping from SAP manifest."""
    spk2utts = defaultdict(list)
    with open(manifest_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            dur = float(row["duration"])
            if dur < 0.3 or dur > 30:
                continue
            spk2utts[row["speaker"]].append({
                "id": row["id"],
                "text": row["norm_text_without_disfluency"],
            })

    texts = {}
    for spk, utts in spk2utts.items():
        if len(utts) < 2:
            continue
        count = 0
        for i, ref_utt in enumerate(utts):
            if max_pairs_per_speaker is not None and count >= max_pairs_per_speaker:
                break
            tgt_utt = utts[(i + 1) % len(utts)]
            pair_id = f"{ref_utt['id']}__to__{tgt_utt['id']}"
            texts[pair_id] = tgt_utt["text"]
            count += 1
    return texts


def load_l2arctic_gt_texts(data_root, max_pairs_per_speaker=None):
    """Reconstruct pair_id -> gt_text mapping from L2-ARCTIC."""
    texts = {}
    speaker_dirs = sorted([
        d for d in os.listdir(data_root)
        if os.path.isdir(os.path.join(data_root, d))
        and os.path.isdir(os.path.join(data_root, d, "wav"))
    ])

    for spk in speaker_dirs:
        txt_dir = os.path.join(data_root, spk, "transcript")
        wav_dir = os.path.join(data_root, spk, "wav")
        if not os.path.isdir(txt_dir):
            continue

        utts = []
        for wav_file in sorted(os.listdir(wav_dir)):
            if not wav_file.endswith(".wav"):
                continue
            utt_id = wav_file.replace(".wav", "")
            txt_path = os.path.join(txt_dir, utt_id + ".txt")
            if not os.path.exists(txt_path):
                continue
            with open(txt_path) as f:
                text = f.read().strip()
            if not text:
                continue

            info = torchaudio.info(os.path.join(wav_dir, wav_file))
            dur = info.num_frames / info.sample_rate
            if dur < 0.5 or dur > 30:
                continue

            utts.append({"id": utt_id, "text": text})

        if len(utts) < 2:
            continue
        count = 0
        for i, ref_utt in enumerate(utts):
            if max_pairs_per_speaker is not None and count >= max_pairs_per_speaker:
                break
            tgt_utt = utts[(i + 1) % len(utts)]
            pair_id = f"{spk}_{ref_utt['id']}__to__{tgt_utt['id']}"
            texts[pair_id] = tgt_utt["text"]
            count += 1
    return texts


# =========================================================================
# WER / CER evaluation (uses same Whisper as F5-TTS eval)
# =========================================================================

def run_wer_worker(args):
    """
    WER/CER worker for a single GPU. Matches utils_eval.py Whisper usage.
    args = (rank, test_set, ckpt_dir)
    test_set = list of (gen_wav, ref_wav, gt_text)
    """
    rank, test_set, ckpt_dir = args

    from zhon.hanzi import punctuation as zh_punctuation
    from jiwer import process_words
    from transformers import WhisperProcessor, WhisperForConditionalGeneration

    torch.cuda.set_device(rank)

    model_name = "openai/whisper-large-v3" if not ckpt_dir else ckpt_dir
    processor = WhisperProcessor.from_pretrained(model_name)
    whisper_model = WhisperForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.float16
    ).cuda(rank).eval()

    punctuation_all = zh_punctuation + string.punctuation
    results = []

    for gen_wav, ref_wav, truth in tqdm(test_set, desc=f"WER [GPU {rank}]"):
        audio, sr = torchaudio.load(gen_wav)
        if sr != 16000:
            audio = torchaudio.functional.resample(audio, sr, 16000)
        audio_np = audio.squeeze(0).numpy()

        input_features = processor(
            audio_np, sampling_rate=16000, return_tensors="pt"
        ).input_features.to(device=f"cuda:{rank}", dtype=torch.float16)

        with torch.no_grad():
            predicted_ids = whisper_model.generate(
                input_features, language="en", task="transcribe"
            )
        hypo = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

        # Normalize (same as utils_eval.py)
        raw_truth, raw_hypo = truth, hypo
        for x in punctuation_all:
            truth = truth.replace(x, "")
            hypo = hypo.replace(x, "")
        truth = truth.replace("  ", " ").lower().strip()
        hypo = hypo.replace("  ", " ").lower().strip()

        if not truth:
            continue

        measures = process_words(truth, hypo)

        results.append({
            "wav": Path(gen_wav).stem,
            "truth": raw_truth,
            "hypo": raw_hypo,
            "truth_norm": truth,
            "hypo_norm": hypo,
            "wer": measures.wer,
        })

    return results


def run_wer_on_originals(args):
    """
    WER worker for original (un-reconstructed) audio.
    args = (rank, audio_text_pairs, ckpt_dir)
    audio_text_pairs = list of (audio_path, gt_text, group_label)
    """
    rank, audio_text_pairs, ckpt_dir = args

    from zhon.hanzi import punctuation as zh_punctuation
    from jiwer import process_words
    from transformers import WhisperProcessor, WhisperForConditionalGeneration

    torch.cuda.set_device(rank)

    model_name = "openai/whisper-large-v3" if not ckpt_dir else ckpt_dir
    processor = WhisperProcessor.from_pretrained(model_name)
    whisper_model = WhisperForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.float16
    ).cuda(rank).eval()

    punctuation_all = zh_punctuation + string.punctuation
    results = []

    for audio_path, truth, group in tqdm(audio_text_pairs, desc=f"Orig WER [GPU {rank}]"):
        audio, sr = torchaudio.load(audio_path)
        if audio.shape[0] > 1:
            audio = audio.mean(0, keepdim=True)
        if sr != 16000:
            audio = torchaudio.functional.resample(audio, sr, 16000)
        audio_np = audio.squeeze(0).numpy()

        input_features = processor(
            audio_np, sampling_rate=16000, return_tensors="pt"
        ).input_features.to(device=f"cuda:{rank}", dtype=torch.float16)

        with torch.no_grad():
            predicted_ids = whisper_model.generate(
                input_features, language="en", task="transcribe"
            )
        hypo = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

        raw_truth, raw_hypo = truth, hypo
        for x in punctuation_all:
            truth = truth.replace(x, "")
            hypo = hypo.replace(x, "")
        truth = truth.replace("  ", " ").lower().strip()
        hypo = hypo.replace("  ", " ").lower().strip()

        if not truth:
            continue

        measures = process_words(truth, hypo)
        results.append({
            "audio": audio_path,
            "truth": raw_truth,
            "hypo": raw_hypo,
            "truth_norm": truth,
            "hypo_norm": hypo,
            "wer": measures.wer,
            "group": group,
        })

    return results

def run_orig_mos(items, gpu_nums):
    """
    Compute UTMOS on original (un-reconstructed) audio with per-group breakdowns.
    items: list of (audio_path, gt_text, group_label)
    """
    if isinstance(gpu_nums, str):
        gpu_nums = eval(gpu_nums)
    device = f"cuda:{gpu_nums[0]}" if gpu_nums else "cuda:0"

    predictor = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True)
    predictor = predictor.to(device).eval()

    results = []
    groups = defaultdict(list)

    for audio_path, gt_text, group in tqdm(items, desc="Original UTMOS"):
        wav, sr = torchaudio.load(audio_path)
        if wav.shape[0] > 1:
            wav = wav.mean(0, keepdim=True)
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)

        with torch.no_grad():
            score = predictor(wav.to(device), sr=16000).item()

        results.append({
            "audio": audio_path,
            "utmos": score,
            "group": group,
        })
        groups[group].append(score)

    all_scores = [r["utmos"] for r in results]
    mean_mos = np.mean(all_scores)

    print(f"\n{'='*65}")
    print(f"  ORIGINAL AUDIO UTMOS")
    print(f"  UTMOS: {mean_mos:.4f} +/- {np.std(all_scores):.4f}    (n={len(all_scores)})")
    print(f"{'='*65}")
    if groups:
        print(f"\n  Per-group breakdown:")
        for g in sorted(groups.keys()):
            scores = groups[g]
            print(f"    {g:20s}  UTMOS={np.mean(scores):.4f} +/- {np.std(scores):.4f}  (n={len(scores)})")
    print()

    return {
        "mean_utmos": float(mean_mos),
        "std_utmos": float(np.std(all_scores)),
        "num_samples": len(all_scores),
        "per_sample": results,
    }


# =========================================================================
# SIM evaluation (uses same ECAPA-TDNN as F5-TTS eval)
# =========================================================================

def run_sim_worker(args):
    """
    Speaker similarity worker for a single GPU.
    Uses ECAPA_TDNN_SMALL + WavLM (same as utils_eval.py run_sim).
    args = (rank, test_set, ckpt_dir)
    """
    rank, test_set, ckpt_dir = args
    device = f"cuda:{rank}"

    ckpt_dir = "../checkpoints/UniSpeech/wavlm_large_finetune.pth"
    model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type="wavlm_large", config_path=None)
    state_dict = torch.load(ckpt_dir, weights_only=True, map_location="cpu")
    model.load_state_dict(state_dict["model"], strict=False)
    model = model.cuda(device).eval()

    results = []
    for gen_wav, ref_wav, _ in tqdm(test_set, desc=f"SIM [GPU {rank}]"):
        wav1, sr1 = torchaudio.load(gen_wav)
        wav2, sr2 = torchaudio.load(ref_wav)

        wav1 = wav1.cuda(device)
        wav2 = wav2.cuda(device)

        if sr1 != 16000:
            wav1 = torchaudio.transforms.Resample(sr1, 16000).cuda(device)(wav1)
        if sr2 != 16000:
            wav2 = torchaudio.transforms.Resample(sr2, 16000).cuda(device)(wav2)

        with torch.no_grad():
            emb1 = model(wav1)
            emb2 = model(wav2)

        sim = F.cosine_similarity(emb1, emb2)[0].item()
        results.append({
            "wav": Path(gen_wav).stem,
            "sim": sim,
        })

    return results


# =========================================================================
# Aggregation and reporting
# =========================================================================

def aggregate_wer_results(results, metadata):
    """Compute corpus-level and per-group WER/CER with breakdowns."""
    from jiwer import process_words, cer as compute_cer

    group_type = get_group_type(metadata)
    all_truths, all_hypos = [], []
    groups = defaultdict(lambda: {"truths": [], "hypos": []})

    for r in results:
        pair_id = r["wav"]
        truth_n = r["truth_norm"]
        hypo_n = r["hypo_norm"]
        all_truths.append(truth_n)
        all_hypos.append(hypo_n)

        if pair_id in metadata:
            group = get_group_key(metadata[pair_id])
            groups[group]["truths"].append(truth_n)
            groups[group]["hypos"].append(hypo_n)

    if not all_truths:
        print("No valid results to aggregate.")
        return

    corpus_wer = process_words(all_truths, all_hypos).wer
    corpus_cer = compute_cer(all_truths, all_hypos)

    print(f"\n{'='*65}")
    print(f"  WER: {corpus_wer*100:.2f}%    CER: {corpus_cer*100:.2f}%    (n={len(all_truths)})")
    print(f"{'='*65}")

    if groups:
        print(f"\n  Per-{group_type} breakdown:")
        for group_name in sorted(groups.keys()):
            g = groups[group_name]
            g_wer = process_words(g["truths"], g["hypos"]).wer
            g_cer = compute_cer(g["truths"], g["hypos"])
            print(f"    {group_name:20s}  WER={g_wer*100:6.2f}%  CER={g_cer*100:6.2f}%  (n={len(g['truths'])})")
    print()

    return {"corpus_wer": corpus_wer, "corpus_cer": corpus_cer, "num_samples": len(all_truths)}


def aggregate_sim_results(results, metadata):
    """Compute corpus-level and per-group speaker similarity."""
    group_type = get_group_type(metadata)
    all_sims = []
    groups = defaultdict(list)

    for r in results:
        pair_id = r["wav"]
        all_sims.append(r["sim"])
        if pair_id in metadata:
            group = get_group_key(metadata[pair_id])
            groups[group].append(r["sim"])

    if not all_sims:
        print("No valid results to aggregate.")
        return

    mean_sim = np.mean(all_sims)

    print(f"\n{'='*65}")
    print(f"  SIM: {mean_sim:.4f} +/- {np.std(all_sims):.4f}    (n={len(all_sims)})")
    print(f"{'='*65}")

    if groups:
        print(f"\n  Per-{group_type} breakdown:")
        for group_name in sorted(groups.keys()):
            sims = groups[group_name]
            print(f"    {group_name:20s}  SIM={np.mean(sims):.4f} +/- {np.std(sims):.4f}  (n={len(sims)})")
    print()

    return {"mean_sim": mean_sim, "std_sim": float(np.std(all_sims)), "num_samples": len(all_sims)}


def aggregate_orig_wer_results(results):
    """Aggregate original audio WER results with per-group breakdowns."""
    from jiwer import process_words, cer as compute_cer

    all_truths = [r["truth_norm"] for r in results]
    all_hypos = [r["hypo_norm"] for r in results]
    groups = defaultdict(lambda: {"truths": [], "hypos": []})
    for r in results:
        groups[r["group"]]["truths"].append(r["truth_norm"])
        groups[r["group"]]["hypos"].append(r["hypo_norm"])

    if not all_truths:
        print("No valid results.")
        return

    corpus_wer = process_words(all_truths, all_hypos).wer
    corpus_cer = compute_cer(all_truths, all_hypos)

    print(f"\n{'='*65}")
    print(f"  ORIGINAL AUDIO ASR PERFORMANCE")
    print(f"  WER: {corpus_wer*100:.2f}%    CER: {corpus_cer*100:.2f}%    (n={len(all_truths)})")
    print(f"{'='*65}")
    if groups:
        print(f"\n  Per-group breakdown:")
        for g in sorted(groups.keys()):
            d = groups[g]
            g_wer = process_words(d["truths"], d["hypos"]).wer
            g_cer = compute_cer(d["truths"], d["hypos"])
            print(f"    {g:20s}  WER={g_wer*100:6.2f}%  CER={g_cer*100:6.2f}%  (n={len(d['truths'])})")
    print()

    return {"corpus_wer": corpus_wer, "corpus_cer": corpus_cer}


# =========================================================================
# Original audio data loaders (for orig_wer baseline)
# =========================================================================

def load_sap_originals(manifest_csv, sap_data_root):
    """Load (audio_path, gt_text, etiology) for all SAP utterances."""
    items = []
    with open(manifest_csv) as f:
        for row in csv.DictReader(f):
            dur = float(row["duration"])
            if dur < 0.3 or dur > 30:
                continue
            audio = os.path.join(sap_data_root, row["audio_filepath"])
            text = row["norm_text_without_disfluency"]
            etiology = row.get("etiology", "unknown")
            items.append((audio, text, etiology))
    return items


def load_l2arctic_originals(data_root):
    """Load (audio_path, gt_text, L1) for all L2-ARCTIC utterances."""
    items = []
    for spk in sorted(os.listdir(data_root)):
        wav_dir = os.path.join(data_root, spk, "wav")
        txt_dir = os.path.join(data_root, spk, "transcript")
        if not os.path.isdir(wav_dir) or not os.path.isdir(txt_dir):
            continue
        l1 = L2ARCTIC_SPEAKER_L1.get(spk, "Unknown")
        for wav_file in sorted(os.listdir(wav_dir)):
            if not wav_file.endswith(".wav"):
                continue
            utt_id = wav_file.replace(".wav", "")
            txt_path = os.path.join(txt_dir, utt_id + ".txt")
            if not os.path.exists(txt_path):
                continue
            with open(txt_path) as f:
                text = f.read().strip()
            if not text:
                continue
            info = torchaudio.info(os.path.join(wav_dir, wav_file))
            dur = info.num_frames / info.sample_rate
            if dur < 0.5 or dur > 30:
                continue
            items.append((os.path.join(wav_dir, wav_file), text, l1))
    return items


# =========================================================================
# Multi-GPU distribution helper
# =========================================================================

def distribute_to_gpus(data, gpus):
    """Split data list across GPU workers. Returns list of (gpu_id, data_slice)."""
    n = len(gpus)
    if n == 1:
        return [(gpus[0], data)]
    chunk = len(data) // n + 1
    return [(gpus[i], data[i*chunk:(i+1)*chunk]) for i in range(n)]


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate atypical speech generation (SAP / L2-ARCTIC)")
    parser.add_argument("-e", "--eval_type", required=True,
                        choices=["wer", "sim", "orig_wer", "orig_mos"],
                        help="wer: WER+CER on generated audio, sim: speaker similarity, "
                             "orig_wer: WER of original (un-reconstructed) audio")
    parser.add_argument("-g", "--gen_wav_dir", type=str, default=None,
                        help="Directory of generated wav files (required for wer/sim)")
    parser.add_argument("-t", "--testset", type=str, default=None,
                        choices=["sap_dev", "sap_train", "l2arctic"],
                        help="Dataset (required for orig_wer and for loading gt texts)")
    parser.add_argument("-n", "--gpu_nums", type=str, default="[0]",
                        help="GPU list, e.g. '[0,1,2,3]'")
    parser.add_argument("-d", "--data_root", type=str, default=None,
                        help="Dataset root directory")
    parser.add_argument("-m", "--manifest", type=str, default=None,
                        help="SAP manifest CSV path (only for SAP testsets)")
    parser.add_argument("--sim_ckpt", type=str, default="",
                        help="Path to ECAPA-TDNN checkpoint")
    parser.add_argument("--asr_ckpt", type=str, default="",
                        help="Path to Whisper checkpoint (empty = download from HF)")
    parser.add_argument("--max_pairs_per_speaker", type=int, default=None,
                        help="Must match the value used during inference")

    args = parser.parse_args()
    gpus = eval(args.gpu_nums)

    # ---- orig_wer: evaluate ASR on original (un-reconstructed) audio ----
    if args.eval_type == "orig_wer":
        if args.testset is None:
            parser.error("--testset is required for orig_wer")

        if args.testset.startswith("sap"):
            if args.data_root is None:
                parser.error("--data_root is required for SAP")
            manifest = args.manifest
            if manifest is None:
                split = "Dev" if "dev" in args.testset else "Train"
                manifest = os.path.join(args.data_root, "manifest", f"{split}.csv")
            items = load_sap_originals(manifest, args.data_root)
        elif args.testset == "l2arctic":
            if args.data_root is None:
                parser.error("--data_root is required for L2-ARCTIC")
            items = load_l2arctic_originals(args.data_root)

        print(f"Evaluating original audio WER on {len(items)} utterances...")
        distributed = distribute_to_gpus(items, gpus)
        worker_args = [(gpu, chunk, args.asr_ckpt) for gpu, chunk in distributed]

        if len(gpus) == 1:
            all_results = [run_wer_on_originals(worker_args[0])]
        else:
            with Pool(len(gpus)) as pool:
                all_results = pool.map(run_wer_on_originals, worker_args)

        results = [r for batch in all_results for r in batch]
        summary = aggregate_orig_wer_results(results)

        # Save
        out_dir = args.data_root or "."
        out_path = os.path.join(out_dir, f"orig_wer_{args.testset}.json")
        with open(out_path, "w") as f:
            json.dump({"summary": summary, "per_sample": results}, f, indent=2, ensure_ascii=False)
        print(f"Saved to {out_path}")
        return
    
    if args.eval_type == "orig_mos":
        if args.testset is None:
            parser.error("--testset is required for orig_mos")

        if args.testset.startswith("sap"):
            if args.data_root is None:
                parser.error("--data_root is required for SAP")
            manifest = args.manifest
            if manifest is None:
                split = "Dev" if "dev" in args.testset else "Train"
                manifest = os.path.join(args.data_root, "manifest", f"{split}.csv")
            items = load_sap_originals(manifest, args.data_root)
        elif args.testset == "l2arctic":
            if args.data_root is None:
                parser.error("--data_root is required for L2-ARCTIC")
            items = load_l2arctic_originals(args.data_root)

        print(f"Computing UTMOS on {len(items)} original utterances...")
        summary = run_orig_mos(items, args.gpu_nums)

        out_dir = args.data_root or "."
        out_path = os.path.join(out_dir, f"orig_mos_{args.testset}.json")
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"Saved to {out_path}")
        return

    # ---- wer / sim: evaluate generated audio ----
    if args.gen_wav_dir is None:
        parser.error("--gen_wav_dir is required for wer/sim")

    metadata = load_pair_metadata(args.gen_wav_dir)

    if args.eval_type == "wer":
        # Load ground truth texts
        if args.testset is None:
            # Try to infer from gen_wav_dir path
            for name in ["sap_dev", "sap_train", "l2arctic"]:
                if name in args.gen_wav_dir:
                    args.testset = name
                    break
            if args.testset is None:
                parser.error("Cannot infer --testset from gen_wav_dir. Please specify --testset.")

        if args.testset.startswith("sap"):
            if args.data_root is None:
                parser.error("--data_root required for SAP WER eval")
            manifest = args.manifest
            if manifest is None:
                split = "Dev" if "dev" in args.testset else "Train"
                manifest = os.path.join(args.data_root, "manifest", f"{split}.csv")
            gt_texts = load_sap_gt_texts(manifest, args.data_root,
                                          max_pairs_per_speaker=args.max_pairs_per_speaker)
        elif args.testset == "l2arctic":
            if args.data_root is None:
                parser.error("--data_root required for L2-ARCTIC WER eval")
            gt_texts = load_l2arctic_gt_texts(args.data_root,
                                               max_pairs_per_speaker=args.max_pairs_per_speaker)

        test_set, pair_ids = build_test_set_with_text(args.gen_wav_dir, metadata, gt_texts)
        print(f"Evaluating WER on {len(test_set)} generated files...")

        distributed = distribute_to_gpus(test_set, gpus)
        worker_args = [(gpu, chunk, args.asr_ckpt) for gpu, chunk in distributed]

        if len(gpus) == 1:
            all_results = [run_wer_worker(worker_args[0])]
        else:
            with Pool(len(gpus)) as pool:
                all_results = pool.map(run_wer_worker, worker_args)

        results = [r for batch in all_results for r in batch]
        summary = aggregate_wer_results(results, metadata)

        out_path = os.path.join(args.gen_wav_dir, "eval_wer_results.json")
        with open(out_path, "w") as f:
            json.dump({"summary": summary, "per_sample": results}, f, indent=2, ensure_ascii=False)
        print(f"Saved to {out_path}")

    elif args.eval_type == "sim":
        test_set, pair_ids = build_test_set(args.gen_wav_dir, metadata)
        print(f"Evaluating SIM on {len(test_set)} generated files...")

        distributed = distribute_to_gpus(test_set, gpus)
        worker_args = [(gpu, chunk, args.sim_ckpt) for gpu, chunk in distributed]

        if len(gpus) == 1:
            all_results = [run_sim_worker(worker_args[0])]
        else:
            with Pool(len(gpus)) as pool:
                all_results = pool.map(run_sim_worker, worker_args)

        results = [r for batch in all_results for r in batch]
        summary = aggregate_sim_results(results, metadata)

        out_path = os.path.join(args.gen_wav_dir, "eval_sim_results.json")
        with open(out_path, "w") as f:
            json.dump({"summary": summary, "per_sample": results}, f, indent=2, ensure_ascii=False)
        print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()