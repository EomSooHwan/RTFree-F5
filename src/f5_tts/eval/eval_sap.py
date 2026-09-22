"""
Evaluate generated SAP dysarthric reconstruction outputs.
Metrics: WER, CER (via Whisper), SIM (via WavLM-TDNN or ECAPA-TDNN), UTMOS.

Usage:
  # WER + CER
  python src/f5_tts/eval/eval_sap.py -e wer -g <gen_wav_dir> -m <manifest_csv> -d <sap_data_root> -n "[0,1,2,3]"
  # Speaker similarity
  python src/f5_tts/eval/eval_sap.py -e sim -g <gen_wav_dir> -m <manifest_csv> -d <sap_data_root> -n "[0,1,2,3]"
  # UTMOS (can also use the existing eval_utmos.py)
  python src/f5_tts/eval/eval_utmos.py --audio_dir <gen_wav_dir>
"""

import argparse
import csv
import json
import os
from collections import defaultdict

import numpy as np


def load_sap_pairs(manifest_csv, sap_data_root):
    """
    Load cross-utterance pairs from SAP manifest.
    Returns dict: pair_id -> {ref_audio, ref_text, tgt_audio, tgt_text, speaker, etiology}
    
    Pair IDs follow the convention: {ref_utt_id}__to__{tgt_utt_id}
    which matches what get_sap_metainfo() produces.
    """
    # First build speaker -> utterances mapping
    spk2utts = defaultdict(list)
    with open(manifest_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            dur = float(row["duration"])
            if dur < 0.3 or dur > 30:
                continue
            spk2utts[row["speaker"]].append({
                "id": row["id"],
                "audio": os.path.join(sap_data_root, row["audio_filepath"]),
                "text": row["norm_text_without_disfluency"],
                "speaker": row["speaker"],
                "etiology": row.get("etiology", "unknown"),
            })

    # Build cross-utterance pairs (same logic as get_sap_metainfo)
    pairs = {}
    for spk, utts in spk2utts.items():
        if len(utts) < 2:
            continue
        for i, ref_utt in enumerate(utts):
            tgt_utt = utts[(i + 1) % len(utts)]
            pair_id = f"{ref_utt['id']}__to__{tgt_utt['id']}"
            pairs[pair_id] = {
                "ref_audio": ref_utt["audio"],
                "ref_text": ref_utt["text"],
                "tgt_audio": tgt_utt["audio"],
                "tgt_text": tgt_utt["text"],
                "speaker": spk,
                "etiology": ref_utt["etiology"],
            }
    return pairs


def eval_wer_cer(gen_wav_dir, pairs, gpu_nums):
    """Compute WER and CER using Whisper on generated audio."""
    import torch
    from faster_whisper import WhisperModel

    # Pick first GPU
    if isinstance(gpu_nums, str):
        gpu_nums = eval(gpu_nums)
    device_id = gpu_nums[0] if gpu_nums else 0

    model = WhisperModel("large-v3", device=f"cuda:{device_id}", compute_type="float16")

    results = []
    from jiwer import wer as compute_wer, cer as compute_cer

    gen_files = sorted([f for f in os.listdir(gen_wav_dir) if f.endswith(".wav")])
    print(f"Evaluating WER/CER on {len(gen_files)} files...")

    all_refs = []
    all_hyps = []

    for fname in gen_files:
        pair_id = fname.replace(".wav", "")
        if pair_id not in pairs:
            print(f"  Warning: {pair_id} not found in manifest pairs, skipping")
            continue

        ref_text = pairs[pair_id]["tgt_text"].lower().strip()
        gen_path = os.path.join(gen_wav_dir, fname)

        segments, info = model.transcribe(gen_path, language="en")
        hyp_text = " ".join([seg.text for seg in segments]).lower().strip()

        all_refs.append(ref_text)
        all_hyps.append(hyp_text)

        results.append({
            "pair_id": pair_id,
            "ref_text": ref_text,
            "hyp_text": hyp_text,
            "speaker": pairs[pair_id]["speaker"],
            "etiology": pairs[pair_id]["etiology"],
        })

    # Compute corpus-level metrics
    corpus_wer = compute_wer(all_refs, all_hyps)
    corpus_cer = compute_cer(all_refs, all_hyps)

    # Per-etiology breakdown
    etiology_groups = defaultdict(lambda: {"refs": [], "hyps": []})
    for r in results:
        etiology_groups[r["etiology"]]["refs"].append(r["ref_text"])
        etiology_groups[r["etiology"]]["hyps"].append(r["hyp_text"])

    print(f"\n{'='*60}")
    print(f"Overall WER: {corpus_wer:.4f} ({corpus_wer*100:.2f}%)")
    print(f"Overall CER: {corpus_cer:.4f} ({corpus_cer*100:.2f}%)")
    print(f"Total pairs evaluated: {len(results)}")
    print(f"\nPer-etiology breakdown:")
    for etiology, group in sorted(etiology_groups.items()):
        e_wer = compute_wer(group["refs"], group["hyps"])
        e_cer = compute_cer(group["refs"], group["hyps"])
        print(f"  {etiology}: WER={e_wer*100:.2f}%, CER={e_cer*100:.2f}% (n={len(group['refs'])})")
    print(f"{'='*60}\n")

    # Save detailed results
    out_path = os.path.join(gen_wav_dir, "sap_wer_cer_results.json")
    with open(out_path, "w") as f:
        json.dump({
            "corpus_wer": corpus_wer,
            "corpus_cer": corpus_cer,
            "num_samples": len(results),
            "per_sample": results,
        }, f, indent=2)
    print(f"Saved detailed results to {out_path}")


def eval_sim(gen_wav_dir, pairs, gpu_nums):
    """Compute speaker similarity between generated audio and original dysarthric reference."""
    import torch
    import torchaudio
    from speechbrain.inference.speaker import EncoderClassifier

    if isinstance(gpu_nums, str):
        gpu_nums = eval(gpu_nums)
    device_id = gpu_nums[0] if gpu_nums else 0
    device = f"cuda:{device_id}"

    # Load speaker verification model
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": device},
    )

    def get_embedding(audio_path):
        sig, sr = torchaudio.load(audio_path)
        if sig.shape[0] > 1:
            sig = sig.mean(0, keepdim=True)
        if sr != 16000:
            sig = torchaudio.functional.resample(sig, sr, 16000)
        with torch.no_grad():
            emb = classifier.encode_batch(sig.to(device))
        return emb.squeeze().cpu()

    gen_files = sorted([f for f in os.listdir(gen_wav_dir) if f.endswith(".wav")])
    print(f"Evaluating speaker similarity on {len(gen_files)} files...")

    results = []
    all_sims = []

    for fname in gen_files:
        pair_id = fname.replace(".wav", "")
        if pair_id not in pairs:
            continue

        gen_path = os.path.join(gen_wav_dir, fname)
        ref_path = pairs[pair_id]["ref_audio"]

        if not os.path.exists(ref_path):
            print(f"  Warning: ref audio not found: {ref_path}")
            continue

        gen_emb = get_embedding(gen_path)
        ref_emb = get_embedding(ref_path)

        sim = torch.nn.functional.cosine_similarity(gen_emb, ref_emb, dim=0).item()
        all_sims.append(sim)

        results.append({
            "pair_id": pair_id,
            "similarity": sim,
            "speaker": pairs[pair_id]["speaker"],
            "etiology": pairs[pair_id]["etiology"],
        })

    mean_sim = np.mean(all_sims)

    # Per-etiology breakdown
    etiology_groups = defaultdict(list)
    for r in results:
        etiology_groups[r["etiology"]].append(r["similarity"])

    print(f"\n{'='*60}")
    print(f"Overall Speaker Similarity: {mean_sim:.4f}")
    print(f"Total pairs evaluated: {len(results)}")
    print(f"\nPer-etiology breakdown:")
    for etiology, sims in sorted(etiology_groups.items()):
        print(f"  {etiology}: SIM={np.mean(sims):.4f} ± {np.std(sims):.4f} (n={len(sims)})")
    print(f"{'='*60}\n")

    out_path = os.path.join(gen_wav_dir, "sap_sim_results.json")
    with open(out_path, "w") as f:
        json.dump({
            "mean_similarity": mean_sim,
            "num_samples": len(results),
            "per_sample": results,
        }, f, indent=2)
    print(f"Saved detailed results to {out_path}")


def eval_orig_wer(manifest_csv, sap_data_root, gpu_nums):
    """
    Bonus: compute WER/CER of original dysarthric audio.
    This quantifies how bad ASR is on dysarthric speech (motivates the paper).
    """
    import csv
    from faster_whisper import WhisperModel
    from jiwer import wer as compute_wer, cer as compute_cer

    if isinstance(gpu_nums, str):
        gpu_nums = eval(gpu_nums)
    device_id = gpu_nums[0] if gpu_nums else 0

    model = WhisperModel("large-v3", device=f"cuda:{device_id}", compute_type="float16")

    all_refs, all_hyps = [], []
    etiology_groups = defaultdict(lambda: {"refs": [], "hyps": []})

    with open(manifest_csv) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"Computing WER on {len(rows)} original dysarthric utterances...")
    for row in rows:
        dur = float(row["duration"])
        if dur < 0.3 or dur > 30:
            continue

        audio_path = os.path.join(sap_data_root, row["audio_filepath"])
        ref_text = row["norm_text_without_disfluency"].lower().strip()

        segments, _ = model.transcribe(audio_path, language="en")
        hyp_text = " ".join([seg.text for seg in segments]).lower().strip()

        all_refs.append(ref_text)
        all_hyps.append(hyp_text)
        etiology_groups[row.get("etiology", "unknown")]["refs"].append(ref_text)
        etiology_groups[row.get("etiology", "unknown")]["hyps"].append(hyp_text)

    print(f"\n{'='*60}")
    print(f"ORIGINAL DYSARTHRIC SPEECH ASR PERFORMANCE")
    print(f"Overall WER: {compute_wer(all_refs, all_hyps)*100:.2f}%")
    print(f"Overall CER: {compute_cer(all_refs, all_hyps)*100:.2f}%")
    print(f"\nPer-etiology:")
    for etiology, group in sorted(etiology_groups.items()):
        print(f"  {etiology}: WER={compute_wer(group['refs'], group['hyps'])*100:.2f}%, "
              f"CER={compute_cer(group['refs'], group['hyps'])*100:.2f}% (n={len(group['refs'])})")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate SAP dysarthric reconstruction")
    parser.add_argument("-e", "--eval_type", required=True, choices=["wer", "sim", "orig_wer"],
                        help="wer: WER+CER on generated, sim: speaker similarity, orig_wer: WER of original audio")
    parser.add_argument("-g", "--gen_wav_dir", type=str, default=None,
                        help="Directory of generated wav files (required for wer/sim)")
    parser.add_argument("-m", "--manifest", type=str, required=True,
                        help="Path to SAP manifest CSV (e.g. manifest/Dev.csv)")
    parser.add_argument("-d", "--data_root", type=str, required=True,
                        help="Root of SAP dataset (SpeechAccessibility_Research_Release/)")
    parser.add_argument("-n", "--gpu_nums", type=str, default="[0]")

    args = parser.parse_args()

    if args.eval_type in ["wer", "sim"] and args.gen_wav_dir is None:
        parser.error("--gen_wav_dir is required for wer/sim evaluation")

    if args.eval_type == "orig_wer":
        eval_orig_wer(args.manifest, args.data_root, args.gpu_nums)
        return

    pairs = load_sap_pairs(args.manifest, args.data_root)
    print(f"Loaded {len(pairs)} cross-utterance pairs from manifest")

    if args.eval_type == "wer":
        eval_wer_cer(args.gen_wav_dir, pairs, args.gpu_nums)
    elif args.eval_type == "sim":
        eval_sim(args.gen_wav_dir, pairs, args.gpu_nums)


if __name__ == "__main__":
    main()