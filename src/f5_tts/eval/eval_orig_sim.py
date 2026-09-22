"""
Evaluate speaker similarity (SIM) on original audio pairs from the same speaker.

This provides a baseline/ceiling for what SIM scores to expect when comparing
same-speaker utterances in SAP (dysarthric) and L2-ARCTIC (accented) datasets.

Usage:
    python eval_orig_sim.py -t sap_dev -d /path/to/SAP -m /path/to/manifest.csv -n "[0,1,2,3]"
    python eval_orig_sim.py -t l2arctic -d /path/to/L2_ARCTIC -n "[0,1,2,3]"
"""

import argparse
import csv
import json
import os
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from tqdm import tqdm

# Assuming this script is run from the F5-TTS project root
import sys
sys.path.insert(0, os.getcwd())

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
# Build same-speaker pairs for SIM evaluation
# =========================================================================

def build_sap_pairs(manifest_csv, sap_data_root, max_pairs_per_speaker=None):
    """
    Build same-speaker pairs from SAP manifest for SIM baseline evaluation.
    
    Returns:
        pairs: list of (audio1_path, audio2_path, speaker, etiology)
    """
    spk2utts = defaultdict(list)
    spk2etiology = {}
    
    with open(manifest_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            dur = float(row["duration"])
            if dur < 0.3 or dur > 30:
                continue
            spk = row["speaker"]
            spk2utts[spk].append(os.path.join(sap_data_root, row["audio_filepath"]))
            spk2etiology[spk] = row.get("etiology", "unknown")
    
    pairs = []
    for spk, utts in spk2utts.items():
        if len(utts) < 2:
            continue
        
        etiology = spk2etiology[spk]
        count = 0
        
        for i in range(len(utts)):
            if max_pairs_per_speaker is not None and count >= max_pairs_per_speaker:
                break
            
            # Pair with next utterance (circular)
            j = (i + 1) % len(utts)
            pairs.append((utts[i], utts[j], spk, etiology))
            count += 1
    
    print(f"SAP: Built {len(pairs)} same-speaker pairs from {len(spk2utts)} speakers")
    return pairs


def build_l2arctic_pairs(data_root, max_pairs_per_speaker=None):
    """
    Build same-speaker pairs from L2-ARCTIC for SIM baseline evaluation.
    
    Returns:
        pairs: list of (audio1_path, audio2_path, speaker, l1)
    """
    pairs = []
    
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
        
        # Collect valid utterances
        utts = []
        wav_files = sorted([f for f in os.listdir(wav_dir) if f.endswith(".wav")])
        
        for wav_file in wav_files:
            utt_id = wav_file.replace(".wav", "")
            txt_path = os.path.join(txt_dir, utt_id + ".txt")
            
            if not os.path.exists(txt_path):
                continue
            
            wav_path = os.path.join(wav_dir, wav_file)
            
            # Duration filter
            try:
                info = torchaudio.info(wav_path)
                dur = info.num_frames / info.sample_rate
                if dur < 0.5 or dur > 30:
                    continue
            except Exception:
                continue
            
            utts.append(wav_path)
        
        if len(utts) < 2:
            continue
        
        l1 = L2ARCTIC_SPEAKER_L1.get(spk, "Unknown")
        count = 0
        
        for i in range(len(utts)):
            if max_pairs_per_speaker is not None and count >= max_pairs_per_speaker:
                break
            
            j = (i + 1) % len(utts)
            pairs.append((utts[i], utts[j], spk, l1))
            count += 1
    
    print(f"L2-ARCTIC: Built {len(pairs)} same-speaker pairs from {len(speaker_dirs)} speakers")
    return pairs


# =========================================================================
# SIM computation worker
# =========================================================================

def run_sim_worker(args):
    """
    Compute SIM for a batch of pairs on a single GPU.
    
    args: (rank, pairs, sim_ckpt_path)
    pairs: list of (audio1_path, audio2_path, speaker, group)
    """
    rank, pairs, sim_ckpt_path = args
    device = f"cuda:{rank}"
    
    # Load model
    model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type="wavlm_large", config_path=None)
    state_dict = torch.load(sim_ckpt_path, weights_only=True, map_location="cpu")
    model.load_state_dict(state_dict["model"], strict=False)
    model = model.to(device).eval()
    
    results = []
    
    for audio1_path, audio2_path, speaker, group in tqdm(pairs, desc=f"SIM [GPU {rank}]"):
        try:
            wav1, sr1 = torchaudio.load(audio1_path)
            wav2, sr2 = torchaudio.load(audio2_path)
        except Exception as e:
            print(f"  Warning: Failed to load audio pair: {e}")
            continue
        
        # Ensure mono
        if wav1.shape[0] > 1:
            wav1 = wav1.mean(0, keepdim=True)
        if wav2.shape[0] > 1:
            wav2 = wav2.mean(0, keepdim=True)
        
        wav1 = wav1.to(device)
        wav2 = wav2.to(device)
        
        # Resample to 16kHz if needed
        if sr1 != 16000:
            wav1 = torchaudio.transforms.Resample(sr1, 16000).to(device)(wav1)
        if sr2 != 16000:
            wav2 = torchaudio.transforms.Resample(sr2, 16000).to(device)(wav2)
        
        with torch.no_grad():
            emb1 = model(wav1)
            emb2 = model(wav2)
        
        sim = F.cosine_similarity(emb1, emb2)[0].item()
        
        results.append({
            "audio1": Path(audio1_path).stem,
            "audio2": Path(audio2_path).stem,
            "speaker": speaker,
            "group": group,
            "sim": sim,
        })
    
    return results


# =========================================================================
# Aggregation and reporting
# =========================================================================

def aggregate_results(results, group_type="group"):
    """Compute corpus-level and per-group SIM statistics."""
    all_sims = [r["sim"] for r in results]
    
    groups = defaultdict(list)
    speakers = defaultdict(list)
    
    for r in results:
        groups[r["group"]].append(r["sim"])
        speakers[r["speaker"]].append(r["sim"])
    
    mean_sim = np.mean(all_sims)
    std_sim = np.std(all_sims)
    
    print(f"\n{'='*70}")
    print(f"  ORIGINAL SAME-SPEAKER SIM (Baseline)")
    print(f"  SIM: {mean_sim:.4f} ± {std_sim:.4f}    (n={len(all_sims)})")
    print(f"{'='*70}")
    
    # Per-group breakdown
    if groups:
        print(f"\n  Per-{group_type} breakdown:")
        group_stats = {}
        for group_name in sorted(groups.keys()):
            sims = groups[group_name]
            g_mean = np.mean(sims)
            g_std = np.std(sims)
            print(f"    {group_name:20s}  SIM={g_mean:.4f} ± {g_std:.4f}  (n={len(sims)})")
            group_stats[group_name] = {"mean": g_mean, "std": g_std, "n": len(sims)}
    
    # Per-speaker statistics (summary)
    speaker_means = [np.mean(sims) for sims in speakers.values()]
    print(f"\n  Per-speaker SIM distribution:")
    print(f"    Mean of speaker means: {np.mean(speaker_means):.4f}")
    print(f"    Std of speaker means:  {np.std(speaker_means):.4f}")
    print(f"    Min speaker mean:      {np.min(speaker_means):.4f}")
    print(f"    Max speaker mean:      {np.max(speaker_means):.4f}")
    print()
    
    return {
        "mean_sim": float(mean_sim),
        "std_sim": float(std_sim),
        "num_pairs": len(all_sims),
        "per_group": group_stats if groups else {},
        "num_speakers": len(speakers),
    }


def distribute_to_gpus(data, gpus):
    """Split data across GPUs."""
    n = len(gpus)
    if n == 1:
        return [(gpus[0], data)]
    
    chunk_size = len(data) // n + 1
    return [(gpus[i], data[i * chunk_size:(i + 1) * chunk_size]) for i in range(n)]


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate SIM on original same-speaker pairs (baseline)"
    )
    parser.add_argument("-t", "--testset", required=True,
                        choices=["sap_dev", "sap_train", "l2arctic"],
                        help="Dataset to evaluate")
    parser.add_argument("-d", "--data_root", required=True,
                        help="Dataset root directory")
    parser.add_argument("-m", "--manifest", type=str, default=None,
                        help="SAP manifest CSV path (required for SAP)")
    parser.add_argument("-n", "--gpu_nums", type=str, default="[0]",
                        help="GPU list, e.g. '[0,1,2,3]'")
    parser.add_argument("--sim_ckpt", type=str,
                        default="../checkpoints/UniSpeech/wavlm_large_finetune.pth",
                        help="Path to ECAPA-TDNN checkpoint")
    parser.add_argument("--max_pairs_per_speaker", type=int, default=10,
                        help="Maximum pairs per speaker (default: 10)")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Output JSON path (default: <data_root>/orig_sim_<testset>.json)")
    
    args = parser.parse_args()
    gpus = eval(args.gpu_nums)
    
    # Build pairs
    if args.testset.startswith("sap"):
        if args.manifest is None:
            split = "Dev" if "dev" in args.testset else "Train"
            args.manifest = os.path.join(args.data_root, "manifest", f"{split}.csv")
        
        pairs = build_sap_pairs(
            args.manifest, 
            args.data_root,
            max_pairs_per_speaker=args.max_pairs_per_speaker
        )
        group_type = "etiology"
        
    elif args.testset == "l2arctic":
        pairs = build_l2arctic_pairs(
            args.data_root,
            max_pairs_per_speaker=args.max_pairs_per_speaker
        )
        group_type = "L1"
    
    if not pairs:
        print("ERROR: No pairs found!")
        return
    
    # Check checkpoint exists
    if not os.path.exists(args.sim_ckpt):
        print(f"ERROR: SIM checkpoint not found: {args.sim_ckpt}")
        print("Please provide --sim_ckpt path to wavlm_large_finetune.pth")
        return
    
    print(f"\nEvaluating SIM on {len(pairs)} same-speaker pairs...")
    print(f"Using checkpoint: {args.sim_ckpt}")
    
    # Distribute across GPUs
    distributed = distribute_to_gpus(pairs, gpus)
    worker_args = [(gpu, chunk, args.sim_ckpt) for gpu, chunk in distributed]
    
    # Run evaluation
    if len(gpus) == 1:
        all_results = [run_sim_worker(worker_args[0])]
    else:
        with Pool(len(gpus)) as pool:
            all_results = pool.map(run_sim_worker, worker_args)
    
    # Flatten results
    results = [r for batch in all_results for r in batch]
    
    # Aggregate and print
    summary = aggregate_results(results, group_type)
    
    # Save
    output_path = args.output or os.path.join(args.data_root, f"orig_sim_{args.testset}.json")
    with open(output_path, "w") as f:
        json.dump({
            "summary": summary,
            "per_sample": results,
        }, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()