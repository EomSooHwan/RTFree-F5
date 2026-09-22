#!/usr/bin/env python3
"""
Debug script to diagnose why SIM is ~0.99 for atypical speech evaluation.

Checks:
1. Are gen_wav and ref_wav pointing to different files?
2. Are the audio durations plausible?
3. Are the audio waveforms actually different?
4. Sample a few pairs and compute SIM manually for verification
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torchaudio
import torch.nn.functional as F
from pathlib import Path

def load_metadata(gen_wav_dir):
    meta_path = os.path.join(gen_wav_dir, "_pair_metadata.json")
    if not os.path.exists(meta_path):
        print(f"ERROR: No metadata file at {meta_path}")
        sys.exit(1)
    with open(meta_path) as f:
        return json.load(f)

def analyze_paths(gen_wav_dir, metadata, max_samples=10):
    """Check if generated and reference paths are correctly different."""
    print("\n" + "="*70)
    print("PATH ANALYSIS")
    print("="*70)
    
    gen_files = sorted([f for f in os.listdir(gen_wav_dir) 
                        if f.endswith(".wav") and not f.startswith("_")])
    
    issues = []
    for i, fname in enumerate(gen_files[:max_samples]):
        pair_id = fname.replace(".wav", "")
        if pair_id not in metadata:
            print(f"  WARNING: {pair_id} not in metadata")
            continue
            
        gen_path = os.path.join(gen_wav_dir, fname)
        ref_path = metadata[pair_id]["ref_audio"]
        
        # Check if paths resolve to same file
        gen_real = os.path.realpath(gen_path)
        ref_real = os.path.realpath(ref_path)
        
        same_file = gen_real == ref_real
        if same_file:
            issues.append(pair_id)
            
        print(f"\n  Pair {i+1}: {pair_id}")
        print(f"    Gen: {gen_path}")
        print(f"    Ref: {ref_path}")
        print(f"    Same file? {'YES - BUG!' if same_file else 'No (correct)'}")
    
    if issues:
        print(f"\n  CRITICAL: {len(issues)} pairs point to same file!")
    else:
        print(f"\n  OK: All checked pairs point to different files")
    
    return len(issues) == 0

def analyze_durations(gen_wav_dir, metadata, max_samples=10):
    """Check if durations are plausible."""
    print("\n" + "="*70)
    print("DURATION ANALYSIS")
    print("="*70)
    
    gen_files = sorted([f for f in os.listdir(gen_wav_dir) 
                        if f.endswith(".wav") and not f.startswith("_")])
    
    suspiciously_similar = 0
    
    for i, fname in enumerate(gen_files[:max_samples]):
        pair_id = fname.replace(".wav", "")
        if pair_id not in metadata:
            continue
            
        gen_path = os.path.join(gen_wav_dir, fname)
        ref_path = metadata[pair_id]["ref_audio"]
        
        if not os.path.exists(ref_path):
            print(f"  WARNING: ref audio missing: {ref_path}")
            continue
        
        gen_info = torchaudio.info(gen_path)
        ref_info = torchaudio.info(ref_path)
        
        gen_dur = gen_info.num_frames / gen_info.sample_rate
        ref_dur = ref_info.num_frames / ref_info.sample_rate
        
        dur_diff = abs(gen_dur - ref_dur)
        similar = dur_diff < 0.1  # Less than 100ms difference
        
        if similar:
            suspiciously_similar += 1
        
        print(f"\n  Pair {i+1}: {pair_id}")
        print(f"    Gen duration: {gen_dur:.3f}s")
        print(f"    Ref duration: {ref_dur:.3f}s")
        print(f"    Difference: {dur_diff:.3f}s {'(SUSPICIOUS!)' if similar else ''}")
    
    if suspiciously_similar > max_samples * 0.5:
        print(f"\n  WARNING: {suspiciously_similar}/{max_samples} pairs have very similar durations")
        print("  This might indicate the model is copying the reference instead of generating new content")

def analyze_waveform_similarity(gen_wav_dir, metadata, max_samples=5):
    """Check if waveforms are actually different."""
    print("\n" + "="*70)
    print("WAVEFORM SIMILARITY ANALYSIS")
    print("="*70)
    
    gen_files = sorted([f for f in os.listdir(gen_wav_dir) 
                        if f.endswith(".wav") and not f.startswith("_")])
    
    for i, fname in enumerate(gen_files[:max_samples]):
        pair_id = fname.replace(".wav", "")
        if pair_id not in metadata:
            continue
            
        gen_path = os.path.join(gen_wav_dir, fname)
        ref_path = metadata[pair_id]["ref_audio"]
        
        if not os.path.exists(ref_path):
            continue
        
        gen_wav, gen_sr = torchaudio.load(gen_path)
        ref_wav, ref_sr = torchaudio.load(ref_path)
        
        # Resample to same rate
        target_sr = 16000
        if gen_sr != target_sr:
            gen_wav = torchaudio.functional.resample(gen_wav, gen_sr, target_sr)
        if ref_sr != target_sr:
            ref_wav = torchaudio.functional.resample(ref_wav, ref_sr, target_sr)
        
        # Ensure mono
        gen_wav = gen_wav.mean(0) if gen_wav.dim() > 1 and gen_wav.shape[0] > 1 else gen_wav.squeeze(0)
        ref_wav = ref_wav.mean(0) if ref_wav.dim() > 1 and ref_wav.shape[0] > 1 else ref_wav.squeeze(0)
        
        # Truncate to shorter length
        min_len = min(len(gen_wav), len(ref_wav))
        gen_wav = gen_wav[:min_len]
        ref_wav = ref_wav[:min_len]
        
        # Compute correlation
        correlation = torch.corrcoef(torch.stack([gen_wav, ref_wav]))[0, 1].item()
        
        # Compute MSE
        mse = F.mse_loss(gen_wav, ref_wav).item()
        
        print(f"\n  Pair {i+1}: {pair_id}")
        print(f"    Waveform correlation: {correlation:.4f}")
        print(f"    Waveform MSE: {mse:.6f}")
        
        if correlation > 0.95:
            print("    >>> HIGH CORRELATION: Waveforms are nearly identical!")
        elif correlation > 0.5:
            print("    >>> Moderate correlation (could be speaker similarity)")
        else:
            print("    >>> Low correlation (waveforms are different - expected)")

def verify_sim_calculation(gen_wav_dir, metadata, max_samples=5):
    """Manually compute SIM on a few samples to verify."""
    print("\n" + "="*70)
    print("MANUAL SIM VERIFICATION")
    print("="*70)
    
    try:
        # Try to import the ECAPA model
        sys.path.insert(0, os.getcwd())
        from f5_tts.eval.ecapa_tdnn import ECAPA_TDNN_SMALL
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type="wavlm_large", config_path=None)
        model = model.to(device).eval()
        
        gen_files = sorted([f for f in os.listdir(gen_wav_dir) 
                            if f.endswith(".wav") and not f.startswith("_")])
        
        sims = []
        for i, fname in enumerate(gen_files[:max_samples]):
            pair_id = fname.replace(".wav", "")
            if pair_id not in metadata:
                continue
                
            gen_path = os.path.join(gen_wav_dir, fname)
            ref_path = metadata[pair_id]["ref_audio"]
            
            if not os.path.exists(ref_path):
                continue
            
            wav1, sr1 = torchaudio.load(gen_path)
            wav2, sr2 = torchaudio.load(ref_path)
            
            wav1 = wav1.to(device)
            wav2 = wav2.to(device)
            
            if sr1 != 16000:
                wav1 = torchaudio.transforms.Resample(sr1, 16000).to(device)(wav1)
            if sr2 != 16000:
                wav2 = torchaudio.transforms.Resample(sr2, 16000).to(device)(wav2)
            
            with torch.no_grad():
                emb1 = model(wav1)
                emb2 = model(wav2)
            
            sim = F.cosine_similarity(emb1, emb2)[0].item()
            sims.append(sim)
            
            print(f"\n  Pair {i+1}: {pair_id}")
            print(f"    SIM: {sim:.4f}")
            
        if sims:
            print(f"\n  Mean SIM: {np.mean(sims):.4f}")
            print(f"  This should match the eval_atypical.py output")
            
    except ImportError as e:
        print(f"  Could not import ECAPA model: {e}")
        print("  Run this script from the F5-TTS project root directory")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("gen_wav_dir", help="Directory containing generated wav files")
    parser.add_argument("--max-samples", type=int, default=10)
    args = parser.parse_args()
    
    print(f"Analyzing: {args.gen_wav_dir}")
    
    metadata = load_metadata(args.gen_wav_dir)
    print(f"Loaded metadata for {len(metadata)} pairs")
    
    # Run analyses
    paths_ok = analyze_paths(args.gen_wav_dir, metadata, args.max_samples)
    analyze_durations(args.gen_wav_dir, metadata, args.max_samples)
    analyze_waveform_similarity(args.gen_wav_dir, metadata, min(5, args.max_samples))
    
    if paths_ok:
        print("\n" + "="*70)
        print("DIAGNOSIS")
        print("="*70)
        print("""
If paths are correct but SIM is still ~0.99, likely causes:

1. MODEL OUTPUT ISSUE: The model might be copying the reference audio 
   instead of generating new content. Check:
   - Listen to a few generated files
   - Compare gen audio content to ref audio content
   - The text/content should be DIFFERENT even if voice is similar

2. MEL EXTRACTION BUG: In inference, this line extracts generated portion:
   gen = gen[ref_mel_lens[i]:total_mel_lens[i], :]
   
   If ref_mel_lens or total_mel_lens are wrong, you might be extracting
   the reference portion instead of the generated portion.

3. DEBUGGING SUGGESTION: Add this to eval_infer_batch_atypical.py:
   print(f"ref_mel_lens[{i}]={ref_mel_lens[i]}, total={total_mel_lens[i]}")
   
   The generated portion should be from ref_mel_lens to total_mel_lens.
   If ref_mel_lens == 0, you're including the reference in the output.
""")

if __name__ == "__main__":
    main()