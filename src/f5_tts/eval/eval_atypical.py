# Evaluate on the atypical-speech testsets (SAP dysarthric, L2-ARCTIC non-native) with per-group breakdowns
# (etiology for SAP, L1 for L2-ARCTIC). WER is corpus-level over all utterances; SIM and UTMOS are averaged.

import argparse
import ast
import json
import os
import sys


sys.path.append(os.getcwd())

import multiprocessing as mp
from collections import defaultdict

import numpy as np
import torch
import torchaudio
from tqdm import tqdm

from f5_tts.eval.utils_atypical import ATYPICAL_TESTSETS, cross_utterance_pairs, load_utterances
from f5_tts.eval.utils_eval import run_asr_wer, run_sim


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e",
        "--eval_task",
        type=str,
        default="wer",
        choices=["wer", "sim", "orig_wer", "orig_sim", "orig_mos"],
        help="wer/sim: generated wavs vs. target text / reference speaker; orig_*: the original recordings",
    )
    parser.add_argument("-t", "--testset", type=str, required=True, choices=ATYPICAL_TESTSETS)
    parser.add_argument("-d", "--data_root", type=str, default=None, help="dataset root (needed for orig_* tasks)")
    parser.add_argument(
        "-m", "--manifest", type=str, default=None, help="SAP manifest csv, default manifest/{Dev,Train}.csv"
    )
    parser.add_argument("-g", "--gen_wav_dir", type=str, default=None, help="generated wavs (wer / sim)")
    parser.add_argument(
        "-n", "--gpu_nums", type=str, default="8", help="Number of GPUs to use (e.g., 8) or GPU list (e.g., [0,1,2,3])"
    )
    parser.add_argument(
        "--max_pairs_per_speaker", type=int, default=None, help="orig_sim: same-speaker pairs per speaker"
    )
    parser.add_argument("--wavlm_ckpt", type=str, default="../checkpoints/UniSpeech/wavlm_large_finetune.pth")
    parser.add_argument("--local", action="store_true", help="Use local custom checkpoint directory")
    return parser.parse_args()


def parse_gpu_nums(gpu_nums_str):
    if gpu_nums_str.startswith("[") and gpu_nums_str.endswith("]"):
        return ast.literal_eval(gpu_nums_str)
    return list(range(int(gpu_nums_str)))


def split_jobs(test_set, gpus):
    if len(gpus) == 1:
        return [(gpus[0], test_set)]
    per_job = len(test_set) // len(gpus) + 1
    return [(gpu, test_set[i * per_job : (i + 1) * per_job]) for i, gpu in enumerate(gpus)]


def run_pool(fn, jobs):
    if len(jobs) == 1:
        return fn(jobs[0])
    with mp.Pool(processes=len(jobs)) as pool:
        return [r for results in pool.map(fn, jobs) for r in results]


def run_utmos(args):
    rank, test_set = args  # (wav, _, _) items
    device = f"cuda:{rank}"
    predictor = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True).to(device)
    results = []
    for wav_path, _, _ in tqdm(test_set):
        wav, sr = torchaudio.load(wav_path)
        wav = wav.mean(0, keepdim=True) if wav.shape[0] > 1 else wav
        with torch.no_grad():
            score = predictor(wav.to(device), sr).item()
        results.append({"wav": os.path.splitext(os.path.basename(wav_path))[0], "utmos": score})
    return results


def corpus_wer(results):
    from jiwer import process_words

    return process_words([r["truth_norm"] for r in results], [r["hypo_norm"] for r in results]).wer


def summarize(metric, results, group_of):
    """Overall and per-group metric. `results` carry `wav` (utt or pair id) and the metric field."""
    aggregate = corpus_wer if metric == "wer" else (lambda rs: float(np.mean([r[metric] for r in rs])))
    by_group = defaultdict(list)
    for r in results:
        by_group[group_of.get(r["wav"], "unknown")].append(r)
    summary = {metric: aggregate(results), "num_samples": len(results), "per_group": {}}
    print(f"\nTotal {len(results)} samples")
    print(f"{metric.upper()}: {summary[metric]:.4f}")
    for group in sorted(by_group):
        summary["per_group"][group] = {metric: aggregate(by_group[group]), "num_samples": len(by_group[group])}
        print(f"  {group:20s} {metric.upper()}: {summary['per_group'][group][metric]:.4f}  (n={len(by_group[group])})")
    return summary


def main():
    args = get_args()
    gpus = parse_gpu_nums(args.gpu_nums)
    asr_ckpt_dir = "../checkpoints/openai/whisper-large-v3" if args.local else ""

    if args.eval_task in ["wer", "sim"]:
        assert args.gen_wav_dir, "--gen_wav_dir is required"
        with open(os.path.join(args.gen_wav_dir, "_pair_metadata.json")) as f:
            metadata = json.load(f)  # written by eval_infer_batch.py
        pair_ids = sorted(
            f[: -len(".wav")]
            for f in os.listdir(args.gen_wav_dir)
            if f.endswith(".wav") and f[: -len(".wav")] in metadata
        )
        # (gen_wav, ref_wav, target text), the format of utils_eval.run_asr_wer / run_sim
        test_set = [
            (os.path.join(args.gen_wav_dir, f"{p}.wav"), metadata[p]["ref_audio"], metadata[p]["tgt_text"])
            for p in pair_ids
        ]
        group_of = {p: metadata[p]["group"] for p in pair_ids}
        out_dir, task = args.gen_wav_dir, args.eval_task
    else:
        assert args.data_root, "--data_root is required for orig_* tasks"
        spk2utts = load_utterances(args.testset, args.data_root, args.manifest)
        if args.eval_task == "orig_sim":  # same-speaker pairs of original recordings
            metainfo, metadata = cross_utterance_pairs(spk2utts, args.max_pairs_per_speaker)
            test_set = [(ref_wav, tgt_wav, "") for _, _, ref_wav, _, tgt_wav in metainfo]
            group_of = {
                os.path.splitext(os.path.basename(ref_wav))[0]: metadata[p]["group"] for p, _, ref_wav, _, _ in metainfo
            }
        else:
            utts = [u for spk_utts in spk2utts.values() for u in spk_utts]
            test_set = [(u["audio"], u["audio"], u["text"]) for u in utts]
            group_of = {os.path.splitext(os.path.basename(u["audio"]))[0]: u["group"] for u in utts}
        out_dir, task = args.data_root, args.eval_task[len("orig_") :]

    if task == "wer":
        results = run_pool(run_asr_wer, [(rank, "en", sub, asr_ckpt_dir) for rank, sub in split_jobs(test_set, gpus)])
    elif task == "sim":
        results = run_pool(run_sim, [(rank, sub, args.wavlm_ckpt) for rank, sub in split_jobs(test_set, gpus)])
    elif task == "mos":
        results = run_pool(run_utmos, [(rank, sub) for rank, sub in split_jobs(test_set, gpus)])

    metric = "utmos" if task == "mos" else task
    summary = summarize(metric, results, group_of)

    result_path = os.path.join(out_dir, f"_{args.eval_task}_results.json")
    with open(result_path, "w") as f:
        json.dump({"summary": summary, "per_sample": results}, f, indent=2, ensure_ascii=False)
    print(f"{metric.upper()} results saved to {result_path}")


if __name__ == "__main__":
    main()
