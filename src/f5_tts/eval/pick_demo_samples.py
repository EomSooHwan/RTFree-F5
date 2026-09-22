# Pick presentation samples from the atypical-speech evaluation outputs (SAP / L2-ARCTIC).
#
# For every cross-utterance pair that both systems generated, the script collects the ground-truth target
# recording, the F5-TTS output (oracle transcript) and the RTFree-F5 output, ranks the pairs by how clearly they
# show the paper's point, and copies the top-k triples (loudness-matched, 24 kHz) to <out_dir>/<rank>_<pair_id>/.
#
# Ranking (higher is better):
#   score = WER(F5-TTS oracle) - WER(RTFree-F5)            intelligibility gap, the main story
#         + w_sim   * SIM(RTFree-F5)                        identity still preserved
#         + w_utmos * UTMOS(RTFree-F5) / 5                  natural output
# subject to WER(RTFree-F5) <= --max_rtfree_wer, --min_words <= #words, --min_secs <= target duration <= --max_secs.
# Per-sample metrics are read from the _wer/_sim/_utmos result files in each generation directory (old
# `eval_wer_results.json` names are accepted too); missing WER / SIM are computed on the fly on --gpu.
#
# Usage:
#   python src/f5_tts/eval/pick_demo_samples.py -t sap_dev -d <SAP_ROOT> \
#       --rtfree_dir results/RTFree_F5_model_last/sap_dev_rtfree/seed0_euler_nfe32_vocos_ss-1_cfg2.0_speed1.0_gt-dur \
#       --oracle_dir results/F5TTS_v1_Base_1250000/sap_dev_oracle/seed0_euler_nfe32_vocos_ss-1_cfg2.0_speed1.0_gt-dur \
#       -o demo/sap -k 5
# Listen to the top candidates before choosing; the ranking only pre-selects.

import argparse
import json
import os
import sys


sys.path.append(os.getcwd())

import soundfile as sf
import torchaudio

from f5_tts.eval.utils_atypical import ATYPICAL_TESTSETS, cross_utterance_pairs, load_utterances


TARGET_SR = 24000
TARGET_RMS = 0.1  # same loudness normalization as F5-TTS inference


def load_per_sample(gen_dir, names, key):
    """{pair_id: value} from the first existing result file among `names` (json with per_sample, or jsonl)."""
    for name in names:
        path = os.path.join(gen_dir, name)
        if not os.path.exists(path):
            continue
        if name.endswith(".jsonl"):
            rows = [json.loads(line) for line in open(path) if line.startswith("{")]
        else:
            rows = json.load(open(path))["per_sample"]
        return {r["wav"]: r for r in rows if key in r}
    return {}


def compute_missing(gen_dir, pairs, metric, gpu, wavlm_ckpt):
    """WER / SIM for the pairs of `gen_dir` lacking a result file."""
    from f5_tts.eval.utils_eval import run_asr_wer, run_sim

    test_set = [(os.path.join(gen_dir, f"{p}.wav"), m["ref_audio"], m["tgt_text"]) for p, m in pairs.items()]
    print(f"{metric.upper()} results not found in {gen_dir}, computing for {len(test_set)} samples on GPU {gpu}")
    results = run_asr_wer((gpu, "en", test_set, "")) if metric == "wer" else run_sim((gpu, test_set, wavlm_ckpt))
    return {r["wav"]: r for r in results}


def load_wav(path):
    wav, sr = torchaudio.load(path)
    wav = wav.mean(0) if wav.shape[0] > 1 else wav[0]
    if sr != TARGET_SR:
        wav = torchaudio.functional.resample(wav, sr, TARGET_SR)
    return wav


def export(src, dst, normalize):
    wav = load_wav(src)
    if normalize:
        wav = wav * TARGET_RMS / (wav.pow(2).mean().sqrt() + 1e-8)
        wav = wav / max(1.0, wav.abs().max())  # avoid clipping
    sf.write(dst, wav.numpy(), TARGET_SR)


def main():
    parser = argparse.ArgumentParser(description="Pick presentation samples (ground truth / F5-TTS oracle / RTFree-F5)")
    parser.add_argument("-t", "--testset", required=True, choices=ATYPICAL_TESTSETS)
    parser.add_argument("-d", "--data_root", required=True, help="SAP release dir or L2-ARCTIC root")
    parser.add_argument("-m", "--manifest", default=None, help="SAP manifest csv, default manifest/{Dev,Train}.csv")
    parser.add_argument("--rtfree_dir", required=True, help="generated wavs of RTFree-F5 (mode rtfree)")
    parser.add_argument("--oracle_dir", required=True, help="generated wavs of F5-TTS with oracle transcripts")
    parser.add_argument("-o", "--out_dir", required=True)
    parser.add_argument("-k", "--top_k", type=int, default=5)
    parser.add_argument("--max_rtfree_wer", type=float, default=0.0, help="RTFree-F5 output must be this intelligible")
    parser.add_argument("--min_words", type=int, default=5)
    parser.add_argument("--min_secs", type=float, default=2.5, help="target utterance duration range")
    parser.add_argument("--max_secs", type=float, default=8.0)
    parser.add_argument("--w_sim", type=float, default=0.5)
    parser.add_argument("--w_utmos", type=float, default=0.25)
    parser.add_argument("--group", default=None, help="restrict to an etiology (SAP) or L1 (L2-ARCTIC)")
    parser.add_argument("--no_normalize", action="store_true", help="copy the audio without loudness matching")
    parser.add_argument("--gpu", type=int, default=0, help="GPU for computing missing WER / SIM")
    parser.add_argument("--wavlm_ckpt", default="../checkpoints/UniSpeech/wavlm_large_finetune.pth")
    args = parser.parse_args()

    # pairs: ground-truth target audio / text per pair id, as generated by eval_infer_batch.py
    _, metadata = cross_utterance_pairs(load_utterances(args.testset, args.data_root, args.manifest))
    generated = {
        p
        for p in metadata
        if os.path.exists(os.path.join(args.rtfree_dir, f"{p}.wav"))
        and os.path.exists(os.path.join(args.oracle_dir, f"{p}.wav"))
    }
    pairs = {p: metadata[p] for p in generated}
    print(f"{len(pairs)} pairs generated by both systems")

    metrics = {}
    for name, gen_dir in [("rtfree", args.rtfree_dir), ("oracle", args.oracle_dir)]:
        wer = load_per_sample(gen_dir, ["_wer_results.json", "eval_wer_results.json", "_wer_results.jsonl"], "wer")
        if not wer:
            wer = compute_missing(gen_dir, pairs, "wer", args.gpu, args.wavlm_ckpt)
        sim = load_per_sample(gen_dir, ["_sim_results.json", "eval_sim_results.json", "_sim_results.jsonl"], "sim")
        if not sim and name == "rtfree":
            sim = compute_missing(gen_dir, pairs, "sim", args.gpu, args.wavlm_ckpt)
        utmos = load_per_sample(gen_dir, ["_utmos_results.jsonl"], "utmos")
        metrics[name] = dict(wer=wer, sim=sim, utmos=utmos)

    candidates = []
    for p, m in pairs.items():
        if p not in metrics["rtfree"]["wer"] or p not in metrics["oracle"]["wer"]:
            continue
        if args.group and m["group"] != args.group:
            continue
        secs = sf.info(m["tgt_audio"]).duration
        n_words = len(m["tgt_text"].split())
        rt_wer, or_wer = metrics["rtfree"]["wer"][p]["wer"], metrics["oracle"]["wer"][p]["wer"]
        rt_sim = metrics["rtfree"]["sim"].get(p, {}).get("sim", 0.0)
        rt_utmos = metrics["rtfree"]["utmos"].get(p, {}).get("utmos", 0.0)
        if rt_wer > args.max_rtfree_wer or n_words < args.min_words or not (args.min_secs <= secs <= args.max_secs):
            continue
        score = (or_wer - rt_wer) + args.w_sim * rt_sim + args.w_utmos * rt_utmos / 5
        candidates.append(
            dict(
                pair_id=p,
                score=score,
                speaker=m["speaker"],
                group=m["group"],
                target_secs=secs,
                target_text=m["tgt_text"],
                wer_oracle=or_wer,
                wer_rtfree=rt_wer,
                hypo_oracle=metrics["oracle"]["wer"][p].get("hypo"),
                hypo_rtfree=metrics["rtfree"]["wer"][p].get("hypo"),
                sim_oracle=metrics["oracle"]["sim"].get(p, {}).get("sim"),
                sim_rtfree=rt_sim,
                utmos_oracle=metrics["oracle"]["utmos"].get(p, {}).get("utmos"),
                utmos_rtfree=rt_utmos,
                ground_truth=m["tgt_audio"],
                reference=m["ref_audio"],
            )
        )
    candidates.sort(key=lambda c: -c["score"])
    print(f"{len(candidates)} candidates pass the filters")
    if not candidates:
        print("Relax --max_rtfree_wer / --min_words / --min_secs / --max_secs.")
        return

    os.makedirs(args.out_dir, exist_ok=True)
    fmt = lambda v: "-" if v is None else f"{v:.2f}"  # noqa: E731
    print(
        f"\n{'rank':>4} {'pair':40s} {'group':16s} {'secs':>5} {'WER F5':>7} {'WER RT':>7} {'SIM RT':>7} {'MOS RT':>7}"
    )
    for rank, c in enumerate(candidates[: args.top_k], 1):
        print(
            f"{rank:>4} {c['pair_id'][:40]:40s} {c['group'][:16]:16s} {c['target_secs']:5.1f} "
            f"{fmt(c['wer_oracle']):>7} {fmt(c['wer_rtfree']):>7} {fmt(c['sim_rtfree']):>7} {fmt(c['utmos_rtfree']):>7}"
        )
        print(f"     text  : {c['target_text']}")
        print(f"     F5-TTS: {c['hypo_oracle']}")
        print(f"     RTFree: {c['hypo_rtfree']}")

        sample_dir = os.path.join(args.out_dir, f"{rank}_{c['pair_id']}")
        os.makedirs(sample_dir, exist_ok=True)
        export(c["ground_truth"], os.path.join(sample_dir, "1_ground_truth.wav"), not args.no_normalize)
        export(
            os.path.join(args.oracle_dir, f"{c['pair_id']}.wav"),
            os.path.join(sample_dir, "2_f5tts_oracle.wav"),
            not args.no_normalize,
        )
        export(
            os.path.join(args.rtfree_dir, f"{c['pair_id']}.wav"),
            os.path.join(sample_dir, "3_rtfree_f5.wav"),
            not args.no_normalize,
        )
        export(c["reference"], os.path.join(sample_dir, "0_reference.wav"), not args.no_normalize)
        with open(os.path.join(sample_dir, "info.json"), "w") as f:
            json.dump(c, f, indent=2, ensure_ascii=False)

    with open(os.path.join(args.out_dir, "candidates.json"), "w") as f:
        json.dump(candidates, f, indent=2, ensure_ascii=False)
    print(
        f"\nTop-{min(args.top_k, len(candidates))} triples written to {args.out_dir} (all candidates in candidates.json)"
    )


if __name__ == "__main__":
    main()
