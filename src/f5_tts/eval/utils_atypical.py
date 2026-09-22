"""
Atypical-speech test sets used in the RTFree-F5 paper:
  - SAP (Speech Accessibility Project, dysarthric speech), Dev / Train manifests
  - L2-ARCTIC (non-native English, 24 speakers x 6 L1 backgrounds)

Evaluation is cross-utterance: the reference is one utterance of a speaker, the target text (and, with
use_truth_duration, the target duration) come from another utterance of the same speaker.
"""

import csv
import os
from collections import defaultdict

import soundfile as sf


L2ARCTIC_SPEAKER_L1 = {
    "ABA": "Arabic", "SKA": "Arabic", "YBAA": "Arabic", "ZHAA": "Arabic",
    "BWC": "Mandarin", "LXC": "Mandarin", "NCC": "Mandarin", "TXHC": "Mandarin",
    "ASI": "Hindi", "RRBI": "Hindi", "SVBI": "Hindi", "TNI": "Hindi",
    "HJK": "Korean", "HKK": "Korean", "YDCK": "Korean", "YKWK": "Korean",
    "EBVS": "Spanish", "ERMS": "Spanish", "MBMPS": "Spanish", "NJS": "Spanish",
    "HQTV": "Vietnamese", "PNV": "Vietnamese", "THV": "Vietnamese", "TLV": "Vietnamese",
}  # fmt: skip

ATYPICAL_TESTSETS = ["sap_dev", "sap_train", "l2arctic"]


# utterances: {speaker: [{"id", "audio", "text", "group"}]}, group = etiology (SAP) | L1 (L2-ARCTIC)


def load_sap_utterances(manifest_csv, data_root, min_duration=0.3, max_duration=30):
    spk2utts = defaultdict(list)
    with open(manifest_csv) as f:
        for row in csv.DictReader(f):
            text = row["norm_text_without_disfluency"].strip()
            if not text or not (min_duration <= float(row["duration"]) <= max_duration):
                continue
            spk2utts[row["speaker"]].append(
                dict(
                    id=row["id"],
                    audio=os.path.join(data_root, row["audio_filepath"]),
                    text=text,
                    group=row.get("etiology", "unknown"),
                )
            )
    return spk2utts


def load_l2arctic_utterances(data_root, min_duration=0.5, max_duration=30):
    """Expects data_root/<SPEAKER>/{wav/<utt>.wav, transcript/<utt>.txt}."""
    spk2utts = defaultdict(list)
    for spk in sorted(os.listdir(data_root)):
        wav_dir, txt_dir = os.path.join(data_root, spk, "wav"), os.path.join(data_root, spk, "transcript")
        if not (os.path.isdir(wav_dir) and os.path.isdir(txt_dir)):
            continue
        for wav_file in sorted(f for f in os.listdir(wav_dir) if f.endswith(".wav")):
            utt = wav_file[: -len(".wav")]
            txt_path = os.path.join(txt_dir, utt + ".txt")
            if not os.path.exists(txt_path):
                continue
            with open(txt_path) as f:
                text = f.read().strip()
            wav_path = os.path.join(wav_dir, wav_file)
            if not text or not (min_duration <= sf.info(wav_path).duration <= max_duration):
                continue
            spk2utts[spk].append(
                dict(id=f"{spk}_{utt}", audio=wav_path, text=text, group=L2ARCTIC_SPEAKER_L1.get(spk, "Unknown"))
            )
    return spk2utts


def load_utterances(testset, data_root, manifest=None):
    if testset in ["sap_dev", "sap_train"]:
        split = "Dev" if testset == "sap_dev" else "Train"
        manifest = manifest or os.path.join(data_root, "manifest", f"{split}.csv")
        return load_sap_utterances(manifest, data_root)
    elif testset == "l2arctic":
        return load_l2arctic_utterances(data_root)
    raise ValueError(f"unknown atypical testset {testset}, choose from {ATYPICAL_TESTSETS}")


def cross_utterance_pairs(spk2utts, max_pairs_per_speaker=None):
    """
    Pair each utterance (reference) with the next utterance of the same speaker (target), circularly.
    Returns
        metainfo: [(pair_id, ref_text, ref_wav, " " + tgt_text, tgt_wav)], the format of get_inference_prompt()
        metadata: {pair_id: {"speaker", "group", "ref_audio", "tgt_audio", "tgt_text"}}, saved next to the generated wavs
    """
    metainfo, metadata = [], {}
    for spk, utts in spk2utts.items():
        if len(utts) < 2:
            continue
        for i, ref in enumerate(utts):
            if max_pairs_per_speaker is not None and i >= max_pairs_per_speaker:
                break
            tgt = utts[(i + 1) % len(utts)]
            pair_id = f"{ref['id']}__to__{tgt['id']}"
            metainfo.append((pair_id, ref["text"], ref["audio"], " " + tgt["text"], tgt["audio"]))
            metadata[pair_id] = dict(
                speaker=spk, group=ref["group"], ref_audio=ref["audio"], tgt_audio=tgt["audio"], tgt_text=tgt["text"]
            )
    return metainfo, metadata


def get_atypical_metainfo(testset, data_root, manifest=None, max_pairs_per_speaker=None):
    spk2utts = load_utterances(testset, data_root, manifest)
    metainfo, metadata = cross_utterance_pairs(spk2utts, max_pairs_per_speaker)
    n_spk = sum(len(u) >= 2 for u in spk2utts.values())
    print(f"{testset}: {len(metainfo)} cross-utterance pairs from {n_spk} speakers")
    return metainfo, metadata
