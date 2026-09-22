import math
import os
import random
import string
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio
from tqdm import tqdm

from f5_tts.eval.ecapa_tdnn import ECAPA_TDNN_SMALL
from f5_tts.model.modules import MelSpec
from f5_tts.model.utils import convert_char_to_pinyin


# seedtts testset metainfo: utt, prompt_text, prompt_wav, gt_text, gt_wav
def get_seedtts_testset_metainfo(metalst):
    f = open(metalst)
    lines = f.readlines()
    f.close()
    metainfo = []
    for line in lines:
        if len(line.strip().split("|")) == 5:
            utt, prompt_text, prompt_wav, gt_text, gt_wav = line.strip().split("|")
        elif len(line.strip().split("|")) == 4:
            utt, prompt_text, prompt_wav, gt_text = line.strip().split("|")
            gt_wav = os.path.join(os.path.dirname(metalst), "wavs", utt + ".wav")
        if not os.path.isabs(prompt_wav):
            prompt_wav = os.path.join(os.path.dirname(metalst), prompt_wav)
        metainfo.append((utt, prompt_text, prompt_wav, gt_text, gt_wav))
    return metainfo


# librispeech test-clean metainfo: gen_utt, ref_txt, ref_wav, gen_txt, gen_wav
def get_librispeech_test_clean_metainfo(metalst, librispeech_test_clean_path):
    f = open(metalst)
    lines = f.readlines()
    f.close()
    metainfo = []
    for line in lines:
        ref_utt, ref_dur, ref_txt, gen_utt, gen_dur, gen_txt = line.strip().split("\t")

        # ref_txt = ref_txt[0] + ref_txt[1:].lower() + '.'  # if use librispeech test-clean (no-pc)
        ref_spk_id, ref_chaptr_id, _ = ref_utt.split("-")
        ref_wav = os.path.join(librispeech_test_clean_path, ref_spk_id, ref_chaptr_id, ref_utt + ".flac")

        # gen_txt = gen_txt[0] + gen_txt[1:].lower() + '.'  # if use librispeech test-clean (no-pc)
        gen_spk_id, gen_chaptr_id, _ = gen_utt.split("-")
        gen_wav = os.path.join(librispeech_test_clean_path, gen_spk_id, gen_chaptr_id, gen_utt + ".flac")

        metainfo.append((gen_utt, ref_txt, ref_wav, " " + gen_txt, gen_wav))

    return metainfo


# padded to max length mel batch
def padded_mel_batch(ref_mels):
    max_mel_length = torch.LongTensor([mel.shape[-1] for mel in ref_mels]).amax()
    padded_ref_mels = []
    for mel in ref_mels:
        padded_ref_mel = F.pad(mel, (0, max_mel_length - mel.shape[-1]), value=0)
        padded_ref_mels.append(padded_ref_mel)
    padded_ref_mels = torch.stack(padded_ref_mels)
    padded_ref_mels = padded_ref_mels.permute(0, 2, 1)
    return padded_ref_mels


# get prompts from metainfo containing: utt, prompt_text, prompt_wav, gt_text, gt_wav


def get_inference_prompt(
    metainfo,
    speed=1.0,
    tokenizer="pinyin",
    polyphone=True,
    target_sample_rate=24000,
    n_fft=1024,
    win_length=1024,
    n_mel_channels=100,
    hop_length=256,
    mel_spec_type="vocos",
    target_rms=0.1,
    use_truth_duration=False,
    infer_batch_size=1,
    num_buckets=200,
    min_secs=3,
    max_secs=40,
    target_text_only=False,  # RTFree-F5: condition on the target text only (prompt_text is still used for duration)
):
    prompts_all = []

    min_tokens = min_secs * target_sample_rate // hop_length
    max_tokens = max_secs * target_sample_rate // hop_length

    batch_accum = [0] * num_buckets
    utts, ref_rms_list, ref_mels, ref_mel_lens, total_mel_lens, final_text_list = (
        [[] for _ in range(num_buckets)] for _ in range(6)
    )

    mel_spectrogram = MelSpec(
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mel_channels=n_mel_channels,
        target_sample_rate=target_sample_rate,
        mel_spec_type=mel_spec_type,
    )

    for utt, prompt_text, prompt_wav, gt_text, gt_wav in tqdm(metainfo, desc="Processing prompts..."):
        # Audio
        ref_audio, ref_sr = torchaudio.load(prompt_wav)
        ref_rms = torch.sqrt(torch.mean(torch.square(ref_audio)))
        if ref_rms < target_rms:
            ref_audio = ref_audio * target_rms / ref_rms
        assert ref_audio.shape[-1] > 5000, f"Empty prompt wav: {prompt_wav}, or torchaudio backend issue."
        if ref_sr != target_sample_rate:
            resampler = torchaudio.transforms.Resample(ref_sr, target_sample_rate)
            ref_audio = resampler(ref_audio)

        # Text
        if len(prompt_text[-1].encode("utf-8")) == 1:
            prompt_text = prompt_text + " "
        text = [gt_text] if target_text_only else [prompt_text + gt_text]
        if tokenizer == "pinyin":
            text_list = convert_char_to_pinyin(text, polyphone=polyphone)
        else:
            text_list = text

        # to mel spectrogram
        ref_mel = mel_spectrogram(ref_audio)
        ref_mel = ref_mel.squeeze(0)

        # Duration, mel frame length
        ref_mel_len = ref_mel.shape[-1]

        if use_truth_duration:
            gt_audio, gt_sr = torchaudio.load(gt_wav)
            if gt_sr != target_sample_rate:
                resampler = torchaudio.transforms.Resample(gt_sr, target_sample_rate)
                gt_audio = resampler(gt_audio)
            total_mel_len = ref_mel_len + int(gt_audio.shape[-1] / hop_length / speed)

            # # test vocoder resynthesis
            # ref_audio = gt_audio
        else:
            ref_text_len = len(prompt_text.encode("utf-8"))
            gen_text_len = len(gt_text.encode("utf-8"))
            total_mel_len = ref_mel_len + int(ref_mel_len / ref_text_len * gen_text_len / speed)

        # deal with batch
        assert infer_batch_size > 0, "infer_batch_size should be greater than 0."
        if not (min_tokens <= total_mel_len <= max_tokens):  # can happen for the atypical-speech testsets
            print(
                f"Skipping {utt}: {total_mel_len * hop_length / target_sample_rate:.1f}s out of [{min_secs}, {max_secs}]s"
            )
            continue
        bucket_i = math.floor((total_mel_len - min_tokens) / (max_tokens - min_tokens + 1) * num_buckets)

        utts[bucket_i].append(utt)
        ref_rms_list[bucket_i].append(ref_rms)
        ref_mels[bucket_i].append(ref_mel)
        ref_mel_lens[bucket_i].append(ref_mel_len)
        total_mel_lens[bucket_i].append(total_mel_len)
        final_text_list[bucket_i].extend(text_list)

        batch_accum[bucket_i] += total_mel_len

        if batch_accum[bucket_i] >= infer_batch_size:
            # print(f"\n{len(ref_mels[bucket_i][0][0])}\n{ref_mel_lens[bucket_i]}\n{total_mel_lens[bucket_i]}")
            prompts_all.append(
                (
                    utts[bucket_i],
                    ref_rms_list[bucket_i],
                    padded_mel_batch(ref_mels[bucket_i]),
                    ref_mel_lens[bucket_i],
                    total_mel_lens[bucket_i],
                    final_text_list[bucket_i],
                )
            )
            batch_accum[bucket_i] = 0
            (
                utts[bucket_i],
                ref_rms_list[bucket_i],
                ref_mels[bucket_i],
                ref_mel_lens[bucket_i],
                total_mel_lens[bucket_i],
                final_text_list[bucket_i],
            ) = [], [], [], [], [], []

    # add residual
    for bucket_i, bucket_frames in enumerate(batch_accum):
        if bucket_frames > 0:
            prompts_all.append(
                (
                    utts[bucket_i],
                    ref_rms_list[bucket_i],
                    padded_mel_batch(ref_mels[bucket_i]),
                    ref_mel_lens[bucket_i],
                    total_mel_lens[bucket_i],
                    final_text_list[bucket_i],
                )
            )
    # not only leave easy work for last workers
    random.seed(666)
    random.shuffle(prompts_all)

    return prompts_all


# get wav_res_ref_text of seed-tts test metalst
# https://github.com/BytedanceSpeech/seed-tts-eval


def get_seed_tts_test(metalst, gen_wav_dir, gpus):
    f = open(metalst)
    lines = f.readlines()
    f.close()

    test_set_ = []
    for line in tqdm(lines):
        if len(line.strip().split("|")) == 5:
            utt, prompt_text, prompt_wav, gt_text, gt_wav = line.strip().split("|")
        elif len(line.strip().split("|")) == 4:
            utt, prompt_text, prompt_wav, gt_text = line.strip().split("|")

        if not os.path.exists(os.path.join(gen_wav_dir, utt + ".wav")):
            continue
        gen_wav = os.path.join(gen_wav_dir, utt + ".wav")
        if not os.path.isabs(prompt_wav):
            prompt_wav = os.path.join(os.path.dirname(metalst), prompt_wav)

        test_set_.append((gen_wav, prompt_wav, gt_text))

    num_jobs = len(gpus)
    if num_jobs == 1:
        return [(gpus[0], test_set_)]

    wav_per_job = len(test_set_) // num_jobs + 1
    test_set = []
    for i in range(num_jobs):
        test_set.append((gpus[i], test_set_[i * wav_per_job : (i + 1) * wav_per_job]))

    return test_set


# get librispeech test-clean cross sentence test


def get_librispeech_test(metalst, gen_wav_dir, gpus, librispeech_test_clean_path, eval_ground_truth=False):
    f = open(metalst)
    lines = f.readlines()
    f.close()

    test_set_ = []
    for line in tqdm(lines):
        ref_utt, ref_dur, ref_txt, gen_utt, gen_dur, gen_txt = line.strip().split("\t")

        if eval_ground_truth:
            gen_spk_id, gen_chaptr_id, _ = gen_utt.split("-")
            gen_wav = os.path.join(librispeech_test_clean_path, gen_spk_id, gen_chaptr_id, gen_utt + ".flac")
        else:
            if not os.path.exists(os.path.join(gen_wav_dir, gen_utt + ".wav")):
                raise FileNotFoundError(f"Generated wav not found: {gen_utt}")
            gen_wav = os.path.join(gen_wav_dir, gen_utt + ".wav")

        ref_spk_id, ref_chaptr_id, _ = ref_utt.split("-")
        ref_wav = os.path.join(librispeech_test_clean_path, ref_spk_id, ref_chaptr_id, ref_utt + ".flac")

        test_set_.append((gen_wav, ref_wav, gen_txt))

    num_jobs = len(gpus)
    if num_jobs == 1:
        return [(gpus[0], test_set_)]

    wav_per_job = len(test_set_) // num_jobs + 1
    test_set = []
    for i in range(num_jobs):
        test_set.append((gpus[i], test_set_[i * wav_per_job : (i + 1) * wav_per_job]))

    return test_set


# ASR transcripts for the reference audios (F5-TTS "ASR" baseline: Whisper large-v3 replaces the oracle transcript)


def precompute_asr_transcripts(metainfo, cache_path, accelerator, device="cuda"):
    """Transcribe every prompt_wav in metainfo (main process only), cache to json, return {prompt_wav: transcript}."""
    import json

    if not os.path.exists(cache_path) and accelerator.is_main_process:
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
        whisper = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3", torch_dtype=torch.float16)
        whisper = whisper.to(device).eval()

        transcripts = {}
        for prompt_wav in tqdm(sorted({item[2] for item in metainfo}), desc="ASR transcription of references"):
            audio, sr = torchaudio.load(prompt_wav)
            audio = audio.mean(0) if audio.shape[0] > 1 else audio[0]
            if sr != 16000:
                audio = torchaudio.functional.resample(audio, sr, 16000)
            features = processor(audio.numpy(), sampling_rate=16000, return_tensors="pt").input_features
            with torch.no_grad():
                ids = whisper.generate(
                    features.to(device=device, dtype=torch.float16), language="en", task="transcribe"
                )
            transcripts[prompt_wav] = processor.batch_decode(ids, skip_special_tokens=True)[0].strip()

        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(transcripts, f, indent=2, ensure_ascii=False)
        del whisper, processor
        torch.cuda.empty_cache()

    accelerator.wait_for_everyone()
    with open(cache_path) as f:
        return json.load(f)


# load asr model


def load_asr_model(lang, ckpt_dir=""):
    if lang == "zh":
        from funasr import AutoModel

        model = AutoModel(
            model=os.path.join(ckpt_dir, "paraformer-zh"),
            # vad_model = os.path.join(ckpt_dir, "fsmn-vad"),
            # punc_model = os.path.join(ckpt_dir, "ct-punc"),
            # spk_model = os.path.join(ckpt_dir, "cam++"),
            disable_update=True,
        )  # following seed-tts setting
    elif lang == "en":
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        model_name = "openai/whisper-large-v3" if ckpt_dir == "" else ckpt_dir
        processor = WhisperProcessor.from_pretrained(model_name)
        whisper_model = WhisperForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float16)
        whisper_model = whisper_model.cuda()
        whisper_model.eval()
        model = (whisper_model, processor)
    return model


# WER Evaluation, the way Seed-TTS does


def run_asr_wer(args):
    rank, lang, test_set, ckpt_dir = args

    if lang == "zh":
        import zhconv

        torch.cuda.set_device(rank)
    elif lang == "en":
        torch.cuda.set_device(rank)
    else:
        raise NotImplementedError("lang support only 'zh' (funasr paraformer-zh), 'en' (HF whisper-large-v3), for now.")

    asr_model = load_asr_model(lang, ckpt_dir=ckpt_dir)

    if lang == "en":
        whisper_model, whisper_processor = asr_model

    from zhon.hanzi import punctuation

    punctuation_all = punctuation + string.punctuation
    wer_results = []

    from jiwer import process_words

    for gen_wav, prompt_wav, truth in tqdm(test_set):
        if lang == "zh":
            res = asr_model.generate(input=gen_wav, batch_size_s=300, disable_pbar=True)
            hypo = res[0]["text"]
            hypo = zhconv.convert(hypo, "zh-cn")
        elif lang == "en":
            audio, sr = torchaudio.load(gen_wav)
            if sr != 16000:
                audio = torchaudio.functional.resample(audio, sr, 16000)
            audio = audio.squeeze(0).numpy()
            input_features = whisper_processor(audio, sampling_rate=16000, return_tensors="pt").input_features.to(
                device=f"cuda:{rank}", dtype=torch.float16
            )
            with torch.no_grad():
                predicted_ids = whisper_model.generate(
                    input_features,
                    language="en",
                    task="transcribe",
                )
            hypo = whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

        raw_truth = truth
        raw_hypo = hypo

        for x in punctuation_all:
            truth = truth.replace(x, "")
            hypo = hypo.replace(x, "")

        truth = truth.replace("  ", " ")
        hypo = hypo.replace("  ", " ")

        if lang == "zh":
            truth = " ".join([x for x in truth])
            hypo = " ".join([x for x in hypo])
        elif lang == "en":
            truth = truth.lower()
            hypo = hypo.lower()

        measures = process_words(truth, hypo)
        wer = measures.wer

        # ref_list = truth.split(" ")
        # subs = measures.substitutions / len(ref_list)
        # dele = measures.deletions / len(ref_list)
        # inse = measures.insertions / len(ref_list)

        wer_results.append(
            {
                "wav": Path(gen_wav).stem,
                "truth": raw_truth,
                "hypo": raw_hypo,
                "wer": wer,
                "truth_norm": truth,
                "hypo_norm": hypo,
            }
        )

    return wer_results


# SIM Evaluation


def run_sim(args):
    rank, test_set, ckpt_dir = args
    device = f"cuda:{rank}"

    model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type="wavlm_large", config_path=None)
    state_dict = torch.load(ckpt_dir, weights_only=True, map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict["model"], strict=False)

    use_gpu = True if torch.cuda.is_available() else False
    if use_gpu:
        model = model.cuda(device)
    model.eval()

    sim_results = []
    for gen_wav, prompt_wav, truth in tqdm(test_set):
        wav1, sr1 = torchaudio.load(gen_wav)
        wav2, sr2 = torchaudio.load(prompt_wav)

        if use_gpu:
            wav1 = wav1.cuda(device)
            wav2 = wav2.cuda(device)

        if sr1 != 16000:
            resample1 = torchaudio.transforms.Resample(orig_freq=sr1, new_freq=16000)
            if use_gpu:
                resample1 = resample1.cuda(device)
            wav1 = resample1(wav1)
        if sr2 != 16000:
            resample2 = torchaudio.transforms.Resample(orig_freq=sr2, new_freq=16000)
            if use_gpu:
                resample2 = resample2.cuda(device)
            wav2 = resample2(wav2)

        with torch.no_grad():
            emb1 = model(wav1)
            emb2 = model(wav2)

        sim = F.cosine_similarity(emb1, emb2)[0].item()
        # print(f"VSim score between two audios: {sim:.4f} (-1.0, 1.0).")
        sim_results.append(
            {
                "wav": Path(gen_wav).stem,
                "sim": sim,
            }
        )

    return sim_results
