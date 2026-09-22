import os
import sys


sys.path.append(os.getcwd())

import argparse
import time
from importlib.resources import files

import torch
import torchaudio
from accelerate import Accelerator
from hydra.utils import get_class
from omegaconf import OmegaConf
from tqdm import tqdm

from f5_tts.eval.utils_eval import (
    get_inference_prompt,
    get_librispeech_test_clean_metainfo,
    get_seedtts_testset_metainfo,
)
from f5_tts.infer.utils_infer import load_checkpoint, load_vocoder
from f5_tts.model import CFM
from f5_tts.model.utils import get_tokenizer
import json

accelerator = Accelerator()
device = f"cuda:{accelerator.process_index}"


use_ema = True
target_rms = 0.1


rel_path = str(files("f5_tts").joinpath("../../"))

def get_sap_metainfo(manifest_csv, sap_data_root, max_pairs_per_speaker=5):
    """
    Build cross-utterance pairs from SAP manifest.
    Returns list of tuples matching the format expected by get_inference_prompt:
        (pair_id, ref_audio_path, ref_text, tgt_audio_path, tgt_text)
    
    Args:
        max_pairs_per_speaker: Limit pairs per speaker to keep eval tractable.
                               Set to None for all pairs.
    """
    import csv
    from collections import defaultdict

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
            })

    pairs = []
    for spk, utts in spk2utts.items():
        if len(utts) < 2:
            continue
        count = 0
        for i, ref_utt in enumerate(utts):
            if max_pairs_per_speaker and count >= max_pairs_per_speaker:
                break
            tgt_utt = utts[(i + 1) % len(utts)]
            pair_id = f"{ref_utt['id']}__to__{tgt_utt['id']}"
            pairs.append((
                pair_id,
                ref_utt["audio"],       # ref_audio_path
                ref_utt["text"],        # ref_text (for F5-TTS baseline)
                tgt_utt["audio"],       # tgt_audio_path (for duration estimation)
                tgt_utt["text"],        # tgt_text (target transcription)
            ))
            count += 1

    print(f"SAP: built {len(pairs)} cross-utterance pairs from "
          f"{sum(1 for u in spk2utts.values() if len(u) >= 2)} speakers")
    return pairs

def precompute_asr_transcripts(metainfo, cache_path, target_sample_rate=24000):
    """
    Run Whisper on all reference audios and cache transcripts.
    Only runs on main process; others wait and load cache.
    """
    if os.path.exists(cache_path):
        if accelerator.is_main_process:
            print(f"Loading ASR transcript cache from {cache_path}")
        accelerator.wait_for_everyone()
        import json
        with open(cache_path) as f:
            return json.load(f)

    if accelerator.is_main_process:
        import json
        from transformers import WhisperProcessor, WhisperForConditionalGeneration

        print(f"Precomputing ASR transcripts for {len(metainfo)} references...")
        processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
        whisper_model = WhisperForConditionalGeneration.from_pretrained(
            "openai/whisper-large-v3", torch_dtype=torch.float16
        ).to(device).eval()

        # Deduplicate: multiple pairs may share the same ref audio
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
            ).input_features.to(device=device, dtype=torch.float16)

            with torch.no_grad():
                predicted_ids = whisper_model.generate(
                    input_features, language="en", task="transcribe"
                )
            hypo = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            transcripts[audio_path] = hypo.strip()

        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(transcripts, f, indent=2, ensure_ascii=False)
        print(f"Saved ASR cache: {len(transcripts)} entries -> {cache_path}")

        del whisper_model, processor
        torch.cuda.empty_cache()

    accelerator.wait_for_everyone()

    import json
    with open(cache_path) as f:
        return json.load(f)

def main():
    parser = argparse.ArgumentParser(description="batch inference")

    parser.add_argument("-s", "--seed", default=None, type=int)
    parser.add_argument("-n", "--expname", required=True)
    parser.add_argument("-c", "--ckptstep", default=None, type=int,
                        help="Checkpoint step number. Used to construct default ckpt path if --ckpt_path is not given.")

    parser.add_argument("-nfe", "--nfestep", default=32, type=int)
    parser.add_argument("-o", "--odemethod", default="euler")
    parser.add_argument("-ss", "--swaysampling", default=-1, type=float)

    parser.add_argument("-t", "--testset", required=True)
    parser.add_argument(
        "-p", "--librispeech_test_clean_path", default=f"{rel_path}/data/LibriSpeech/test-clean", type=str
    )

    # AFTER the existing --librispeech_test_clean_path arg (line 53):
    parser.add_argument("--sap_data_root", type=str,
                        default=f"{rel_path}/data/SpeechAccessibility_Research_Release",
                        help="Root directory of SAP dataset")
    parser.add_argument("--sap_manifest", type=str, default=None,
                        help="Path to SAP manifest CSV. Defaults to <sap_data_root>/manifest/Dev.csv")

    parser.add_argument("--local", action="store_true", help="Use local vocoder checkpoint directory")
    
    parser.add_argument("--reffree", action="store_true", help="Use RefFree mode")
    parser.add_argument("--speech_encoder", type=str, default="microsoft/wavlm-large")

    parser.add_argument("--ckpt_path", type=str, default=None,
                        help="Explicit path to checkpoint file. Overrides --ckptstep based path construction.")
    parser.add_argument("--config", type=str, default=None,
                        help="Config name to load (e.g. F5TTS_v1_Base). Defaults to --expname if not given.")

    parser.add_argument("--ref_text_mode", type=str, default="oracle",
                            choices=["oracle", "asr"],
                            help="How to obtain reference text: oracle=ground truth, asr=Whisper")

    args = parser.parse_args()

    seed = args.seed
    exp_name = args.expname
    ckpt_step = args.ckptstep

    nfe_step = args.nfestep
    ode_method = args.odemethod
    sway_sampling_coef = args.swaysampling

    testset = args.testset

    infer_batch_size = 1  # max frames. 1 for ddp single inference (recommended)
    cfg_strength = 2.0
    speed = 1.0
    use_truth_duration = False
    no_ref_audio = False

    # Config: use --config if provided, otherwise fall back to exp_name
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

    if testset == "ls_pc_test_clean":
        metalst = rel_path + "/data/librispeech_pc_test_clean_cross_sentence.lst"
        librispeech_test_clean_path = args.librispeech_test_clean_path
        metainfo = get_librispeech_test_clean_metainfo(metalst, librispeech_test_clean_path)

    elif testset == "seedtts_test_zh":
        metalst = rel_path + "/data/seedtts_testset/zh/meta.lst"
        metainfo = get_seedtts_testset_metainfo(metalst)

    elif testset == "seedtts_test_en":
        metalst = rel_path + "/data/seedtts_testset/en/meta.lst"
        metainfo = get_seedtts_testset_metainfo(metalst)

    elif testset == "sap_dev":
        sap_manifest = args.sap_manifest or os.path.join(args.sap_data_root, "manifest", "Dev.csv")
        metainfo = get_sap_metainfo(sap_manifest, args.sap_data_root)

    elif testset == "sap_train":
        sap_manifest = args.sap_manifest or os.path.join(args.sap_data_root, "manifest", "Train.csv")
        metainfo = get_sap_metainfo(sap_manifest, args.sap_data_root)
        
    # ---- ASR transcript mode: replace ref_text with Whisper output ----
    if args.ref_text_mode == "asr" and not args.reffree:
        cache_path = f"{rel_path}/data/asr_cache_{testset}.json"
        asr_cache = precompute_asr_transcripts(metainfo, cache_path)
        metainfo = [
            (utt, asr_cache.get(prompt_wav, prompt_text), prompt_wav, gt_text, gt_wav)
            for utt, prompt_text, prompt_wav, gt_text, gt_wav in metainfo
        ]
        if accelerator.is_main_process:
            print(f"Replaced ref_text with ASR transcripts ({len(asr_cache)} cached)")
        
    metainfo_dict = {}
    for item in metainfo:
        utt = item[0]  # utterance id
        if testset == "ls_pc_test_clean":
            # item format: (utt, ref_audio, ref_text, gen_audio, gen_text, ...)
            metainfo_dict[utt] = {"ref_audio": item[2]}
        elif testset.startswith("seedtts"):
            # item format varies - check get_seedtts_testset_metainfo
            metainfo_dict[utt] = {"ref_audio": item[2]}

    # Resolve checkpoint step label for output directory naming
    if args.ckpt_path:
        # Derive a label from the checkpoint filename for output dir naming
        ckpt_basename = os.path.splitext(os.path.basename(args.ckpt_path))[0]  # e.g. "model_last"
        ckpt_step_label = ckpt_basename  # use full basename as label
    else:
        if ckpt_step is None:
            raise ValueError("Either --ckpt_path or --ckptstep must be provided.")
        ckpt_step_label = str(ckpt_step)

    # path to save generated wavs
    # Determine mode label for output dir
    if args.reffree:
        mode_label = "reffree"
    else:
        mode_label = args.ref_text_mode  # "oracle" or "asr"

    # path to save generated wavs
    output_dir = (
        f"{rel_path}/"
        f"results/{exp_name}_{ckpt_step_label}/{testset}_{mode_label}/"
        f"seed{seed}_{ode_method}_nfe{nfe_step}_{mel_spec_type}"
        f"{f'_ss{sway_sampling_coef}' if sway_sampling_coef else ''}"
        f"_cfg{cfg_strength}_speed{speed}"
        f"{'_gt-dur' if use_truth_duration else ''}"
        f"{'_no-ref-audio' if no_ref_audio else ''}"
    )

    # -------------------------------------------------#

    prompts_all = get_inference_prompt(
        metainfo,
        speed=speed,
        tokenizer=tokenizer,
        target_sample_rate=target_sample_rate,
        n_mel_channels=n_mel_channels,
        hop_length=hop_length,
        mel_spec_type=mel_spec_type,
        target_rms=target_rms,
        use_truth_duration=use_truth_duration,
        infer_batch_size=infer_batch_size,
    )

    # Vocoder model
    local = args.local
    if mel_spec_type == "vocos":
        vocoder_local_path = "../checkpoints/charactr/vocos-mel-24khz"
    elif mel_spec_type == "bigvgan":
        vocoder_local_path = "../checkpoints/bigvgan_v2_24khz_100band_256x"
    vocoder = load_vocoder(vocoder_name=mel_spec_type, is_local=local, local_path=vocoder_local_path)

    # Tokenizer
    vocab_char_map, vocab_size = get_tokenizer(dataset_name, tokenizer)

    # Model
    model = CFM(
        transformer=model_cls(**model_arc, text_num_embeds=vocab_size, mel_dim=n_mel_channels),
        mel_spec_kwargs=dict(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mel_channels=n_mel_channels,
            target_sample_rate=target_sample_rate,
            mel_spec_type=mel_spec_type,
        ),
        odeint_kwargs=dict(
            method=ode_method,
        ),
        vocab_char_map=vocab_char_map,
        speech_encoder_name=args.speech_encoder if args.reffree else None,
    ).to(device)

    # Resolve checkpoint path
    if args.ckpt_path:
        ckpt_path = args.ckpt_path
        if not os.path.exists(ckpt_path):
            raise ValueError(f"Provided --ckpt_path does not exist: {ckpt_path}")
    else:
        ckpt_prefix = rel_path + f"/ckpts/{exp_name}/model_{ckpt_step}"
        if os.path.exists(ckpt_prefix + ".pt"):
            ckpt_path = ckpt_prefix + ".pt"
        elif os.path.exists(ckpt_prefix + ".safetensors"):
            ckpt_path = ckpt_prefix + ".safetensors"
        else:
            print("Loading from self-organized training checkpoints rather than released pretrained.")
            ckpt_prefix = rel_path + f"/{model_cfg.ckpts.save_dir}/model_{ckpt_step}"
            if os.path.exists(ckpt_prefix + ".pt"):
                ckpt_path = ckpt_prefix + ".pt"
            elif os.path.exists(ckpt_prefix + ".safetensors"):
                ckpt_path = ckpt_prefix + ".safetensors"
            else:
                raise ValueError("The checkpoint does not exist or cannot be found in given location.")

    dtype = torch.float32 if mel_spec_type == "bigvgan" else None
    model = load_checkpoint(model, ckpt_path, device, dtype=dtype, use_ema=use_ema)

    if not os.path.exists(output_dir) and accelerator.is_main_process:
        os.makedirs(output_dir)

    # start batch inference
    accelerator.wait_for_everyone()
    start = time.time()

    with accelerator.split_between_processes(prompts_all) as prompts:
        for prompt in tqdm(prompts, disable=not accelerator.is_local_main_process):
            utts, ref_rms_list, ref_mels, ref_mel_lens, total_mel_lens, final_text_list = prompt
            if args.reffree:
                # get_inference_prompt returns text as "ref_text gen_text"
                # We need only gen_text for RefFree mode
                # Reconstruct gen-only text from metainfo
                gen_text_list = []
                for utt in utts:
                    # Get gen_text from metainfo (index depends on testset format)
                    for item in metainfo:
                        if item[0] == utt:
                            if testset == "ls_pc_test_clean":
                                gen_text_list.append(item[3])  # gen_text field
                            elif testset.startswith("seedtts"):
                                gen_text_list.append(item[3])  # gen_text field
                            break
                final_text_list = gen_text_list
                
            all_exist = all(os.path.exists(f"{output_dir}/{utt}.wav") for utt in utts)
            if all_exist:
                continue
            ref_mels = ref_mels.to(device)
            ref_mel_lens = torch.tensor(ref_mel_lens, dtype=torch.long).to(device)
            total_mel_lens = torch.tensor(total_mel_lens, dtype=torch.long).to(device)
            
            ref_audio_tensor = None
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
                
                # Pad to same length and stack
                ref_audio_sample_lens = torch.tensor(
                    [a.shape[0] for a in ref_audio_list], dtype=torch.long
                ).to(device)
                max_len = ref_audio_sample_lens.amax().item()
                ref_audio_tensor = torch.stack([
                    torch.nn.functional.pad(a, (0, max_len - a.shape[0])) for a in ref_audio_list
                ]).to(device)
            else:
                ref_audio_sample_lens = None

            # Inference
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
                # Final result
                for i, gen in enumerate(generated):
                    gen = gen[ref_mel_lens[i] : total_mel_lens[i], :].unsqueeze(0)
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
        timediff = time.time() - start
        print(f"Done batch inference in {timediff / 60:.2f} minutes.")


if __name__ == "__main__":
    main()