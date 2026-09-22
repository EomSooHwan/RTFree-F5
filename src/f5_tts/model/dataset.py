import json
from importlib.resources import files

import torch
import torch.nn.functional as F
import torchaudio
from datasets import Dataset as Dataset_
from datasets import load_from_disk
from torch import nn
from torch.utils.data import Dataset, Sampler
from tqdm import tqdm

from f5_tts.model.modules import MelSpec
from f5_tts.model.utils import default


class HFDataset(Dataset):
    def __init__(
        self,
        hf_dataset: Dataset,
        target_sample_rate=24_000,
        n_mel_channels=100,
        hop_length=256,
        n_fft=1024,
        win_length=1024,
        mel_spec_type="vocos",
    ):
        self.data = hf_dataset
        self.target_sample_rate = target_sample_rate
        self.hop_length = hop_length

        self.mel_spectrogram = MelSpec(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mel_channels=n_mel_channels,
            target_sample_rate=target_sample_rate,
            mel_spec_type=mel_spec_type,
        )

    def get_frame_len(self, index):
        row = self.data[index]
        audio = row["audio"]["array"]
        sample_rate = row["audio"]["sampling_rate"]
        return audio.shape[-1] / sample_rate * self.target_sample_rate / self.hop_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data[index]
        audio = row["audio"]["array"]

        # logger.info(f"Audio shape: {audio.shape}")

        sample_rate = row["audio"]["sampling_rate"]
        duration = audio.shape[-1] / sample_rate

        if duration > 30 or duration < 0.3:
            return self.__getitem__((index + 1) % len(self.data))

        audio_tensor = torch.from_numpy(audio).float()

        if sample_rate != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, self.target_sample_rate)
            audio_tensor = resampler(audio_tensor)

        audio_tensor = audio_tensor.unsqueeze(0)  # 't -> 1 t')

        mel_spec = self.mel_spectrogram(audio_tensor)

        mel_spec = mel_spec.squeeze(0)  # '1 d t -> d t'

        text = row["text"]

        return dict(
            mel_spec=mel_spec,
            text=text,
        )


class CustomDataset(Dataset):
    def __init__(
        self,
        custom_dataset: Dataset,
        durations=None,
        target_sample_rate=24_000,
        hop_length=256,
        n_mel_channels=100,
        n_fft=1024,
        win_length=1024,
        mel_spec_type="vocos",
        preprocessed_mel=False,
        mel_spec_module: nn.Module | None = None,
    ):
        self.data = custom_dataset
        self.durations = durations
        self.target_sample_rate = target_sample_rate
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.win_length = win_length
        self.mel_spec_type = mel_spec_type
        self.preprocessed_mel = preprocessed_mel

        if not preprocessed_mel:
            self.mel_spectrogram = default(
                mel_spec_module,
                MelSpec(
                    n_fft=n_fft,
                    hop_length=hop_length,
                    win_length=win_length,
                    n_mel_channels=n_mel_channels,
                    target_sample_rate=target_sample_rate,
                    mel_spec_type=mel_spec_type,
                ),
            )

    def get_frame_len(self, index):
        if (
            self.durations is not None
        ):  # Please make sure the separately provided durations are correct, otherwise 99.99% OOM
            return self.durations[index] * self.target_sample_rate / self.hop_length
        return self.data[index]["duration"] * self.target_sample_rate / self.hop_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        while True:
            row = self.data[index]
            audio_path = row["audio_path"]
            text = row["text"]
            duration = row["duration"]

            # filter by given length
            if 0.3 <= duration <= 30:
                break  # valid

            index = (index + 1) % len(self.data)

        if self.preprocessed_mel:
            mel_spec = torch.tensor(row["mel_spec"])
        else:
            audio, source_sample_rate = torchaudio.load(audio_path)

            # make sure mono input
            if audio.shape[0] > 1:
                audio = torch.mean(audio, dim=0, keepdim=True)

            # resample if necessary
            if source_sample_rate != self.target_sample_rate:
                resampler = torchaudio.transforms.Resample(source_sample_rate, self.target_sample_rate)
                audio = resampler(audio)

            # to mel spectrogram
            mel_spec = self.mel_spectrogram(audio)
            mel_spec = mel_spec.squeeze(0)  # '1 d t -> d t'

        return {
            "mel_spec": mel_spec,
            "text": text,
        }


# Dynamic Batch Sampler
class DynamicBatchSampler(Sampler[list[int]]):
    """Extension of Sampler that will do the following:
    1.  Change the batch size (essentially number of sequences)
        in a batch to ensure that the total number of frames are less
        than a certain threshold.
    2.  Make sure the padding efficiency in the batch is high.
    3.  Shuffle batches each epoch while maintaining reproducibility.
    """

    def __init__(
        self, sampler: Sampler[int], frames_threshold: int, max_samples=0, random_seed=None, drop_residual: bool = False
    ):
        self.sampler = sampler
        self.frames_threshold = frames_threshold
        self.max_samples = max_samples
        self.random_seed = random_seed
        self.epoch = 0

        indices, batches = [], []
        data_source = self.sampler.data_source

        for idx in tqdm(
            self.sampler, desc="Sorting with sampler... if slow, check whether dataset is provided with duration"
        ):
            indices.append((idx, data_source.get_frame_len(idx)))
        indices.sort(key=lambda elem: elem[1])

        batch = []
        batch_frames = 0
        for idx, frame_len in tqdm(
            indices, desc=f"Creating dynamic batches with {frames_threshold} audio frames per gpu"
        ):
            if batch_frames + frame_len <= self.frames_threshold and (max_samples == 0 or len(batch) < max_samples):
                batch.append(idx)
                batch_frames += frame_len
            else:
                if len(batch) > 0:
                    batches.append(batch)
                if frame_len <= self.frames_threshold:
                    batch = [idx]
                    batch_frames = frame_len
                else:
                    batch = []
                    batch_frames = 0

        if not drop_residual and len(batch) > 0:
            batches.append(batch)

        del indices
        self.batches = batches

        # Ensure even batches with accelerate BatchSamplerShard cls under frame_per_batch setting
        self.drop_last = True

    def set_epoch(self, epoch: int) -> None:
        """Sets the epoch for this sampler."""
        self.epoch = epoch

    def __iter__(self):
        # Use both random_seed and epoch for deterministic but different shuffling per epoch
        if self.random_seed is not None:
            g = torch.Generator()
            g.manual_seed(self.random_seed + self.epoch)
            # Use PyTorch's random permutation for better reproducibility across PyTorch versions
            indices = torch.randperm(len(self.batches), generator=g).tolist()
            batches = [self.batches[i] for i in indices]
        else:
            batches = self.batches
        return iter(batches)

    def __len__(self):
        return len(self.batches)


# Load dataset


def load_dataset(
    dataset_name: str,
    tokenizer: str = "pinyin",
    dataset_type: str = "CustomDataset",
    audio_type: str = "raw",
    mel_spec_module: nn.Module | None = None,
    mel_spec_kwargs: dict = dict(),
    cross_utterance: bool = False,
    speaker_id_key: str = "speaker_id",
) -> CustomDataset | HFDataset:
    """
    dataset_type    - "CustomDataset" if you want to use tokenizer name and default data path to load for train_dataset
                    - "CustomDatasetPath" if you just want to pass the full path to a preprocessed dataset without relying on tokenizer
    """

    print("Loading dataset ...")

    if dataset_type == "CustomDataset":
        rel_data_path = str(files("f5_tts").joinpath(f"../../data/{dataset_name}_{tokenizer}"))
        if audio_type == "raw":
            try:
                train_dataset = load_from_disk(f"{rel_data_path}/raw")
            except:  # noqa: E722
                train_dataset = Dataset_.from_file(f"{rel_data_path}/raw.arrow")
            preprocessed_mel = False
        elif audio_type == "mel":
            train_dataset = Dataset_.from_file(f"{rel_data_path}/mel.arrow")
            preprocessed_mel = True
        with open(f"{rel_data_path}/duration.json", "r", encoding="utf-8") as f:
            data_dict = json.load(f)
        durations = data_dict["duration"]
        if cross_utterance: 
            train_dataset = CrossUtteranceDataset(
                train_dataset,  # the raw arrow dataset
                durations=durations,
                mel_spec_module=mel_spec_module,
                speaker_id_key=speaker_id_key,
                **mel_spec_kwargs,
            )
        else: 
            train_dataset = CustomDataset(
                train_dataset,
                durations=durations,
                preprocessed_mel=preprocessed_mel,
                mel_spec_module=mel_spec_module,
                **mel_spec_kwargs,
            )

    elif dataset_type == "CustomDatasetPath":
        try:
            train_dataset = load_from_disk(f"{dataset_name}/raw")
        except:  # noqa: E722
            train_dataset = Dataset_.from_file(f"{dataset_name}/raw.arrow")

        with open(f"{dataset_name}/duration.json", "r", encoding="utf-8") as f:
            data_dict = json.load(f)
        durations = data_dict["duration"]
        train_dataset = CustomDataset(
            train_dataset, durations=durations, preprocessed_mel=preprocessed_mel, **mel_spec_kwargs
        )

    elif dataset_type == "HFDataset":
        print(
            "Should manually modify the path of huggingface dataset to your need.\n"
            + "May also the corresponding script cuz different dataset may have different format."
        )
        pre, post = dataset_name.split("_")
        train_dataset = HFDataset(
            load_dataset(f"{pre}/{pre}", split=f"train.{post}", cache_dir=str(files("f5_tts").joinpath("../../data"))),
        )

    return train_dataset


# collation


def collate_fn(batch):
    mel_specs = [item["mel_spec"].squeeze(0) for item in batch]
    mel_lengths = torch.LongTensor([spec.shape[-1] for spec in mel_specs])
    max_mel_length = mel_lengths.amax()

    padded_mel_specs = []
    for spec in mel_specs:
        padding = (0, max_mel_length - spec.size(-1))
        padded_spec = F.pad(spec, padding, value=0)
        padded_mel_specs.append(padded_spec)

    mel_specs = torch.stack(padded_mel_specs)

    text = [item["text"] for item in batch]
    text_lengths = torch.LongTensor([len(item) for item in text])

    return dict(
        mel=mel_specs,
        mel_lengths=mel_lengths,  # records for padding mask
        text=text,
        text_lengths=text_lengths,
    )


class CrossUtteranceDataset(Dataset):
    """
    Samples pairs of utterances from the same speaker.
    Returns: ref_audio (for speech encoder), tgt_mel, tgt_text (for flow matching)
    Text is returned as raw string - tokenization happens in CFM.forward() via vocab_char_map.
    """
    def __init__(
        self,
        custom_dataset,
        durations=None,
        target_sample_rate=24_000,
        hop_length=256,
        n_mel_channels=100,
        n_fft=1024,
        win_length=1024,
        mel_spec_type="vocos",
        mel_spec_module=None,
        speaker_id_key="speaker_id",
    ):
        self.data = custom_dataset
        self.durations = durations
        self.target_sample_rate = target_sample_rate
        self.hop_length = hop_length
        
        self.mel_spectrogram = default(
            mel_spec_module,
            MelSpec(n_fft=n_fft, hop_length=hop_length, win_length=win_length,
                    n_mel_channels=n_mel_channels, target_sample_rate=target_sample_rate,
                    mel_spec_type=mel_spec_type),
        )
        
        # Build speaker -> indices mapping
        from collections import defaultdict
        self.spk2idx = defaultdict(list)
        for i in range(len(self.data)):
            row = self.data[i]
            spk = row.get(speaker_id_key, row.get("speaker", "unknown"))
            dur = durations[i] if durations else row.get("duration", 10)
            if 0.3 <= dur <= 30:
                self.spk2idx[spk].append(i)
        
        # Only keep speakers with 2+ utterances (needed for cross-utterance pairing)
        self.valid_idx = [i for spk, idxs in self.spk2idx.items() if len(idxs) >= 2 for i in idxs]
        print(f"CrossUtteranceDataset: {len(self.valid_idx)} samples, "
              f"{sum(1 for idxs in self.spk2idx.values() if len(idxs)>=2)} speakers")
    
    def get_frame_len(self, index):
        i = self.valid_idx[index]
        dur = self.durations[i] if self.durations else self.data[i].get("duration", 10)
        return dur * self.target_sample_rate / self.hop_length
    
    def __len__(self):
        return len(self.valid_idx)
    
    def __getitem__(self, index):
        import random
        i = self.valid_idx[index]
        row = self.data[i]
        spk = row.get("speaker_id", row.get("speaker", "unknown"))
        
        # Sample different utterance from same speaker
        candidates = [x for x in self.spk2idx[spk] if x != i]
        j = random.choice(candidates) if candidates else i
        partner = self.data[j]
        
        # Load reference audio (for speech encoder input)
        ref_audio, sr = torchaudio.load(row["audio_path"])
        if ref_audio.shape[0] > 1:
            ref_audio = ref_audio.mean(0, keepdim=True)
        if sr != self.target_sample_rate:
            ref_audio = torchaudio.transforms.Resample(sr, self.target_sample_rate)(ref_audio)
        
        # Load target audio -> mel (for flow matching target)
        tgt_audio, sr = torchaudio.load(partner["audio_path"])
        if tgt_audio.shape[0] > 1:
            tgt_audio = tgt_audio.mean(0, keepdim=True)
        if sr != self.target_sample_rate:
            tgt_audio = torchaudio.transforms.Resample(sr, self.target_sample_rate)(tgt_audio)
        
        ref_mel = self.mel_spectrogram(ref_audio).squeeze(0)
        tgt_mel = self.mel_spectrogram(tgt_audio).squeeze(0)
        
        return {
            "ref_audio": ref_audio.squeeze(0),  # (samples,) for speech encoder
            "ref_mel": ref_mel,                  # (n_mel, frames) for ref_len calculation
            "tgt_mel": tgt_mel,                  # (n_mel, frames) for flow matching
            "tgt_text": partner["text"],         # raw string, tokenized in CFM.forward()
        }
        
def cross_utterance_collate_fn(batch):
    # 1. Concatenate ref and tgt per sample
    full_mels = [torch.cat([b["ref_mel"], b["tgt_mel"]], dim=1) for b in batch]
    
    # 2. Get lengths
    ref_mel_lengths = torch.LongTensor([b["ref_mel"].shape[-1] for b in batch])
    tgt_mel_lengths = torch.LongTensor([b["tgt_mel"].shape[-1] for b in batch])
    full_mel_lengths = ref_mel_lengths + tgt_mel_lengths
    
    # 3. Pad the concatenated sequences
    max_full_len = full_mel_lengths.amax()
    padded_full_mels = []
    for mel in full_mels:
        padded_full_mels.append(F.pad(mel, (0, max_full_len - mel.shape[-1]), value=0))
    full_mels = torch.stack(padded_full_mels)
    
    # 4. Handle ref audio — track original lengths before padding
    ref_audio_lens = torch.LongTensor([b["ref_audio"].shape[-1] for b in batch])
    max_ref_audio = ref_audio_lens.amax().item()
    ref_audios = torch.stack([F.pad(b["ref_audio"], (0, max_ref_audio - b["ref_audio"].shape[-1])) for b in batch])
    
    return {
        "ref_audio": ref_audios,
        "ref_audio_lens": ref_audio_lens,      # NEW
        "mel": full_mels,
        "mel_lengths": full_mel_lengths,
        "ref_mel_lengths": ref_mel_lengths,
        "tgt_text": [b["tgt_text"] for b in batch],
    }