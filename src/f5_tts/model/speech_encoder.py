"""
RTFree-F5 speech conditioning: a frozen self-supervised speech encoder (WavLM) whose
frame-level features are projected into the F5-TTS text-conditioning space by a small MLP.

ein notation:
b - batch
nw - raw wave length
n - frame sequence
d - dimension
"""
# ruff: noqa: F722 F821

from __future__ import annotations

import torch
import torch.nn.functional as F
import torchaudio
from torch import nn


class SpeechEncoderWavLM(nn.Module):
    """Frozen WavLM feature extractor. Returns last-layer hidden states and their valid lengths."""

    def __init__(self, model_name="microsoft/wavlm-large", sample_rate=16000):
        super().__init__()
        from transformers import WavLMModel

        self.encoder = WavLMModel.from_pretrained(model_name)
        self.output_dim = self.encoder.config.hidden_size
        self.sample_rate = sample_rate
        self.freeze()

    def freeze(self):
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False

    def train(self, mode=True):  # stay in eval mode regardless of parent .train() calls
        super().train(mode)
        self.encoder.eval()
        return self

    @torch.no_grad()
    def forward(self, audio: float["b nw"], audio_lens: int["b"], sample_rate: int):
        audio = audio.float()
        if sample_rate != self.sample_rate:
            audio = torchaudio.functional.resample(audio, sample_rate, self.sample_rate)
            audio_lens = (audio_lens * self.sample_rate / sample_rate).long().clamp(max=audio.shape[-1])
        audio = audio / (audio.abs().amax(dim=-1, keepdim=True) + 1e-8)  # peak normalize

        attention_mask = torch.arange(audio.shape[-1], device=audio.device)[None, :] < audio_lens[:, None]
        feats = self.encoder(audio, attention_mask=attention_mask.long()).last_hidden_state
        feat_lens = self.encoder._get_feat_extract_output_lengths(audio_lens).clamp(max=feats.shape[1])
        return feats, feat_lens


class Projector(nn.Module):
    """Two-layer MLP with LayerNorm mapping SSL features to the text-conditioning space (Eq. 4 in the paper)."""

    def __init__(self, input_dim, output_dim, hidden_dim=None, out_init_scale=0.1):
        super().__init__()
        hidden_dim = hidden_dim or output_dim
        self.mlp = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, output_dim))
        self.mlp[-1].weight.data *= out_init_scale  # small output init for training stability
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x: float["b n d"]) -> float["b n d"]:
        return self.norm(self.mlp(x))


def align_to_frames(feats: float["n d"], num_frames: int) -> float["n' d"]:
    """Linearly interpolate a feature sequence (e.g. 50 Hz WavLM) to `num_frames` (mel frame rate)."""
    if feats.shape[0] == num_frames:
        return feats
    return F.interpolate(feats.T[None], size=num_frames, mode="linear", align_corners=False)[0].T
