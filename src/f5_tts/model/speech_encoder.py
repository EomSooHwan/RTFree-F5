"""
RefFree-F5: Speech encoder and projector.
NEW FILE: Add to f5_tts/model/speech_encoder.py
"""

import torch
import torch.nn.functional as F
import torchaudio
from torch import nn


class SpeechEncoderWavLM(nn.Module):
    def __init__(self, model_name="microsoft/wavlm-large", freeze=True, target_sample_rate=16000):
        super().__init__()
        from transformers import WavLMModel
        self.encoder = WavLMModel.from_pretrained(model_name)
        self.output_dim = self.encoder.config.hidden_size
        self.target_sample_rate = target_sample_rate
        if freeze:
            self.freeze()
    
    def freeze(self):
        for p in self.encoder.parameters():
            p.requires_grad = False
    
    def unfreeze(self):
        for p in self.encoder.parameters():
            p.requires_grad = True
    
    def forward(self, audio, sample_rate=None, attention_mask=None):
        if sample_rate and sample_rate != self.target_sample_rate:
            audio = torchaudio.functional.resample(audio, sample_rate, self.target_sample_rate)
        audio = audio / (audio.abs().max(dim=-1, keepdim=True).values + 1e-8)
        with torch.no_grad():
            out = self.encoder(audio, attention_mask=attention_mask).last_hidden_state
        return out


class Projector(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or output_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.mlp[-1].weight.data *= 0.1
        self.norm = nn.LayerNorm(output_dim)
    
    def forward(self, x):
        x = self.mlp(x)
        return self.norm(x)


def align_features(features, target_len):
    if features.shape[1] == target_len:
        return features
    return F.interpolate(features.transpose(1,2), size=target_len, mode='linear', align_corners=False).transpose(1,2)