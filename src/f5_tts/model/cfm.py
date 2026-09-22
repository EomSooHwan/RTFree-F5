"""
ein notation:
b - batch
n - sequence
nt - text sequence
nw - raw wave length
d - dimension
"""
# ruff: noqa: F722 F821

from __future__ import annotations

from random import random
from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torchdiffeq import odeint

from f5_tts.model.modules import MelSpec
from f5_tts.model.speech_encoder import Projector, SpeechEncoderWavLM, align_to_frames
from f5_tts.model.utils import (
    default,
    exists,
    get_epss_timesteps,
    lens_to_mask,
    list_str_to_idx,
    list_str_to_tensor,
    mask_from_frac_lengths,
)


class CFM(nn.Module):
    def __init__(
        self,
        transformer: nn.Module,
        sigma=0.0,
        odeint_kwargs: dict = dict(
            # atol = 1e-5,
            # rtol = 1e-5,
            method="euler"  # 'midpoint'
        ),
        audio_drop_prob=0.3,
        cond_drop_prob=0.2,
        num_channels=None,
        mel_spec_module: nn.Module | None = None,
        mel_spec_kwargs: dict = dict(),
        frac_lengths_mask: tuple[float, float] = (0.7, 1.0),
        vocab_char_map: dict[str:int] | None = None,
        speech_encoder_name: str | None = None,  # RTFree-F5: e.g. "microsoft/wavlm-large"; None for vanilla F5-TTS
        projector_hidden_dim: int | None = None,  # RTFree-F5: projector hidden size, defaults to text_dim
    ):
        super().__init__()

        self.frac_lengths_mask = frac_lengths_mask

        # mel spec
        self.mel_spec = default(mel_spec_module, MelSpec(**mel_spec_kwargs))
        num_channels = default(num_channels, self.mel_spec.n_mel_channels)
        self.num_channels = num_channels

        # classifier-free guidance
        self.audio_drop_prob = audio_drop_prob
        self.cond_drop_prob = cond_drop_prob

        # transformer
        self.transformer = transformer
        dim = transformer.dim
        self.dim = dim

        # conditional flow related
        self.sigma = sigma

        # sampling related
        self.odeint_kwargs = odeint_kwargs

        # vocab map for tokenization
        self.vocab_char_map = vocab_char_map

        # RTFree-F5: frozen speech encoder + projector replacing the reference-transcript conditioning
        self.speech_encoder = None
        self.projector = None
        if speech_encoder_name is not None:
            self.speech_encoder = SpeechEncoderWavLM(speech_encoder_name)
            text_dim = transformer.text_embed.text_embed.embedding_dim
            self.projector = Projector(self.speech_encoder.output_dim, text_dim, projector_hidden_dim)

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def is_rtfree(self):
        return self.speech_encoder is not None

    def set_training_stage(self, stage: int):
        """RTFree-F5 two-stage training. Stage 1: projector only. Stage 2: projector + DiT, text encoder kept frozen."""
        assert self.is_rtfree, "set_training_stage requires a speech encoder (RTFree-F5 model)"
        assert stage in (1, 2)
        for p in self.transformer.parameters():
            p.requires_grad = stage == 2
        for p in self.transformer.text_embed.parameters():
            p.requires_grad = False
        for p in self.projector.parameters():
            p.requires_grad = True
        self.speech_encoder.freeze()
        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"RTFree-F5 training stage {stage}: {n_trainable / 1e6:.2f}M trainable parameters")

    def encode_speech(self, ref_audio: float["b nw"], ref_audio_lens: int["b"], ref_lens: int["b"]) -> float["b n d"]:
        """
        Reference audio -> projected SSL features in the text-conditioning space (H_ref in the paper).
        Each sample's features are linearly interpolated from the encoder frame rate to its
        reference mel length ref_lens[i]; output is zero-padded to max(ref_lens).
        """
        feats, feat_lens = self.speech_encoder(ref_audio, ref_audio_lens, self.mel_spec.target_sample_rate)
        feats = self.projector(feats.to(next(self.projector.parameters()).dtype))
        aligned = [align_to_frames(feats[i, : feat_lens[i]], int(ref_lens[i])) for i in range(feats.shape[0])]
        return pad_sequence(aligned, batch_first=True)

    @torch.no_grad()
    def sample(
        self,
        cond: float["b n d"] | float["b nw"],
        text: int["b nt"] | list[str],
        duration: int | int["b"],
        *,
        lens: int["b"] | None = None,
        steps=32,
        cfg_strength=1.0,
        sway_sampling_coef=None,
        seed: int | None = None,
        max_duration=65536,
        vocoder: Callable[[float["b d n"]], float["b nw"]] | None = None,
        use_epss=True,
        no_ref_audio=False,
        duplicate_test=False,
        t_inter=0.1,
        edit_mask=None,
        ref_audio: float["b nw"] | None = None,  # RTFree-F5: reference waveforms for the speech encoder
        ref_audio_lens: int["b"] | None = None,  # RTFree-F5: valid samples per reference waveform
    ):
        self.eval()
        # raw wave

        if cond.ndim == 2:
            cond = self.mel_spec(cond)
            cond = cond.permute(0, 2, 1)
            assert cond.shape[-1] == self.num_channels

        cond = cond.to(next(self.parameters()).dtype)

        batch, cond_seq_len, device = *cond.shape[:2], cond.device
        if not exists(lens):
            lens = torch.full((batch,), cond_seq_len, device=device, dtype=torch.long)

        # text

        if isinstance(text, list):
            if exists(self.vocab_char_map):
                text = list_str_to_idx(text, self.vocab_char_map).to(device)
            else:
                text = list_str_to_tensor(text).to(device)
            assert text.shape[0] == batch

        # duration

        cond_mask = lens_to_mask(lens)
        if edit_mask is not None:
            cond_mask = cond_mask & edit_mask

        if isinstance(duration, int):
            duration = torch.full((batch,), duration, device=device, dtype=torch.long)

        duration = torch.maximum(
            torch.maximum((text != -1).sum(dim=-1), lens) + 1, duration
        )  # duration at least text/audio prompt length plus one token, so something is generated
        duration = duration.clamp(max=max_duration)
        max_duration = duration.amax()

        # duplicate test corner for inner time step oberservation
        if duplicate_test:
            test_cond = F.pad(cond, (0, 0, cond_seq_len, max_duration - 2 * cond_seq_len), value=0.0)

        cond = F.pad(cond, (0, 0, 0, max_duration - cond_seq_len), value=0.0)
        if no_ref_audio:
            cond = torch.zeros_like(cond)

        cond_mask = F.pad(cond_mask, (0, max_duration - cond_mask.shape[-1]), value=False)
        cond_mask = cond_mask.unsqueeze(-1)
        step_cond = torch.where(
            cond_mask, cond, torch.zeros_like(cond)
        )  # allow direct control (cut cond audio) with lens passed in

        # RTFree-F5: speech features of the reference replace the reference transcript; `text` is the target text only
        speech_cond, ref_lens = None, None
        if ref_audio is not None:
            assert self.is_rtfree, "ref_audio given but model has no speech encoder"
            if ref_audio_lens is None:
                ref_audio_lens = torch.full((batch,), ref_audio.shape[-1], device=device, dtype=torch.long)
            ref_lens = lens
            speech_cond = self.encode_speech(ref_audio.to(device), ref_audio_lens.to(device), ref_lens)

        if batch > 1:
            mask = lens_to_mask(duration)
        else:  # save memory and speed up, as single inference need no mask currently
            mask = None

        # neural ode

        def fn(t, x):
            # at each step, conditioning is fixed
            # step_cond = torch.where(cond_mask, cond, torch.zeros_like(cond))

            # predict flow (cond)
            if cfg_strength < 1e-5:
                pred = self.transformer(
                    x=x,
                    cond=step_cond,
                    text=text,
                    time=t,
                    mask=mask,
                    drop_audio_cond=False,
                    drop_text=False,
                    cache=True,
                    speech_cond=speech_cond,
                    ref_lens=ref_lens,
                )
                return pred

            # predict flow (cond and uncond), for classifier-free guidance
            pred_cfg = self.transformer(
                x=x,
                cond=step_cond,
                text=text,
                time=t,
                mask=mask,
                cfg_infer=True,
                cache=True,
                speech_cond=speech_cond,
                ref_lens=ref_lens,
            )
            pred, null_pred = torch.chunk(pred_cfg, 2, dim=0)
            return pred + (pred - null_pred) * cfg_strength

        # noise input
        # to make sure batch inference result is same with different batch size, and for sure single inference
        # still some difference maybe due to convolutional layers
        y0 = []
        for dur in duration:
            if exists(seed):
                torch.manual_seed(seed)
            y0.append(torch.randn(dur, self.num_channels, device=self.device, dtype=step_cond.dtype))
        y0 = pad_sequence(y0, padding_value=0, batch_first=True)

        t_start = 0

        # duplicate test corner for inner time step oberservation
        if duplicate_test:
            t_start = t_inter
            y0 = (1 - t_start) * y0 + t_start * test_cond
            steps = int(steps * (1 - t_start))

        if t_start == 0 and use_epss:  # use Empirically Pruned Step Sampling for low NFE
            t = get_epss_timesteps(steps, device=self.device, dtype=step_cond.dtype)
        else:
            t = torch.linspace(t_start, 1, steps + 1, device=self.device, dtype=step_cond.dtype)
        if sway_sampling_coef is not None:
            t = t + sway_sampling_coef * (torch.cos(torch.pi / 2 * t) - 1 + t)

        trajectory = odeint(fn, y0, t, **self.odeint_kwargs)
        self.transformer.clear_cache()

        sampled = trajectory[-1]
        out = sampled
        out = torch.where(cond_mask, cond, out)

        if exists(vocoder):
            out = out.permute(0, 2, 1)
            out = vocoder(out)

        return out, trajectory

    def forward(
        self,
        inp: float["b n d"] | float["b nw"],  # mel or raw wave
        text: int["b nt"] | list[str],
        *,
        lens: int["b"] | None = None,
        noise_scheduler: str | None = None,
        ref_audio: float["b nw"] | None = None,  # RTFree-F5 cross-utterance training: reference waveforms
        ref_audio_lens: int["b"] | None = None,  # RTFree-F5: valid samples per reference waveform
        ref_lens: int["b"] | None = None,  # RTFree-F5: reference mel frames; inp = [ref mel ; target mel]
    ):
        # handle raw wave
        if inp.ndim == 2:
            inp = self.mel_spec(inp)
            inp = inp.permute(0, 2, 1)
            assert inp.shape[-1] == self.num_channels

        batch, seq_len, dtype, device, _σ1 = *inp.shape[:2], inp.dtype, self.device, self.sigma

        # handle text as string
        if isinstance(text, list):
            if exists(self.vocab_char_map):
                text = list_str_to_idx(text, self.vocab_char_map).to(device)
            else:
                text = list_str_to_tensor(text).to(device)
            assert text.shape[0] == batch

        # lens and mask
        if not exists(lens):  # if lens not acquired by trainer from collate_fn
            lens = torch.full((batch,), seq_len, device=device)
        mask = lens_to_mask(lens, length=seq_len)

        if exists(ref_lens):
            # RTFree-F5 cross-utterance infilling: the whole reference is context, the whole target is predicted
            rand_span_mask = ~lens_to_mask(ref_lens, length=seq_len) & mask
        else:
            # get a random span to mask out for training conditionally
            frac_lengths = torch.zeros((batch,), device=self.device).float().uniform_(*self.frac_lengths_mask)
            rand_span_mask = mask_from_frac_lengths(lens, frac_lengths)
            if exists(mask):
                rand_span_mask &= mask

        # mel is x1
        x1 = inp

        # x0 is gaussian noise
        x0 = torch.randn_like(x1)

        # time step
        time = torch.rand((batch,), dtype=dtype, device=self.device)
        # TODO. noise_scheduler

        # sample xt (φ_t(x) in the paper)
        t = time.unsqueeze(-1).unsqueeze(-1)
        φ = (1 - t) * x0 + t * x1
        flow = x1 - x0

        # only predict what is within the random mask span for infilling
        cond = torch.where(rand_span_mask[..., None], torch.zeros_like(x1), x1)

        # transformer and cfg training with a drop rate
        drop_audio_cond = random() < self.audio_drop_prob  # p_drop in voicebox paper
        if random() < self.cond_drop_prob:  # p_uncond in voicebox paper
            drop_audio_cond = True
            drop_text = True
        else:
            drop_text = False

        # RTFree-F5: projected reference speech features (kept even when text is dropped, as in the paper's training)
        speech_cond = None
        if exists(ref_audio):
            assert exists(ref_lens) and exists(ref_audio_lens)
            speech_cond = self.encode_speech(ref_audio, ref_audio_lens, ref_lens)

        # apply mask will use more memory; might adjust batchsize or batchsampler long sequence threshold
        pred = self.transformer(
            x=φ,
            cond=cond,
            text=text,
            time=time,
            drop_audio_cond=drop_audio_cond,
            drop_text=drop_text,
            mask=mask,
            speech_cond=speech_cond,
            ref_lens=ref_lens,
        )

        # flow matching loss
        loss = F.mse_loss(pred, flow, reduction="none")
        loss = loss[rand_span_mask]

        return loss.mean(), cond, pred
