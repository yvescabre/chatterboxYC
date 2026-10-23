"""Pure-torch, Core ML friendly wrappers around the three on-device Nano modules.

On device: t3 (GPT2_small), s3gen.flow (meanflow, 2 CFM steps), s3gen.mel2wav (HiFT).
Off device: text BPE tokenization, the sampling loop, s3gen.tokenizer, speaker encoder,
the voice encoder (voice conditionals are precomputed offline) and Perth watermarking
(out of scope for this spike).
"""
import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .common import HIFT_UPSAMPLE, MAX_CTX, N_CFM_TIMESTEPS


# --------------------------------------------------------------------------------------
# T3
# --------------------------------------------------------------------------------------
class T3Prefix(nn.Module):
    """Builds the T3 prefix embeddings: [speaker | prompt speech | text | start speech]."""

    def __init__(self, t3):
        super().__init__()
        self.text_emb = nn.Embedding.from_pretrained(t3.text_emb.weight.detach().clone(), freeze=True)
        self.speech_emb = nn.Embedding.from_pretrained(t3.speech_emb.weight.detach().clone(), freeze=True)
        self.spkr_enc = nn.Linear(t3.cond_enc.spkr_enc.in_features, t3.cond_enc.spkr_enc.out_features)
        self.spkr_enc.load_state_dict(t3.cond_enc.spkr_enc.state_dict())
        self.start_speech_token = int(t3.hp.start_speech_token)

    def forward(self, text_tokens, speaker_emb, cond_prompt_speech_tokens):
        speaker = self.spkr_enc(speaker_emb.reshape(-1, self.spkr_enc.in_features))[:, None, :]
        prompt = self.speech_emb(cond_prompt_speech_tokens.long())
        text = self.text_emb(text_tokens.long())
        start_token = torch.ones_like(text_tokens[:, :1]) * self.start_speech_token
        start = self.speech_emb(start_token)
        return torch.cat([speaker, prompt, text, start], dim=1)


class T3Stateful(nn.Module):
    """GPT2_small backbone with Core ML states for the key/value cache.

    inputs_embeds  [1, L, 768]
    position_start [1] int32      - absolute position of the first query token
    causal_mask    [1, 1, L, S]   - additive mask; S == past_kv_len + L
    states         k_cache/v_cache [n_layer, 1, n_head, MAX_CTX, head_dim]
    output         speech logits of the last position, [1, 1, 6563]
    `text_head` is dropped (unused at inference).
    """

    def __init__(self, t3, max_ctx: int = MAX_CTX):
        super().__init__()
        cfg = t3.cfg
        self.n_layer = cfg.n_layer
        self.n_head = cfg.n_head
        self.n_embd = cfg.hidden_size
        self.head_dim = self.n_embd // self.n_head
        self.max_ctx = max_ctx
        self.h = t3.tfmr.h
        self.ln_f = t3.tfmr.ln_f
        self.speech_head = t3.speech_head
        self.register_buffer("wpe", t3.tfmr.wpe.weight[:max_ctx].detach().clone())
        cache_shape = (self.n_layer, 1, self.n_head, max_ctx, self.head_dim)
        self.register_buffer("k_cache", torch.zeros(cache_shape))
        self.register_buffer("v_cache", torch.zeros(cache_shape))

    def _split_heads(self, x):
        b, l, _ = x.shape
        return x.view(b, l, self.n_head, self.head_dim).transpose(1, 2)

    def forward(self, inputs_embeds, position_start, causal_mask):
        q_len = inputs_embeds.shape[1]
        end_step = causal_mask.shape[-1]
        past_len = end_step - q_len

        positions = position_start.long() + torch.arange(q_len, device=inputs_embeds.device)
        hidden = inputs_embeds + self.wpe.index_select(0, positions).unsqueeze(0)

        for i, block in enumerate(self.h):
            residual = hidden
            x = block.ln_1(hidden)
            qkv = block.attn.c_attn(x)
            q, k, v = qkv.split(self.n_embd, dim=2)
            q = self._split_heads(q)
            k = self._split_heads(k)
            v = self._split_heads(v)

            self.k_cache[i, :, :, past_len:end_step, :] = k
            self.v_cache[i, :, :, past_len:end_step, :] = v
            keys = self.k_cache[i, :, :, :end_step, :]
            values = self.v_cache[i, :, :, :end_step, :]

            scores = torch.matmul(q, keys.transpose(-1, -2)) / math.sqrt(self.head_dim)
            scores = scores + causal_mask
            attn = torch.matmul(torch.softmax(scores, dim=-1), values)
            attn = attn.transpose(1, 2).reshape(1, q_len, self.n_embd)
            hidden = residual + block.attn.c_proj(attn)

            residual = hidden
            hidden = residual + block.mlp(block.ln_2(hidden))

        hidden = self.ln_f(hidden)
        return self.speech_head(hidden[:, -1:, :])


def t3_reference_logits(t3, prefix_embeds, step_tokens):
    """Reference (unwrapped) T3: prefix forward + incremental steps. Returns a list of logits."""
    out = []
    res = t3.tfmr(inputs_embeds=prefix_embeds, use_cache=True)
    past = res.past_key_values
    out.append(t3.speech_head(res[0][:, -1:, :]))
    for tok in step_tokens:
        emb = t3.speech_emb(tok.view(1, 1))
        res = t3.tfmr(inputs_embeds=emb, past_key_values=past, use_cache=True)
        past = res.past_key_values
        out.append(t3.speech_head(res[0]))
    return out


# --------------------------------------------------------------------------------------
# Flow (meanflow, 2 CFM steps)
# --------------------------------------------------------------------------------------
class FlowWrapper(nn.Module):
    """speech tokens + offline voice conditionals + noise -> mel.

    Replicates `S3Token2Mel.forward` / `CausalMaskedDiffWithXvec.inference` with
    `finalize=True`, `meanflow=True`, `n_timesteps=2`.

    inputs
      token        [1, T]            int32, generated speech tokens (prompt excluded)
      prompt_token [1, Tp]           int32, from conds.gen
      prompt_token_len [1]           int32, from conds.gen
      prompt_feat  [1, 2*Tp, 80]     from conds.gen
      embedding    [1, 192]          speaker x-vector from conds.gen
      initial_noise [1, 80, 2*(Tp+T)] CFM latent noise before noised-mel replacement
      noise        [1, 80, 2*T]      the explicit noised_mels argument
    output
      mel          [1, 80, 2*T]
    """

    def __init__(self, flow, n_timesteps: int = N_CFM_TIMESTEPS):
        super().__init__()
        self.flow = flow
        self.n_timesteps = n_timesteps
        t_span = torch.linspace(0, 1, n_timesteps + 1)
        self.register_buffer("t_span", t_span)

    def forward(self, token, prompt_token, prompt_token_len, prompt_feat, embedding, initial_noise, noise):
        flow = self.flow
        embedding = F.normalize(embedding, dim=1)
        embedding = flow.spk_embed_affine_layer(embedding)

        tokens = torch.cat([prompt_token.long(), token.long()], dim=1)
        n_tok = tokens.shape[1]
        token_len = prompt_token_len.long() + (tokens.shape[1] - prompt_token.shape[1])
        emb = flow.input_embedding(tokens)

        h, h_masks = flow.encoder(emb, token_len)
        h = flow.encoder_proj(h)
        mu = h.transpose(1, 2).contiguous()

        mel_len1 = prompt_feat.shape[1]
        conds = torch.cat([prompt_feat, torch.zeros_like(mu.transpose(1, 2))[:, mel_len1:, :]], dim=1)
        conds = conds.transpose(1, 2)

        mask = h_masks.to(mu)
        x = torch.cat([initial_noise[:, :, :mel_len1], noise], dim=2)
        speaker = embedding.unsqueeze(-1) + torch.zeros_like(x)
        conds = torch.cat([speaker, conds], dim=1)
        for i in range(self.n_timesteps):
            t = self.t_span[i].view(1)
            r = self.t_span[i + 1].view(1)
            dxdt = flow.decoder.estimator.forward(x, mask=mask, mu=mu, t=t, cond=conds, r=r)
            x = x + (r - t) * dxdt

        return x[:, :, mel_len1:]


# --------------------------------------------------------------------------------------
# HiFT vocoder
# --------------------------------------------------------------------------------------
class HiFTWrapper(nn.Module):
    """mel -> waveform, with all randomness lifted into explicit inputs and with
    `torch.stft`/`torch.istft` replaced by real-valued conv/DFT-basis implementations.

    inputs
      mel        [1, 80, T]
      phase      [1, nb_harmonics+1, 1]  uniform(-pi, pi) sine phases (index 0 is zeroed)
      sine_noise [1, nb_harmonics+1, N]  standard normal, N = 120*T
      src_noise  [1, 1, N]               standard normal
    output
      wav        [1, N]
    """

    def __init__(self, mel2wav, use_torch_stft: bool = False):
        super().__init__()
        self.g = mel2wav
        self.use_torch_stft = use_torch_stft
        self.n_fft = mel2wav.istft_params["n_fft"]
        self.hop = mel2wav.istft_params["hop_len"]
        self.pad = self.n_fft // 2
        self.n_bins = self.n_fft // 2 + 1
        self.harmonics = mel2wav.nb_harmonics

        window = mel2wav.stft_window.detach().clone()
        n, k = self.n_fft, self.n_fft
        idx_n = torch.arange(n, dtype=torch.float64)
        idx_k = torch.arange(self.n_bins, dtype=torch.float64)
        ang = 2 * math.pi * idx_k[:, None] * idx_n[None, :] / n
        # analysis: real/imag = conv1d with windowed cos/-sin basis
        self.register_buffer("stft_cos", (torch.cos(ang).float() * window).unsqueeze(1))
        self.register_buffer("stft_sin", (-torch.sin(ang).float() * window).unsqueeze(1))
        # synthesis: full spectrum inverse DFT basis (conjugate symmetry rebuilt in forward)
        idx_kk = torch.arange(k, dtype=torch.float64)
        ang_full = 2 * math.pi * idx_n[:, None] * idx_kk[None, :] / n
        self.register_buffer("idft_cos", torch.cos(ang_full).float())
        self.register_buffer("idft_sin", torch.sin(ang_full).float())
        self.register_buffer("window", window)
        self.register_buffer("ola_weight", torch.eye(self.n_fft).view(self.n_fft, 1, self.n_fft))
        phase_mask = torch.ones(1, self.harmonics + 1, 1)
        phase_mask[:, 0, :] = 0.0
        self.register_buffer("phase_mask", phase_mask)

    # ---- real-valued STFT / ISTFT ----
    def _stft(self, x):
        if self.use_torch_stft:
            return self.g._stft(x)
        xp = F.pad(x.unsqueeze(1), (self.pad, self.pad), mode="reflect")
        real = F.conv1d(xp, self.stft_cos, stride=self.hop)
        imag = F.conv1d(xp, self.stft_sin, stride=self.hop)
        return real, imag

    def _istft(self, magnitude, phase):
        if self.use_torch_stft:
            return self.g._istft(magnitude, phase)
        magnitude = torch.clip(magnitude, max=1e2)
        real = magnitude * torch.cos(phase)
        imag = magnitude * torch.sin(phase)
        # rebuild the full spectrum by conjugate symmetry
        full_real = torch.cat([real, real.flip(1)[:, 1:-1, :]], dim=1)
        full_imag = torch.cat([imag, -imag.flip(1)[:, 1:-1, :]], dim=1)
        frames = (torch.matmul(self.idft_cos, full_real) - torch.matmul(self.idft_sin, full_imag)) / self.n_fft
        frames = frames * self.window.view(1, -1, 1)
        wav = F.conv_transpose1d(frames, self.ola_weight, stride=self.hop)
        wsq = (self.window ** 2).view(1, -1, 1).expand(1, self.n_fft, frames.shape[-1])
        norm = F.conv_transpose1d(wsq, self.ola_weight, stride=self.hop)
        wav = wav / (norm + 1e-11)
        return wav[:, 0, self.pad:-self.pad]

    def _source(self, mel, phase, sine_noise, src_noise):
        g = self.g
        f0 = g.f0_predictor(mel)
        f0_up = g.f0_upsamp(f0[:, None]).transpose(1, 2)  # [1, N, 1]
        f0_t = f0_up.transpose(1, 2)  # [1, 1, N]

        sine = g.m_source.l_sin_gen
        mult = torch.arange(1, self.harmonics + 2, dtype=f0_t.dtype, device=f0_t.device).view(1, -1, 1)
        f_mat = f0_t * mult / sine.sampling_rate
        theta = 2 * math.pi * (torch.cumsum(f_mat, dim=-1) % 1)
        sine_waves = sine.sine_amp * torch.sin(theta + phase * self.phase_mask)
        uv = (f_mat[:, :1, :] * 0 + (f0_t > sine.voiced_threshold).to(f0_t.dtype))
        uv_h = (f0_t > sine.voiced_threshold).to(f0_t.dtype).expand(-1, self.harmonics + 1, -1)
        noise_amp = uv_h * sine.noise_std + (1 - uv_h) * sine.sine_amp / 3
        sine_waves = sine_waves * uv_h + noise_amp * sine_noise

        sine_merge = g.m_source.l_tanh(g.m_source.l_linear(sine_waves.transpose(1, 2)))
        s = sine_merge.transpose(1, 2) * 0 + sine_merge.transpose(1, 2)
        _ = (uv, src_noise)  # noise branch is unused by HiFTGenerator.decode
        return s, f0

    def forward(self, mel, phase, sine_noise, src_noise):
        g = self.g
        s, _ = self._source(mel, phase, sine_noise, src_noise)

        s_real, s_imag = self._stft(s.squeeze(1))
        s_stft = torch.cat([s_real, s_imag], dim=1)

        x = g.conv_pre(mel)
        for i in range(g.num_upsamples):
            x = F.leaky_relu(x, g.lrelu_slope)
            x = g.ups[i](x)
            if i == g.num_upsamples - 1:
                x = g.reflection_pad(x)
            si = g.source_downs[i](s_stft)
            si = g.source_resblocks[i](si)
            x = x + si
            xs = None
            for j in range(g.num_kernels):
                block = g.resblocks[i * g.num_kernels + j]
                xs = block(x) if xs is None else xs + block(x)
            x = xs / g.num_kernels

        x = F.leaky_relu(x)
        x = g.conv_post(x)
        magnitude = torch.exp(x[:, : self.n_bins, :])
        ph = torch.sin(x[:, self.n_bins:, :])
        wav = self._istft(magnitude, ph)
        return torch.clamp(wav, -g.audio_limit, g.audio_limit)


def hift_noise_for(mel: torch.Tensor, seed: int = 0, harmonics: int = 8):
    """Draws the HiFT randomness in exactly the order `HiFTGenerator.inference` draws it."""
    n_samples = mel.shape[-1] * HIFT_UPSAMPLE
    g = torch.Generator().manual_seed(seed)
    phase = torch.rand((1, harmonics + 1, 1), generator=g) * (2 * math.pi) - math.pi
    sine_noise = torch.randn((1, harmonics + 1, n_samples), generator=g)
    src_noise = torch.randn((1, 1, n_samples), generator=g)
    return phase, sine_noise, src_noise


def patched_hift_reference(mel2wav, mel, phase, sine_noise, src_noise):
    """Runs the original HiFTGenerator.inference with the randomness forced to fixed values."""
    import contextlib

    from chatterbox.models.s3gen import hifigan

    sine = mel2wav.m_source.l_sin_gen

    def fake_forward(f0):
        f_mat = torch.zeros((f0.size(0), sine.harmonic_num + 1, f0.size(-1)))
        for i in range(sine.harmonic_num + 1):
            f_mat[:, i: i + 1, :] = f0 * (i + 1) / sine.sampling_rate
        theta = 2 * np.pi * (torch.cumsum(f_mat, dim=-1) % 1)
        ph = phase.clone()
        ph[:, 0, :] = 0
        waves = sine.sine_amp * torch.sin(theta + ph)
        uv = (f0 > sine.voiced_threshold).type(torch.float32).expand(-1, sine.harmonic_num + 1, -1)
        noise_amp = uv * sine.noise_std + (1 - uv) * sine.sine_amp / 3
        waves = waves * uv + noise_amp * sine_noise
        return waves, uv[:, :1, :], None

    orig_sine = type(sine).forward
    orig_randn_like = torch.randn_like

    def fake_randn_like(t, *a, **kw):
        if t.shape == src_noise.shape:
            return src_noise
        return orig_randn_like(t, *a, **kw)

    with contextlib.ExitStack() as stack:
        type(sine).forward = lambda self, f0: fake_forward(f0)
        torch.randn_like = fake_randn_like
        stack.callback(lambda: setattr(type(sine), "forward", orig_sine))
        stack.callback(lambda: setattr(torch, "randn_like", orig_randn_like))
        wav, _ = mel2wav.inference(speech_feat=mel, cache_source=torch.zeros(1, 1, 0))
    _ = hifigan
    return wav
