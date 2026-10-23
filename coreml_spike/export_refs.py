#!/usr/bin/env python3
"""Establish Linux PyTorch parity and export Mac-side Core ML reference fixtures."""
import argparse
import json
import pathlib
import sys

import numpy as np
import torch

from coreml_spike.common import (
    LINES,
    MAX_CTX,
    N_CFM_TIMESTEPS,
    artifacts_dir,
    causal_mask,
    load_nano,
    max_abs_err,
)
from coreml_spike.wrappers import (
    FlowWrapper,
    HiFTWrapper,
    T3Prefix,
    T3Stateful,
    hift_noise_for,
    patched_hift_reference,
)


S3GEN_SIL = 4299


def _tokens(model, text):
    from chatterbox.tts import punc_norm

    normalized = punc_norm(text)
    ids = model.tokenizer(normalized, return_tensors="pt", padding=True, truncation=True).input_ids
    return normalized, ids.to(model.device)


def _to_np(x):
    return x.detach().cpu().numpy()


def _sample_numpy(logits, history, rng):
    logits = np.asarray(logits, dtype=np.float64).copy()
    for token in set(history):
        if logits[token] < 0:
            logits[token] *= 1.2
        else:
            logits[token] /= 1.2
    logits /= 0.8
    k = min(1000, logits.size)
    if k < logits.size:
        threshold = np.partition(logits, -k)[-k]
        logits[logits < threshold] = -np.inf
    order = np.argsort(logits)[::-1]
    sorted_logits = logits[order]
    sorted_probs = np.exp(sorted_logits - np.max(sorted_logits))
    sorted_probs /= sorted_probs.sum()
    cumulative = np.cumsum(sorted_probs)
    remove = cumulative > 0.95
    remove[1:] = remove[:-1].copy()
    remove[0] = False
    sorted_logits[remove] = -np.inf
    probs = np.exp(sorted_logits - np.max(sorted_logits))
    probs /= probs.sum()
    return int(order[rng.choice(order.size, p=probs)])


def _sample_t3(model, prefix, stateful, seed=0, max_tokens=24):
    """Run source and wrapped T3 sampling with the same NumPy RNG and processors."""
    t3 = model.t3
    rng_ref = np.random.default_rng(seed)
    rng_wrapped = np.random.default_rng(seed)
    start = torch.tensor([[t3.hp.start_speech_token]], dtype=torch.long, device=t3.device)
    prefix_len = prefix.shape[1]

    ref = t3.tfmr(inputs_embeds=prefix, use_cache=True)
    ref_past = ref.past_key_values
    logits_ref = t3.speech_head(ref[0][:, -1:, :])[0, -1].detach().cpu().numpy()

    stateful.k_cache.zero_()
    stateful.v_cache.zero_()
    mask = causal_mask(prefix_len, prefix_len, dtype=prefix.dtype).to(prefix.device)
    logits_wrapped = stateful(
        prefix,
        torch.tensor([0], dtype=torch.int32, device=prefix.device),
        mask,
    )[0, -1].detach().cpu().numpy()
    history_ref = [int(start.item())]
    history_wrapped = [int(start.item())]
    generated_ref, generated_wrapped = [], []
    for i in range(max_tokens):
        token_ref = _sample_numpy(logits_ref, history_ref, rng_ref)
        token_wrapped = _sample_numpy(logits_wrapped, history_wrapped, rng_wrapped)
        generated_ref.append(token_ref)
        generated_wrapped.append(token_wrapped)
        if token_ref == t3.hp.stop_speech_token or token_wrapped == t3.hp.stop_speech_token:
            break

        ref_token = torch.tensor([[token_ref]], dtype=torch.long, device=t3.device)
        ref_out = t3.tfmr(
            inputs_embeds=t3.speech_emb(ref_token),
            past_key_values=ref_past,
            use_cache=True,
        )
        ref_past = ref_out.past_key_values
        logits_ref = t3.speech_head(ref_out[0][:, -1:, :])[0, -1].detach().cpu().numpy()

        wr_token = torch.tensor([[token_wrapped]], dtype=torch.long, device=t3.device)
        pos = prefix_len + i
        mask = causal_mask(1, pos + 1, dtype=prefix.dtype).to(prefix.device)
        logits_wrapped = stateful(
            t3.speech_emb(wr_token),
            torch.tensor([pos], dtype=torch.int32, device=prefix.device),
            mask,
        )[0, -1].detach().cpu().numpy()
        history_ref.append(token_ref)
        history_wrapped.append(token_wrapped)

    return generated_ref, generated_wrapped


def _fixed_flow_noise(prompt_feat, token_count, seed):
    frames = prompt_feat.shape[1] + token_count * 2
    gen = torch.Generator().manual_seed(seed)
    initial_noise = torch.randn((1, 80, frames), generator=gen)
    noised_mels = torch.randn((1, 80, token_count * 2), generator=gen)
    return initial_noise, noised_mels


def _flow_reference(flow, token, conds, initial_noise, noised_mels):
    original_randn_like = torch.randn_like

    def fixed_randn_like(tensor, *args, **kwargs):
        if tuple(tensor.shape) == tuple(initial_noise.shape):
            return initial_noise.to(device=tensor.device, dtype=tensor.dtype)
        return original_randn_like(tensor, *args, **kwargs)

    torch.randn_like = fixed_randn_like
    try:
        return flow.inference(
            token=token,
            token_len=torch.tensor([token.shape[1]], dtype=torch.long),
            prompt_token=conds["prompt_token"],
            prompt_token_len=conds["prompt_token_len"],
            prompt_feat=conds["prompt_feat"],
            prompt_feat_len=conds["prompt_feat_len"],
            embedding=conds["embedding"],
            finalize=True,
            n_timesteps=N_CFM_TIMESTEPS,
            noised_mels=noised_mels,
            meanflow=True,
        )[0]
    finally:
        torch.randn_like = original_randn_like


def _flow_wrapped(flow_wrapper, token, conds, initial_noise, noised_mels):
    return flow_wrapper(
        token,
        conds["prompt_token"],
        conds["prompt_token_len"],
        conds["prompt_feat"],
        conds["embedding"],
        initial_noise,
        noised_mels,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    out_dir = artifacts_dir()
    torch.manual_seed(0)
    model = load_nano(args.device)
    t3 = model.t3
    t3_prefix = T3Prefix(t3).eval()
    t3_stateful = T3Stateful(t3).eval()
    flow_wrapper = FlowWrapper(model.s3gen.flow).eval()
    hift_wrapper = HiFTWrapper(model.s3gen.mel2wav).eval()
    parity = {"components": {}, "pipeline": []}

    # Tokenization fixtures for the Mac end-to-end runner.
    tokenized_lines = []
    all_text_tokens = []
    for i, text in enumerate(LINES):
        normalized, tokens = _tokens(model, text)
        all_text_tokens.append(tokens)
        tokenized_lines.append(
            {"index": i, "text": text, "normalized": normalized, "token_ids": tokens[0].tolist()}
        )
    (out_dir / "tokenized_texts.json").write_text(
        json.dumps(tokenized_lines, indent=2, ensure_ascii=False) + "\n"
    )

    # Prefix parity (original prepare_input_embeds versus T3Prefix).
    normalized, first_text_tokens = _tokens(model, LINES[0])
    conds_t3 = model.conds.t3
    with torch.inference_mode():
        prefix_expected, _ = t3.prepare_input_embeds(
            t3_cond=conds_t3,
            text_tokens=first_text_tokens,
            speech_tokens=torch.full_like(first_text_tokens[:, :1], t3.hp.start_speech_token),
        )
        prefix_actual = t3_prefix(
            first_text_tokens,
            conds_t3.speaker_emb,
            conds_t3.cond_prompt_speech_tokens,
        )
    prefix_err = max_abs_err(prefix_expected, prefix_actual)
    parity["components"]["t3_prefix"] = {"max_abs_error": prefix_err, "passed": prefix_err < 1e-5}
    np.savez_compressed(
        out_dir / "t3_prefix_ref.npz",
        text_tokens=_to_np(first_text_tokens),
        speaker_emb=_to_np(conds_t3.speaker_emb),
        cond_prompt_speech_tokens=_to_np(conds_t3.cond_prompt_speech_tokens),
        inputs_embeds=_to_np(prefix_expected),
    )

    # T3 state/cache parity, both full prefix and cached incremental tokens.
    fixed_tokens = torch.tensor([[13, 83, 119]], dtype=torch.long, device=t3.device)
    with torch.inference_mode():
        t3_stateful.k_cache.zero_()
        t3_stateful.v_cache.zero_()
        q_len = prefix_expected.shape[1]
        mask = causal_mask(q_len, q_len, dtype=prefix_expected.dtype).to(t3.device)
        wrapped_logits = t3_stateful(
            prefix_expected,
            torch.tensor([0], dtype=torch.int32, device=t3.device),
            mask,
        )
        ref_output = t3.tfmr(inputs_embeds=prefix_expected, use_cache=True)
        expected_logits = t3.speech_head(ref_output[0][:, -1:, :])
        errors = [max_abs_err(expected_logits, wrapped_logits)]
        ref_past = ref_output.past_key_values
        incremental_ref, incremental_wrapped = [], []
        for i in range(fixed_tokens.shape[1]):
            token = fixed_tokens[:, i : i + 1]
            position = q_len + i
            ref_output = t3.tfmr(
                inputs_embeds=t3.speech_emb(token),
                past_key_values=ref_past,
                use_cache=True,
            )
            ref_past = ref_output.past_key_values
            expected = t3.speech_head(ref_output[0][:, -1:, :])
            imask = causal_mask(1, position + 1, dtype=prefix_expected.dtype).to(t3.device)
            actual = t3_stateful(
                t3.speech_emb(token),
                torch.tensor([position], dtype=torch.int32, device=t3.device),
                imask,
            )
            errors.append(max_abs_err(expected, actual))
            incremental_ref.append(_to_np(expected))
            incremental_wrapped.append(_to_np(actual))
        t3_error = max(errors)
    parity["components"]["t3_stateful"] = {
        "max_abs_error": t3_error,
        "passed": t3_error < 1e-3,
        "prefix_error": errors[0],
        "incremental_errors": errors[1:],
    }
    np.savez_compressed(
        out_dir / "t3_stateful_ref.npz",
        inputs_embeds=_to_np(prefix_expected),
        position_start=np.array([0], dtype=np.int32),
        causal_mask=_to_np(causal_mask(q_len, q_len, dtype=prefix_expected.dtype)),
        speech_logits=_to_np(expected_logits),
        incremental_tokens=_to_np(fixed_tokens),
        incremental_logits=np.concatenate(incremental_ref, axis=1),
    )

    # Export the token embedding table used by the later Swift-side sampling loop.
    np.save(out_dir / "speech_emb_fp16.npy", t3.speech_emb.weight.detach().cpu().numpy().astype(np.float16))

    # Flow parity with the exact two-step meanflow implementation and explicit noise.
    conds_gen = {
        "prompt_token": model.conds.gen["prompt_token"].long(),
        "prompt_token_len": model.conds.gen["prompt_token_len"].long(),
        "prompt_feat": model.conds.gen["prompt_feat"].float(),
        "prompt_feat_len": model.conds.gen["prompt_feat_len"],
        "embedding": model.conds.gen["embedding"].float(),
    }
    np.savez_compressed(
        out_dir / "conds.npz",
        t3_speaker_emb=_to_np(conds_t3.speaker_emb),
        t3_prompt_speech_tokens=_to_np(conds_t3.cond_prompt_speech_tokens),
        gen_prompt_token=_to_np(conds_gen["prompt_token"]),
        gen_prompt_token_len=_to_np(conds_gen["prompt_token_len"]),
        gen_prompt_feat=_to_np(conds_gen["prompt_feat"]),
        gen_embedding=_to_np(conds_gen["embedding"]),
    )
    flow_tokens = torch.tensor([[17, 42, 99]], dtype=torch.int32, device=model.device)
    initial_noise, noised_mels = _fixed_flow_noise(conds_gen["prompt_feat"], flow_tokens.shape[1], 123)
    with torch.inference_mode():
        flow_expected = _flow_reference(model.s3gen.flow, flow_tokens, conds_gen, initial_noise, noised_mels)
        flow_actual = _flow_wrapped(flow_wrapper, flow_tokens, conds_gen, initial_noise, noised_mels)
    flow_error = max_abs_err(flow_expected, flow_actual)
    parity["components"]["flow"] = {"max_abs_error": flow_error, "passed": flow_error < 1e-4}
    np.savez_compressed(
        out_dir / "flow_ref.npz",
        token=_to_np(flow_tokens),
        prompt_token=_to_np(conds_gen["prompt_token"]),
        prompt_token_len=_to_np(conds_gen["prompt_token_len"]),
        prompt_feat=_to_np(conds_gen["prompt_feat"]),
        embedding=_to_np(conds_gen["embedding"]),
        initial_noise=_to_np(initial_noise),
        noise=_to_np(noised_mels),
        mel=_to_np(flow_expected),
    )

    # HiFT parity with its source randomness made explicit; compare real-DFT wrapper to
    # the original complex STFT/ISTFT implementation.
    phase, sine_noise, src_noise = hift_noise_for(flow_expected, seed=321, harmonics=8)
    with torch.inference_mode():
        wav_expected = patched_hift_reference(
            model.s3gen.mel2wav, flow_expected, phase, sine_noise, src_noise
        )
        wav_actual = hift_wrapper(flow_expected, phase, sine_noise, src_noise)
    hift_error = max_abs_err(wav_expected, wav_actual)
    parity["components"]["hift"] = {"max_abs_error": hift_error, "passed": hift_error < 1e-4}
    np.savez_compressed(
        out_dir / "hift_ref.npz",
        mel=_to_np(flow_expected),
        phase=_to_np(phase),
        sine_noise=_to_np(sine_noise),
        src_noise=_to_np(src_noise),
        wav=_to_np(wav_expected),
    )

    # Full token-identical source/wrapper pipeline for each requested text. Both samplers
    # use seed 0, identical NumPy sampling logic, and T3 logits as input.
    for i, (text, text_tokens) in enumerate(zip(LINES, all_text_tokens)):
        with torch.inference_mode():
            prefix, _ = t3.prepare_input_embeds(
                t3_cond=conds_t3,
                text_tokens=text_tokens,
                speech_tokens=torch.full_like(text_tokens[:, :1], t3.hp.start_speech_token),
            )
            sampled_ref, sampled_wrapper = _sample_t3(model, prefix, t3_stateful, seed=0)
            tokens_match = sampled_ref == sampled_wrapper
            kept = [tok for tok in sampled_ref if tok < 6561]
            speech_tokens = torch.tensor(
                [kept + [S3GEN_SIL] * 3], dtype=torch.int32, device=model.device
            )
            initial, noise = _fixed_flow_noise(conds_gen["prompt_feat"], len(kept) + 3, 700 + i)
            mel_ref = _flow_reference(model.s3gen.flow, speech_tokens, conds_gen, initial, noise)
            mel_wrapped = _flow_wrapped(flow_wrapper, speech_tokens, conds_gen, initial, noise)
            phase, sine_noise, src_noise = hift_noise_for(mel_ref, seed=900 + i, harmonics=8)
            wav_ref = patched_hift_reference(
                model.s3gen.mel2wav, mel_ref, phase, sine_noise, src_noise
            )
            wav_wrapped = hift_wrapper(mel_wrapped, phase, sine_noise, src_noise)
        mel_err = max_abs_err(mel_ref, mel_wrapped)
        wav_err = max_abs_err(wav_ref, wav_wrapped)
        parity["pipeline"].append(
            {
                "line": i,
                "sampled_tokens": sampled_ref,
                "wrapper_tokens": sampled_wrapper,
                "tokens_match": tokens_match,
                "flow_max_abs_error": mel_err,
                "wav_max_abs_error": wav_err,
            }
        )
        np.savez_compressed(
            out_dir / f"pipeline_line_{i}.npz",
            text_tokens=_to_np(text_tokens),
            sampled_tokens=np.asarray(sampled_ref, dtype=np.int32),
            speech_tokens=_to_np(speech_tokens),
            mel=_to_np(mel_ref),
            wav=_to_np(wav_ref),
        )
        print(
            f"[pipeline] line={i} tokens_match={tokens_match} "
            f"flow_max_abs={mel_err:.8g} wav_max_abs={wav_err:.8g}"
        )

    parity["all_passed"] = (
        all(item["passed"] for item in parity["components"].values())
        and all(
            item["tokens_match"]
            and item["flow_max_abs_error"] < 1e-4
            and item["wav_max_abs_error"] < 1e-4
            for item in parity["pipeline"]
        )
    )
    (out_dir / "parity.json").write_text(json.dumps(parity, indent=2) + "\n")
    print(json.dumps(parity, indent=2))
    print(f"[refs] wrote reference fixtures to {out_dir}")
    if not parity["all_passed"]:
        raise SystemExit("Parity gate failed; do not convert until the mismatch is resolved.")


if __name__ == "__main__":
    main()
