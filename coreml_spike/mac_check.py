#!/usr/bin/env python3
"""Mac-only validation, latency sampling, and NumPy-sampled end-to-end demo."""
import argparse
import json
import pathlib
import statistics
import time
import wave

import coremltools as ct
import numpy as np

from coreml_spike.common import HIFT_UPSAMPLE, MAX_CTX, MAX_SPEECH_TOKENS, artifacts_dir, max_abs_err


SPEECH_START = 6561
STOP_SPEECH = 6562
SPEECH_VOCAB = 6563
S3GEN_SIL = 4299
SAMPLE_RATE = 24000


def _read_npz(name):
    return np.load(artifacts_dir() / name, allow_pickle=False)


def _model(component, compute_units=ct.ComputeUnit.ALL):
    path = artifacts_dir() / f"{component}_fp16.mlpackage"
    return ct.models.MLModel(str(path), compute_units=compute_units)


def _filter_inputs(package, inputs):
    declared = {item.name for item in package.get_spec().description.input}
    return {name: value for name, value in inputs.items() if name in declared}


def _numpy_mask(q_len, kv_len):
    past = kv_len - q_len
    q = np.arange(q_len)[:, None] + past
    k = np.arange(kv_len)[None, :]
    mask = np.where(k <= q, 0.0, -1.0e4).astype(np.float32)
    return mask[None, None]


def _as_f32(value):
    return np.asarray(value, dtype=np.float32)


def _timed_predict(component, package, inputs, repeats=5):
    inputs = _filter_inputs(package, inputs)
    samples = []
    for _ in range(repeats):
        state = package.make_state() if component == "t3" else None
        t0 = time.perf_counter()
        if state is None:
            package.predict(inputs)
        else:
            package.predict(inputs, state=state)
        samples.append(time.perf_counter() - t0)
    return float(statistics.median(samples))


def _stateful_predict(package, inputs, state=None):
    if state is None:
        state = package.make_state()
    output = package.predict(inputs, state=state)
    return output, state


def _sample_numpy(logits, history, rng, temperature=0.8, top_k=1000, top_p=0.95, repetition_penalty=1.2):
    logits = np.asarray(logits, dtype=np.float64).copy()
    for token in set(history):
        if logits[token] < 0:
            logits[token] *= repetition_penalty
        else:
            logits[token] /= repetition_penalty
    logits /= temperature
    k = min(top_k, logits.size)
    if k < logits.size:
        threshold = np.partition(logits, -k)[-k]
        logits[logits < threshold] = -np.inf
    order = np.argsort(logits)[::-1]
    sorted_logits = logits[order]
    probabilities = np.exp(sorted_logits - np.max(sorted_logits))
    probabilities /= probabilities.sum()
    cumulative = np.cumsum(probabilities)
    remove = cumulative > top_p
    remove[1:] = remove[:-1].copy()
    remove[0] = False
    sorted_logits[remove] = -np.inf
    probabilities = np.exp(sorted_logits - np.max(sorted_logits))
    probabilities /= probabilities.sum()
    return int(order[rng.choice(order.size, p=probabilities)])


def _sample_speech_tokens(prefix, t3_model, state, speech_emb, rng, max_tokens=MAX_SPEECH_TOKENS - 3):
    length = prefix.shape[1]
    max_tokens = min(max_tokens, MAX_CTX - length)
    if max_tokens < 1:
        raise ValueError(f"T3 prefix length {length} leaves no room for speech tokens")
    output = None
    for offset in range(0, length, 512):
        chunk = _as_f32(prefix[:, offset : offset + 512])
        chunk_len = chunk.shape[1]
        inputs = {
            "inputs_embeds": chunk,
            "position_start": np.array([offset], dtype=np.int32),
            "causal_mask": _numpy_mask(chunk_len, offset + chunk_len),
        }
        output, state = _stateful_predict(t3_model, inputs, state)
    logits = output["speech_logits"][0, -1]
    tokens = []
    history = [SPEECH_START]
    for index in range(max_tokens):
        token = _sample_numpy(logits, history, rng)
        if token == STOP_SPEECH:
            break
        tokens.append(token)
        history.append(token)
        position = length + index
        next_inputs = {
            "inputs_embeds": _as_f32(speech_emb[token][None, None, :]),
            "position_start": np.array([position], dtype=np.int32),
            "causal_mask": _numpy_mask(1, position + 1),
        }
        output, state = _stateful_predict(t3_model, next_inputs, state)
        logits = output["speech_logits"][0, -1]
    # The sampler may not have queried the final sampled token if max_tokens was reached,
    # but all returned IDs are valid for the later speech-token filtering step.
    return tokens, state


def _write_wav(path, audio):
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(pcm.tobytes())


def check_references():
    cases = {
        "prefix": ("t3_prefix_ref.npz", ["text_tokens", "speaker_emb", "cond_prompt_speech_tokens"], "inputs_embeds", "inputs_embeds"),
        "t3": ("t3_stateful_ref.npz", ["inputs_embeds", "position_start", "causal_mask"], "speech_logits", "speech_logits"),
        "flow": ("flow_ref.npz", ["token", "prompt_token", "prompt_token_len", "prompt_feat", "embedding", "initial_noise", "noise"], "mel", "mel"),
        "hift": ("hift_ref.npz", ["mel", "phase", "sine_noise", "src_noise"], "wav", "wav"),
    }
    results = {}
    for component, (ref_name, input_names, output_name, expected_name) in cases.items():
        refs = _read_npz(ref_name)
        inputs = {name: refs[name] for name in input_names}
        if component in ("prefix", "t3"):
            for name in ("text_tokens", "cond_prompt_speech_tokens", "position_start"):
                if name in inputs:
                    inputs[name] = np.asarray(inputs[name], dtype=np.int32)
        if component == "flow":
            inputs["token"] = np.asarray(inputs["token"], dtype=np.int32)
            inputs["prompt_token"] = np.asarray(inputs["prompt_token"], dtype=np.int32)
            inputs["prompt_token_len"] = np.asarray(inputs["prompt_token_len"], dtype=np.int32)
        package = _model(component)
        inputs = _filter_inputs(package, inputs)
        if component == "t3":
            predicted, state = _stateful_predict(package, inputs)
            prefix_error = max_abs_err(predicted[output_name], refs[expected_name])
            speech_emb = np.load(artifacts_dir() / "speech_emb_fp16.npy", allow_pickle=False)
            incremental = []
            prefix_len = inputs["inputs_embeds"].shape[1]
            for index, token in enumerate(np.asarray(refs["incremental_tokens"]).reshape(-1)):
                position = prefix_len + index
                step_inputs = {
                    "inputs_embeds": _as_f32(speech_emb[int(token)][None, None, :]),
                    "position_start": np.array([position], dtype=np.int32),
                    "causal_mask": _numpy_mask(1, position + 1),
                }
                step_output, state = _stateful_predict(package, step_inputs, state)
                incremental.append(step_output[output_name])
            incremental_error = max_abs_err(
                np.concatenate(incremental, axis=1), refs["incremental_logits"]
            )
            error = max(prefix_error, incremental_error)
            results[component] = {
                "max_abs_error": error,
                "prefix_max_abs_error": prefix_error,
                "incremental_max_abs_error": incremental_error,
            }
        else:
            predicted = package.predict(inputs)
            error = max_abs_err(predicted[output_name], refs[expected_name])
            results[component] = {"max_abs_error": error}
        print(f"[reference] {component}: max_abs_error={error:.9g}")

        latency = {}
        for label, unit in (("ALL", ct.ComputeUnit.ALL), ("CPU_ONLY", ct.ComputeUnit.CPU_ONLY)):
            runner = package if label == "ALL" else _model(component, compute_units=unit)
            latency[label] = _timed_predict(component, runner, inputs, repeats=5)
            print(f"[latency] {component} {label}: median={latency[label]:.6f}s (5 runs)")
            if runner is not package:
                del runner
        results[component]["median_seconds"] = latency
    return results


def run_pipeline():
    out_dir = artifacts_dir()
    conds = _read_npz("conds.npz")
    speech_emb = np.load(out_dir / "speech_emb_fp16.npy", allow_pickle=False)
    tokenized = json.loads((out_dir / "tokenized_texts.json").read_text())
    prefix_model = _model("prefix")
    t3_model = _model("t3")
    flow_model = _model("flow")
    hift_model = _model("hift")
    rng = np.random.default_rng(0)

    cond_prompt = np.asarray(conds["t3_prompt_speech_tokens"], dtype=np.int32)
    speaker_emb = _as_f32(conds["t3_speaker_emb"])
    gen_prompt_token = np.asarray(conds["gen_prompt_token"], dtype=np.int32)
    gen_prompt_token_len = np.asarray(conds["gen_prompt_token_len"], dtype=np.int32)
    gen_prompt_feat = _as_f32(conds["gen_prompt_feat"])
    gen_embedding = _as_f32(conds["gen_embedding"])

    results = []
    for line in tokenized:
        index = int(line["index"])
        text_tokens = np.asarray([line["token_ids"]], dtype=np.int32)
        t0 = time.perf_counter()
        prefix = prefix_model.predict(
            {
                "text_tokens": text_tokens,
                "speaker_emb": speaker_emb,
                "cond_prompt_speech_tokens": cond_prompt,
            }
        )["inputs_embeds"]
        state = t3_model.make_state()
        sampled, _ = _sample_speech_tokens(prefix, t3_model, state, speech_emb, rng)
        valid = [token for token in sampled if token < 6561]
        speech_tokens = np.asarray([valid + [S3GEN_SIL] * 3], dtype=np.int32)
        out_len = speech_tokens.shape[1] * 2
        prompt_frames = gen_prompt_feat.shape[1]
        flow_inputs = {
            "token": speech_tokens,
            "prompt_token": gen_prompt_token,
            "prompt_token_len": gen_prompt_token_len,
            "prompt_feat": gen_prompt_feat,
            "embedding": gen_embedding,
            "initial_noise": rng.standard_normal((1, 80, prompt_frames + out_len), dtype=np.float32),
            "noise": rng.standard_normal((1, 80, out_len), dtype=np.float32),
        }
        mel = flow_model.predict(flow_inputs)["mel"]
        with _read_npz("hift_ref.npz") as hift_refs:
            hcount = int(hift_refs["phase"].shape[1])
        samples = mel.shape[-1] * HIFT_UPSAMPLE
        hift_inputs = {
            "mel": _as_f32(mel),
            "phase": rng.uniform(-np.pi, np.pi, size=(1, hcount, 1)).astype(np.float32),
            "sine_noise": rng.standard_normal((1, hcount, samples), dtype=np.float32),
            "src_noise": rng.standard_normal((1, 1, samples), dtype=np.float32),
        }
        wav = hift_model.predict(_filter_inputs(hift_model, hift_inputs))["wav"][0]
        wav_path = out_dir / f"mac_{index}.wav"
        _write_wav(wav_path, wav)
        elapsed = time.perf_counter() - t0
        duration = len(wav) / SAMPLE_RATE
        rtf = elapsed / duration if duration else float("inf")
        results.append(
            {
                "line": index,
                "sampled_tokens": sampled,
                "speech_tokens": valid,
                "time_to_audio_seconds": elapsed,
                "audio_seconds": duration,
                "rtf": rtf,
                "wav": str(wav_path),
            }
        )
        print(
            f"[pipeline] line={index} time_to_audio={elapsed:.3f}s "
            f"audio={duration:.3f}s RTF={rtf:.3f} wav={wav_path}"
        )
    (out_dir / "mac_pipeline_results.json").write_text(json.dumps(results, indent=2) + "\n")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--references-only", action="store_true")
    args = parser.parse_args()
    if hasattr(ct.utils, "_macos_version") and ct.utils._macos_version() < (15, 0):
        raise SystemExit("Core ML stateful models require macOS 15 / iOS 18.")
    refs = check_references()
    if not args.references_only:
        run_pipeline()
    print(json.dumps(refs, indent=2))


if __name__ == "__main__":
    main()
