#!/usr/bin/env python3
"""Convert parity-checked Nano wrappers to iOS 18 Core ML mlprogram packages."""
import argparse
import gc
import json
import traceback

import coremltools as ct
import numpy as np
import torch
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.frontend.torch.ops import _get_inputs
from coremltools.converters.mil.frontend.torch.torch_op_registry import register_torch_op
from coremltools.optimize.coreml._config import CompressionGranularity

from coreml_spike.common import (
    HIFT_UPSAMPLE,
    MAX_CTX,
    MAX_SPEECH_TOKENS,
    MAX_TEXT_LEN,
    N_CFM_TIMESTEPS,
    artifacts_dir,
    dir_size,
    human,
    load_nano,
)
from coreml_spike.wrappers import FlowWrapper, HiFTWrapper, T3Prefix, T3Stateful


@register_torch_op
def view_as(context, node):
    x, other = _get_inputs(context, node, expected=2)
    context.add(mb.reshape(x=x, shape=mb.shape(x=other), name=node.name))


def _tensor(name, shape, dtype=None):
    return ct.TensorType(name=name, shape=shape, dtype=dtype)


def _convert_component(component, out_dir):
    model = load_nano("cpu")
    t3 = model.t3
    conds = model.conds.gen
    output_type = None
    if component == "prefix":
        module = T3Prefix(t3).eval()
        text_len = ct.RangeDim(1, MAX_TEXT_LEN, default=16)
        input_types = [
            _tensor("text_tokens", (1, text_len), np.int32),
            _tensor("speaker_emb", (1, 256), np.float32),
            _tensor("cond_prompt_speech_tokens", (1, 375), np.int32),
        ]
        example_inputs = (
            torch.ones((1, 16), dtype=torch.long),
            model.conds.t3.speaker_emb,
            model.conds.t3.cond_prompt_speech_tokens,
        )
        states = None
        output_type = _tensor("inputs_embeds", None)
    elif component == "t3":
        module = T3Stateful(t3).eval()
        l_dim = ct.RangeDim(1, 512, default=16)
        s_dim = ct.RangeDim(1, MAX_CTX, default=16)
        input_types = [
            _tensor("inputs_embeds", (1, l_dim, 768), np.float32),
            _tensor("position_start", (1,), np.int32),
            _tensor("causal_mask", (1, 1, l_dim, s_dim), np.float32),
        ]
        example_inputs = (
            torch.zeros((1, 16, 768), dtype=torch.float32),
            torch.zeros((1,), dtype=torch.int32),
            torch.zeros((1, 1, 16, 16), dtype=torch.float32),
        )
        state_shape = (t3.cfg.n_layer, 1, t3.cfg.n_head, MAX_CTX, t3.cfg.hidden_size // t3.cfg.n_head)
        states = [
            ct.StateType(name="k_cache", wrapped_type=ct.TensorType(shape=state_shape)),
            ct.StateType(name="v_cache", wrapped_type=ct.TensorType(shape=state_shape)),
        ]
        output_type = _tensor("speech_logits", None)
    elif component == "flow":
        module = FlowWrapper(model.s3gen.flow, N_CFM_TIMESTEPS).eval()
        token_len = ct.RangeDim(1, MAX_SPEECH_TOKENS, default=3)
        prompt_len = ct.RangeDim(1, 600, default=int(conds["prompt_token"].shape[1]))
        feat_len = ct.RangeDim(1, 1200, default=int(conds["prompt_feat"].shape[1]))
        all_frames = ct.RangeDim(2, 2400, default=int(conds["prompt_feat"].shape[1] + 6))
        gen_frames = ct.RangeDim(2, 1200, default=6)
        input_types = [
            _tensor("token", (1, token_len), np.int32),
            _tensor("prompt_token", (1, prompt_len), np.int32),
            _tensor("prompt_token_len", (1,), np.int32),
            _tensor("prompt_feat", (1, feat_len, 80), np.float32),
            _tensor("embedding", (1, 192), np.float32),
            _tensor("initial_noise", (1, 80, all_frames), np.float32),
            _tensor("noise", (1, 80, gen_frames), np.float32),
        ]
        example_inputs = (
            torch.ones((1, 3), dtype=torch.int32),
            conds["prompt_token"].long(),
            conds["prompt_token_len"].int(),
            conds["prompt_feat"].float(),
            conds["embedding"].float(),
            torch.zeros((1, 80, 506), dtype=torch.float32),
            torch.zeros((1, 80, 6), dtype=torch.float32),
        )
        states = None
        output_type = _tensor("mel", None)
    elif component == "hift":
        module = HiFTWrapper(model.s3gen.mel2wav).eval()
        mel_len = ct.RangeDim(1, MAX_SPEECH_TOKENS * 2, default=6)
        sample_len = ct.RangeDim(HIFT_UPSAMPLE, MAX_SPEECH_TOKENS * 2 * HIFT_UPSAMPLE, default=6 * HIFT_UPSAMPLE)
        harmonics = model.s3gen.mel2wav.nb_harmonics + 1
        input_types = [
            _tensor("mel", (1, 80, mel_len), np.float32),
            _tensor("phase", (1, harmonics, 1), np.float32),
            _tensor("sine_noise", (1, harmonics, sample_len), np.float32),
            _tensor("src_noise", (1, 1, sample_len), np.float32),
        ]
        example_inputs = (
            torch.zeros((1, 80, 6), dtype=torch.float32),
            torch.zeros((1, harmonics, 1), dtype=torch.float32),
            torch.zeros((1, harmonics, 6 * HIFT_UPSAMPLE), dtype=torch.float32),
            torch.zeros((1, 1, 6 * HIFT_UPSAMPLE), dtype=torch.float32),
        )
        states = None
        output_type = _tensor("wav", None)
    else:
        raise ValueError(component)

    torch_model = torch.jit.trace(module, example_inputs, strict=False)
    converted = ct.convert(
        torch_model,
        source="pytorch",
        inputs=input_types,
        outputs=[output_type],
        states=states,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=ct.precision.FLOAT16,
    )
    fp16_path = out_dir / f"{component}_fp16.mlpackage"
    converted.save(str(fp16_path))
    sizes = {"fp16_bytes": dir_size(fp16_path), "fp16": human(dir_size(fp16_path))}

    try:
        linear_config = ct.optimize.coreml.OptimizationConfig(
            global_config=ct.optimize.coreml.OpLinearQuantizerConfig(
                mode="linear_symmetric",
                dtype=np.int8,
                granularity=CompressionGranularity.PER_CHANNEL,
            )
        )
        quantized = ct.optimize.coreml.linear_quantize_weights(converted, linear_config)
        int8_path = out_dir / f"{component}_int8.mlpackage"
        quantized.save(str(int8_path))
        sizes["int8_bytes"] = dir_size(int8_path)
        sizes["int8"] = human(sizes["int8_bytes"])
        del quantized
    except Exception as exc:
        sizes["int8_error"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    gc.collect()

    try:
        palette_config = ct.optimize.coreml.OptimizationConfig(
            global_config=ct.optimize.coreml.OpPalettizerConfig(
                mode="kmeans",
                nbits=6,
            )
        )
        palettized = ct.optimize.coreml.palettize_weights(converted, palette_config)
        palette_path = out_dir / f"{component}_6bit.mlpackage"
        palettized.save(str(palette_path))
        sizes["6bit_bytes"] = dir_size(palette_path)
        sizes["6bit"] = human(sizes["6bit_bytes"])
        del palettized
    except Exception as exc:
        sizes["6bit_error"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    del converted, torch_model, module, model
    gc.collect()
    return sizes


def _quantize_existing(component, out_dir):
    fp16_path = out_dir / f"{component}_fp16.mlpackage"
    model = ct.models.MLModel(str(fp16_path), skip_model_load=True)
    sizes = {"fp16_bytes": dir_size(fp16_path), "fp16": human(dir_size(fp16_path))}
    try:
        path = out_dir / f"{component}_int8.mlpackage"
        if not path.exists():
            config = ct.optimize.coreml.OptimizationConfig(
                global_config=ct.optimize.coreml.OpLinearQuantizerConfig(
                    mode="linear_symmetric",
                    dtype=np.int8,
                    granularity=CompressionGranularity.PER_CHANNEL,
                )
            )
            quantized = ct.optimize.coreml.linear_quantize_weights(model, config)
            quantized.save(str(path))
        sizes["int8_bytes"] = dir_size(path)
        sizes["int8"] = human(sizes["int8_bytes"])
    except Exception as exc:
        sizes["int8_error"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    try:
        config = ct.optimize.coreml.OptimizationConfig(
            global_config=ct.optimize.coreml.OpPalettizerConfig(
                mode="kmeans",
                nbits=6,
            )
        )
        palettized = ct.optimize.coreml.palettize_weights(model, config)
        path = out_dir / f"{component}_6bit.mlpackage"
        palettized.save(str(path))
        sizes["6bit_bytes"] = dir_size(path)
        sizes["6bit"] = human(sizes["6bit_bytes"])
    except Exception as exc:
        sizes["6bit_error"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    return sizes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--component",
        choices=("prefix", "t3", "flow", "hift", "all"),
        default="all",
    )
    parser.add_argument("--quantize-existing", action="store_true")
    args = parser.parse_args()
    out_dir = artifacts_dir()
    parity_path = out_dir / "parity.json"
    if not parity_path.exists():
        raise SystemExit("Reference parity is required first: run python -m coreml_spike.export_refs")
    parity = json.loads(parity_path.read_text())
    if not parity.get("all_passed"):
        raise SystemExit("Parity gate failed; refusing Core ML conversion.")

    components = ("prefix", "t3", "flow", "hift") if args.component == "all" else (args.component,)
    report_path = out_dir / "conversion_status.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    for component in components:
        print(f"[convert] starting {component}")
        try:
            if args.quantize_existing:
                report[component] = {"status": "converted", "sizes": _quantize_existing(component, out_dir)}
            else:
                report[component] = {"status": "converted", "sizes": _convert_component(component, out_dir)}
            print(f"[convert] {component}: {report[component]['sizes']}")
        except Exception as exc:
            report[component] = {
                "status": "blocked",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            print(f"[convert] BLOCKED {component}: {type(exc).__name__}: {exc}")
            traceback.print_exc()
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        gc.collect()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
