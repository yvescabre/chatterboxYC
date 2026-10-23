# Chatterbox-Nano Core ML spike

Time-boxed feasibility work, not product code. The source tree is isolated under
`coreml_spike/`; reference arrays, converted packages, logs, and generated WAVs live
outside the repository in `~/nano-spike/coreml/` by default. Override that location
with `NANO_SPIKE_DIR`.

## Runtime boundary

The proposed on-device modules are:

1. `T3Stateful`: GPT2-small (`hidden_size=768`, 12 layers / 12 heads), with 1024-token
   key/value states and a 6563-way speech head. The unused T3 `text_head` is not wrapped.
2. `FlowWrapper`: S3Gen meanflow with two CFM Euler steps. Speech tokens, prompt token /
   mel features, speaker embedding, initial context noise, and generated-region noise
   are inputs, so voice conditionals can be swapped.
3. `HiFTWrapper`: HiFT mel-to-waveform. F0 prediction and source synthesis remain in the
   wrapper; sine phase and source-noise arrays are explicit inputs. The vocoder's real
   DFT implementation uses real-valued convolution/matmul rather than complex STFT ops.

`T3Prefix` is an additional small model. It projects the offline speaker embedding,
embeds the 375 conditioning speech tokens and BPE text, and appends the start-speech
embedding. `speech_emb_fp16.npy` exports the speech embedding table used for generated
tokens in the sampling loop. Offline `s3gen.tokenizer`, `s3gen.speaker_encoder`, and
`ve` are not part of Core ML; their already-computed conditionals are exported as
`conds.npz`. Swift is expected to own BPE tokenization and sampling in production.

The Perth watermark is out of scope for this spike. It must be considered separately
before shipping generated audio.

## Linux: parity fixtures and conversion

Use the Chatterbox repository root as the working directory. In this spike environment:

```bash
source ~/venvs/chatterbox/bin/activate
python -m pip install coremltools==8.3
export NANO_SPIKE_DIR=~/nano-spike/coreml
python -m coreml_spike.export_refs
```

`export_refs` loads Nano in CPU mode, tokenizes the three requested lines, compares each
wrapper to the original PyTorch module using fixed inputs/noise, and writes
`parity.json`, `tokenized_texts.json`, `conds.npz`, `speech_emb_fp16.npy`, and compressed
NPZ input/output references to `NANO_SPIKE_DIR`. It also samples T3 with seed 0 using
the same NumPy sampler for the source and cached wrapper, then compares the flow and
HiFT stages on identical tokens. Conversion is gated on those parity results.

Run conversions after parity passes:

```bash
python -m coreml_spike.convert --component prefix
python -m coreml_spike.convert --component t3
python -m coreml_spike.convert --component flow
python -m coreml_spike.convert --component hift
# or: python -m coreml_spike.convert --component all
```

Each conversion targets `mlprogram`, iOS 18, and FP16 compute. The converter also
attempts per-channel symmetric INT8 weight quantization with
`ct.optimize.coreml.linear_quantize_weights` and 6-bit k-means palettization. Sizes and
conversion errors are recorded in `conversion_status.json`. To re-run compression on
already converted packages, use `--quantize-existing` with a component name.

Latest Linux conversion results (package sizes):

| Component | FP16 | INT8 per-channel | 6-bit palette |
| --- | ---: | ---: | ---: |
| T3 prefix | 83.6 MiB | 41.9 MiB | 31.4 MiB |
| Stateful T3 | 173.6 MiB | 87.2 MiB | 65.4 MiB |
| HiFT | 39.9 MiB | 20.2 MiB | 15.2 MiB |
| Flow | blocked | — | — |

Flow conversion stops at the Core ML Tools 8.3 PyTorch frontend error:
`RuntimeError: PyTorch convert function for op 'view_as' not implemented.` No Flow
package is available for Mac end-to-end checks yet. No Core ML prediction is attempted
on Linux.

## Mac: Core ML references, timing, and audio

On macOS 15 or later, install a Python environment with Core ML Tools 8.3, NumPy, and
PyTorch 2.6.0, make the repository importable, and copy/reference the artifact directory
created on Linux. Then run:

```bash
source ~/venvs/chatterbox/bin/activate
export NANO_SPIKE_DIR=~/nano-spike/coreml
python -m coreml_spike.mac_check
```

The Mac runner loads each FP16 `.mlpackage`, reports max absolute error against its NPZ
reference, and measures median inference latency across five runs with
`ComputeUnit.ALL` and `ComputeUnit.CPU_ONLY`. It then runs prefix -> stateful T3 sampling
-> two-step meanflow -> HiFT for all three lines using NumPy sampling seed 0, writes
`mac_0.wav`, `mac_1.wav`, and `mac_2.wav` at 24 kHz, and reports time-to-audio and RTF.
Use `--references-only` to omit the end-to-end WAV run.

The NumPy sampler uses temperature 0.8, top-k 1000, top-p 0.95, repetition penalty 1.2,
stops at speech token 6562, removes generated IDs >= 6561 before S3Gen, and appends
three S3Gen silence tokens (4299). The Mac timing runner is a research harness, not a
production Swift implementation.

## Known feasibility limits

Core ML prediction requires macOS/iOS. The Linux pass can establish PyTorch parity and
attempt conversion, but cannot establish device correctness, latency, memory use, or
audio quality. In particular, the stateful GPT2 cache must be validated on iOS 18 or
later, and the full Nano model's performance on an iPad remains unknown until the Mac
and device checks are run. Flow's unsupported `view_as` conversion op also remains an
open blocker for the complete pipeline.
