# DiT-BlockSkip

A research implementation of **Memory-Efficient Fine-Tuning Diffusion Transformers via Dynamic Patch Sampling and Block Skipping** (Park et al., CVPR 2026 / arXiv:2603.20755).

This repository implements the two core ideas described in the paper:

1. **Timestep-aware dynamic patch sampling**: choose a crop size linearly between `s_min` and `s_max` as a function of diffusion timestep, round the crop to the VAE scale, then resize it back to `s_min x s_min`.
2. **Block skipping with residual precomputation**: skip the first `n` and last `m` transformer blocks during fine-tuning, while storing the residual `f_out - f_in` produced by the skipped blocks during a no-grad precomputation stage.
3. **Cross-attention block selection**: evaluate semantic changes when image-to-text cross-attention is masked in early/late block prefixes and choose `(n,m)` under `n+m=k`.
4. **LoRA on active blocks**: train only the non-skipped transformer blocks with PEFT-style low-rank adapters.

The paper reports FLUX with 57 DiT blocks (19 double-stream + 38 single-stream), skip ratios of 30/40/50%, batch size 1, gradient accumulation 4, 500 iterations, AdamW and LR 1e-4; its reported FLUX training resolution is 256x256 for DiT-BlockSkip. See the paper/supplement for exact experimental settings.

**Important:** the authors' official code was not publicly discoverable when this implementation was prepared. The core method is therefore implemented from the published paper and supplement, while the Diffusers/FLUX adapter is engineering glue and may need minor API adjustments as Diffusers evolves.

## Layout

```text
src/dit_blockskip/
  patch_sampling.py      # dynamic crop + resize
  residual_store.py      # residual feature cache with deterministic keys
  block_selection.py     # cross-attention mask selection utilities
  lora.py                # lightweight LoRA modules
  flux_adapter.py        # FLUX block traversal + residual-aware execution
  losses.py              # conditional flow matching objective
  memory.py              # CUDA memory instrumentation
  data.py                # image/prompt dataset helpers
  trainer.py             # two-stage precompute + fine-tuning loop
configs/flux_blockskip.yaml
scripts/precompute_residuals.py
scripts/train_flux.py
scripts/select_blocks.py
scripts/sanity_check.py
tests/
```

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

## Minimal workflow

### 1. Prepare data

Create a JSONL file with one object per image:

```json
{"image":"/data/subject/0001.jpg","prompt":"a photo of a person"}
```

### 2. Select the skip split

For FLUX, the paper's reported CustomConcept101 selections include:

- 30%: `(n,m)=(7,10)` with DINO
- 40%: `(13,10)` with DINO
- 50%: `(19,10)` with DINO

The code also supports re-running the selection procedure rather than hard-coding these values.

### 3. Precompute residuals

```bash
python scripts/precompute_residuals.py \
  --config configs/flux_blockskip.yaml \
  --data /data/train.jsonl \
  --output /data/residuals
```

### 4. Fine-tune

```bash
accelerate launch scripts/train_flux.py \
  --config configs/flux_blockskip.yaml \
  --data /data/train.jsonl \
  --residuals /data/residuals
```

### 5. Sanity check

```bash
python scripts/sanity_check.py
```

## Method notes

For timestep `t` and maximum timestep `T`, the paper defines

`crop_size = s_min + (t/T) * (s_max-s_min)`

and then rounds it to a multiple of the VAE downsampling scale. The resulting crop is resized to `s_min x s_min`. This preserves the training tensor shape while varying the visible spatial context with noise level.

For skipped consecutive blocks, the paper stores

`Delta f = f_out - f_in`

and in fine-tuning uses

`f_next = f_active + Delta f`

so gradients do not need to traverse the frozen skipped blocks.

## License

This repository contains an independent implementation. The paper is CC BY 4.0; model checkpoints and external libraries retain their own licenses.
