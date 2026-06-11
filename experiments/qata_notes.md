# QATA Notes

## 2026-06-03 Minimal QATA Implementation

### Modified Files

- `model/quality_aggregation.py`
- `model/make_model_clipreid.py`
- `config/defaults.py`
- `configs/vit_clipreid.yml`
- `configs/vit_clipreid_ilids.yml`
- `configs/vit_clipreid_qata.yml`
- `configs/vit_clipreid_ilids_qata.yml`

### Configuration Difference

- Baseline configs keep `MODEL.QATA.ENABLED: False`.
- QATA configs set `MODEL.QATA.ENABLED: True`.
- MARS QATA output directory: `logs/mars_vit_clip_reid_qata`.
- iLIDS-VID QATA output directory: `logs/ilids_vit_clip_reid_qata`.

### Scope

- Replaced only the model-internal video-level pooling for `img_feature` and `img_feature_proj`.
- Did not modify TMD aggregation, CLIP-Memory generation, dense inference, or stage2 processor code.

### Sanity Checks

- `python3 -m py_compile model/quality_aggregation.py model/make_model_clipreid.py config/defaults.py`: passed.
- Config merge check with `/data1/lgf/miniconda3/envs/tfclip/bin/python`: passed.
- `QualityWeightedPooling` smoke test: output shape and weight normalization passed.

### Experiment Commands

MARS:

```bash
CUDA_VISIBLE_DEVICES=0 /data1/lgf/miniconda3/envs/tfclip/bin/python train.py --config_file configs/vit_clipreid_qata.yml
```

iLIDS-VID:

```bash
CUDA_VISIBLE_DEVICES=0 /data1/lgf/miniconda3/envs/tfclip/bin/python train.py --config_file configs/vit_clipreid_ilids_qata.yml
```

### Results

- Training not started in this implementation pass.
- mAP / Rank-1 / Rank-5: pending.

## 2026-06-03 Sanity Check

### Commands

```bash
git status --short
git diff --name-only
python -m py_compile model/quality_aggregation.py
python -m py_compile model/make_model_clipreid.py
python -m py_compile config/defaults.py
/data1/lgf/miniconda3/envs/tfclip/bin/python -m py_compile model/quality_aggregation.py
/data1/lgf/miniconda3/envs/tfclip/bin/python -m py_compile model/make_model_clipreid.py
/data1/lgf/miniconda3/envs/tfclip/bin/python -m py_compile config/defaults.py
/data1/lgf/miniconda3/envs/tfclip/bin/python /tmp/check_qata_build.py --config_file configs/vit_clipreid_qata.yml
/data1/lgf/miniconda3/envs/tfclip/bin/python /tmp/check_qata_forward_cpu.py
```

### Results

- `python -m py_compile ...`: failed because `python` is not available in the shell.
- tfclip env `py_compile`: passed for `model/quality_aggregation.py`, `model/make_model_clipreid.py`, and `config/defaults.py`.
- MARS QATA config merge: passed.
- Model build with the real project path reached CLIP loading, but failed at `clip_model.to("cuda")` because no CUDA GPUs are available in this session.
- CPU-only temporary shape check passed:
  - baseline config output feature shape: `(1, 2048)`
  - QATA config output feature shape: `(1, 2048)`
  - QATA config has `qpool_768` and `qpool_512`.
- MARS dataset metadata loaded successfully.
- Fetching one real dataloader batch did not complete in a reasonable time in this environment, so the temporary process was terminated. No training was started and no checkpoint was saved.

### Notes

- Strict file-scope check found two additional modified baseline config files: `configs/vit_clipreid.yml` and `configs/vit_clipreid_ilids.yml`, where QATA is explicitly disabled. These changes are harmless but outside the narrower sanity-check file list.
- `AGENTS.md` is untracked in the worktree and was not modified during this check.

## 2026-06-03 Formal Minimal QATA Experiments

### Commands

GPU check:

```bash
nvidia-smi
```

MARS QATA:

```bash
CUDA_VISIBLE_DEVICES=2 /data1/lgf/miniconda3/envs/tfclip/bin/python train.py --config_file configs/vit_clipreid_qata.yml OUTPUT_DIR logs/qata_mars_20260603_214227
```

iLIDS-VID QATA:

```bash
CUDA_VISIBLE_DEVICES=3 /data1/lgf/miniconda3/envs/tfclip/bin/python train.py --config_file configs/vit_clipreid_ilids_qata.yml OUTPUT_DIR logs/qata_ilids_20260603_214227
```

### Results

| Dataset | Method | Config | Output Dir | mAP | Rank-1 | Rank-5 | Best Epoch | Train Time | Notes |
|---|---|---|---|---|---|---|---|---|---|
| MARS | Baseline | `configs/vit_clipreid.yml` | `logs/mars_vit_clip_reid_newprompt+dense_meanp` | 88.9 | 93.0 | 98.1 | 56 | 3:28:27.841264 | Existing log; best perform 181.9%. |
| MARS | Minimal QATA | `configs/vit_clipreid_qata.yml` | `logs/qata_mars_20260603_214227` | 88.2 | 92.3 | 97.0 | 32 | 3:24:10.848671 | GPU 2; no OOM; best perform 180.5%. |
| iLIDS-VID | Baseline | `configs/vit_clipreid_ilids.yml` | `logs/ilids_vit_clip_reid` | 76.9 | 81.2 | 86.7 | 28 | 2:13:16.290315 | Existing log; best perform 158.2%. |
| iLIDS-VID | Minimal QATA | `configs/vit_clipreid_ilids_qata.yml` | `logs/qata_ilids_20260603_214227` | 75.6 | 79.8 | 86.8 | 26 | 2:14:50.775733 | GPU 3; no OOM; best perform 155.4%. |

### Initial Analysis

- Minimal QATA did not improve either dataset under the current settings.
- MARS dropped by 0.7 mAP, 0.7 Rank-1, and 1.1 Rank-5 at the logged best epoch.
- iLIDS-VID dropped by 1.3 mAP and 1.4 Rank-1, while Rank-5 was nearly unchanged.
- The trend is consistent across MARS and iLIDS-VID: train accuracy becomes high, but validation plateaus below baseline.
- This result is not strong enough to justify directly extending QATA into TMD, CLIP-Memory, or dense inference yet.
- Recommended next step: diagnose QATA frame weights and try a conservative residual/regularized variant before expanding scope.

## 2026-06-04 Minimal QATA Diagnosis

### Full-Run Log Recheck

Compared logs:

- MARS baseline: `logs/mars_vit_clip_reid_newprompt+dense_meanp/train_log.txt`
- MARS Minimal QATA: `logs/qata_mars_20260603_214227/train_log.txt`
- iLIDS baseline: `logs/ilids_vit_clip_reid/train_log.txt`
- iLIDS Minimal QATA: `logs/qata_ilids_20260603_214227/train_log.txt`

Key findings:

- Best epoch moves earlier with QATA.
  - MARS baseline best epoch: 56; Minimal QATA best epoch: 32.
  - iLIDS baseline best epoch: 28; Minimal QATA best epoch: 26.
- Training loss and training accuracy are very close to baseline at matched epochs.
  - MARS epoch 5 near-end loss: baseline 7.211 vs QATA 7.206.
  - MARS epoch 30 near-end loss: baseline 3.990 vs QATA 3.963.
  - iLIDS epoch 5 near-end loss: baseline 5.456 vs QATA 5.455.
  - iLIDS epoch 30 near-end loss: baseline 2.881 vs QATA 2.804.
- Validation is not an early-stop or under-training issue.
  - MARS QATA reaches 88.2 mAP at epoch 32 and then stays around 87.9-88.2 through epoch 80, but Rank-1 remains below baseline.
  - iLIDS QATA reaches the best combined score at epoch 26; later mAP hovers around 75.1-76.1 while Rank-1 drops to about 78.3-79.5.
- Overfitting signal is mild but visible: train accuracy saturates, while validation plateaus below baseline. This is especially clear on iLIDS.
- Curve behavior is not wildly unstable; the issue is a lower plateau rather than catastrophic divergence.

Interpretation:

- Minimal QATA is optimization-stable, but it does not provide a better temporal aggregation signal than mean pooling.
- The degradation is likely caused by adding a weak/unregularized quality estimator that perturbs well-tuned CLIP video features without a direct quality supervision signal.

### Diagnostic Code Added

Additional switchable diagnostics:

- `MODEL.QATA.LOG_STATS = False`
- `MODEL.QATA.STATS_FILE = "qata_weight_stats.txt"`

When both `MODEL.QATA.ENABLED=True` and `MODEL.QATA.LOG_STATS=True`, stage2 training records epoch-averaged QATA weight statistics for:

- `img`: weights from `qpool_768`
- `proj`: weights from `qpool_512`

Metrics written per epoch:

- mean
- std
- max
- min
- entropy
- effective frames = `exp(entropy)`
- top1 weight
- top2 weight

The diagnostics are off by default and do not change baseline training or normal QATA training.

### Short Diagnostic Runs

Commands:

```bash
CUDA_VISIBLE_DEVICES=2 /data1/lgf/miniconda3/envs/tfclip/bin/python train.py --config_file configs/vit_clipreid_qata.yml OUTPUT_DIR logs/qata_diag_mars_20260604_154211 MODEL.QATA.LOG_STATS True MODEL.QATA.STATS_FILE qata_weight_stats.txt SOLVER.STAGE2.MAX_EPOCHS 5 SOLVER.STAGE2.EVAL_PERIOD 5
CUDA_VISIBLE_DEVICES=3 /data1/lgf/miniconda3/envs/tfclip/bin/python train.py --config_file configs/vit_clipreid_ilids_qata.yml OUTPUT_DIR logs/qata_diag_ilids_20260604_154211 MODEL.QATA.LOG_STATS True MODEL.QATA.STATS_FILE qata_weight_stats.txt SOLVER.STAGE2.MAX_EPOCHS 5 SOLVER.STAGE2.EVAL_PERIOD 5
```

Diagnostic outputs:

| Dataset | Output Dir | Epochs | Diagnostic File | mAP@5 | Rank-1@5 | Train Time |
|---|---|---:|---|---:|---:|---|
| MARS | `logs/qata_diag_mars_20260604_154211` | 5 | `qata_weight_stats.txt` | 77.6 | 85.0 | 0:14:36.171842 |
| iLIDS-VID | `logs/qata_diag_ilids_20260604_154211` | 5 | `qata_weight_stats.txt` | 53.9 | 58.0 | 0:06:58.849779 |

Weight statistics summary at epoch 5:

| Dataset | Branch | Mean | Std | Max | Min | Entropy | Effective Frames | Top1 | Top2 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MARS | img | 0.1250 | 0.0110 | 0.1571 | 0.0956 | 2.0755 | 7.9687 | 0.1418 | 0.2766 |
| MARS | proj | 0.1250 | 0.0147 | 0.1648 | 0.0828 | 2.0724 | 7.9438 | 0.1464 | 0.2845 |
| iLIDS-VID | img | 0.1250 | 0.0108 | 0.1571 | 0.0983 | 2.0757 | 7.9702 | 0.1419 | 0.2766 |
| iLIDS-VID | proj | 0.1250 | 0.0118 | 0.1576 | 0.0937 | 2.0749 | 7.9640 | 0.1427 | 0.2783 |

For `SEQ_LEN=8`, uniform pooling has:

- weight mean: 0.125
- entropy: `ln(8)=2.0794`
- effective frames: 8
- top1: 0.125
- top2: 0.25

Diagnosis:

- No weight collapse was observed.
- The weights are very close to uniform on both MARS and iLIDS.
- `proj` becomes slightly more non-uniform than `img`, especially on MARS, but the difference is weak.
- Minimal QATA is therefore not failing because it selects one frame too aggressively. It is closer to a noisy near-mean pooling module that adds extra parameters and small feature perturbations without learning reliable quality separation.

### Second-Round Variant Design

Recommended candidate A: Residual QATA.

Formula:

```text
pooled = mean_pool + alpha * (qpool - mean_pool)
```

Design:

- Add `MODEL.QATA.MODE = "plain" | "residual" | "warmup"`.
- Add `MODEL.QATA.ALPHA = 0.1`.
- First version should use fixed alpha, not learnable alpha.
- Apply only to the same two locations: `img_feature` and `img_feature_proj`.

Why:

- It preserves the baseline mean representation as the main path.
- It allows QATA to contribute only a small correction.
- It is easy to ablate with `alpha=0.0/0.1/0.2`.
- It directly addresses the current finding: QATA is not harmful through collapse, but through weak noisy perturbation.

Recommended candidate B: Warmup QATA.

Formula:

```text
pooled = (1 - beta) * mean_pool + beta * qpool
```

Design:

- Add `MODEL.QATA.WARMUP_EPOCHS`, e.g. 10 or 20.
- Add `MODEL.QATA.TARGET_BETA`, e.g. 0.2.
- `beta` linearly increases from 0 to target beta.

Risk:

- Requires passing current epoch or a setter into the model, so it touches processor/model interaction more than Residual QATA.
- Better as a second step after fixed residual alpha.

Recommended candidate C: High-temperature QATA.

Design:

- Test `MODEL.QATA.TEMP = 2.0, 3.0, 5.0`.

Risk:

- Current weights are already near-uniform at temp 1.0, so raising temperature is unlikely to fix the main issue.
- Useful only as a control experiment to verify that even softer weights behave like mean pooling.

Candidate D: Entropy regularization.

Design:

- Keep as a config design only for now.
- If future diagnostics show collapse, add entropy regularization to discourage overly sharp weights.

Risk:

- Current diagnosis does not show collapse. Adding entropy loss now would push weights even closer to uniform and likely not help.

### Recommended Next Step

Run Residual QATA first:

| Priority | Dataset | Variant | Alpha | Temp | Notes |
|---:|---|---|---:|---:|---|
| 1 | MARS | Residual QATA | 0.1 | 1.0 | Main low-risk check. |
| 2 | iLIDS-VID | Residual QATA | 0.1 | 1.0 | Confirm cross-dataset trend. |
| 3 | MARS | Residual QATA | 0.2 | 1.0 | Only if alpha 0.1 is not worse. |
| 4 | iLIDS-VID | Residual QATA | 0.2 | 1.0 | Only if alpha 0.1 is not worse. |
| 5 | MARS | Plain QATA | 0.0 | 3.0 | Low-priority control; likely close to mean. |

Stop for now:

- Extending QATA to TMD.
- Extending QATA to CLIP-Memory generation.
- Extending QATA to dense inference.
- Entropy regularization full training.
- Large grids over reduction/dropout/temperature before Residual QATA is tested.

## 2026-06-04 Residual QATA Implementation

### Goal

Implement a conservative second-round QATA variant:

```text
pooled = mean_pool + alpha * (qpool - mean_pool)
```

The first version uses fixed `alpha=0.1`. QATA remains switchable and still only replaces the two minimal pooling locations:

- `img_feature.mean(1)`
- `img_feature_proj.mean(1)`

No changes were made to TMD, CLIP-Memory generation, or dense inference.

### Modified Files

- `config/defaults.py`
- `model/quality_aggregation.py`
- `model/make_model_clipreid.py`
- `configs/vit_clipreid_qata_residual_a01.yml`
- `configs/vit_clipreid_ilids_qata_residual_a01.yml`
- `experiments/qata_notes.md`

### Config Additions

Added under `MODEL.QATA`:

- `MODE = "plain"`
- `ALPHA = 0.1`

Compatibility:

- `MODEL.QATA.ENABLED=False`: baseline mean pooling.
- `MODEL.QATA.ENABLED=True`, `MODE="plain"`: first-round Minimal QATA.
- `MODEL.QATA.ENABLED=True`, `MODE="residual"`: Residual QATA.

### Implementation Notes

`QualityWeightedPooling` now accepts:

- `mode="plain" | "residual"`
- `alpha`

Forward behavior:

- `plain`: `pooled = qpool`
- `residual`: `pooled = mean_pool + alpha * (qpool - mean_pool)`

Weights are still returned when requested, so `MODEL.QATA.LOG_STATS=True` continues to work.

### Checks

Commands:

```bash
/data1/lgf/miniconda3/envs/tfclip/bin/python -m py_compile model/quality_aggregation.py model/make_model_clipreid.py processor/processor_clipreid_stage2.py config/defaults.py
/data1/lgf/miniconda3/envs/tfclip/bin/python -c "from config import cfg; files=['configs/vit_clipreid_qata_residual_a01.yml','configs/vit_clipreid_ilids_qata_residual_a01.yml']; ..."
/data1/lgf/miniconda3/envs/tfclip/bin/python -c "import torch; from model.quality_aggregation import QualityWeightedPooling; ..."
CUDA_VISIBLE_DEVICES=2 /data1/lgf/miniconda3/envs/tfclip/bin/python /tmp/check_residual_qata_sanity.py
```

Results:

- `py_compile`: passed.
- Config merge: passed.
  - MARS residual config: `ENABLED=True`, `MODE=residual`, `ALPHA=0.1`, `TEMP=1.0`, `LOG_STATS=True`, `OUTPUT_DIR=logs/qata_residual_a01_mars`.
  - iLIDS residual config: `ENABLED=True`, `MODE=residual`, `ALPHA=0.1`, `TEMP=1.0`, `LOG_STATS=True`, `OUTPUT_DIR=logs/qata_residual_a01_ilids`.
- QPool formula smoke test: passed.
  - qpool output shape: `(2, 6)`
  - residual output shape: `(2, 6)`
  - weights shape: `(2, 8)`
  - residual formula max error: `0.0`
  - weight consistency max error: `0.0`
- Synthetic 2-iteration train sanity: passed on GPU 2.
  - Output dir: `logs/qata_residual_a01_sanity`
  - Dummy losses: `6.431396`, `5.420898`
  - `qpool_768_mode=residual`
  - `qpool_768_alpha=0.1`
  - `latest_weight_keys=['img', 'proj']`
  - `qata_weight_stats.txt` was written successfully.

Synthetic sanity weight stats:

| Epoch | Branch | Mean | Std | Max | Min | Entropy | Effective Frames | Top1 | Top2 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | img | 0.1250 | 0.002839 | 0.130463 | 0.120512 | 2.079180 | 7.997906 | 0.129316 | 0.257232 |
| 1 | proj | 0.1250 | 0.002125 | 0.128914 | 0.121121 | 2.079293 | 7.998810 | 0.128271 | 0.255085 |

### Recommended Full Training Commands

MARS:

```bash
CUDA_VISIBLE_DEVICES=<free_gpu> /data1/lgf/miniconda3/envs/tfclip/bin/python train.py --config_file configs/vit_clipreid_qata_residual_a01.yml OUTPUT_DIR logs/qata_residual_a01_mars_<timestamp>
```

iLIDS-VID:

```bash
CUDA_VISIBLE_DEVICES=<free_gpu> /data1/lgf/miniconda3/envs/tfclip/bin/python train.py --config_file configs/vit_clipreid_ilids_qata_residual_a01.yml OUTPUT_DIR logs/qata_residual_a01_ilids_<timestamp>
```

## Residual QATA alpha=0.1 Full Training - 2026-06-04

### Pre-checks

- `git status --short`: clean.
- MARS config checked:
  - `MODEL.QATA.ENABLED=True`
  - `MODEL.QATA.MODE="residual"`
  - `MODEL.QATA.ALPHA=0.1`
  - `MODEL.QATA.TEMP=1.0`
  - `MODEL.QATA.LOG_STATS=True`
- iLIDS config checked with the same QATA settings.
- MARS output directory was unique and did not overwrite baseline/plain QATA logs.

### MARS Command

```bash
CUDA_VISIBLE_DEVICES=2 /data1/lgf/miniconda3/envs/tfclip/bin/python train.py --config_file configs/vit_clipreid_qata_residual_a01.yml OUTPUT_DIR logs/qata_residual_a01_mars_20260604_185152
```

### Results

| Dataset | Method | Config | Output Dir | mAP | Rank-1 | Rank-5 | Best Epoch | Train Time | Weight Stats | Notes |
| ------- | ------ | ------ | ---------- | --: | -----: | -----: | ---------: | ---------- | ------------ | ----- |
| MARS | Baseline | `configs/vit_clipreid.yml` | `logs/mars_vit_clip_reid_newprompt+dense_meanp` | 88.9 | 93.0 | 98.1 | 56 | 3:28:27.841264 | n/a | Original mean pooling baseline. |
| MARS | Plain Minimal QATA | `configs/vit_clipreid_qata.yml` | `logs/qata_mars_20260603_214227` | 88.2 | 92.3 | 97.0 | 32 | 3:24:10.848671 | Near-uniform, no collapse | Direct qpool replacement hurt MARS. |
| MARS | Residual QATA alpha=0.1 | `configs/vit_clipreid_qata_residual_a01.yml` | `logs/qata_residual_a01_mars_20260604_185152` | 89.2 | 93.0 | 98.1 | 42 | 4:21:59.837291 | Epoch 80 img eff 7.9244/top1 0.1445; proj eff 7.8204/top1 0.1583 | Saved `best_model.pth.tar`, `checkpoint_ep.pth.tar`, `train_log.txt`, and `qata_weight_stats.txt`. |
| iLIDS-VID | Baseline | `configs/vit_clipreid_ilids.yml` | `logs/ilids_vit_clip_reid` | 76.9 | 81.2 | 86.7 | 28 | 2:13:16.290315 | n/a | Original mean pooling baseline. |
| iLIDS-VID | Plain Minimal QATA | `configs/vit_clipreid_ilids_qata.yml` | `logs/qata_ilids_20260603_214227` | 75.6 | 79.8 | 86.8 | 26 | 2:14:50.775733 | Near-uniform, no collapse | Direct qpool replacement hurt iLIDS mAP/R1. |
| iLIDS-VID | Residual QATA alpha=0.1 | `configs/vit_clipreid_ilids_qata_residual_a01.yml` | not started | n/a | n/a | n/a | n/a | n/a | n/a | Blocked after MARS because `nvidia-smi` failed and `torch.cuda.is_available()` returned False. |

### MARS Observations

- Residual QATA alpha=0.1 is clearly better than plain QATA on MARS:
  - vs plain QATA: +1.0 mAP, +0.7 Rank-1, +1.1 Rank-5.
  - vs baseline: +0.3 mAP, equal Rank-1, equal Rank-5.
- Best log line reports `Best Perform 182.4%, achieved at epoch 42`.
- Validation reached a stable plateau from roughly epoch 52 onward:
  - several later epochs reported 89.2 / 93.0 / 98.1.
- QATA weights did not collapse:
  - epoch 80 img: entropy 2.0698, effective frames 7.9244, top1 0.1445, top2 0.2820.
  - epoch 80 proj: entropy 2.0562, effective frames 7.8204, top1 0.1583, top2 0.3038.
- The learned weights remain close to uniform but slightly sharper, especially on `img_feature_proj`. The gain is likely from a small residual correction rather than strong frame selection.

### iLIDS Status

After MARS finished, GPU checks failed:

```bash
nvidia-smi
```

returned:

```text
NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver.
```

and:

```bash
/data1/lgf/miniconda3/envs/tfclip/bin/python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count())"
```

returned:

```text
False
0
```

Therefore iLIDS Residual QATA was not started to avoid an invalid run.

## Residual QATA alpha=0.1 MARS Repeat Attempt - 2026-06-05 01:44 CST

### Pre-checks

- `git status --short`: clean.
- Code path checked:
  - `MODEL.QATA.ENABLED=False` keeps baseline mean pooling.
  - `MODEL.QATA.ENABLED=True`, `MODE="plain"` keeps first-round plain QATA behavior.
  - `MODEL.QATA.ENABLED=True`, `MODE="residual"`, `ALPHA=0.1` uses `mean_pool + alpha * (qpool - mean_pool)`.
- Config merge for `configs/vit_clipreid_qata_residual_a01.yml`:
  - `MODEL.QATA.ENABLED=True`
  - `MODEL.QATA.MODE=residual`
  - `MODEL.QATA.ALPHA=0.1`
  - `MODEL.QATA.TEMP=1.0`
  - `MODEL.QATA.LOG_STATS=True`
  - default `OUTPUT_DIR=logs/qata_residual_a01_mars`

### GPU Status

Repeat training was not started because GPU/CUDA was unavailable during the pre-check.

```bash
nvidia-smi
```

returned:

```text
NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver.
```

and:

```bash
/data1/lgf/miniconda3/envs/tfclip/bin/python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count())"
```

returned:

```text
False
0
```

Next command to run after GPU recovery:

```bash
CUDA_VISIBLE_DEVICES=<free_gpu> /data1/lgf/miniconda3/envs/tfclip/bin/python train.py --config_file configs/vit_clipreid_qata_residual_a01.yml OUTPUT_DIR logs/qata_residual_a01_mars_repeat_<timestamp>
```

## Residual QATA alpha=0.1 MARS Repeat Completed - 2026-06-05

### Command

```bash
CUDA_VISIBLE_DEVICES=2 /data1/lgf/miniconda3/envs/tfclip/bin/python train.py --config_file configs/vit_clipreid_qata_residual_a01.yml OUTPUT_DIR logs/qata_residual_a01_mars_repeat_20260605_022703
```

This run used escalated permissions because the sandboxed environment could not access NVML/CUDA, while the escalated pre-check showed 4 available RTX 4090 GPUs. GPU 2 was selected.

### Result

| Dataset | Method | Config | Output Dir | mAP | Rank-1 | Rank-5 | Best Epoch | Train Time | Notes |
|---|---|---|---|---:|---:|---:|---:|---|---|
| MARS | Residual QATA alpha=0.1 repeat | `configs/vit_clipreid_qata_residual_a01.yml` | `logs/qata_residual_a01_mars_repeat_20260605_022703` | 89.1 | 93.3 | 97.8 | 42 | 4:21:46.407493 | Log-selected best by `mAP + Rank-1`; highest mAP plateau reached 89.2 from epoch 52 onward. |

Additional peak-mAP checkpoints:

| Epoch | mAP | Rank-1 | Rank-5 |
|---:|---:|---:|---:|
| 52 | 89.2 | 93.0 | 97.9 |
| 56 | 89.2 | 93.0 | 98.1 |
| 64 | 89.2 | 93.0 | 98.1 |
| 66 | 89.2 | 93.0 | 98.1 |
| 68 | 89.2 | 93.0 | 98.1 |
| 74 | 89.2 | 93.0 | 98.1 |
| 76 | 89.2 | 93.0 | 98.1 |
| 78 | 89.2 | 93.0 | 98.1 |
| 80 | 89.2 | 93.0 | 98.1 |

Artifacts checked:

- `best_model.pth.tar`: present.
- `checkpoint_ep.pth.tar`: present.
- `train_log.txt`: present.
- `qata_weight_stats.txt`: present.

### Comparison

| Method | mAP | Rank-1 | Rank-5 | Best Epoch |
|---|---:|---:|---:|---:|
| Baseline | 88.9 | 93.0 | 98.1 | 56 |
| Plain QATA | 88.2 | 92.3 | 97.0 | 32 |
| Residual QATA alpha=0.1 first run | 89.2 | 93.0 | 98.1 | 42 |
| Residual QATA alpha=0.1 repeat | 89.2 | 93.0 | 98.1 | 56/64/66/68/74/76/78/80 |

### Weight Stats

Epoch 80 repeat:

- `img`: effective frames 7.9244, top1 0.1445, top2 0.2820.
- `proj`: effective frames 7.8204, top1 0.1583, top2 0.3038.

This almost exactly matches the first Residual QATA alpha=0.1 run:

- first run epoch 80 `img`: effective frames about 7.92, top1 about 0.1445.
- first run epoch 80 `proj`: effective frames about 7.82, top1 about 0.1583.

Interpretation:

- QATA weights are still close to uniform, not collapsed.
- `img_feature_proj` is consistently sharper than `img_feature`, but still conservative.
- Residual QATA alpha=0.1 behaves as a small correction to mean pooling rather than hard frame selection.
- Compared with Plain QATA, Residual QATA is much more stable and recovers baseline performance while giving a repeatable mAP improvement on MARS.

### Current Recommendation

- Residual QATA alpha=0.1 on MARS is worth keeping as the strongest feature-aggregation result so far.
- The repeated MARS result supports the +0.3 mAP claim in the peak-mAP plateau sense, although the log-selected `mAP + Rank-1` best epoch is 42 with mAP 89.1 / Rank-1 93.3.
- Before running alpha=0.2, run iLIDS alpha=0.1 to verify cross-dataset behavior.
- Do not expand to TMD, dense inference, or CLIP-Memory yet. Quality-Aware CLIP-Memory Construction can be prepared conceptually, but should wait until iLIDS alpha=0.1 is known.

## iLIDS Residual QATA alpha=0.1 Attempt - 2026-06-05

### Pre-checks

- `git status --short`: only `experiments/qata_notes.md` was modified from experiment recording; no core code changes.
- `configs/vit_clipreid_ilids_qata_residual_a01.yml` checked:
  - `MODEL.QATA.ENABLED=True`
  - `MODEL.QATA.MODE="residual"`
  - `MODEL.QATA.ALPHA=0.1`
  - `MODEL.QATA.TEMP=1.0`
  - `MODEL.QATA.LOG_STATS=True`
  - `OUTPUT_DIR='logs/qata_residual_a01_ilids'`
- No existing `logs/qata_residual_a01_ilids_*` output directory was found, so a timestamped output directory would not overwrite prior logs.

### GPU Status

Training was not started because the required GPU pre-check failed, even with escalated permissions:

```bash
nvidia-smi
```

returned:

```text
Failed to initialize NVML: Driver/library version mismatch
NVML library version: 535.309
```

Next command after GPU/NVML recovery:

```bash
CUDA_VISIBLE_DEVICES=<free_gpu> /data1/lgf/miniconda3/envs/tfclip/bin/python train.py --config_file configs/vit_clipreid_ilids_qata_residual_a01.yml OUTPUT_DIR logs/qata_residual_a01_ilids_<timestamp>
```

## iLIDS Residual QATA alpha=0.1 Full Run - 2026-06-07

### Pre-checks

- `git status --short`: `experiments/qata_notes.md` already modified for experiment notes; no core code changes were made for this run.
- `nvidia-smi`: GPU 0 and GPU 2 were idle; GPU 0 was selected.
- Config checked: `configs/vit_clipreid_ilids_qata_residual_a01.yml`
  - `MODEL.QATA.ENABLED=True`
  - `MODEL.QATA.MODE="residual"`
  - `MODEL.QATA.ALPHA=0.1`
  - `MODEL.QATA.TEMP=1.0`
  - `MODEL.QATA.LOG_STATS=True`

The first started run at `logs/qata_residual_a01_ilids_20260607_030733` was interrupted around epoch 4 and kept as an incomplete log. The completed full run used a new directory:

```bash
CUDA_VISIBLE_DEVICES=0 /data1/lgf/miniconda3/envs/tfclip/bin/python train.py --config_file configs/vit_clipreid_ilids_qata_residual_a01.yml OUTPUT_DIR logs/qata_residual_a01_ilids_20260607_031850
```

### Result

| Dataset | Method | Config | Output Dir | mAP | Rank-1 | Rank-5 | Best Epoch | Train Time | Notes |
|---|---|---|---|---:|---:|---:|---:|---|---|
| iLIDS-VID | Residual QATA alpha=0.1 | `configs/vit_clipreid_ilids_qata_residual_a01.yml` | `logs/qata_residual_a01_ilids_20260607_031850` | 76.2 | 80.3 | 86.2 | 28 | 2:16:23 | Better than plain QATA mAP, below current baseline. |

Artifacts checked:

- `best_model.pth.tar`: present.
- `checkpoint_ep.pth.tar`: present.
- `train_log.txt`: present.
- `qata_weight_stats.txt`: present.

### Comparison

| Method | mAP | Rank-1 | Rank-5 | Best Epoch |
|---|---:|---:|---:|---:|
| Baseline | 76.9 | 81.2 | 86.7 | 28 |
| Plain QATA | 75.6 | 79.8 | 86.8 | 26 |
| Residual QATA alpha=0.1 | 76.2 | 80.3 | 86.2 | 28 |

### Weight Stats

Best epoch 28:

- `img`: effective frames 7.9863, top1 0.1344, top2 0.2653.
- `proj`: effective frames 7.9787, top1 0.1363, top2 0.2686.

Final epoch 80:

- `img`: effective frames 7.9850, top1 0.1340, top2 0.2649.
- `proj`: effective frames 7.9810, top1 0.1351, top2 0.2666.

Interpretation:

- QATA weights remain very close to uniform for the whole iLIDS run.
- The `proj` branch is slightly sharper than `img`, but the difference is small.
- Compared with MARS, iLIDS weights are even more uniform, so Residual QATA behaves almost like mean pooling plus a very small learned perturbation.
- The validation curve peaks around epoch 26-28, then gradually settles near mAP 75.1-75.3 after epoch 50, indicating no late recovery.

### Conclusion

- iLIDS Residual QATA alpha=0.1 is better than plain QATA in mAP and Rank-1: `76.2/80.3` vs `75.6/79.8`.
- It does not beat the current iLIDS baseline: `76.2/80.3/86.2` vs `76.9/81.2/86.7`.
- The cross-dataset trend is mixed: MARS improves stably by +0.3 mAP, while iLIDS remains below baseline by -0.7 mAP.
- Since iLIDS baseline/protocol is already known to be less reliable than MARS, this does not fully invalidate Residual QATA, but it weakens the case for continuing ordinary feature aggregation.
- Do not run alpha=0.2 as the next priority unless a cheap confirmatory run is needed. The better next research direction is preparing Quality-Aware CLIP-Memory Construction conceptually, without implementing it yet.

## MARS Residual QATA alpha=0.2 Full Run - 2026-06-07

### Pre-checks

- `nvidia-smi`: GPU 0 was occupied by the running iLIDS QATA job; GPU 2 was idle and selected for this run.
- No core code was modified.
- Added config: `configs/vit_clipreid_qata_residual_a02.yml`.
- Config source: copied from `configs/vit_clipreid_qata_residual_a01.yml`.
- Config difference:
  - `MODEL.QATA.ENABLED=True`
  - `MODEL.QATA.MODE="residual"`
  - `MODEL.QATA.ALPHA=0.2`
  - `MODEL.QATA.TEMP=1.0`
  - `MODEL.QATA.LOG_STATS=True`
- `OUTPUT_DIR` was overridden in the training command.

Sanity check passed:

```bash
/data1/lgf/miniconda3/envs/tfclip/bin/python -m py_compile config/defaults.py model/quality_aggregation.py model/make_model_clipreid.py processor/processor_clipreid_stage2.py
```

Training command:

```bash
CUDA_VISIBLE_DEVICES=2 /data1/lgf/miniconda3/envs/tfclip/bin/python train.py --config_file configs/vit_clipreid_qata_residual_a02.yml OUTPUT_DIR logs/qata_residual_a02_mars_20260607_032430
```

Start/end:

- Start: `2026-06-07 03:24:51 +0800`
- End: `2026-06-07 06:51:25 +0800`
- Wall-clock elapsed: `3:26:34`
- Exit status: `0`

### Result

| Dataset | Method | Config | Output Dir | mAP | Rank-1 | Rank-5 | Best Epoch | Train Time | Notes |
|---|---|---|---|---:|---:|---:|---:|---|---|
| MARS | Residual QATA alpha=0.2 | `configs/vit_clipreid_qata_residual_a02.yml` | `logs/qata_residual_a02_mars_20260607_032430` | 89.2 | 93.1 | 97.9 | 54 | 3:26:34 | mAP ties alpha=0.1; Rank-1 +0.1; Rank-5 -0.2. |

Artifacts checked:

- `best_model.pth.tar`: present.
- `checkpoint_ep.pth.tar`: present.
- `train_log.txt`: present.
- `qata_weight_stats.txt`: present.
- `run_meta.txt`: present.

### Comparison

| Method | mAP | Rank-1 | Rank-5 | Notes |
|---|---:|---:|---:|---|
| Baseline | 88.9 | 93.0 | 98.1 | Reference baseline |
| Plain QATA | 88.2 | 92.3 | 97.0 | Worse than baseline |
| Residual QATA alpha=0.1 first | 89.2 | 93.0 | 98.1 | Stable gain in mAP |
| Residual QATA alpha=0.1 repeat | 89.2 | 93.0 | 98.1 | Reproduced |
| Residual QATA alpha=0.2 | 89.2 | 93.1 | 97.9 | No clear improvement over alpha=0.1 |

### Weight Stats

Final epoch 80:

- `img`: mean 0.1250, std 0.0236, effective frames 7.8430, top1 0.1536, top2 0.2975.
- `proj`: mean 0.1250, std 0.0399, effective frames 7.5704, top1 0.1801, top2 0.3388.

Interpretation:

- The weights remain smooth and close to uniform overall.
- Compared with a purely uniform 8-frame average, the `img` branch is still a weak perturbation.
- The `proj` branch is more selective than `img`, but still far from hard frame selection.

### Conclusion

- Alpha `0.2` is not clearly better than alpha `0.1`: mAP is tied, Rank-1 is slightly higher, and Rank-5 is slightly lower.
- Alpha `0.2` is still above baseline on mAP and Rank-1, but not on Rank-5.
- This does not prove QATA can only work as a very weak residual correction, because alpha `0.2` did not materially collapse. It does suggest that simply increasing residual strength is not reliably beneficial.
- Keep alpha `0.1` as the safer fixed default for MARS.
- Next ablations worth running: alpha `0.05` and learnable residual alpha initialized near `0.1`.
- Quality-Aware CLIP-Memory Construction can be prepared as the next phase, but should use a separate config flag and start from the safer residual setting rather than hard-coding alpha `0.2`.

## Quality-Aware CLIP-Memory Construction Design - 2026-06-07

Scope for this note:

- Pause ordinary feature aggregation tuning.
- Do not run alpha `0.05`, learnable alpha, temperature grid, or TMD expansion.
- Do not modify core code or configs in this step.
- Only analyze the current code path and design the next-stage Quality-Aware CLIP-Memory Construction.

### A. Current CLIP-Memory Construction

The CLIP-Memory used by stage2 is constructed in `processor/processor_clipreid_stage2.py`.

Relevant locations:

- `do_train_stage2(...)`: main stage2 training function.
- `generate_cluster_features(labels, features)`: local helper at lines 50-64.
- CLIP-Memory generation block: lines 162-192.
- Training use of generated memory: line 220 passes `text_features2=cluster_features` into the model.

Current variables and shapes:

- `image_features`: Python list of per-sequence projected CLIP image features on CPU.
- `labels`: Python list of identity labels.
- `labels_list`: stacked labels, comment says `[8256]`.
- `image_features_list`: stacked features, comment says `[8256, 512]`.
- `cluster_features`: output of `generate_cluster_features`, shape `[num_classes, 512]`.

The memory construction is two-level mean pooling:

1. Sequence-level temporal mean in `model/make_model_clipreid.py` when `get_image=True`.
   - For ViT-B/16, `image_features_proj[:, 0]` gives frame-level projected CLS tokens with shape `[B*T, 512]`.
   - It is reshaped to `[B, T, 512]`.
   - `img_feature_proj.mean(1)` returns `[B, 512]`.
   - This happens at lines 236-240.

2. Identity-level prototype mean in `generate_cluster_features`.
   - Features for the same ID are collected.
   - `torch.stack(centers[idx], dim=0).mean(0)` builds one prototype per ID.
   - This is an equal-weight mean over all sequences of that ID.

Dense clip feature aggregation:

- In the CLIP-Memory generation block, dense input with shape `[b, n, s, c, h, w]` is reshaped to `[b*n, s, c, h, w]`.
- The model returns one `[512]` feature per dense clip.
- Those dense clip features are flattened and averaged by `torch.mean(image_feature, 0, keepdim=True)` before insertion into memory.
- Dense inference has the same equal-weight pattern in `do_inference_dense`: model output is flattened and averaged by `torch.mean(feat, 0, keepdim=True)`.

The current QATA feature module does not affect CLIP-Memory construction:

- The normal training forward path applies QATA to `img_feature` and `img_feature_proj` when `cfg.MODEL.QATA.ENABLED=True`.
- The `get_image=True` path still uses direct `mean(1)` for projected image features.
- Therefore current Residual QATA experiments changed the model's normal sequence features, but not the precomputed CLIP-Memory prototypes.

### B. CLIP-Memory and Video Feature Relationship

Memory initialization:

- At the start of `do_train_stage2`, before epoch 1, the processor runs `train_loader_stage1` under `torch.no_grad()`.
- It calls `model(img, get_image=True)` to encode each sequence into a `[512]` projected video feature.
- `generate_cluster_features` averages those sequence features by ID.
- The result is detached: `cluster_features = generate_cluster_features(...).detach()`.

Memory update:

- There is no epoch-wise or gradient-based update of `cluster_features`.
- The tensor is fixed for the whole stage2 run.
- Gradients update the model and SSP module, but not the memory tensor.

SSP / I2T / frame loss dependency:

- During training, `cluster_features` is passed to `model(..., text_features2=cluster_features)`.
- Inside `model/make_model_clipreid.py`, `text_features2` is expanded from `[num_classes, 512]` to `[B, num_classes, 512]`.
- `image_features_proj_raw` is reshaped to `[B, T, tokens, 512]`.
- `video_feature_project = image_features_proj_raw2.mean(1)` averages dense projected token features over frames, producing `[B, tokens, 512]`.
- `self.SSP(text_features2, video_feature_project)` generates an image-specific prompt correction for the memory.
- Final I2T logits are `torch.einsum("bd,bkd->bk", img_feature_proj, text_features2)`, shape `[B, num_classes]`.
- In `loss/make_loss.py`, `i2tscore` is supervised by cross entropy / label smoothing and added with weight `cfg.MODEL.I2T_LOSS_WEIGHT`.
- Frame loss uses `score[3]` from the TMD/frame classifier branch and does not directly consume memory.
- ID and triplet losses consume `[img_feature, img_feature_proj, cls_f_tp]`; they are not directly memory losses, but they share the same forward pass and parameters.

Final training influence:

- CLIP-Memory affects training through the SSP-conditioned I2T classification loss.
- It encourages the current video projected feature `img_feature_proj` to align with the correct identity prototype after SSP adaptation.
- Because the memory is fixed, any bias in its construction persists for the entire stage2 optimization.

### C. Why Low-Quality Frames Can Pollute CLIP-Memory

Features entering memory:

- The memory uses only projected CLIP image features from the `get_image=True` path.
- For ViT-B/16, this is the projected CLS token `image_features_proj[:, 0]`, dimension 512.
- Each sequence contributes exactly one `[512]` feature to the ID memory.

Equal-weight operations:

- Frame-level mean: `[B, T, 512] -> [B, 512]` by `mean(1)` in `get_image=True`.
- Dense-clip mean: `[n, 512] -> [1, 512]` by `torch.mean(..., 0)`.
- Identity-level mean: all sequence features for one ID are averaged equally in `generate_cluster_features`.
- SSP conditioning also uses equal temporal averaging of dense token features: `video_feature_project = image_features_proj_raw2.mean(1)`.

Pollution path:

- If one sequence contains occlusion, blur, detector shift, background dominance, or repeated low-information frames, its `mean(1)` feature moves away from a clean identity representation.
- That noisy sequence feature is appended to `image_features` with the same weight as clean sequences.
- `generate_cluster_features` then averages it equally into the ID prototype.
- The resulting identity prototype becomes a biased memory target.
- During stage2, the I2T loss repeatedly pulls training samples toward this biased prototype, so the noise affects optimization beyond the original bad frames.

This is a stronger motivation than ordinary feature aggregation:

- Ordinary feature aggregation only changes current batch representation.
- Memory construction changes the fixed identity target used by every epoch of stage2.
- A small correction during memory construction can have a persistent effect without altering inference behavior.

### D. Candidate Designs

#### Version 1: Memory Construction Only

Goal:

- Add residual quality pooling only when constructing `cluster_features`.
- Do not change the normal model forward used for training.
- Do not change final inference.
- Do not change TMD, dense inference, or frame loss.

Design:

- Replace only the `get_image=True` memory-construction temporal mean with residual quality pooling over projected CLS features.
- Current memory feature:
  - `proj_frames = image_features_proj[:, 0].view(B, T, 512)`
  - `seq_feature = proj_frames.mean(1)`
- Proposed memory feature:
  - `mean = proj_frames.mean(1)`
  - `qpool = sum(softmax(q(proj_frames)) * proj_frames)`
  - `seq_feature = mean + MEMORY_ALPHA * (qpool - mean)`
- Then keep `generate_cluster_features` unchanged, so ID prototypes are still one vector per ID.

Expected code impact in a future implementation:

- Add a memory-specific path that can return quality-pooled projected features for `get_image=True`.
- Prefer reusing `QualityWeightedPooling(dim=512, mode="residual", alpha=MEMORY_ALPHA)`.
- Log memory-specific stats separately, e.g. `qata_memory_weight_stats.txt`.

Important implementation detail:

- A pure processor-side change is awkward because `model(img, get_image=True)` already returns pooled `[B, 512]`; the frame-level `[B, T, 512]` features are no longer available.
- The cleanest minimal implementation is to extend the model's `get_image=True` path with an optional memory pooling mode, while leaving the normal training/inference forward unchanged.

Risk:

- Low. It touches only the pre-stage2 memory generation path.
- It is independently ablatable.
- If disabled, baseline behavior should be exactly preserved.

#### Version 2: Memory Construction + SSP Alignment Weighting

Goal:

- Build cleaner memory as in Version 1.
- Additionally reduce the I2T/SSP alignment penalty from low-quality sequences and emphasize high-quality sequences.

Design:

- During normal stage2 forward, obtain a sequence quality scalar per sample, for example:
  - confidence from `top1` / entropy of quality weights,
  - mean predicted quality score,
  - or a detached quality margin.
- Apply this scalar to the I2T loss only:
  - compute I2T cross entropy with `reduction="none"`;
  - multiply each sample loss by a normalized quality weight;
  - average over batch.
- Leave ID loss, triplet loss, and frame loss unchanged at first.

Required future changes:

- `model.forward` must expose per-sample quality weights or a detached quality scalar.
- `processor_clipreid_stage2.py` must pass that scalar into the loss function.
- `loss/make_loss.py` must support weighted I2T loss.

Risk:

- Medium. Loss weighting can change optimization dynamics more than memory construction.
- Bad quality estimates early in training could suppress useful hard samples.
- It adds another confound: improvement could come from sample reweighting rather than memory quality.

Best use:

- Only after Version 1 shows nontrivial benefit.

#### Version 3: Memory Construction + Multi-Prototype Memory

Goal:

- Avoid forcing each ID into a single prototype.
- Represent different views, cameras, clothing states, tracklet qualities, or appearance modes with multiple prototypes.

Design options:

- Quality split:
  - high-quality prototype and low/medium-quality prototype per ID.
- Camera/view split:
  - one prototype per ID-camera or ID-view when metadata is reliable.
- Clustering:
  - run K-means or online clustering over sequence features per ID.
  - produce `[num_classes, K, 512]` prototypes.

Training use:

- Current logits are `[B, num_classes]`.
- Multi-prototype memory would produce either `[B, num_classes, K]` similarities or a flattened `[B, num_classes*K]` logit matrix.
- For class-level supervision, reduce prototypes by max, logsumexp, quality-weighted sum, or attention to recover `[B, num_classes]`.

Risk:

- High. It changes target structure and loss semantics.
- IDs with few tracklets may not support K prototypes.
- Prototype assignment can be unstable and dataset-specific.
- It increases implementation and debugging cost.

Best use:

- Not the next minimal experiment.
- Consider only if Version 1 improves MARS but saturates or shows view-dependent failure cases.

### E. Recommended Minimal Next Experiment

Best next step: Version 1, Memory Construction Only.

Reasons:

- It targets the most likely pollution point: fixed identity prototypes.
- It does not alter normal training forward, inference aggregation, TMD, dense inference, or the loss definition.
- It is compatible with current Residual QATA but can be ablated independently.
- Training cost is almost unchanged; only the one-time memory generation pass adds a tiny MLP/softmax over frames.
- It can preserve baseline exactly with `APPLY_MEMORY_CONSTRUCTION=False`.

Recommended default:

- `APPLY_FEATURE_AGG=False`
- `APPLY_MEMORY_CONSTRUCTION=True`
- `MEMORY_MODE="residual"`
- `MEMORY_ALPHA=0.1`
- `MEMORY_TEMP=1.0`
- `REUSE_QPOOL=False` for the first implementation, to avoid coupling memory construction to the feature aggregation module state.

### F. Suggested Config Design

Current config has only global QATA fields:

- `MODEL.QATA.ENABLED`
- `MODEL.QATA.MODE`
- `MODEL.QATA.ALPHA`
- `MODEL.QATA.REDUCTION`
- `MODEL.QATA.DROPOUT`
- `MODEL.QATA.TEMP`
- `MODEL.QATA.RETURN_WEIGHTS`
- `MODEL.QATA.LOG_STATS`
- `MODEL.QATA.STATS_FILE`

Suggested future extension:

```yaml
MODEL:
  QATA:
    ENABLED: True

    APPLY_FEATURE_AGG: False
    APPLY_MEMORY_CONSTRUCTION: True

    MODE: "residual"
    ALPHA: 0.1
    TEMP: 1.0

    MEMORY_MODE: "residual"
    MEMORY_ALPHA: 0.1
    MEMORY_TEMP: 1.0
    MEMORY_USE_RESIDUAL: True
    REUSE_QPOOL: False

    LOG_STATS: True
    STATS_FILE: "qata_weight_stats.txt"
    MEMORY_LOG_STATS: True
    MEMORY_STATS_FILE: "qata_memory_weight_stats.txt"
```

Notes:

- `APPLY_FEATURE_AGG` decouples ordinary feature aggregation from memory construction.
- `APPLY_MEMORY_CONSTRUCTION` enables the new path independently.
- `MEMORY_ALPHA` should not silently reuse `ALPHA`, because current experiments show feature aggregation alpha tuning is not the main direction.
- `REUSE_QPOOL=False` is safer initially. Shared qpool parameters could entangle memory construction with training-time feature aggregation.
- `MEMORY_USE_RESIDUAL` is mostly redundant if `MEMORY_MODE` supports `"mean"`, `"plain"`, and `"residual"`, but it can make ablation configs explicit.

### G. Minimal Experiment Matrix

Run MARS first.

Reason:

- MARS has stable repeated residual results.
- The baseline and a=0.1 repeat are reproducible.
- The memory construction hypothesis should first be tested where variance is lower and prior QATA signal is positive.
- iLIDS should be confirmatory after MARS, because the current iLIDS result is more volatile and below baseline even for residual feature QATA.

Minimal MARS matrix:

| Experiment | Feature Aggregation | Memory Construction | Expected Purpose |
|---|---|---|---|
| Baseline | mean | mean memory | Reference: 88.9 / 93.0 / 98.1 |
| Residual QATA feature only | residual alpha=0.1 | mean memory | Existing best feature-only result: 89.2 / 93.0 / 98.1 |
| Quality-Aware Memory only | mean | residual memory alpha=0.1 | Tests whether cleaner memory helps without changing inference features |
| Feature + Memory | residual alpha=0.1 | residual memory alpha=0.1 | Tests compatibility and additivity |

Run order:

1. MARS Quality-Aware Memory only.
2. If MARS Memory only improves mAP or Rank-1 without Rank-5 damage, run MARS Feature + Memory.
3. If either MARS memory run is positive, run iLIDS Quality-Aware Memory only.
4. Only then consider iLIDS Feature + Memory.

### H. Risk Analysis

Most likely bug sources:

- Shape mistakes in the `get_image=True` path, especially `[B*T, 512] -> [B, T, 512]`.
- Dense input path `[b, n, s, c, h, w]`, where memory construction currently averages dense clip features after model output.
- Accidentally enabling feature aggregation when the intended ablation is memory-only.
- Reusing training qpool parameters for memory construction, which can create optimizer/state ambiguity.
- Passing memory stats into the existing `latest_qata_weights` logger and mixing feature stats with memory stats.
- Changing `cluster_features` shape without updating `text_features2.unsqueeze(0).expand(B, -1, -1)`.

Positive signals:

- MARS Memory only beats baseline by at least +0.2 mAP without Rank-1/Rank-5 degradation.
- MARS Feature + Memory beats feature-only residual QATA, especially if mAP exceeds 89.2 or Rank-5 recovers to 98.1.
- Memory weight stats show mild but non-collapsed selectivity, e.g. effective frames remain high but lower than uniform.
- I2T loss or `Acc_clip` improves without ID/triplet instability.

Stop signals:

- Memory only underperforms baseline similarly to Plain QATA.
- Feature + Memory is worse than feature-only residual QATA.
- Memory weights collapse to one frame early.
- Gains appear only in Rank-1 while mAP drops, suggesting over-sharpening.
- iLIDS degrades further after a positive but tiny MARS change, indicating poor cross-dataset robustness.

If memory version does not improve:

- It does not necessarily invalidate the quality hypothesis.
- Possible explanations:
  - The current `get_image=True` projected CLS features are already robust enough.
  - The learned quality estimator has no direct supervision and stays near-uniform.
  - Equal sequence-level ID averaging may be the bigger issue than frame-level averaging.
  - SSP adaptation may compensate for noisy prototypes already.
  - The main bottleneck may be identity-level multi-modality rather than low-quality frame contamination.
- In that case, do not return to broad feature aggregation tuning. The next conceptual direction would be sequence-level memory weighting or multi-prototype memory, not more alpha/temperature sweeps.

### Final Recommendation

Implement Version 1: Memory Construction Only.

Minimal future implementation target:

- Add independent config switches for feature aggregation and memory construction.
- Keep default baseline behavior unchanged.
- Add a memory-specific residual quality pooling path for `get_image=True` projected CLS features.
- Use `MEMORY_ALPHA=0.1`, `MEMORY_TEMP=1.0`, `MEMORY_MODE="residual"`.
- Log memory weights separately from feature QATA weights.
- First run only MARS Memory only, then MARS Feature + Memory if positive.

Do not implement SSP alignment weighting or multi-prototype memory until the Memory Construction Only ablation shows a measurable MARS gain.

## 2026-06-07 Memory Construction Feasibility Check

Purpose: verify whether a Quality-Aware CLIP-Memory Construction variant would accidentally build fixed memory with a randomly initialized qpool. This check only used code reading and log/note analysis. No code was changed, no training was started, and no GPU task was run.

### 1. Is `cluster_features` Built Once?

File: `processor/processor_clipreid_stage2.py`

Function: `do_train_stage2()`

Relevant logic:

- `generate_cluster_features()` is defined inside `do_train_stage2()` at lines 50-64.
- CLIP-Memory construction starts before the epoch loop at lines 162-192.
- The code iterates over `train_loader_stage1` under `torch.no_grad()`, collects `model(img, get_image=True)` outputs, stacks them as `image_features_list`, then calls:
  - `cluster_features = generate_cluster_features(labels_list.cpu().numpy(), image_features_list).detach()`
- The epoch loop starts after this at line 195.
- During training, `cluster_features` is only passed into:
  - `model(x=img, ..., text_features2=cluster_features)`
- Search found no later recomputation or online update of `cluster_features`.

Conclusion: `cluster_features` is constructed once at the start of `do_train_stage2()`, detached, and fixed for the full stage2 training run.

### 2. What Does `generate_cluster_features()` Receive?

`generate_cluster_features(labels, features)` receives sequence-level features:

- Normal path:
  - `image_feature = model(img, get_image=True)`
  - each `img_feat` is appended to `image_features`
  - stacked result is `image_features_list`, with code comment `[8256, 512]`
- Dense path:
  - `model(img, get_image=True)` returns multiple sequence features
  - these are flattened and averaged by `torch.mean(image_feature, 0, keepdim=True)` before appending

Inside `generate_cluster_features()`:

- Features are grouped by identity label.
- Each identity center is computed by equal mean:
  - `torch.stack(centers[idx], dim=0).mean(0)`
- Output shape is `[num_classes, 512]`.

Conclusion: `generate_cluster_features()` only has access to `[N, 512]` sequence-level features. It cannot perform frame-level quality weighting unless the upstream `get_image=True` path is changed to expose `[B, T, 512]` frame-level features or pre-pooled quality-aware sequence features.

### 3. What Does `model(img, get_image=True)` Return?

File: `model/make_model_clipreid.py`

Relevant logic:

- The `get_image=True` branch starts at line 225.
- Input video `x` has shape `[B, T, C, H, W]`.
- Frames are flattened to `[B*T, C, H, W]`.
- For ViT-B/16:
  - `image_features, image_features_proj = self.image_encoder(x)`
  - `img_feature_proj = image_features_proj[:, 0]`
  - reshape to `[B, T, 512]`
  - `img_feature_proj = img_feature_proj.mean(1)`
  - return `[B, 512]`
- For RN50:
  - the projected feature is similarly reshaped to `[B, T, 512]`
  - then `mean(1)` is returned.

Important: the current `get_image=True` path does not use `qpool_512`, even when `MODEL.QATA.ENABLED=True`. It always returns mean-pooled sequence-level projected features.

Conclusion: current CLIP-Memory is built from mean-pooled `[B, 512]` sequence features. To use frame-level memory QATA, `get_image=True` would need a new mode or return path that exposes `[B, T, 512]`, or applies a controlled quality pooling before returning `[B, 512]`.

### 4. Is qpool Trained at Memory Construction Time?

File: `train.py`

Relevant logic:

- `train.py` loads config and dataloaders.
- It constructs the model with `make_model(...)`.
- It constructs loss, optimizer, and scheduler.
- It directly calls `do_train_stage2(...)`.
- Search found no `model.load_param(...)`, no `load_state_dict(...)`, and no resume/checkpoint loading before `do_train_stage2()`.

File: `model/make_model_clipreid.py`

- `qpool_768` and `qpool_512` are instantiated only if `cfg.MODEL.QATA.ENABLED=True`.
- Their `FrameQualityEstimator` is a newly initialized MLP:
  - LayerNorm
  - Linear
  - GELU
  - Dropout
  - Linear

Conclusion:

- In a normal stage2 training launch, qpool is random at the moment CLIP-Memory is constructed.
- Plain baseline has no qpool modules at all because they are only created when QATA is enabled.
- Unless a trained QATA checkpoint is explicitly loaded before memory construction, learned-qpool memory construction would use random quality scores.

### 5. Current Residual QATA Feature-Only Experiment

Current Residual QATA feature-only behavior:

- The memory construction path still uses `get_image=True`, which mean-pools `img_feature_proj` and ignores qpool.
- During the stage2 training loop, normal forward uses qpool for:
  - `img_feature` `[B, T, 768] -> [B, 768]`
  - `img_feature_proj` `[B, T, 512] -> [B, 512]`
- qpool is trained end-to-end through the normal training losses.
- However, the fixed `cluster_features` memory was already created before qpool training began.

Conclusion: in current Residual QATA feature-only experiments, qpool learns during stage2 but does not affect CLIP-Memory construction. The memory remains the initial fixed mean-pooled memory for the whole run.

### 6. Risk: Random qpool Building Fixed Memory

The concern is valid.

If a future "Memory Construction Only" implementation directly uses the learnable qpool during the current stage2 memory-construction block, then:

- qpool has not yet been trained.
- the resulting frame weights are random or weakly structured by random initialization.
- `cluster_features` is detached and fixed.
- random quality bias would be frozen into the identity memory for the entire training run.

Residual alpha can bound the damage, but it does not solve the conceptual issue. With `alpha=0.1`, the memory would be close to mean memory, so the expected benefit may be too weak while still introducing uncontrolled random perturbation.

### 7. Alternative Designs

#### A. Deterministic Quality Memory

Do not use a learnable qpool for memory construction. Use deterministic frame quality scores computed from frame-level projected features.

Possible scores:

- Feature norm: higher norm as higher confidence.
- Frame consistency: cosine similarity between each frame feature and the sequence mean.
- Distance to sequence center: downweight frames far from the sequence consensus.
- CLIP confidence or logit margin, if a stable text/identity reference is available.

Required code change if implemented later:

- Add a controlled `get_image=True` variant that returns frame-level `[B, T, 512]`, or add a separate method for memory feature extraction.
- Compute deterministic weights before sequence pooling.
- Keep identity-level aggregation in `generate_cluster_features()` unchanged initially.

Pros:

- No random qpool.
- No extra pretraining pass.
- Clean memory-only ablation.
- Most compatible with the current fixed-memory design.

Cons:

- Heuristic quality may be weak.
- Consistency weighting may favor redundant/easy frames rather than truly discriminative frames.

#### B. Two-Pass Memory Construction

Protocol:

1. Train Residual QATA feature-only and save a checkpoint.
2. Start a second stage2 run.
3. Load the trained qpool before CLIP-Memory construction.
4. Build quality-aware memory using the trained qpool.
5. Train stage2 with that memory.

Pros:

- Uses learned quality weights rather than random weights.
- Tests the intended learnable quality-memory idea more directly.

Cons:

- More complex protocol.
- Less fair against one-pass baseline unless an equivalent two-pass baseline is added.
- Current `train.py` has no checkpoint-loading path before `do_train_stage2()`, so implementation would require additional controlled loading logic.
- More expensive and harder to explain in ablations.

#### C. Online Memory Update

Periodically recompute `cluster_features` during training using the current model/qpool.

Pros:

- Avoids keeping the initial random-qpool memory forever after later updates.
- Could adapt memory to improving features.

Cons:

- Larger processor change.
- Expensive because memory generation scans `train_loader_stage1`.
- Risk of target drift and unstable I2T supervision.
- More complicated with AMP/distributed training.
- Not suitable as the next minimal experiment.

#### D. Frame-Level Return + Small Residual Alpha

Modify `get_image=True` to return frame-level features and build memory with:

`memory_feature = mean + alpha * (qpool - mean)`

with very small `alpha`, e.g. 0.1.

Pros:

- Damage from random qpool is bounded.
- Easy to align with the current Residual QATA formula.

Cons:

- qpool is still random if no checkpoint is loaded.
- Expected gain may be tiny because alpha is small.
- This is not a clean quality-aware memory test; it mostly tests whether small random perturbation is harmless.

### 8. Recommendation

Current Version 1: Memory Construction Only should not be directly implemented with the learnable qpool in the existing stage2 start block.

Reason: under the current training flow, qpool is randomly initialized when CLIP-Memory is built, and the resulting `cluster_features` are fixed for the whole run. This would risk freezing random quality weights into the identity memory.

Minimal reliable alternative:

- Implement Deterministic Quality Memory first.
- Use frame-level projected CLS features `[B, T, 512]`.
- Weight frames by deterministic frame consistency or distance-to-sequence-mean.
- Keep feature aggregation QATA disabled for the first memory-only ablation.
- Keep identity-level `generate_cluster_features()` as equal mean over sequence features for the first version.

Recommended next experiment:

1. First preserve or commit the current QATA feature-only results and code state.
2. Open a separate branch or separate config family for memory experiments.
3. Implement deterministic memory-only construction on MARS.
4. Compare against:
   - baseline mean feature + mean memory
   - Residual QATA feature-only + mean memory
5. Only if deterministic memory shows a signal should we consider Two-Pass learned-qpool memory.

Experiments to avoid for now:

- Learned qpool memory construction in the current one-pass stage2 setup.
- Online memory update.
- Extending QATA to TMD, CLIP-Memory SSP internals, or dense inference before memory construction is isolated.

Revised conclusion: the previous "Memory Construction Only" idea needs a stronger constraint. It is feasible only if the memory quality scores are deterministic or if qpool is loaded from a trained checkpoint before memory construction. The cleanest next step is Deterministic Quality Memory, not random/untrained qpool memory.

## Memory Weight Diagnosis Plan - 2026-06-10

Current checkout note:

- Current branch observed during this diagnosis: `exp-qata`.
- The saved MARS memory-only run exists under `logs/memory_consistency_a01_mars_20260608_023602`.
- The current checkout does not contain `configs/vit_clipreid_memory_consistency_a01.yml` or the memory-construction implementation in `processor/processor_clipreid_stage2.py`.
- Therefore this section is based on saved logs and prior recorded implementation behavior. Before implementing any new memory variant, switch back to or restore `exp-memory-consistency`.

### Existing Memory Stats

Read from `logs/memory_consistency_a01_mars_20260608_023602/memory_construction_stats.txt`:

```text
enabled=True
mode=consistency_residual
alpha=0.100000
tau=1.000000
num_ids=625
mean_entropy=2.300781
mean_effective_samples=13.273438
mean_top1_weight=0.130371
mean_samples_per_id=13.276800
max_weight_sum_error=0.000000
```

Interpretation:

- `mean_effective_samples` is almost identical to `mean_samples_per_id`, so the memory weights are effectively uniform within most identities.
- `mean_top1_weight=0.1304` alone is not enough to judge sharpness, because IDs have different sample counts. The stronger evidence is the effective-sample statistic.
- Current consistency-aware memory is therefore nearly the original equal-mean memory plus a very small residual perturbation.

Missing statistics:

- Per-ID cosine similarity mean/std/min/max.
- Global cosine similarity histogram/range.
- Per-ID score range before softmax.
- Weight distribution under alternative temperatures.
- Outlier counts per ID.
- Correlation between low-consistency sequences and noisy tracklets.
- Per-ID sample count distribution, especially `mean(1 / samples_per_id)` for a proper top-1 uniform baseline.

### Why tau=1.0 Produced Nearly Uniform Weights

For one ID with `n` samples, if the best sequence score is `Delta` larger than all others, the approximate top-1 softmax weight is:

```text
p_top1 = exp(Delta / tau) / (exp(Delta / tau) + n - 1)
```

With `tau=1.0`, cosine-score gaps must be large to create meaningful selection. In CLIP feature space, same-ID sequence features are likely already close to their ID mean, so `Delta` is probably small. That makes `softmax(cos / 1.0)` almost uniform.

Representative top-1 weights for `n=13`:

| Score Gap Delta | tau=1.0 | tau=0.1 | tau=0.05 | tau=0.01 |
|---:|---:|---:|---:|---:|
| 0.005 | 0.077 | 0.081 | 0.084 | 0.121 |
| 0.010 | 0.078 | 0.084 | 0.092 | 0.185 |
| 0.020 | 0.078 | 0.092 | 0.111 | 0.381 |
| 0.050 | 0.081 | 0.121 | 0.185 | 0.925 |
| 0.100 | 0.084 | 0.185 | 0.381 | 0.999 |
| 0.200 | 0.092 | 0.381 | 0.820 | 1.000 |

This table shows why blindly running `MEMORY_TEMP=0.1` is risky:

- If actual score gaps are around `0.005-0.02`, `tau=0.1` will still be weak.
- If actual score gaps are around `0.05-0.1`, `tau=0.05` or `0.01` may collapse to one sequence.
- Without the real cosine distribution, temperature selection is guesswork.

### CPU-Only Diagnostic Script Design

Goal: simulate memory weights without launching full training.

Required inputs:

- Stage2 config.
- A checkpoint/model capable of producing `model(img, get_image=True)` sequence-level features.
- `train_loader_stage1` for MARS.

If sequence-level features `[N, 512]` are not already saved, exact diagnosis requires re-extracting them. This can be attempted CPU-only, but CLIP ViT feature extraction on CPU may be very slow. If CPU extraction is impractical, run only a short feature-extraction job later after explicit GPU approval; do not start full training.

Diagnostic procedure:

1. Build dataloader and model.
2. Extract `features: [N, 512]` and labels from `train_loader_stage1` using the same `get_image=True` path used for CLIP-Memory.
3. For each ID:
   - compute equal mean prototype.
   - L2-normalize features and prototype.
   - compute cosine similarities.
   - record similarity mean/std/min/max/range.
   - compute weights for `tau in {1.0, 0.5, 0.1, 0.05, 0.01}`.
   - record entropy, effective samples, top1/top2 weight, and bottom1 weight.
   - count outliers below `mean - std`, `mean - 2 * std`, and bottom `10%`.
4. Aggregate global stats and write:
   - `memory_similarity_stats.csv`
   - `memory_tau_simulation.csv`
   - optional histograms as text bins.

Decision criteria:

- If cosine similarity ranges are extremely narrow, consistency score has weak quality discrimination and memory construction should stop or switch score type.
- If ranges are meaningful but tau=1.0 is too smooth, try lower temperature only after simulation shows non-collapse.
- If outliers exist but top weights are still diffuse, use outlier suppression or robust mean instead of top-heavy softmax.

### Candidate Variants

Candidate A: Low-Temperature Consistency Memory

- Use `MEMORY_TEMP=0.1` or `0.05`.
- Keep `MEMORY_ALPHA=0.1` first; only use `0.2` if simulation remains too weak.
- Risk: may overweight the most central/redundant sequence rather than the highest-quality sequence.
- Do not run full training until weight simulation shows effective samples are reduced but not collapsed.

Candidate B: Outlier-Suppressed Memory

- Compute cosine similarity to ID mean.
- Drop the lowest `k%` samples or samples below `mean - std`.
- Average the remaining samples.
- Motivation: the actual problem is likely low-quality outliers, not choosing a single best sequence.
- This is more aligned with robustness than softmax sharpening.

Candidate C: Residual Robust Memory

- Use trimmed mean or top-p mean, then residual update:
  - `memory = mean + alpha * (robust_mean - mean)`
- Keep alpha small, such as `0.1`.
- This avoids strong prototype drift and is easier to defend as a robust estimator.

Candidate D: Stop Memory Construction

- If cosine similarity has weak discrimination and no meaningful outlier pattern, stop this route.
- Focus on Residual QATA feature-only, where MARS already showed stable positive signal.

### Recommendation

- Do not run iLIDS memory-only.
- Do not blindly run `MEMORY_ALPHA` or `MEMORY_TEMP` sweeps.
- The next step should be CPU-only or feature-extraction-only memory weight simulation.
- If simulation shows real outliers, prioritize `Outlier-Suppressed Memory` or `Residual Robust Memory` over pure low-temperature softmax.
- If simulation only shows narrow same-ID cosine ranges, stop memory construction and return to Residual QATA feature-only for visualization, ablation, and paper analysis.
- Current best publishable direction remains Residual QATA feature-only a=0.1, because it has a cleaner ablation and stable MARS gain.

## QATA Stage Summary - 2026-06-10

Full standalone report:

- `experiments/qata_stage_summary.md`

Scope:

- Current branch: `exp-qata`.
- Memory consistency branch is paused.
- No new training was run for this summary.
- No model, processor, or config code was modified.

### Unified Results

| Dataset | Method | Config | Output Dir | mAP | Rank-1 | Rank-5 | Best Epoch | Conclusion |
|---|---|---|---|---:|---:|---:|---:|---|
| MARS | Baseline | `configs/vit_clipreid.yml` | `logs/mars_vit_clip_reid_newprompt+dense_meanp` | 88.9 | 93.0 | 98.1 | 56 | Current main baseline |
| MARS | Plain QATA | `configs/vit_clipreid_qata.yml` | `logs/qata_mars_20260603_214227` | 88.2 | 92.3 | 97.0 | 32 | Clear regression |
| MARS | Residual QATA a=0.1 | `configs/vit_clipreid_qata_residual_a01.yml` | `logs/qata_residual_a01_mars_20260604_185152` | 89.2 | 93.0 | 98.1 | 42 | +0.3 mAP, Rank metrics preserved |
| MARS | Residual QATA a=0.1 repeat | `configs/vit_clipreid_qata_residual_a01.yml` | `logs/qata_residual_a01_mars_repeat_20260605_022703` | 89.2 | 93.0 | 98.1 | 56/64/66/68/74/76/78/80 | Reproduced +0.3 mAP plateau |
| MARS | Residual QATA a=0.2 | `configs/vit_clipreid_qata_residual_a02.yml` | `logs/qata_residual_a02_mars_20260607_032430` | 89.2 | 93.1 | 97.9 | 54 | Not clearly better than a=0.1 |
| MARS | Consistency Memory a=0.1 | `configs/vit_clipreid_memory_consistency_a01.yml` | `logs/memory_consistency_a01_mars_20260608_023602` | 88.7 | 92.7 | 97.4 | 48 | Below baseline; memory weights nearly uniform |
| iLIDS-VID | Baseline | `configs/vit_clipreid_ilids.yml` | `logs/ilids_vit_clip_reid` | 76.9 | 81.2 | 86.7 | 28 | Current reproduced baseline |
| iLIDS-VID | Plain QATA | `configs/vit_clipreid_ilids_qata.yml` | `logs/qata_ilids_20260603_214227` | 75.6 | 79.8 | 86.8 | 26 | mAP / Rank-1 regression |
| iLIDS-VID | Residual QATA a=0.1 | `configs/vit_clipreid_ilids_qata_residual_a01.yml` | `logs/qata_residual_a01_ilids_20260607_031850` | 76.2 | 80.3 | 86.2 | 28 | Better than plain QATA, below baseline |

Diagnostic runs:

| Dataset | Method | Output Dir | Epochs | Key Observation |
|---|---|---|---:|---|
| MARS | Plain QATA weight diag | `logs/qata_diag_mars_20260604_154211` | 5 | effective frames close to 8; top1 around 0.14; weights near-uniform |
| iLIDS-VID | Plain QATA weight diag | `logs/qata_diag_ilids_20260604_154211` | 5 | effective frames close to 8; top1 around 0.14; weights near-uniform |

### Main Interpretation

Plain QATA failed because it directly replaced a strong mean-pooling baseline with a learnable qpool that did not learn reliable quality separation. The diagnostic runs showed near-uniform weights rather than collapse. This means the added module mostly introduced noise and unconstrained feature perturbation, especially harmful for an already tuned CLIP video representation.

Residual QATA is more stable because it keeps mean pooling as the dominant path:

```text
pooled = mean_pool + alpha * (qpool - mean_pool)
```

With `alpha=0.1`, the learned qpool can only make a bounded correction. This explains why it recovers baseline behavior and improves MARS mAP, while plain QATA degrades both MARS and iLIDS.

### MARS Meaning and Limits

MARS gives the strongest positive result:

- Residual QATA a=0.1: stable `89.2` mAP across two runs.
- Baseline: `88.9` mAP.
- Rank-1 / Rank-5 are essentially preserved.

This supports keeping Residual QATA as a low-risk feature aggregation module. The limitation is that the gain is small, Rank metrics do not consistently improve, and qpool weights remain close to uniform. The current evidence supports “conservative residual aggregation,” not strong frame-quality selection.

### iLIDS Interpretation

iLIDS Residual QATA a=0.1 improves over plain QATA but remains below baseline:

- Baseline: `76.9 / 81.2 / 86.7`
- Plain QATA: `75.6 / 79.8 / 86.8`
- Residual QATA: `76.2 / 80.3 / 86.2`

Possible causes:

- iLIDS is smaller and more volatile.
- The current iLIDS baseline/protocol is already less reliable than MARS.
- QATA weights on iLIDS are even closer to uniform.
- The residual correction does not provide late-epoch generalization recovery.

This weakens the case for expanding ordinary feature-level QATA, but does not erase the reproducible MARS signal.

### Memory Consistency Interpretation

Consistency Memory a=0.1 underperformed baseline:

- Result: `88.7 / 92.7 / 97.4`
- Baseline: `88.9 / 93.0 / 98.1`

Memory stats:

- mean effective samples: `13.2734`
- mean samples per ID: `13.2768`
- mean top1 weight: `0.1304`

The effective sample count is almost identical to the sample count, so the memory weighting nearly degenerates to equal mean. Current consistency-to-ID-mean scoring either has weak discrimination or `tau=1.0` is too smooth. Since memory is built once and fixed at stage2 start, this version does not justify more blind memory alpha/temp sweeps.

### Recommendation

Continue QATA only in a narrowed form:

- Keep Residual QATA feature-only as the strongest current result.
- Stop plain QATA.
- Pause memory consistency.
- Do not run iLIDS memory-only.
- Do not expand to TMD, SSP internals, CLIP-Memory, or dense inference yet.

If writing a paper, QATA can be a lightweight residual temporal aggregation submodule and a useful ablation story. It should not be positioned as the sole core contribution, a strong quality selector, or evidence for CLIP-Memory learning.

### Next Three Directions

1. Low risk: Residual QATA visualization and statistical analysis.
   - Use existing logs/checkpoints.
   - Visualize high-weight and low-weight frames.
   - Determine whether QATA learns quality or acts as regularization.

2. Medium risk: Residual QATA with deterministic quality priors.
   - Add feature norm, frame-to-sequence consistency, or temporal stability as weak hints.
   - Keep residual form to avoid plain-QATA instability.

3. Medium-high risk: Robust Memory instead of softmax memory.
   - Use trimmed mean / outlier suppression / residual robust mean.
   - Only revisit after feature/similarity diagnostics, not blind temp/alpha sweeps.

Short-term decision:

- Do not run more training immediately.
- Consolidate Residual QATA feature-only analysis, visualization, and paper framing first.

## Multi-prototype CLIP-Memory MARS Result - 2026-06-10

Branch: `exp-multiproto-memory`

Experiment:

```bash
CUDA_VISIBLE_DEVICES=0 /data1/lgf/miniconda3/envs/tfclip/bin/python train.py \
--config_file configs/vit_clipreid_multiproto_k2_farthest_lse.yml \
OUTPUT_DIR logs/multiproto_k2_farthest_lse_mars_20260610_143142
```

Result:

| Dataset | Method | Config | Output Dir | mAP | Rank-1 | Rank-5 | Best Epoch | Train Time | Notes |
|---|---|---|---|---:|---:|---:|---:|---|---|
| MARS | Multi-prototype K=2 farthest logsumexp | `configs/vit_clipreid_multiproto_k2_farthest_lse.yml` | `logs/multiproto_k2_farthest_lse_mars_20260610_143142` | 83.9 | 90.9 | 97.8 | 62 | 3:37:35 | Feature QATA disabled. Final epoch 80: mAP 84.1 / Rank-1 90.4 / Rank-5 97.8. |

Comparison:

- Baseline: `88.9 / 93.0 / 98.1`
- Residual QATA a=0.1: `89.2 / 93.0 / 98.1`
- Consistency Memory a=0.1: `88.7 / 92.7 / 97.4`
- Multi-prototype K=2 farthest logsumexp: `83.9 / 90.9 / 97.8`

Memory stats:

- num_classes: `625`
- num_prototypes: `2`
- mean samples per ID: `13.2768`
- min/max samples per ID: `1 / 271`
- fallback IDs: `1`
- empty-cluster fallbacks: `0`
- prototype cosine mean/std: `0.9322 / 0.0329`

Conclusion:

This first multi-prototype version is stable but significantly below baseline. The high prototype cosine mean suggests the farthest-two split does not produce sufficiently distinct ID modes in the current feature space. The result should not be used as evidence that multi-prototype memory is invalid yet, because the implementation also changes the similarity path to prototype-level cosine/logsumexp. Next step should be a control diagnostic, not another full blind run.

## Duplicate Mean Multi-prototype Control Status - 2026-06-10

Implemented the control path `MODEL.MEMORY.CLUSTER_MODE="duplicate_mean"`:

- build original per-ID mean memory `[C,D]`
- duplicate to `[C,K,D]`
- keep current multi-prototype logsumexp scoring path
- keep feature QATA disabled

Sanity checks passed:

- `py_compile` passed for `processor/processor_clipreid_stage2.py`, `model/make_model_clipreid.py`, and `config/defaults.py`
- `/tmp/check_multiproto_memory.py` passed:
  - duplicate output `[C,2,D]`
  - `proto[:,0,:] == proto[:,1,:]`
  - with normalization disabled, duplicated prototype equals original single mean
  - logsumexp output `[B,C]`
  - no NaN/Inf

Training was not started because the current instruction disallows permission escalation, and normal-user PyTorch CUDA is unavailable:

```text
cuda available: False
device count: 0
UserWarning: Can't initialize NVML
```

Pending command when normal-user CUDA access is available:

```bash
CUDA_VISIBLE_DEVICES=<free_gpu> /data1/lgf/miniconda3/envs/tfclip/bin/python train.py \
--config_file configs/vit_clipreid_multiproto_k2_dupmean_lse.yml \
OUTPUT_DIR logs/multiproto_k2_dupmean_lse_mars_<timestamp>
```

## Duplicate Mean Multi-prototype Control Result - 2026-06-10

Experiment:

```bash
CUDA_VISIBLE_DEVICES=3 /data1/lgf/miniconda3/envs/tfclip/bin/python train.py \
--config_file configs/vit_clipreid_multiproto_k2_dupmean_lse.yml \
OUTPUT_DIR logs/multiproto_k2_dupmean_lse_mars_20260610_195020
```

Result:

| Dataset | Method | Config | Output Dir | mAP | Rank-1 | Rank-5 | Best Epoch | Train Time | Notes |
|---|---|---|---|---:|---:|---:|---:|---|---|
| MARS | Multi-prototype duplicate mean LSE | `configs/vit_clipreid_multiproto_k2_dupmean_lse.yml` | `logs/multiproto_k2_dupmean_lse_mars_20260610_195020` | 84.6 | 90.7 | 97.3 | 66 | 3:41:23 | Feature QATA disabled. Highest mAP observed was 84.7, but code-selected best is epoch 66 by mAP+Rank-1. |

Memory stats:

- num_classes: `625`
- K: `2`
- mean samples per ID: `13.2768`
- min/max samples per ID: `1 / 271`
- fallback IDs: `0`
- empty-cluster fallbacks: `0`
- prototype cosine mean/std: `1.0000 / 0.0000`
- cluster mode: `duplicate_mean`
- aggregation mode: `logsumexp`

Comparison:

| Method | mAP | Rank-1 | Rank-5 |
|---|---:|---:|---:|
| Baseline | 88.9 | 93.0 | 98.1 |
| Residual QATA a=0.1 | 89.2 | 93.0 | 98.1 |
| Multi-prototype farthest LSE | 83.9 | 90.9 | 97.8 |
| Multi-prototype duplicate mean LSE | 84.6 | 90.7 | 97.3 |

Conclusion:

The duplicate-mean control also drops far below baseline, so the current multi-prototype scoring path is not baseline-equivalent. Since duplicated prototypes are exactly the original mean memory repeated along K, this result shifts the main suspicion from prototype construction alone to score normalization, logsumexp aggregation, shape handling, or another branch consuming `[C,K,D]` memory. Do not run KMeans, max, iLIDS, or QATA combinations until a strict numerical equivalence diagnostic passes.

## Multi-prototype Scoring Equivalence Check and Fix - 2026-06-11

No training was run. No GPU task was run.

The strict duplicate-mean equivalence check found four concrete code-level mismatches:

1. Original single-prototype I2T scoring uses raw dot product:
   `einsum("bd,bkd->bk", img_feature_proj, text_features2)`.
   The previous multi-prototype path used normalized cosine similarity.
2. Previous logsumexp aggregation did not subtract `temp*log(K)`, so duplicate prototypes added a class-independent constant.
3. Previous multi-prototype SSP flattened `[C,K,D]` into `[C*K,D]`, so SSP self-attention saw twice as many memory tokens. Duplicate prototypes therefore did not remain equivalent after SSP.
4. `duplicate_mean` respected `NORMALIZE_PROTOTYPES=True`, so the duplicated memory was not a strict copy of the original mean memory.

Fixes:

- `compute_i2t_scores()` now uses the original raw-dot formula for single and multi memory.
- Corrected logsumexp:
  `temp * logsumexp(sim / temp, dim=K) - temp * log(K)`.
- Multi-prototype SSP now runs on class-level memory `[B,C,D]`, then broadcasts the class prompt to each prototype.
- `duplicate_mean` now strictly copies original per-ID mean features and ignores prototype normalization.
- `MULTI_ENABLED=False` path remains the original single mean path.

Sanity checks:

```bash
/data1/lgf/miniconda3/envs/tfclip/bin/python -m py_compile \
model/make_model_clipreid.py processor/processor_clipreid_stage2.py config/defaults.py

/data1/lgf/miniconda3/envs/tfclip/bin/python /tmp/check_multiproto_equivalence.py
```

Equivalence results:

| Check | Max Abs Diff |
|---|---:|
| duplicate max vs single | 0.0000000000 |
| corrected duplicate logsumexp vs single | 0.0000038147 |
| SSP duplicate max vs SSP single | 0.0000076294 |
| SSP corrected duplicate logsumexp vs SSP single | 0.0000076294 |
| duplicate proto0 vs single mean | 0.0000000000 |
| duplicate proto1 vs single mean | 0.0000000000 |
| CE duplicate max vs CE single | 0.0000000000 |
| CE duplicate logsumexp vs CE single | 0.0000000000 |

Decision:

- The duplicate-mean numerical equivalence check now passes.
- It is valid to rerun duplicate mean with the fixed path.
- Do not run farthest/KMeans/max/iLIDS/Residual-QATA combinations until fixed duplicate mean returns near baseline.

Recommended command, not executed:

```bash
CUDA_VISIBLE_DEVICES=<free_gpu> /data1/lgf/miniconda3/envs/tfclip/bin/python train.py \
--config_file configs/vit_clipreid_multiproto_k2_dupmean_lse.yml \
OUTPUT_DIR logs/multiproto_k2_dupmean_lse_fixed_mars_<timestamp>
```
## Fixed Duplicate Mean Multi-prototype Control Result - 2026-06-11

Experiment:

```bash
CUDA_VISIBLE_DEVICES=2 /data1/lgf/miniconda3/envs/tfclip/bin/python train.py \
--config_file configs/vit_clipreid_multiproto_k2_dupmean_lse.yml \
OUTPUT_DIR logs/multiproto_k2_dupmean_lse_fixed_mars_20260611_015110
```

Result:

| Dataset | Method | Config | Output Dir | mAP | Rank-1 | Rank-5 | Best Epoch | Train Time | Notes |
|---|---|---|---|---:|---:|---:|---:|---|---|
| MARS | Fixed duplicate mean LSE | `configs/vit_clipreid_multiproto_k2_dupmean_lse.yml` | `logs/multiproto_k2_dupmean_lse_fixed_mars_20260611_015110` | 88.9 | 93.0 | 97.3 | 64 | 3:21:09 | QATA disabled. Corrected raw-dot scoring, corrected logsumexp, class-level SSP, strict duplicate mean. |

Memory stats:

- num_classes: `625`
- K: `2`
- mean samples per ID: `13.2768`
- min/max samples per ID: `1 / 271`
- fallback IDs: `0`
- empty-cluster fallbacks: `0`
- prototype cosine mean/std: `1.0000 / 0.0000`
- cluster mode: `duplicate_mean`
- aggregation mode: `logsumexp`

Comparison:

| Method | mAP | Rank-1 | Rank-5 |
|---|---:|---:|---:|
| Baseline | 88.9 | 93.0 | 98.1 |
| Old farthest LSE | 83.9 | 90.9 | 97.8 |
| Old duplicate mean LSE | 84.6 | 90.7 | 97.3 |
| Fixed duplicate mean LSE | 88.9 | 93.0 | 97.3 |

Conclusion:

The fixed duplicate-mean control recovers baseline-level mAP and Rank-1, so the severe old duplicate/farthest drop was caused by the non-equivalent scoring path. The scoring path is now repaired enough to retest farthest-two. Do not proceed to KMeans, max, iLIDS, or Residual QATA combinations until corrected farthest MARS is tested.
