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
