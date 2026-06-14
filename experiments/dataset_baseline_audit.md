# Dataset Baseline Audit

Date: 2026-06-13  
Branch: `exp-multiproto-memory`  
Commit: `e8d15d7`

This audit only reads logs, configs, and dataset code. No training, GPU job, or core-code modification was performed.

## Current Baseline Results

| Dataset | Output Dir | mAP | Rank-1 | Rank-5 | Rank-10 | Best Epoch | Train Time | Paper Result | Gap |
|---|---|---:|---:|---:|---:|---:|---|---|---|
| MARS | existing baseline logs | 88.9 | 93.0 | 98.1 | - | 56 | - | close to paper | acceptable |
| LS-VID | `logs/baseline_lsvid_20260613_045926` | 79.8 | 88.1 | 96.2 | 97.4 | 80 | 3:57:52 | mAP 83.8 / Rank-1 90.4 | mAP -4.0 / Rank-1 -2.3 |
| iLIDS-VID | `logs/baseline_ilidsvid_20260613_025719` | 76.9 | 81.2 | 86.7 | 89.0 | 28 | 2:12:27 | Rank-1 94.5 / Rank-5 99.1 | large Rank-k gap |

## Configuration And Dataset Summary

### LS-VID

Run metadata:

| Item | Value |
|---|---|
| Config | `configs/vit_clipreid_lsvid.yml` |
| Git tracking status | untracked local file |
| `DATASETS.NAMES` | `lsvid` |
| `DATASETS.ROOT_DIR` | `/data1/lgf/TF-CLIP/data/LS-VID` |
| QATA | disabled |
| Multi-prototype memory | disabled |
| Stage2 max epoch | 80 |
| Stage2 batch size | 16 |
| Num instances | 4 |
| Sequence length | 8 |
| Stage2 LR schedule | base LR `5e-6`, steps `[30, 50, 70]`, gamma `0.1` |
| Eval period | 2 epochs |
| Training sample mode | `rrs_train` |
| Validation sample mode during training | `rrs_test`, batch size 30 |
| Dense eval path | implemented in `make_eval_all_dataloader()` / `do_inference_dense()`, but not used by `train.py` during this run |

Loaded LS-VID statistics:

| Split | IDs | Tracklets |
|---|---:|---:|
| train | 842 | 2,831 |
| train dense, sampling step 48 | 842 | 10,106 |
| query | 2,730 | 3,504 |
| gallery | 2,730 | 7,829 |

Additional dataset facts:

- Tracklet frame count range: 60 to 2533, average 199.6.
- `datasets/set/lsvid.py` default `sampling_step=48`.
- Stage2 training uses `dataset.train`, which is replaced by `train_dense` when `sampling_step != 0`, so the actual training sampler sees 10,106 dense train tracklets.
- The code comment in `datasets/set/lsvid.py` says LS-VID has 3772 identities, but the loaded split reports 842 train IDs + 2730 gallery/test IDs = 3572 total. This may be a dataset version, split, or metadata discrepancy worth checking.

### iLIDS-VID

Run metadata:

| Item | Value |
|---|---|
| Config | `configs/vit_clipreid_ilids.yml` |
| `DATASETS.NAMES` | `ilidsvidsequence` |
| `DATASETS.ROOT_DIR` | `/data1/lgf/TF-CLIP/data/iLIDS-VID` |
| `DATASETS.SPLIT` | default `0` |
| QATA | disabled |
| Multi-prototype memory | disabled |
| Stage2 max epoch | 80 |
| Stage2 batch size | 16 |
| Num instances | 4 |
| Sequence length | 8 |
| Sequence stride | 4 |
| Stage2 LR schedule | base LR `5e-6`, steps `[30, 50, 70]`, gamma `0.1` |
| Eval period | 2 epochs |
| Training split used by loader | `trainval`, not only `train` |
| Validation/test sampling | sliding sequence clips from query cam 0 and gallery cam 1 |
| Dense eval path | not used for iLIDS in this training path |

Loaded iLIDS-VID statistics from `meta.json` / `splits.json`:

| Item | Value |
|---|---:|
| identities | 300 |
| cameras | 2 |
| splits | 10 |
| split 0 trainval IDs | 150 |
| split 0 query IDs | 150 |
| split 0 gallery IDs | 150 |
| split 0 trainval sequence clips | 4,679 |
| split 0 query sequence clips, cam 0 | 2,281 |
| split 0 gallery sequence clips, cam 1 | 2,684 |

Protocol details:

- `iLIDSVIDSEQUENCE.__init__()` loads one split via `split_id`, defaulting to split 0.
- Query is generated from camera 0 and gallery from camera 1.
- `Datasequence.load()` shuffles `trainval_pids` and carves out one validation ID, but `make_dataloader()` uses `dataset.trainval` for both stage1 memory construction and stage2 training. Query/gallery IDs are not affected by this shuffle.
- The raw split mat exists at `data/iLIDS-VID/raw/train-test people splits/train_test_splits_ilidsvid.mat`.

## LS-VID Gap Diagnosis

Observed training curve:

| Epoch | mAP | Rank-1 | Rank-5 | Rank-10 |
|---:|---:|---:|---:|---:|
| 20 | 73.7 | 83.6 | 93.7 | 95.7 |
| 40 | 78.9 | 87.5 | 95.5 | 97.3 |
| 52 | 79.6 | 87.9 | 96.0 | 97.4 |
| 60 | 79.6 | 88.1 | 96.0 | 97.4 |
| 80 | 79.8 | 88.1 | 96.2 | 97.4 |

Findings:

1. `configs/vit_clipreid_lsvid.yml` is not tracked by git, while tracked configs only include MARS/iLIDS/QATA/multiproto variants. This strongly suggests the LS-VID config is a local derivative rather than an official LS-VID config restored from the original release package.
2. Best epoch is the final epoch 80. The curve is mostly plateaued after epoch 52, but it still reaches the best mAP at epoch 80, so incomplete convergence remains plausible.
3. Training uses `rrs_test` validation during training, not dense evaluation. Dense inference exists but is not used by `train.py`. If the paper reports dense or multi-clip testing, this can explain part of the mAP/Rank-1 gap.
4. Loaded LS-VID ID count may not match the dataset comment: code comment says 3772 identities, current split loads 3572 total IDs by train+test-pid accounting. This should be checked against the official LS-VID split files.
5. The config uses stage2 batch size 16 and num instances 4, matching the local MARS-style config pattern, but there is no evidence this is the paper's LS-VID-specific setting.

Most likely LS-VID causes, prioritized:

1. Non-official LS-VID config: local MARS-derived hyperparameters and evaluation path may not match paper settings.
2. Evaluation mismatch: current training validation uses `rrs_test`, while dense/multi-clip evaluation is implemented but not used.
3. Possible dataset/split/version mismatch: loaded identity accounting differs from the code comment's LS-VID identity count, even though tracklet counts match the expected train/query/gallery counts.

Secondary causes:

- 80 epochs may be insufficient or the LR step schedule may be suboptimal for LS-VID, especially because best mAP occurs at the last epoch.
- Batch size 16 may be lower than an official LS-VID setting if the paper used larger effective batches or multi-GPU training.
- No explicit LS-VID-specific pretrained checkpoint path is configured; the run relies on the standard baseline initialization path.

## iLIDS-VID Gap Diagnosis

Observed training curve:

| Epoch | mAP | Rank-1 | Rank-5 | Rank-10 |
|---:|---:|---:|---:|---:|
| 20 | 75.4 | 79.7 | 87.1 | 89.7 |
| 28 | 76.9 | 81.2 | 86.7 | 89.0 |
| 40 | 76.7 | 80.1 | 86.5 | 88.6 |
| 60 | 76.5 | 79.5 | 86.1 | 88.6 |
| 80 | 76.4 | 79.5 | 86.0 | 88.5 |

Findings:

1. The code evaluates only one split by default (`DATASETS.SPLIT=0`), while iLIDS-VID commonly uses 10 random train/test splits and reports average Rank-k. This is the strongest explanation for the large paper gap.
2. The paper reports Rank-k but not mAP, so current mAP cannot be directly compared to the paper table.
3. Query/gallery are constructed as camera-specific sliding clips: query cam 0, gallery cam 1. This is a valid protocol choice, but if the paper averaged both directions or used a different clip aggregation convention, Rank-k can shift substantially.
4. The training set is small: 150 trainval IDs. The curve peaks early at epoch 28 and then slowly declines, so simply extending training is unlikely to solve the gap.
5. The current validation/test loader expands each query/gallery ID into many sequence clips, then evaluates clip-level embeddings. If the paper evaluates person-level/video-level aggregation differently, this can produce a large Rank-k mismatch.

Most likely iLIDS-VID causes, prioritized:

1. Split/protocol mismatch: one split 0 vs 10-split average or a different split convention.
2. Query/gallery direction or aggregation mismatch: current query is cam 0 and gallery cam 1, with multiple sliding clips evaluated directly.
3. Evaluation unit mismatch: clip-level sequence evaluation may differ from paper-level track/person aggregation.
4. Dataset preprocessing/version differences: regenerated `meta.json` / `splits.json` may not exactly match the paper release package.
5. Small-dataset instability: with only 150 trainval IDs, a single split can be noisy and should not be treated as a primary claim unless protocol is aligned.

## MARS As Control

MARS baseline reaches mAP 88.9 / Rank-1 93.0 / Rank-5 98.1, close to the expected TF-CLIP behavior. This makes a broad model-forward or loss implementation failure unlikely.

The more likely failure mode is dataset-specific:

- LS-VID: config/evaluation schedule/protocol and possibly dataset version.
- iLIDS-VID: split/protocol/evaluation-unit mismatch.

LS-VID is the better next calibration target because:

- Its gap is moderate, not catastrophic.
- Its best epoch is at the final epoch, giving a clear baseline-only action: extend or recover official schedule.
- Tracklet counts match the expected train/query/gallery counts, so the dataset is less suspicious than iLIDS.

iLIDS-VID should not be a main validation dataset until split averaging and protocol are verified.

## Recommended Next Actions

### LS-VID

Recommended:

1. Recover or reconstruct the official LS-VID TF-CLIP config before running any improvement module.
2. Verify whether the paper uses `rrs_test` or dense evaluation. If dense evaluation is the reported protocol, evaluate the current best checkpoint with the dense path before changing training.
3. If no official config is available, run a baseline-only calibration with longer training, for example 120 epochs, while keeping QATA/MEMORY disabled. A conservative schedule would move LR drops later than `[30, 50, 70]`, because the current best is at epoch 80.

Do not run Residual QATA on LS-VID yet. The current baseline is still 4.0 mAP below the paper, so any module gain would be hard to interpret.

### iLIDS-VID

Recommended:

1. Pause using iLIDS-VID as a main validation dataset.
2. Audit the exact paper protocol: split index vs 10-split average, query/gallery direction, clip/video aggregation, and whether Rank-k is averaged over splits.
3. Implement or script 10-split baseline evaluation only after confirming the expected protocol.
4. Treat current iLIDS results as low-confidence auxiliary evidence only.

Extending epochs is not recommended as the first action. The curve peaks at epoch 28 and then declines/plateaus.

### Overall

Current recommendation:

1. Do not run QATA, Residual QATA, Multi-prototype, or other improvement modules on LS-VID/iLIDS until their baselines are calibrated.
2. Use MARS as the trusted baseline-control dataset for module sanity.
3. Prioritize LS-VID baseline calibration over iLIDS because the LS-VID gap is smaller and more actionable.
4. For iLIDS, solve protocol first; for LS-VID, solve official config/eval mode first.

Decision:

- Continue Residual QATA on LS-VID: not yet recommended.
- Use iLIDS as main validation dataset: not recommended.
- Most stable next step: LS-VID baseline-only calibration with official config or dense-eval verification, not new modules.
