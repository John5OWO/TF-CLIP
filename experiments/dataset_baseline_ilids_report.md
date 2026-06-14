# iLIDS-VID Baseline Reproduction Report

## Attempt 2026-06-13 02:47 +0800

### Purpose

Re-check the original TF-CLIP iLIDS-VID baseline only. No QATA, Residual QATA, Consistency Memory, Multi-prototype Memory, LS-VID, or other new modules should be enabled.

### Environment

| Item | Value |
|---|---|
| Project root | `/data1/lgf/TF-CLIP` |
| Hostname | `ubuntu-Super-Server` |
| Branch | `exp-multiproto-memory` |
| Commit | `e8d15d7 add research decision report for qata and multiproto experiments` |
| `which python` | `/data1/lgf/miniconda3/bin/python` |
| Training python | `/data1/lgf/miniconda3/envs/tfclip/bin/python` |
| `which nvidia-smi` | `/usr/bin/nvidia-smi` |
| Torch | `2.0.1+cu118` |
| Torch CUDA | `11.8` |
| `torch.cuda.is_available()` | `False` |
| Torch device count | `0` |

`nvidia-smi` itself succeeded and showed all four RTX 4090 GPUs idle at the time of the check:

| GPU | Memory | Util |
|---:|---:|---:|
| 0 | 11 MiB / 24564 MiB | 0% |
| 1 | 11 MiB / 24564 MiB | 0% |
| 2 | 11 MiB / 24564 MiB | 0% |
| 3 | 11 MiB / 24564 MiB | 0% |

However, PyTorch in the allowed non-escalated execution environment reported:

```text
torch: 2.0.1+cu118
torch.version.cuda: 11.8
cuda available: False
device count: 0
UserWarning: Can't initialize NVML
```

Per the experiment constraints, training was stopped because `torch.cuda.is_available()` was `False`. No baseline training was launched.

### Git / Workspace

Initial status:

```text
## exp-multiproto-memory...origin/exp-multiproto-memory
```

No uncommitted `model/`, `processor/`, `loss/`, or `config/` modifications were present at the time of the check.

### Baseline Config Check

Config: `configs/vit_clipreid_ilids.yml`

| Key | Value |
|---|---|
| `DATASETS.NAMES` | `ilidsvidsequence` |
| `DATASETS.ROOT_DIR` | `/data1/lgf/TF-CLIP/data/iLIDS-VID` |
| `DATASETS.SPLIT` | `0` |
| `INPUT.SEQ_LEN` | `8` |
| `MODEL.QATA.ENABLED` | `False` |
| `MODEL.MEMORY.MULTI_ENABLED` | `False` |
| `MODEL.QATA.APPLY_MEMORY_CONSTRUCTION` | field absent in this branch/config |

Interpretation:

- Feature-level QATA is disabled.
- Multi-prototype memory is disabled.
- The consistency-memory switch `MODEL.QATA.APPLY_MEMORY_CONSTRUCTION` is not defined in the current `exp-multiproto-memory` branch, so it cannot be accidentally enabled through this config.
- This config is baseline-compatible with respect to the new modules currently present in the branch.

### iLIDS-VID Dataset / Protocol Check

Dataset code:

- Factory mapping: `datasets/make_dataloader_clipreid.py` maps `ilidsvidsequence` to `datasets/set/ilidsvidsequence.py::iLIDSVIDSEQUENCE`.
- Dataset class: `datasets/set/ilidsvidsequence.py`.
- Base split loader: `datasets/set/datasequence.py`.

Observed data paths:

| Item | Status |
|---|---|
| Dataset root `/data1/lgf/TF-CLIP/data/iLIDS-VID` | exists |
| `images/` | exists |
| `meta.json` | exists |
| `splits.json` | exists |
| raw split mat `raw/train-test people splits/train_test_splits_ilidsvid.mat` | exists |

Dataset statistics from `meta.json` / `splits.json`:

| Item | Value |
|---|---:|
| identities | 300 |
| cameras | 2 |
| camera tracklets | 600 |
| png frames in `images/` | 42459 |
| frame count min per camera tracklet | 22 |
| frame count max per camera tracklet | 192 |
| frame count mean per camera tracklet | 70.765 |
| number of splits | 10 |
| split 0 trainval IDs | 150 |
| split 0 query IDs | 150 |
| split 0 gallery IDs | 150 |
| sampled missing files from first/last frame checks | 0 |

Protocol notes:

- `iLIDSVIDSEQUENCE.__init__` loads split `0` by default.
- Query uses cam `0`; gallery uses cam `1`.
- The raw `.mat` split file contains 10 train/test people splits.
- `Datasequence.load()` randomly shuffles `trainval_pids` before taking a validation ID. This affects the train/val partition but not query/gallery IDs.
- The code evaluates only one selected split by default (`DATASETS.SPLIT=0`), while the paper result may use a different split, average over 10 splits, or a different evaluation protocol. This remains a plausible explanation for the large gap to the paper-reported Rank-1.

### Training Status

No training was started.

Reason:

```text
torch.cuda.is_available() == False
```

This violates the pre-run condition in the requested protocol. Because sudo and permission escalation are disallowed, the run was stopped rather than using the workaround that was required in earlier GPU jobs.

### Result Table

| Dataset | Method | Config | Output Dir | mAP | Rank-1 | Rank-5 | Best Epoch | Train Time | Notes |
| ------- | ------ | ------ | ---------- | --: | -----: | -----: | ---------: | ---------- | ----- |
| iLIDS-VID | Baseline | `configs/vit_clipreid_ilids.yml` | not started | n/a | n/a | n/a | n/a | n/a | Stopped before training because PyTorch CUDA was unavailable in the allowed non-escalated environment. |

### Current Judgment

1. This attempt cannot answer whether the iLIDS-VID baseline is still below the paper report because training did not start.
2. Data files and split files appear present; no obvious file-level dataset issue was found.
3. The protocol remains suspicious: default split `0`, random train/val split inside trainval, and possible mismatch with paper split averaging could all contribute to the previous low result.
4. iLIDS-VID should remain a lower-confidence auxiliary dataset until the exact TF-CLIP paper protocol is verified.
5. A valid rerun requires PyTorch CUDA availability under the allowed execution environment, or explicit permission to use the previously necessary non-sandbox execution path.

## Completed Baseline Rerun - 2026-06-13

After explicit continuation, the iLIDS-VID baseline was run with the non-sandbox execution path required for CUDA access on this server. No core code or config file was modified.

Run metadata:

| Item | Value |
|---|---|
| Hostname | `ubuntu-Super-Server` |
| Branch | `exp-multiproto-memory` |
| Commit | `e8d15d7` |
| Config | `configs/vit_clipreid_ilids.yml` |
| Output dir | `logs/baseline_ilidsvid_20260613_025719` |
| Selected GPU | `1` |
| Start time | `2026-06-13 02:57:19 +0800` |
| End time | `2026-06-13 05:09:52 +0800` |
| Exit status | `0` |
| QATA enabled | `False` |
| MEMORY multi enabled | `False` |
| QATA memory construction | field absent |

Saved artifacts:

| File | Status |
|---|---|
| `train_log.txt` | saved |
| `best_model.pth.tar` | saved |
| `checkpoint_ep.pth.tar` | saved |
| `run_meta.txt` | saved |

Result table:

| Dataset | Method | Config | Output Dir | mAP | Rank-1 | Rank-5 | Best Epoch | Train Time | Notes |
| ------- | ------ | ------ | ---------- | --: | -----: | -----: | ---------: | ---------- | ----- |
| iLIDS-VID | Baseline rerun | `configs/vit_clipreid_ilids.yml` | `logs/baseline_ilidsvid_20260613_025719` | 76.9 | 81.2 | 86.7 | 28 | 2:12:27 | Baseline only; QATA/MEMORY disabled; split 0. Rank-10 89.0, Rank-20 93.4. |

Best validation record:

| Epoch | mAP | Rank-1 | Rank-5 | Rank-10 | Rank-20 |
|---:|---:|---:|---:|---:|---:|
| 28 | 76.9 | 81.2 | 86.7 | 89.0 | 93.4 |

Final epoch validation:

| Epoch | mAP | Rank-1 | Rank-5 | Rank-10 | Rank-20 |
|---:|---:|---:|---:|---:|---:|
| 80 | 76.4 | 79.5 | 86.0 | 88.5 | 91.8 |

Judgment:

1. The rerun reproduces the previous local iLIDS-VID baseline almost exactly: mAP 76.9 / Rank-1 81.2 / Rank-5 86.7 at epoch 28.
2. The result remains far below the TF-CLIP paper-reported iLIDS-VID Rank-1 94.5 / Rank-5 99.1.
3. Because data files and split files exist and no obvious missing-frame issue was found, the main unresolved risk is protocol mismatch rather than a simple data-path failure.
4. Likely causes to verify next: split 0 vs 10-split average, query/gallery protocol, whether the paper reports a different split convention, data preprocessing/version differences, or hyperparameter differences specific to iLIDS-VID.
5. Until the exact paper protocol is verified, iLIDS-VID should be treated as a lower-confidence auxiliary dataset rather than a primary claim dataset.
