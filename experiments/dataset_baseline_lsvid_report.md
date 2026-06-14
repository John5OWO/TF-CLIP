# LS-VID Baseline Calibration Report

## Run 2026-06-13 02:49:59 +0800

- Purpose: LS-VID baseline calibration for possible later Residual QATA a=0.1 validation.
- Hostname: ubuntu-Super-Server
- Branch: exp-multiproto-memory
- Commit: e8d15d7
- Torch: 2.0.1+cu118
- CUDA available from torch precheck: False
- NVIDIA-SMI: available; driver 535.309.01; CUDA 12.2; 4 x NVIDIA GeForce RTX 4090
- Selected GPU: 1
- Dataset name: lsvid
- Dataset root: /data1/lgf/Datasets/LS-VID
- Dataset zip found: /data1/lgf/Datasets/LS-VID_V2.zip
- Required extracted path exists: False
- Required train list exists: False
- Required test list exists: False
- Required info_test.mat exists: False
- Config: configs/vit_clipreid_lsvid.yml
- QATA enabled status: False
- MEMORY multi-prototype enabled status: False
- MODEL.QATA.APPLY_MEMORY_CONSTRUCTION: field not present in config/defaults.py

| Dataset | Method | Config | Output Dir | mAP | Rank-1 | Rank-5 | Best Epoch | Train Time | Notes |
| ------- | ------ | ------ | ---------- | --: | -----: | -----: | ---------: | ---------- | ----- |
| LS-VID | TF-CLIP baseline | configs/vit_clipreid_lsvid.yml | logs/baseline_lsvid_20260613_025048 | N/A | N/A | N/A | N/A | 3s | Failed before training: extracted dataset root /data1/lgf/Datasets/LS-VID is missing; only /data1/lgf/Datasets/LS-VID_V2.zip exists. |

## Attempt 2026-06-13 02:50:48 +0800

- Command: `CUDA_VISIBLE_DEVICES=1 /data1/lgf/miniconda3/envs/tfclip/bin/python train.py --config_file configs/vit_clipreid_lsvid.yml OUTPUT_DIR logs/baseline_lsvid_20260613_025048`
- Exit status: 1
- Output dir: logs/baseline_lsvid_20260613_025048
- Train log: logs/baseline_lsvid_20260613_025048/train_log.txt
- Captured stdout/stderr: logs/baseline_lsvid_20260613_025048/train_stdout_stderr.txt
- Run meta: logs/baseline_lsvid_20260613_025048/run_meta.txt
- Failure: `RuntimeError: '/data1/lgf/Datasets/LS-VID' is not available`
- best_model.pth.tar: not created
- checkpoint_ep.pth.tar: not created

## Judgment

1. LS-VID baseline did not complete.
2. It is not yet usable as a Residual QATA validation baseline.
3. Do not start Residual QATA a=0.1 on LS-VID until the baseline completes.
4. Likely cause: dataset path is not prepared for the current `datasets/set/lsvid.py` loader. The zip contains `LS-VID/list_sequence/*` and `LS-VID/test/data/info_test.mat`, while the current loader expects an extracted root containing `info/list_sequence/*` and `info/data/info_test.mat`.

## Run 2026-06-13 08:58:06 +0800

- Purpose: LS-VID baseline calibration for possible later Residual QATA a=0.1 validation.
- Hostname: ubuntu-Super-Server
- Branch: exp-multiproto-memory
- Commit: e8d15d7
- Torch: 2.0.1+cu118
- CUDA execution note: sandbox torch precheck reported no CUDA GPUs, so the actual training command was run in the approved non-sandbox environment.
- GPU: selected GPU 2
- Dataset name: lsvid
- Dataset root used by config: /data1/lgf/TF-CLIP/data/LS-VID
- Extracted dataset root: /data1/lgf/Datasets/LS-VID
- Dataset links used:
  - /data1/lgf/TF-CLIP/data/LS-VID/info/list_sequence -> /data1/lgf/Datasets/LS-VID/list_sequence
  - /data1/lgf/TF-CLIP/data/LS-VID/info/data -> /data1/lgf/Datasets/LS-VID/test/data
  - /data1/lgf/TF-CLIP/data/LS-VID/tracklet_train -> /data1/lgf/Datasets/LS-VID/tracklet_train
  - /data1/lgf/TF-CLIP/data/LS-VID/tracklet_test -> /data1/lgf/Datasets/LS-VID/tracklet_test
- Config: configs/vit_clipreid_lsvid.yml
- QATA enabled status: False
- MEMORY multi-prototype enabled status: False
- MODEL.QATA.APPLY_MEMORY_CONSTRUCTION: field not present in config/defaults.py
- Command: `CUDA_VISIBLE_DEVICES=2 /data1/lgf/miniconda3/envs/tfclip/bin/python train.py --config_file configs/vit_clipreid_lsvid.yml OUTPUT_DIR logs/baseline_lsvid_20260613_045926`
- Exit status: 0
- Output dir: logs/baseline_lsvid_20260613_045926
- Train log: logs/baseline_lsvid_20260613_045926/train_log.txt
- Captured stdout/stderr: logs/baseline_lsvid_20260613_045926/train_stdout_stderr.txt
- Run meta: logs/baseline_lsvid_20260613_045926/run_meta.txt
- best_model.pth.tar: created
- checkpoint_ep.pth.tar: created

| Dataset | Method | Config | Output Dir | mAP | Rank-1 | Rank-5 | Best Epoch | Train Time | Notes |
| ------- | ------ | ------ | ---------- | --: | -----: | -----: | ---------: | ---------- | ----- |
| LS-VID | TF-CLIP baseline | configs/vit_clipreid_lsvid.yml | logs/baseline_lsvid_20260613_045926 | 79.8 | 88.1 | 96.2 | 80 | 3h 58m 03s | Completed original baseline only; Rank-10 97.4. |

## Final Judgment 2026-06-13

1. LS-VID baseline completed normally with exit status 0.
2. The result is usable as the LS-VID baseline reference for later Residual QATA validation.
3. It is reasonable to run Residual QATA a=0.1 on LS-VID next, using this run as the comparison baseline.
4. No abnormal final result was observed. The earlier failures were setup/environment issues: missing extracted dataset root, then sandbox CUDA visibility.
