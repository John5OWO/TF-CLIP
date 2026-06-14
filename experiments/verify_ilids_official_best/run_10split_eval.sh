#!/usr/bin/env bash
set -e

cd /data1/lgf/experiments/verify_ilids_official_best/official_run_workspace

OUT=/data1/lgf/experiments/verify_ilids_official_best
CKPT_DIR=/data1/lgf/experiments/verify_ilids_official_best/official_package/iLIDS_best_code_and_weight/logs_ilids/logs_ilids

source /data1/lgf/miniconda3/bin/activate tfclip

for SPLIT in 0 1 2 3 4 5 6 7 8 9
do
echo "=============================="
echo "Running iLIDS split ${SPLIT}"
echo "=============================="

CUDA_VISIBLE_DEVICES=0 python eval_all.py \
  --config_file configs/official_ilids_eval.yml \
  DATASETS.SPLIT ${SPLIT} \
  TEST.WEIGHT ${CKPT_DIR}/split${SPLIT}/best_model.pth.tar \
  OUTPUT_DIR ${OUT}/logs/split${SPLIT} \
  2>&1 | tee ${OUT}/logs/official_eval_split${SPLIT}.log
done
