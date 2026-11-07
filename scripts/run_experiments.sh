#!/usr/bin/env bash
# Task 3.5 - six runs, each one change away from the reference run (R2).
#   R1 cnn baseline            (backbone change: no pretraining)
#   R2 mnv2 reference          lr 1e-3, light aug, 224, last block unfrozen
#   R3 mnv2 lr 3e-4            (learning-rate change)
#   R4 mnv2 strong aug         (augmentation change)
#   R5 mnv2 frozen only        (frozen-layers change: never unfreeze)
#   R6 mnv2 160px              (image-size change)
#   R7 effnetb0                (backbone change)
# Plus R8 = R2 with MLflow autolog instead of the manual callback (Task 3.3).
set -euo pipefail
cd "$(dirname "$0")/.."
T="python scripts/train.py"
$T --model cnn          --lr 1e-3 --epochs 30           --run-name R1_cnn_baseline
$T --model mobilenetv2  --lr 1e-3                       --run-name R2_mnv2_ref
$T --model mobilenetv2  --lr 3e-4                       --run-name R3_mnv2_lr3e-4
$T --model mobilenetv2  --lr 1e-3 --augment strong      --run-name R4_mnv2_strongaug
$T --model mobilenetv2  --lr 1e-3 --unfreeze-blocks 0   --run-name R5_mnv2_frozen
$T --model mobilenetv2  --lr 1e-3 --img-size 160        --run-name R6_mnv2_160px
$T --model efficientnetb0 --lr 1e-3                     --run-name R7_effb0
$T --model mobilenetv2  --lr 1e-3 --tracking autolog    --run-name R8_mnv2_ref_autolog
python scripts/compare_runs.py
