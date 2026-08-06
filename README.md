# LCMS UNet++ Crack Segmentation

This folder is a self-contained research pipeline for multi-class LCMS road-crack semantic segmentation.

## Structure

- `models/`: modular UNet++ implementation
  - ResNet-50 encoder with ImageNet weights and arbitrary input-channel adaptation
  - ASPP or pyramid context module
  - residual nested decoder
  - SE, CBAM, or attention-gate support
  - optional deep supervision and edge head
- `models/unet3plus.py`: PyTorch UNet 3+ style full-scale skip model for the same training pipeline
- `losses.py`: CrossEntropy+Dice, multi-class Focal, Boundary, and combined crack loss
- `metrics.py`: accuracy, precision, recall, F1/Dice, IoU, CSV logging, plots
- `data.py`: LCMS dataset for `TRAIN/VAL/TEST/IMAGES` and `MASKS`
- `convert_to_binary_dataset.py`: copies a multiclass dataset and converts masks to binary foreground/background PNGs
- `train.py`: AMP training, warmup cosine LR, gradient clipping, EMA, checkpoint saving
- `test.py`: checkpoint evaluation
- `export_onnx.py`: ONNX export
- `visualize.py`: metric visualization over epochs

## Train

```bash
python training_code/lcms_unetpp/train.py \
  --data-path V:\\Devendra\\ASPHALT\\ASPHALT_ACCEPTED\\COMBINED_SPLITTED \
  --output-dir weights/lcms_unetpp \
  --in-channels 3 \
  --num-classes 6 \
  --height 1024 \
  --width 419 \
  --patch-height 512 \
  --patch-width 384 \
  --samples-per-image 4 \
  --anomaly-ratio 0.7 \
  --random-crop-ratio 0.1 \
  --full-image-ratio 0.1 \
  --class-sampling-weights 1:2,2:6,3:5,4:2,5:2 \
  --loss-class-weights 0:0.05,1:2,2:80,3:50,4:4,5:2 \
  --monitor-metric mean_iou \
  --attention gate \
  --context aspp \
  --num-workers 0 \
  --batch-size 4 \
  --epochs 150 \
  --amp
```

The architecture can also be selected from `config.yaml`:

```yaml
model:
  name: hrnet_ocr   # use unetpp, unet3plus, or hrnet_ocr
  hrnet_width: 32
  hrnet_norm: batch
  hrnet_activation: relu
  ocr_mid_channels: 512
  ocr_key_channels: 256
```

The region overlap loss can also be selected there:

```yaml
loss:
  region_loss: tversky   # use dice or tversky
  dice_weight: 0.5       # weight of Dice/Tversky inside CE + region loss
  tversky_alpha: 0.3     # false-positive penalty
  tversky_beta: 0.7      # false-negative penalty; higher favors crack recall
  tversky_gamma: 1.0     # focal Tversky exponent
```

Train from the YAML config:

```powershell
$env:KMP_DUPLICATE_LIB_OK='TRUE'
python train.py --config config.yaml
```

## Binary vs Multiclass Mode

The training and test scripts are configurable for either multiclass segmentation or binary foreground/background segmentation.

Use multiclass when you need separate distress categories:

```yaml
data:
  data_path: F:\Akash\Data\25thjuly\newData_v1
  mask_mode: color
  num_classes: 6
```

Use binary when you only need distress-vs-background:

```yaml
data:
  data_path: F:\Akash\Data\25thjuly\newData_v2
  mask_mode: binary
  num_classes: 2
```

The repo includes two ready-to-run configs:

```text
config_multiclass.yaml
config_binary.yaml
```

### Create Binary Dataset

Create `newData_v2` from the cleaned multiclass `newData_v1`:

```powershell
python convert_to_binary_dataset.py `
  --src F:\Akash\Data\25thjuly\newData_v1 `
  --dst F:\Akash\Data\25thjuly\newData_v2
```

This copies all images unchanged and writes grayscale binary masks:

```text
0   = background
255 = any known foreground class
```

Expected folder layout:

```text
newData_v2/
  TRAIN/
    IMAGES/
    MASKS/
  VAL/
    IMAGES/
    MASKS/
  TEST/
    IMAGES/
    MASKS/
```

If `newData_v2` already exists and you intentionally want to regenerate it:

```powershell
python convert_to_binary_dataset.py `
  --src F:\Akash\Data\25thjuly\newData_v1 `
  --dst F:\Akash\Data\25thjuly\newData_v2 `
  --overwrite
```

### Train Binary UNet++

For binary training from the prepared config:

```powershell
$env:KMP_DUPLICATE_LIB_OK='TRUE'
python train.py --config config_binary.yaml
```

`config_binary.yaml` uses UNet++ with:

```yaml
model:
  name: unetpp

data:
  mask_mode: binary
  num_classes: 2
```

For binary configs, keep single class-weight strings quoted, for example:

```yaml
patch:
  class_sampling_weights: "1:1"
```

If initializing from a 6-class checkpoint, use `init_checkpoint`, not `resume`:

```yaml
checkpoint:
  init_checkpoint: F:/Akash/CodeHub/lcms_unetpp/weights/lcms_unetpp_31stjuly_finetune/best_mean_iou.pth
```

This loads compatible encoder/decoder weights and skips the incompatible 6-class output heads.

Do not use `--resume` when switching from 6 classes to 2 classes. `--resume` expects the same model head shape, optimizer state, scheduler state, and epoch state.

### Train Multiclass UNet++

For multiclass training:

```powershell
$env:KMP_DUPLICATE_LIB_OK='TRUE'
python train.py --config config_multiclass.yaml
```

Or continue using `config.yaml` if it is set to:

```yaml
data:
  mask_mode: color
  num_classes: 6
```

### Test Binary Model

Evaluate binary validation metrics:

```powershell
$env:KMP_DUPLICATE_LIB_OK='TRUE'
python test.py `
  --config config_binary.yaml `
  --checkpoint weights\lcms_unetpp_binary_31july\best_mean_iou.pth `
  --split VAL `
  --output-json weights\lcms_unetpp_binary_31july\val_metrics.json
```

Save binary predictions and overlays:

```powershell
$env:KMP_DUPLICATE_LIB_OK='TRUE'
python test.py `
  --config config_binary.yaml `
  --checkpoint weights\lcms_unetpp_binary_31july\best_mean_iou.pth `
  --split TEST `
  --output-json weights\lcms_unetpp_binary_31july\test_metrics.json `
  --save-predictions `
  --prediction-dir weights\lcms_unetpp_binary_31july\predictions
```

For binary segmentation, `mean_iou` is foreground IoU because background is excluded from the mean when `num_classes > 1`. This is usually the best checkpoint metric for distress-vs-background generalization.

## ClearML Tracking

Training logs to ClearML by default when the `clearml` Python package is installed. The repo-local `clearml.conf` is used automatically unless `CLEARML_CONFIG_FILE` is already set.

ClearML receives:

- resolved hyperparameters and training configuration
- dataset/training metadata
- every epoch metric from `metrics.csv`
- `latest.pth`, `best.pth`, and `best_<monitor_metric>.pth` as output models
- `config.json`, `metrics.csv`, `metrics.png`, `heartbeat.json`, and `train.log` as artifacts

Configure it from `config.yaml`:

```yaml
clearml:
  clearml_project: LCMS Crack Segmentation
  clearml_task_name: lcms_hrnet_ocr_newData_v2
  clearml_tags: hrnet_ocr,newData_v2
  clearml_config_file: clearml.conf
  clearml_output_uri: ""
  clearml_offline: false
  no_clearml_models: false
```

Useful overrides:

```powershell
python train.py --config config.yaml --clearml-task-name my_run
python train.py --config config.yaml --no-clearml
python train.py --config config.yaml --clearml-offline
python train.py --config config.yaml --no-clearml-models
```

To train only from `config.yaml`, edit the values inside `config.yaml` and run only the command above. You do not need to pass the long CLI flags. The config file contains the dataset path, model choice, patch sampling, loss, checkpoint, and training hyperparameters.

Example minimal workflow:

```powershell
# 1. Edit config.yaml
# 2. Start training from the config
$env:KMP_DUPLICATE_LIB_OK='TRUE'
python train.py --config config.yaml
```

CLI flags are optional overrides. For example, this keeps everything from `config.yaml` except the output folder:

```powershell
python train.py --config config.yaml --output-dir weights\my_experiment
```

`--init-checkpoint` loads only compatible tensors. A UNet++ checkpoint can initialize another UNet++ run, but it will not meaningfully initialize HRNet+OCR because the architectures do not share most layer names/shapes.

To switch back to UNet++, change `model.name` to `unetpp` or override from the command line:

```powershell
python train.py --config config.yaml --model unetpp --output-dir weights\lcms_unetpp_from_config
```

For single-channel LCMS images, use `--in-channels 1`. The ResNet first convolution is adapted from RGB pretrained weights by averaging and repeating kernels. Use `--mask-mode color` for RGB class-color masks, `--mask-mode index` for class-id masks, or `--mask-mode binary` only for foreground/background experiments.

Training uses online biased patch sampling by default. About 70% of non-random training patches are centered on anomaly pixels and about 30% are background/mostly-background patches. `--random-crop-ratio` adds fully random crops before the anomaly/background choice. `--full-image-ratio` can replace a small fraction of patch samples with resized full-image samples so the model sees global context during training. Transverse and longitudinal cracks are oversampled by the default class weights. Validation still runs on full-size images.

`best.pth` is still saved using validation `mean_dice`. To save one extra checkpoint for a metric you care about, use `--monitor-metric`. Examples: `--monitor-metric mean_iou`, `--monitor-metric class_3_precision`, `--monitor-metric class_2_f1`, or `--monitor-metric val_loss --monitor-mode min`. The file is saved as `best_<metric>.pth`, for example `best_class_3_precision.pth`.

## UNet 3+ Multiclass Pipeline

This repo also includes a PyTorch UNet 3+ style model inspired by the full-scale skip connection design from [hamidriasat/UNet-3-Plus](https://github.com/hamidriasat/UNet-3-Plus). It is integrated into the existing LCMS pipeline, so it uses the same `train.py`, `test.py`, losses, metrics, ClearML logging, checkpoints, and `TRAIN/VAL/TEST` dataset layout.

The referenced implementation is TensorFlow-oriented and documents UNet 3+ full-scale skip connections with deep supervision. The local implementation keeps those model ideas but is written as a native PyTorch module for this project.

Use the ready config:

```text
config_unet3plus_multiclass.yaml
```

Important config fields:

```yaml
data:
  data_path: F:\Akash\Data\25thjuly\newData_v1
  mask_mode: color
  num_classes: 6

model:
  name: unet3plus
  deep_supervision: true
  norm: group
  activation: silu
  no_edge_head: false
  unet3plus_base_channels: 32
  unet3plus_cat_channels: 32
  unet3plus_dropout: 0.1

training:
  monitor_metric: mean_iou
  monitor_mode: max
  early_stopping_patience: 30

patch:
  foreground_extension_ratio: 0.35
  class_sampling_weights: 1:2,2:6,3:12,4:2,5:12

loss:
  region_loss: tversky
  tversky_alpha: 0.3
  tversky_beta: 0.7
  tversky_gamma: 1.33
  boundary_weight: 0.03
  edge_weight: 0.2
  ohem_start_epoch: 40
  ohem_ratio: 0.25
  lovasz_start_epoch: 150
  lovasz_weight: 0.2
```

The UNet 3+ config enables the requested patch-based multiclass setup:

- Foreground-aware patch extension: `foreground_extension_ratio` crops extra target-centered context and resizes it back to the patch size.
- Class-balanced sampling: `class_sampling_weights` oversamples rare foreground classes.
- Focal Tversky loss: `tversky_gamma > 1` with `tversky_beta > tversky_alpha` favors hard thin-class recall.
- Boundary-aware auxiliary losses: `boundary_weight` and `edge_weight` supervise thin foreground boundaries.
- OHEM: `ohem_start_epoch` turns on hard-pixel mining after the model has started converging.
- Lovasz-Softmax fine-tuning: `lovasz_start_epoch` adds an IoU surrogate loss for the final training stage.
- Test-time augmentation: use `--tta` during `test.py` or `inference.py`.

Train UNet 3+ multiclass:

```powershell
cd F:\Akash\CodeHub\lcms_unetpp
$env:KMP_DUPLICATE_LIB_OK='TRUE'
python train.py --config config_unet3plus_multiclass.yaml
```

Use a different output folder for a comparison run:

```powershell
python train.py `
  --config config_unet3plus_multiclass.yaml `
  --output-dir weights\lcms_unet3plus_multiclass_run2 `
  --clearml-task-name lcms_unet3plus_multiclass_run2
```

If GPU memory is tight, reduce:

```yaml
training:
  batch_size: 2

model:
  unet3plus_base_channels: 24
  unet3plus_cat_channels: 24
```

Evaluate a trained UNet 3+ checkpoint:

```powershell
$env:KMP_DUPLICATE_LIB_OK='TRUE'
python test.py `
  --config config_unet3plus_multiclass.yaml `
  --checkpoint weights\lcms_unet3plus_multiclass\best_mean_iou.pth `
  --split VAL `
  --output-json weights\lcms_unet3plus_multiclass\val_metrics.json
```

Evaluate with horizontal-flip test-time augmentation:

```powershell
$env:KMP_DUPLICATE_LIB_OK='TRUE'
python test.py `
  --config config_unet3plus_multiclass.yaml `
  --checkpoint weights\lcms_unet3plus_multiclass\best_mean_iou.pth `
  --split VAL `
  --tta `
  --output-json weights\lcms_unet3plus_multiclass\val_metrics_tta.json
```

Run prediction-only inference with TTA:

```powershell
$env:KMP_DUPLICATE_LIB_OK='TRUE'
python inference.py `
  --config config_unet3plus_multiclass.yaml `
  --checkpoint weights\lcms_unet3plus_multiclass\best_mean_iou.pth `
  --data-path F:\Akash\Data\25thjuly\newData_v1 `
  --split TEST `
  --tta `
  --prediction-dir weights\lcms_unet3plus_multiclass\predictions_tta
```

Use `mean_iou` for checkpoint selection, then inspect per-class IoU to confirm class `3` and class `5` are not being traded away for the easier classes.

## Fine-tune on newData_v2

Use `--init-checkpoint` for transfer learning from an existing checkpoint. This loads only model weights and starts a fresh optimizer/scheduler, unlike `--resume`, which continues the old run state.

First rebuild `newData_v2` with pixel-balanced TRAIN/VAL/TEST splits so transverse and longitudinal pixels are represented in every split:

```powershell
python split_lcms_dataset.py `
  --root F:\Akash\Data\29thJune\newData_v2 `
  --val-ratio 0.15 `
  --test-ratio 0.15 `
  --pixel-balanced `
  --seed 42
```

The current split summary after this command is:

```text
TRAIN: 706 images
  transverse    60,361 px
  longitudinal  71,938 px

VAL: 152 images
  transverse    13,076 px
  longitudinal  16,318 px

TEST: 152 images
  transverse    13,029 px
  longitudinal  15,841 px
```

Recommended fine-tuning command:

```powershell
$env:KMP_DUPLICATE_LIB_OK='TRUE'
python train.py `
  --data-path F:\Akash\Data\29thJune\newData_v2 `
  --mask-mode color `
  --num-classes 6 `
  --in-channels 3 `
  --init-checkpoint weights\lcms_unetpp_patch_gate_aspp\best.pth `
  --output-dir weights\lcms_unetpp_newData_v2_finetune `
  --batch-size 4 `
  --epochs 180 `
  --height 1024 `
  --width 419 `
  --patch-height 512 `
  --patch-width 384 `
  --samples-per-image 16 `
  --anomaly-ratio 0.9 `
  --random-crop-ratio 0.05 `
  --full-image-ratio 0.15 `
  --min-target-pixels 48 `
  --target-crop-attempts 48 `
  --class-sampling-weights 1:2,2:10,3:10,4:2,5:2 `
  --loss-class-weights 0:0.1,1:1,2:12,3:12,4:2,5:2 `
  --lr 2e-5 `
  --weight-decay 1e-4 `
  --boundary-weight 0.03 `
  --edge-weight 0.05 `
  --focal-weight 0.25 `
  --monitor-metric class_3_iou `
  --attention gate `
  --context aspp `
  --num-workers 0
```

For a boundary-loss ablation, run the same command with a different `--output-dir` and:

```powershell
--boundary-weight 0 --edge-weight 0
```

## Test

```bash
python training_code/lcms_unetpp/test.py \
  --data-path V:\\Devendra\\ASPHALT\\ASPHALT_ACCEPTED\\COMBINED_SPLITTED \
  --checkpoint weights/lcms_unetpp/best.pth \
  --split VAL \
  --output-json weights/lcms_unetpp/val_metrics.json
```

To save predicted masks and overlays:

```bash
python test.py \
  --data-path F:\\Akash\\Data\\4thAugust \
  --checkpoint weights/lcms_unetpp_full_image_run_30thJuly/best.pth \
  --in-channels 3 \
  --num-classes 6 \
  --height 1024 \
  --width 419 \
  --save-predictions \
  --prediction-dir weights/lcms_unetpp_full_image_run_30thJuly/predictions
```

Predicted color masks are written to `predictions/masks`, and image overlays are written to `predictions/overlays`.

## ONNX Export

```bash
python training_code/lcms_unetpp/export_onnx.py \
  --checkpoint weights/lcms_unetpp/best.pth \
  --onnx-path weights/lcms_unetpp/model.onnx \
  --in-channels 3 \
  --num-classes 6 \
  --height 1024 \
  --width 419 \
  --dynamic
```

## test onnx model

```bash
python test_onnx.py `
  --data-path F:\Akash\Data\29thJune\newData_v2 `
  --onnx-path weights\lcms_unetpp_newData_v2_finetune\model.onnx `
  --split TEST `
  --height 1024 `
  --width 419 `
  --num-classes 6 `
  --mask-mode color `
  --output-json weights\lcms_unetpp_newData_v2_finetune\onnx_test_metrics.json `
  --save-predictions `
  --prediction-dir weights\lcms_unetpp_newData_v2_finetune\onnx_predictions
  ```

  ```bash
  python test_onnx.py --data-path F:\Akash\Data\29thJune\newData --onnx-path weights\lcms_unetpp_newData_v2_finetune\model.onnx --split TEST --height 1024 --width 419 --num-classes 6 --mask-mode color --output-json weights\lcms_unetpp_newData_v2_finetune\onnx_test_metrics.json --save-predictions --prediction-dir weights\lcms_unetpp_newData_v2_finetune\onnx_predictions
  ```

  ```python test_onnx_inference.py --data-path F:\Akash\Data\29thJune\newData --onnx-path weights\lcms_unetpp_newData_v2_finetune\model.onnx --split TEST --height 1024 --width 419 --num-classes 6 --prediction-dir weights\lcms_unetpp_newData_v2_finetune\onnx_inference
  ```

  ```python overlay_to_point_cloud.py --prediction-dir weights\lcms_unetpp_newData_v2_finetune\onnx_predictions --output-dir weights\lcms_unetpp_newData_v2_finetune\onnx_predictions\point_clouds --z-source mask --stride 4
  ```

  ```
  python overlay_to_point_cloud.py --prediction-dir weights\lcms_unetpp_newData_v2_finetune\onnx_predictions --range-dir F:\path\to\rangeDataFiltered --raw-range-dir F:\path\to\rangeDataRaw --z-source range --stride 2
  ```

## Plot Metrics

Training writes `metrics.csv` and `metrics.png` automatically. To regenerate plots:

```bash
python training_code/lcms_unetpp/visualize.py \
  --csv weights/lcms_unetpp/metrics.csv \
  --out weights/lcms_unetpp/metrics.png
```

## Debug Logs

Training mirrors console output, tracebacks, Python/C fault traces, environment details, dataset sizes, checkpoints, and metric writes to:

```text
weights/lcms_unetpp_newData/train.log
```

Training also updates this file every train/validation batch:

```text
weights/lcms_unetpp_newData/heartbeat.json
```

If training stops without a traceback, check `heartbeat.json` for the last phase, epoch, and batch. All entry scripts also accept `--log-file` to choose a custom log path.

``` for new data ```

```python train.py --data-path F:\Akash\Data\30thJune\newData_v1 --mask-mode color --num-classes 6 --in-channels 3 --batch-size 8 --epochs 150 --height 1024 --width 419 --patch-height 512 --patch-width 384 --samples-per-image 4 --anomaly-ratio 0.7 --random-crop-ratio 0.1 --full-image-ratio 0.1 --class-sampling-weights 1:2,2:6,3:5,4:2,5:2 --loss-class-weights 0:0.05,1:2,2:80,3:50,4:4,5:2 --monitor-metric class_3_iou --attention gate --context aspp --num-workers 0 --output-dir weights\lcms_unetpp_patch_gate_aspp```

``` for old data ```

```python train.py --data-path F:\Akash\Data\29thJune\SPLIT --mask-mode color --num-classes 6 --in-channels 3 --batch-size 8 --epochs 150 --height 992 --width 416 --patch-height 512 --patch-width 384 --samples-per-image 8 --anomaly-ratio 0.8 --random-crop-ratio 0.1 --full-image-ratio 0.1 --class-sampling-weights 1:2,2:8,3:14,4:2,5:2 --loss-class-weights 0:0.05,1:2,2:100,3:180,4:4,5:2 --monitor-metric class_3_iou --attention gate --context aspp --num-workers 0 --output-dir weights\lcms_unetpp_patch_gate_aspp```

```python train.py --data-path F:\Akash\Data\29thJune\newData_v2 --mask-mode color --num-classes 6 --in-channels 3 --init-checkpoint weights\lcms_unetpp_patch_gate_aspp\best.pth --output-dir weights\lcms_unetpp_newData_v2_finetune --batch-size 4 --epochs 100 --height 1024 --width 419 --patch-height 512 --patch-width 384  --samples-per-image 16 --anomaly-ratio 0.9 --random-crop-ratio 0.05 --full-image-ratio 0.15 --min-target-pixels 48 --target-crop-attempts 48 --class-sampling-weights 1:2,2:10,3:10,4:2,5:2 --loss-class-weights 0:0.1,1:1,2:12,3:12,4:2,5:2 --lr 2e-5 --weight-decay 1e-4 --boundary-weight 0.03 --edge-weight 0.05 --focal-weight 0.25 --monitor-metric class_3_iou --attention gate --context aspp --num-workers 0```

```
TRAIN:
background    241 masks
alligator      20 masks
transverse     69 masks
longitudinal  148 masks
pothole        46 masks
patch          59 masks

VAL:
background     61 masks
alligator       5 masks
transverse     22 masks
longitudinal   32 masks
pothole         8 masks
patch          17 masks
```

```
TRAIN
Class          Masks containing class   Connected instances   Pixels
alligator      20                       2572                  14854
transverse     69                       398                   1005
longitudinal   148                      2023                  4301
pothole        46                       345                   70085
patch          59                       168                   1333198

VAL
Class          Masks containing class   Connected instances   Pixels
alligator      5                        1313                  8079
transverse     22                       194                   547
longitudinal   32                       720                   1806
pothole        8                        27                    3515
patch          17                       41                    196867
```
