# Road Distress Research Implementation

PyTorch + Ultralytics research scaffold for road distress detection, segmentation, geometry-guided severity estimation, uncertainty estimation, maintenance prioritization, evaluation and ablations.

## Scientific cautions
- RDD2022 standard annotations are bounding boxes, not pixel masks or engineering severity labels.
- Attain annotation schemas and severity encodings must be checked against the exact downloaded release.
- PDD2026/PaveSeg annotation formats and class definitions must be verified before conversion.
- RGB images alone cannot provide real-world crack width or pothole depth without calibration, known scale, stereo or depth sensing. Geometry is in pixels unless a valid scale is supplied.
- Maintenance scoring is a research prototype, not an ASTM/IRC/PCI standard. Replace weights with expert/standard-derived values and validate.
- The included random split is a baseline only. For publication, split by road, route, sequence or site to prevent leakage.

## Install
Use Python 3.10/3.11 and a CUDA-matched PyTorch build.
```bash
conda create -n pavement-ai python=3.10 -y
conda activate pavement-ai
pip install -r requirements.txt
pip install -e .
```

## Structure
- `scripts/prepare_rdd.py`: convert normalized RDD CSV to YOLO detection format.
- `scripts/prepare_coco.py`: convert COCO JSON detection or polygon segmentation to YOLO.
- `scripts/train_yolo.py`: train Ultralytics detector or segmenter.
- `scripts/train_severity.py`: train multimodal severity classifier.
- `scripts/evaluate_severity.py`: metrics, confusion matrix and MC-dropout uncertainty.
- `scripts/evaluate_yolo.py`: Ultralytics validation.
- `scripts/run_ablation.py`: image-only, geometry-only and fused severity ablations.
- `scripts/infer.py`: inference.
- `src/road_distress/`: reusable data, geometry, model, metrics and decision modules.

## Dataset setup

### RDD2022
Download from https://github.com/sekilab/RoadDamageDetector. Create a CSV with columns:
`image,class,xmin,ymin,xmax,ymax`
Then:
```bash
python scripts/prepare_rdd.py --images data/raw/rdd/images --csv data/raw/rdd/annotations.csv --out data/processed/rdd --class-map configs/rdd_classes.yaml
```
Edit the class mapping to match the exact source label names.

### COCO-like datasets
Only use this converter if the release really provides COCO JSON (`images`, `annotations`, `categories`):
```bash
python scripts/prepare_coco.py --json annotations.json --images images --out data/processed/pdd --task segment
```
For segmentation, COCO polygon segmentations are supported; RLE masks need an additional conversion path.

### Severity manifest
Create a CSV:
```csv
image,mask,distress_type,severity,road_id,scale_mm_per_pixel
images/a.jpg,masks/a.png,pothole,2,road_01,
images/b.jpg,masks/b.png,longitudinal_crack,1,road_02,0.35
```
Severity must be numeric and consistently encoded (default examples: 0=low, 1=medium, 2=high). Blank mask is allowed for image-only experiments. `scale_mm_per_pixel` must come from calibration, not guesswork. Paths are resolved relative to the manifest file.

## Train detector/segmenter
Edit `configs/yolo.yaml`. Choose `task: detect` with a detection checkpoint such as `yolo11s.pt`, or `task: segment` with a segmentation checkpoint such as `yolo11s-seg.pt`.
```bash
python scripts/train_yolo.py --config configs/yolo.yaml
```

## Train severity
```bash
python scripts/train_severity.py --manifest data/processed/severity_manifest.csv --output outputs/severity --epochs 40
```
The default split is random and is not acceptable as final evidence where images share roads/sequences. Replace it with grouped road/site splits for the paper. The severity model uses ResNet18 image features and mask-derived geometry features. The mask should be aligned to the image.

## Evaluate and infer
```bash
python scripts/evaluate_yolo.py --weights outputs/yolo/pavement/weights/best.pt --data data/processed/rdd/data.yaml --split test
python scripts/evaluate_severity.py --checkpoint outputs/severity/best.pt --manifest data/processed/severity_test.csv
python scripts/infer.py --weights outputs/yolo/pavement/weights/best.pt --image sample.jpg
```

## Ablations
```bash
python scripts/run_ablation.py --manifest data/processed/severity_manifest.csv --output outputs/ablations --epochs 30
```
Reports image-only, geometry-only and image+geometry. Keep splits, seeds and training budgets identical.

## Publication-grade protocol
1. Harmonize category definitions across datasets; retain dataset identity.
2. Freeze a road/site-grouped test set before model development.
3. Report per-class precision, recall, mAP, mIoU/Dice, macro-F1, ordinal MAE, quadratic weighted kappa, ECE/Brier and risk-coverage.
4. Run 3–5 seeds where feasible and report mean ± standard deviation and confidence intervals.
5. Compare against established detector, segmentation and severity baselines.
6. Validate geometry against physical measurements and severity against independent pavement-engineer labels.
7. Validate maintenance priorities against expert decisions or documented intervention records.

This code is an implementation starting point. It does not by itself establish novelty or state-of-the-art performance.
