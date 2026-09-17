# UAV Thermal Person Detector

Person detection in thermal imagery captured from a drone, built for **search and rescue (SAR)** scenarios in rural environments (runways, dirt roads, open fields).

YOLOv8-based models trained on a custom thermal imagery dataset (with some RGB frames) labeled with a single class: `person`.

## Models

Two variants were trained with different goals:

| Model | Architecture | Params | imgsz | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|---|---|---|
| `best_uav_fast.pt` | YOLOv8s | 9.8M | 512 | 0.822 | 0.607 | 0.693 | 0.424 |
| `monet_pop_p2_1024` (best epoch) | YOLOv8m-**p2** | 25.1M | 960 | 0.889 | 0.713 | – | 0.502 |

- **`best_uav_fast`**: lightweight model, built for fast inference (~76 ms/frame on CPU at 512px). Good starting point for limited drone/embedded hardware.
- **`monet_pop_p2_1024`** (see [`notebooks/train_p2_1024.ipynb`](notebooks/train_p2_1024.ipynb)): variant with an extra **P2** detection head (stride 4), specifically designed for very small objects in the frame — a person seen from altitude typically occupies very few pixels. Trained at higher resolution (960px), it achieves better recall and mAP50-95 at the cost of more compute. *This model's weights are not included in this repo (trained and saved on Google Drive); only the training notebook and metrics are included.*

Only `best_uav_fast.pt` is included under [`models/`](models/) due to size and availability.

## Dataset

- Real thermal drone imagery (FLIR-style sensor noise: horizontal banding, vignetting), combined with a few RGB frames.
- Organized by capture scenario: runways (`runway`), dirt roads (`dirtroad`), etc.
- Single class: `person`. Labeled boxes are mostly very small (people seen from altitude), which makes recall the hardest metric to improve.
- Training batch samples (mosaic augmentation) in [`samples/`](samples/):

  ![train_batch0](samples/train_batch0.jpg)

## Limitations

- mAP50-95 (~0.42–0.50) indicates that exact box localization isn't perfect, although detection itself (mAP50) is reasonable.
- The `fast` model's recall (0.61) means it misses people fairly often — **it should not be used as an autonomous decision system**, but as a support/screening tool for a human operator.
- Dataset has few boxes per image and many person-free backgrounds (hard negatives), typical of real SAR scenarios where the positive class is rare.

## Quick usage

```bash
pip install ultralytics

# Inference on an image or folder
yolo predict model=models/best_uav_fast.pt source=path/to/image_or_folder conf=0.25

# Live inference from webcam/camera
yolo predict model=models/best_uav_fast.pt source=0 show=True
```

Or from Python:

```python
from ultralytics import YOLO

model = YOLO("models/best_uav_fast.pt")
results = model.predict("path/to/image.jpg", conf=0.25)
results[0].show()
```

## Training

The notebook [`notebooks/train_p2_1024.ipynb`](notebooks/train_p2_1024.ipynb) contains the full training pipeline for the YOLOv8m-p2 variant on Google Colab (Drive mounting, dataset extraction, `yolo train` with `imgsz=960`, `cos_lr`, resume after session interruptions, and per-epoch metrics analysis).

## License

[MIT](LICENSE)
