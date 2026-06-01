# Scratch Mask Pipeline

This project loads a local leather image, attempts to segment a likely scratch with Segment Anything (SAM), and writes a visible mask overlay to `outputs/`.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Input

Place the source image at `data/input.jpg`.

## Run

```bash
source venv/bin/activate
python src/main.py
```

## Outputs

The pipeline writes:

- `outputs/mask.png`
- `outputs/overlay.png`
- `outputs/overlay_candidate_1.png` (and more if multiple strong candidates are found)
- `outputs/overlay_<image-name>.png` for the other images in the folder

SAM checkpoints are downloaded automatically into `models/`.

## Notes

- `data/input.jpg` is treated as the primary image and maps to `outputs/overlay.png`.
- Other top-level `.jpg`, `.jpeg`, and `.png` files are processed in the same run.
- `BeforeAfter*.jpg` images are cropped to the left-side "before" panel before masking.
