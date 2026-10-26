# Prototypes

Each person improves **one** pipeline step with their own idea.

## How

```bash
git checkout -b <yourname>
mkdir prototypes/<yourname>
cp whiteboard_starter.py prototypes/<yourname>/<yourname>_pipeline.py
# edit ONE function: detect_board, flatten, normalize_lighting, or find_writing
python prototypes/<yourname>/<yourname>_pipeline.py data/phone_photos results_<yourname>
```

Compare `results/` (starter) with `results_<yourname>/` (yours) photo by photo.

## Ideas to start from

**1. Detect board** (`detect_board`)
- Hough lines: find long straight lines, intersect them to get the 4 corners (works when a corner is hidden)
- Color/brightness segmentation: boards are big bright low-saturation regions (HSV threshold), then take the largest blob
- Pick the best quad by score (area + how rectangular it is), not just the first 4-sided contour

**2. Flatten** (`flatten`)
- Use the board's real aspect ratio (e.g. 4×8 ft = 1:2) instead of guessing width/height from the photo
- Refine corners to sub-pixel accuracy with `cv2.cornerSubPix` before warping

**3. Reduce glare** (`normalize_lighting`)
- CLAHE (local contrast equalization) on the L channel in LAB color space
- Detect saturated glare pixels (near 255) and inpaint them with `cv2.inpaint`
- Tune the background blur size; try morphological closing instead of median blur

**4. Find writing** (`find_writing`)
- Sauvola or Otsu thresholding instead of Gaussian adaptive
- Keep marker color: threshold in HSV to separate blue/red/black ink from the white board
- Filter boxes by aspect ratio and fill ratio to drop board frame edges and smudges

## What to bring to the meeting

Fill in your row in `../RESULTS.md`: which step, your idea, which photos it fixed, which it broke.
