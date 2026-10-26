"""
Spot Whiteboard - Mini 1 starter prototype (conventional CV, no ML)
Pipeline: detect board -> flatten (homography) -> reduce glare -> find writing

Usage:
    pip install -r requirements.txt
    python whiteboard_starter.py data/phone_photos results

Each photo produces 4 images in results/:
    <name>_1_edges.jpg    Canny edges used to find the board
    <name>_2_board.jpg    detected board outline (green)
    <name>_3_flat.jpg     board flattened to a straight-on view
    <name>_4_writing.jpg  boxes around detected writing (red)
plus results/summary.csv with one row per photo (bring this to the meeting).

To improve a step: copy this file to prototypes/<yourname>/ and change ONE function.
"""
import sys, os, glob, csv
import cv2
import numpy as np


def order_corners(pts):
    """Order 4 points: top-left, top-right, bottom-right, bottom-left."""
    pts = pts.reshape(4, 2).astype("float32")
    s, d = pts.sum(1), np.diff(pts, axis=1).ravel()
    return np.array([pts[s.argmin()], pts[d.argmin()], pts[s.argmax()], pts[d.argmax()]], dtype="float32")


# STEP 1 - Detect the board (edges + contours)
def detect_board(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=2)  # close gaps in the frame
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_area = img.shape[0] * img.shape[1]
    for c in sorted(contours, key=cv2.contourArea, reverse=True)[:10]:
        if cv2.contourArea(c) < 0.15 * img_area:   # board should be a big part of the image
            break
        approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
        if len(approx) == 4:                       # found a quadrilateral
            return order_corners(approx), edges
    return None, edges


# STEP 2 - Flatten the board (perspective warp / homography)
def flatten(img, corners):
    tl, tr, br, bl = corners
    w = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    h = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype="float32")
    H = cv2.getPerspectiveTransform(corners, dst)
    return cv2.warpPerspective(img, H, (w, h))


# STEP 3 - Reduce glare / uneven lighting (background division)
def normalize_lighting(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    background = cv2.medianBlur(cv2.dilate(gray, np.ones((7, 7), np.uint8)), 51)  # estimate the board's lighting
    return cv2.divide(gray, background, scale=255)                              # flatten bright/dark spots


# STEP 4 - Find the writing (adaptive threshold + morphology)
def binarize(norm):
    """Ink pixels = 255, board = 0. evaluate.py scores this mask against ground truth."""
    return cv2.adaptiveThreshold(norm, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, 25, 15)


def find_writing(norm, flat):
    binary = binarize(norm)
    merged = cv2.dilate(binary, cv2.getStructuringElement(cv2.MORPH_RECT, (25, 7)))  # merge letters into lines
    contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = flat.copy()
    boxes = 0
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w * h > 300:                       # skip tiny specks
            cv2.rectangle(out, (x, y), (x + w, y + h), (0, 0, 255), 2)
            boxes += 1
    return out, boxes


def run(path, out_dir):
    """Run the full pipeline on one photo. Returns a summary row (dict)."""
    name = os.path.splitext(os.path.basename(path))[0]
    row = {"photo": name, "board_found": "", "writing_regions": "", "note": ""}
    img = cv2.imread(path)
    if img is None:
        print(f"[skip] can't read {path}")
        row["note"] = "unreadable file"
        return row
    scale = 1200 / max(img.shape[:2])         # resize for speed and consistent parameters
    img = cv2.resize(img, None, fx=scale, fy=scale) if scale < 1 else img

    corners, edges = detect_board(img)
    cv2.imwrite(f"{out_dir}/{name}_1_edges.jpg", edges)
    if corners is None:
        print(f"[FAIL] {name}: board not found")
        row["board_found"] = "no"
        return row

    vis = img.copy()
    cv2.polylines(vis, [corners.astype(int)], True, (0, 255, 0), 4)
    cv2.imwrite(f"{out_dir}/{name}_2_board.jpg", vis)

    flat = flatten(img, corners)
    cv2.imwrite(f"{out_dir}/{name}_3_flat.jpg", flat)

    boxed, n = find_writing(normalize_lighting(flat), flat)
    cv2.imwrite(f"{out_dir}/{name}_4_writing.jpg", boxed)
    print(f"[OK]   {name}: board found, {n} writing regions")
    row.update(board_found="yes", writing_regions=n)
    return row


if __name__ == "__main__":
    in_dir = sys.argv[1] if len(sys.argv) > 1 else "data/phone_photos"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "results"
    os.makedirs(out_dir, exist_ok=True)
    photos = sorted({p for ext in ("jpg", "jpeg", "png", "JPG", "JPEG", "PNG")
                     for p in glob.glob(f"{in_dir}/*.{ext}")})
    if not photos:
        print(f"No photos found in {in_dir}/ - download them from the shared folder first.")
        sys.exit(1)

    rows = [run(p, out_dir) for p in photos]
    with open(f"{out_dir}/summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["photo", "board_found", "writing_regions", "note"])
        writer.writeheader()
        writer.writerows(rows)

    found = sum(r["board_found"] == "yes" for r in rows)
    print(f"\nDone: board found in {found}/{len(photos)} photos -> {out_dir}/ (see summary.csv)")
