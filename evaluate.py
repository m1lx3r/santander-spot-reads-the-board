"""
Score a pipeline against the synthetic ground truth. Every step is scored on its own,
so you can prove YOUR step got better without the other steps getting in the way.

Usage:
    python evaluate.py whiteboard_starter.py                      # baseline
    python evaluate.py prototypes/mazen/mazen_pipeline.py         # your version
    python evaluate.py <pipeline.py> data/synthetic               # (default photo folder)

Metrics
    Step 1  detect_board        found? + mean corner error in px (1200-px image).
                                "correct" = error under 30 px (2.5% of image width)
    Step 3+4 normalize_lighting  ink F1: compares binarize(normalize_lighting(flat)) to the
             + binarize          true writing mask. Flattened with the TRUE corners, so a
                                 bad detection in step 1 can't hide a good glare fix.
                                 Precision = drawn pixels that are real ink
                                 Recall    = real ink pixels that were found
Writes eval_<pipeline>.csv next to the photos' results.
"""
import sys, os, glob, json, csv, importlib.util
import cv2
import numpy as np

CORRECT_PX = 30
TOL = 2   # pixel tolerance when matching ink


def load_pipeline(path):
    spec = importlib.util.spec_from_file_location("pipeline", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def prepare(img):
    """Same resize the pipeline uses in run()."""
    scale = 1200 / max(img.shape[:2])
    if scale < 1:
        return cv2.resize(img, None, fx=scale, fy=scale), scale
    return img, 1.0


def ink_scores(mod, img, gt, gt_dir, stem, s):
    """Flatten with TRUE corners, run lighting + binarize, compare to the true ink mask."""
    corners = (np.float32(gt["corners_tl_tr_br_bl"]) * s).astype("float32")
    flat = mod.flatten(img, corners)
    h, w = flat.shape[:2]
    pred = mod.binarize(mod.normalize_lighting(flat)) > 0

    dst = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    H_flat = cv2.getPerspectiveTransform(corners, dst)                       # image -> flat
    H = H_flat @ np.diag([s, s, 1.0]) @ np.array(gt["H_board_to_image"])     # board canvas -> flat
    cw, ch = gt["canvas_size"]
    ink = cv2.imread(f"{gt_dir}/{stem}_ink.png", 0)
    gt_ink = cv2.warpPerspective(ink, H, (w, h), flags=cv2.INTER_NEAREST) > 0

    x, y, bw, bh = gt["writing_area_xywh"]
    area = np.zeros((ch, cw), np.uint8)
    area[y + 6:y + bh - 6, x + 6:x + bw - 6] = 1                              # writing surface only
    region = cv2.warpPerspective(area, H, (w, h), flags=cv2.INTER_NEAREST) > 0
    in_img = cv2.warpPerspective(np.ones(img.shape[:2], np.uint8), H_flat, (w, h)) > 0
    region &= cv2.erode(in_img.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0

    pred &= region
    gt_ink &= region
    k = np.ones((2 * TOL + 1, 2 * TOL + 1), np.uint8)
    gt_d = cv2.dilate(gt_ink.astype(np.uint8), k) > 0
    pred_d = cv2.dilate(pred.astype(np.uint8), k) > 0
    precision = (pred & gt_d).sum() / max(pred.sum(), 1)
    recall = (gt_ink & pred_d).sum() / max(gt_ink.sum(), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return precision, recall, f1


def main():
    pipe_path = sys.argv[1] if len(sys.argv) > 1 else "whiteboard_starter.py"
    data = sys.argv[2] if len(sys.argv) > 2 else "data/synthetic"
    gt_dir = f"{data}/gt"
    mod = load_pipeline(pipe_path)
    rows = []
    for jf in sorted(glob.glob(f"{gt_dir}/*.json")):
        stem = os.path.splitext(os.path.basename(jf))[0]
        gt = json.load(open(jf))
        img, s = prepare(cv2.imread(f"{data}/{stem}.jpg"))

        corners, _ = mod.detect_board(img)
        if corners is None:
            found, err = "no", None
        else:
            # accept either the outer frame or the inner writing-surface corners
            x, y, bw, bh = gt["writing_area_xywh"]
            inner = cv2.perspectiveTransform(np.float32([[[x, y], [x + bw, y], [x + bw, y + bh], [x, y + bh]]]),
                                             np.array(gt["H_board_to_image"]))[0]
            pc = np.float32(corners).reshape(4, 2)
            err = min(float(np.linalg.norm(pc - np.float32(c) * s, axis=1).mean())
                      for c in (gt["corners_tl_tr_br_bl"], inner))
            found = "yes" if err < CORRECT_PX else "WRONG"
        p, r, f1 = ink_scores(mod, img, gt, gt_dir, stem, s)
        rows.append(dict(photo=stem.replace("synth_", ""), board=found,
                         corner_err_px="" if err is None else round(err, 1),
                         ink_precision=round(p, 3), ink_recall=round(r, 3), ink_f1=round(f1, 3),
                         glare="yes" if gt["glare"] else ""))

    name = os.path.splitext(os.path.basename(pipe_path))[0]
    out = f"eval_{name}.csv"
    with open(out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)

    print(f"\n{name}")
    print(f"{'photo':<26}{'board':<7}{'err px':>7}{'prec':>7}{'recall':>8}{'ink F1':>8}")
    for r in rows:
        print(f"{r['photo']:<26}{r['board']:<7}{str(r['corner_err_px']):>7}"
              f"{r['ink_precision']:>7.2f}{r['ink_recall']:>8.2f}{r['ink_f1']:>8.2f}")
    ok = sum(r["board"] == "yes" for r in rows)
    wrong = sum(r["board"] == "WRONG" for r in rows)
    f1_all = np.mean([r["ink_f1"] for r in rows])
    f1_glare = np.mean([r["ink_f1"] for r in rows if r["glare"]] or [0])
    print(f"\nStep 1 board correct: {ok}/{len(rows)}   (wrong quad: {wrong}, not found: {len(rows) - ok - wrong})")
    print(f"Ink F1 all photos: {f1_all:.3f}   glare photos only: {f1_glare:.3f}")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
