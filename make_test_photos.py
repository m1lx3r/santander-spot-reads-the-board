"""
Generate a synthetic whiteboard test set WITH ground truth, so we can score every
pipeline step with real numbers (not just "looks ok").

Usage:
    python make_test_photos.py data/synthetic

Writes:
    data/synthetic/synth_##_tags.jpg          the "phone photos"
    data/synthetic/gt/synth_##_tags.json      true board corners + homography board->image
    data/synthetic/gt/synth_##_tags_ink.png   true writing mask (board coordinates)

Rendering model (same as the image-formation lectures):
    pixel = reflectance * illumination + specular glare,  then blur, noise, clip, JPEG
The board is a real 3D rectangle (2.4 m x 1.2 m) projected through a pinhole camera.
"""
import os, sys, json
import cv2
import numpy as np

IMG_W, IMG_H = 2000, 1500          # 4:3 phone photo
F_PX = 1450                         # focal length in pixels (~70 deg horizontal FOV)
BW, BH = 1600, 800                  # writing surface in canvas pixels (2:1 board)
FRAME, TRAY = 28, 30                # aluminum frame + marker tray (canvas pixels)
CW, CH = BW + 2 * FRAME, BH + 2 * FRAME + TRAY
BOARD_M = 2.4                       # board width in meters
PPM = CW / BOARD_M                  # canvas pixels per meter

INK = {"black": (0.12, 0.12, 0.13), "blue": (0.62, 0.26, 0.10),
       "red": (0.16, 0.14, 0.68), "green": (0.22, 0.48, 0.16)}   # BGR reflectance

TEXT = ["Homography: x' = H x  (8 DOF)", "Canny -> contours -> 4 corners",
        "Pinhole: x = f X / Z", "Convolution = flip + slide + sum",
        "F(u,v) = sum f(x,y) e^-j2pi(ux+vy)", "Bayer filter: R G G B",
        "LSI system -> impulse response h", "Wiener filter for deconvolution",
        "TODO: fix glare before OCR", "Midterm review Wednesday",
        "Sampling vs quantization", "CCD vs CMOS readout"]
FONTS = [cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, cv2.FONT_HERSHEY_SIMPLEX,
         cv2.FONT_HERSHEY_DUPLEX, cv2.FONT_HERSHEY_SCRIPT_COMPLEX]

# name, conditions -> one photo each (mimics 4 people x 4 photos)
SHOTS = [
    ("01_straight_clean",      dict()),
    ("02_straight_messy",      dict(messy=True)),
    ("03_angle30_clean",       dict(yaw=30)),
    ("04_angle45_glare",       dict(yaw=45, glare=1.0)),
    ("05_angle45_messy",       dict(yaw=-45, messy=True)),
    ("06_angle60_clean",       dict(yaw=60, dist=2.6)),
    ("07_far_clean",           dict(dist=4.6)),
    ("08_far_glare",           dict(dist=4.2, glare=0.8, yaw=-15)),
    ("09_straight_glare",      dict(glare=1.3)),
    ("10_angle30_dim",         dict(yaw=-30, dim=True)),
    ("11_close_partial",       dict(dist=1.9, shift=0.55)),
    ("12_straight_colored",    dict(colors=["blue", "red", "green", "black"])),
    ("13_angle45_dim_messy",   dict(yaw=40, dim=True, messy=True)),
    ("14_angle30_glare_messy", dict(yaw=-30, glare=1.1, messy=True)),
    ("15_straight_clutter",    dict(clutter=True, dist=3.0)),
    ("16_far_angle45",         dict(dist=4.0, yaw=45)),
]


def elastic(mask, amp, sigma, rng):
    """Wobble strokes with a smooth random displacement field (hand-drawn look)."""
    h, w = mask.shape
    dx = cv2.GaussianBlur(rng.standard_normal((h, w)).astype(np.float32), (0, 0), sigma)
    dy = cv2.GaussianBlur(rng.standard_normal((h, w)).astype(np.float32), (0, 0), sigma)
    dx *= amp / (np.abs(dx).max() + 1e-6)
    dy *= amp / (np.abs(dy).max() + 1e-6)
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return cv2.remap(mask, gx + dx, gy + dy, cv2.INTER_LINEAR)


def draw_board(rng, messy, colors):
    """Board canvas reflectance (CH x CW x 3, float) + true ink mask."""
    surf = 0.93 + 0.02 * cv2.GaussianBlur(rng.standard_normal((CH, CW)).astype(np.float32), (0, 0), 80)
    refl = np.repeat(surf[..., None], 3, axis=2)
    alpha = np.zeros((CH, CW), np.float32)
    ink_col = np.zeros((CH, CW, 3), np.float32)

    # faint ghosts of erased marker (real boards have these; NOT counted as writing)
    ghost = np.zeros((CH, CW), np.uint8)
    for _ in range(rng.integers(3, 7)):
        x, y = int(rng.integers(FRAME, CW - 400)), int(rng.integers(FRAME + 60, FRAME + BH))
        cv2.putText(ghost, TEXT[rng.integers(len(TEXT))][:14], (x, y), FONTS[1], 1.6, 255, 5, cv2.LINE_AA)
    ghost = cv2.GaussianBlur(ghost, (0, 0), 6).astype(np.float32) / 255
    refl *= (1 - 0.10 * ghost)[..., None]

    def stamp(m, color):
        a = m.astype(np.float32) / 255
        sel = a > alpha
        ink_col[sel] = INK[color]
        np.maximum(alpha, a, out=alpha)

    # lines of writing
    lines = rng.choice(len(TEXT), size=6, replace=False)
    y = FRAME + 90
    for i, li in enumerate(lines):
        m = np.zeros((CH, CW), np.uint8)
        font = FONTS[rng.integers(len(FONTS))]
        scale, thick = rng.uniform(1.9, 2.6), int(rng.integers(4, 7))
        x = FRAME + int(rng.integers(40, 140))
        while cv2.getTextSize(TEXT[li], font, scale, thick)[0][0] > BW - 560 and scale > 1.4:
            scale -= 0.1                                   # keep the line on the board
        cv2.putText(m, TEXT[li], (x, y), font, scale, 255, thick, cv2.LINE_AA)
        angle = rng.uniform(-4, 4) if messy else rng.uniform(-1, 1)
        R = cv2.getRotationMatrix2D((x, y), angle, 1.0)
        m = cv2.warpAffine(m, R, (CW, CH))
        m = elastic(m, 6 if messy else 1.5, 25, rng)
        stamp(m, colors[i % len(colors)])
        y += int(rng.integers(110, 135))
        if y > FRAME + BH - 40:
            break

    # a small diagram: box -> arrow -> box, plus an underline
    m = np.zeros((CH, CW), np.uint8)
    bx, by = FRAME + BW - 470, FRAME + 80
    cv2.rectangle(m, (bx, by), (bx + 150, by + 90), 255, 4, cv2.LINE_AA)
    cv2.arrowedLine(m, (bx + 160, by + 45), (bx + 290, by + 45), 255, 4, cv2.LINE_AA, tipLength=0.2)
    cv2.circle(m, (bx + 360, by + 45), 55, 255, 4, cv2.LINE_AA)
    cv2.line(m, (FRAME + 60, FRAME + 110), (FRAME + 620, FRAME + 116), 255, 4, cv2.LINE_AA)
    stamp(elastic(m, 5 if messy else 1.5, 30, rng), colors[-1])

    alpha = cv2.GaussianBlur(alpha, (0, 0), 0.7)
    refl = refl * (1 - alpha[..., None]) + ink_col * alpha[..., None]

    # aluminum frame + tray
    fr = np.zeros((CH, CW), np.uint8)
    cv2.rectangle(fr, (0, 0), (CW - 1, CH - 1), 255, -1)
    cv2.rectangle(fr, (FRAME, FRAME), (FRAME + BW - 1, FRAME + BH - 1), 0, -1)
    refl[fr > 0] = (0.70, 0.71, 0.73)
    refl[FRAME + BH + FRAME - 6:, :] = (0.55, 0.56, 0.58)                       # tray
    cv2.rectangle(refl, (FRAME - 3, FRAME - 3), (FRAME + BW + 2, FRAME + BH + 2), (0.45, 0.45, 0.47), 3)
    cv2.rectangle(refl, (0, 0), (CW - 1, CH - 1), (0.50, 0.50, 0.52), 3)
    ink = np.zeros((CH, CW), np.uint8)
    ink[alpha > 0.5] = 255
    ink[fr > 0] = 0
    return refl.astype(np.float32), ink


def draw_wall(rng, board, clutter):
    """Wall canvas (board mounted in the middle). Returns canvas and board offset."""
    mx, mt, mb = int(1.0 * PPM), int(0.6 * PPM), int(0.9 * PPM)
    H, W = CH + mt + mb, CW + 2 * mx
    tint = rng.uniform(0.7, 0.85) * np.array([0.92, 0.98, 1.0])
    tex = 1 + 0.025 * cv2.GaussianBlur(rng.standard_normal((H, W)).astype(np.float32), (0, 0), 3)
    wall = tint[None, None, :] * tex[..., None]
    wall[H - int(0.12 * PPM):, :] = (0.22, 0.24, 0.26)                         # baseboard / floor
    if clutter:   # cork bulletin board + posters + door frame next to the board
        cv2.rectangle(wall, (mx + CW + 60, mt - 40), (mx + CW + 60 + 620, mt + CH + 80), (0.30, 0.45, 0.62), -1)
        cv2.rectangle(wall, (mx + CW + 60, mt - 40), (mx + CW + 60 + 620, mt + CH + 80), (0.15, 0.2, 0.3), 14)
        for i in range(3):
            c = tuple(float(v) for v in rng.uniform(0.2, 0.9, 3))
            x0, y0 = mx + CW + 110 + i * 190, mt + 60 + (i % 2) * 300
            cv2.rectangle(wall, (x0, y0), (x0 + 150, y0 + 220), c, -1)
        cv2.rectangle(wall, (mx - 600, mt - 200), (mx - 60, H - int(0.12 * PPM)), (0.25, 0.32, 0.42), -1)
        cv2.rectangle(wall, (mx - 600, mt - 200), (mx - 60, H - int(0.12 * PPM)), (0.12, 0.15, 0.2), 18)
    wall[mt:mt + CH, mx:mx + CW] = board
    return wall.astype(np.float32), mx, mt


def pose_homography(W, H, ox, oy, yaw, dist, shift, rng):
    """Homography from wall-canvas pixels to image pixels for a camera looking at the wall."""
    yaw, pitch, roll = np.radians(yaw), np.radians(rng.uniform(-6, 6)), np.radians(rng.uniform(-2, 2))
    Ry = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])
    Rx = np.array([[1, 0, 0], [0, np.cos(pitch), -np.sin(pitch)], [0, np.sin(pitch), np.cos(pitch)]])
    Rz = np.array([[np.cos(roll), -np.sin(roll), 0], [np.sin(roll), np.cos(roll), 0], [0, 0, 1]])
    R = Rz @ Rx @ Ry
    t = np.array([shift * BOARD_M + rng.uniform(-0.05, 0.05), rng.uniform(-0.05, 0.08), dist])

    def project(px, py):
        P = np.array([(px - ox - CW / 2) / PPM, (py - oy - CH / 2) / PPM, 0.0])
        X = R @ P + t
        return X, np.array([F_PX * X[0] / X[2] + IMG_W / 2, F_PX * X[1] / X[2] + IMG_H / 2])

    # shrink the wall until it is fully in front of the camera
    x0, y0, x1, y1 = 0, 0, W - 1, H - 1
    while True:
        pts = [project(x, y) for x, y in [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]]
        if min(p[0][2] for p in pts) > 0.4:
            break
        x0, x1 = x0 + 40, x1 - 40
    src = np.float32([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
    dst = np.float32([p[1] for p in pts])
    return cv2.getPerspectiveTransform(src, dst)


def render(name, cfg, seed, out_dir):
    rng = np.random.default_rng(seed)
    board, ink = draw_board(rng, cfg.get("messy", False), cfg.get("colors", ["black", "blue"]))
    wall, ox, oy = draw_wall(rng, board, cfg.get("clutter", False))
    Hw = pose_homography(wall.shape[1], wall.shape[0], ox, oy, cfg.get("yaw", 0) + rng.uniform(-3, 3),
                         cfg.get("dist", 2.3) * rng.uniform(0.97, 1.03), cfg.get("shift", 0), rng)
    Hb = Hw @ np.array([[1, 0, ox], [0, 1, oy], [0, 0, 1]], dtype=np.float64)   # board canvas -> image

    corners = cv2.perspectiveTransform(np.float32([[[0, 0], [CW, 0], [CW, CH], [0, CH]]]), Hb)[0]
    scale = np.linalg.norm(corners[1] - corners[0]) / CW
    if scale < 0.7:                                    # anti-alias before shrinking (far shots)
        wall = cv2.GaussianBlur(wall, (0, 0), 0.45 / scale)

    room = np.full((IMG_H, IMG_W, 3), 0.55, np.float32)                       # side wall / ceiling
    refl = cv2.warpPerspective(wall, Hw, (IMG_W, IMG_H), dst=room, borderMode=cv2.BORDER_TRANSPARENT)
    board_mask = cv2.warpPerspective(np.ones((CH, CW), np.float32), Hb, (IMG_W, IMG_H))

    # illumination: level x side-to-side gradient x vignetting
    gx, gy = np.meshgrid(np.linspace(-1, 1, IMG_W), np.linspace(-1, 1, IMG_H))
    level = 0.5 if cfg.get("dim") else rng.uniform(0.92, 1.02)
    grad = 1 + rng.uniform(-0.15, 0.15) * gx + rng.uniform(-0.08, 0.08) * gy
    L = (level * grad * (1 - 0.22 * (gx ** 2 + gy ** 2))).astype(np.float32)
    exposure = 225 * (0.85 if cfg.get("glare") else 1.0)   # auto-exposure drops when glare is in frame
    img = refl * L[..., None] * exposure

    # specular glare (additive, glossy board only): window/ceiling light reflection
    g = cfg.get("glare", 0)
    if g:
        G = np.zeros((IMG_H, IMG_W), np.float32)
        for k in range(1 + int(rng.random() < 0.4)):
            u, v = rng.uniform(0.3, 0.7), rng.uniform(0.25, 0.6)
            c = corners[0] * (1 - u) * (1 - v) + corners[1] * u * (1 - v) + corners[2] * u * v + corners[3] * (1 - u) * v
            sx, sy = rng.uniform(150, 260) * scale / 0.9, rng.uniform(80, 150) * scale / 0.9
            G += (g if k == 0 else 0.6 * g) * 260 * np.exp(-(((gx * IMG_W / 2 + IMG_W / 2 - c[0]) / sx) ** 2 +
                                                              ((gy * IMG_H / 2 + IMG_H / 2 - c[1]) / sy) ** 2) / 2)
        img += (G * board_mask)[..., None]

    img *= np.array([rng.uniform(0.9, 1.0), 1.0, rng.uniform(1.0, 1.06)])     # white balance cast
    img = cv2.GaussianBlur(img, (0, 0), 1.2 if cfg.get("dist", 2.3) > 3.5 else 0.7)
    img += rng.normal(0, 6 if cfg.get("dim") else 2.5, img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)

    fname = f"synth_{name}"
    cv2.imwrite(f"{out_dir}/{fname}.jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    cv2.imwrite(f"{out_dir}/gt/{fname}_ink.png", ink)
    with open(f"{out_dir}/gt/{fname}.json", "w") as f:
        json.dump({"corners_tl_tr_br_bl": corners.round(2).tolist(), "H_board_to_image": Hb.tolist(),
                   "canvas_size": [CW, CH], "writing_area_xywh": [FRAME, FRAME, BW, BH],
                   "conditions": name.split("_")[1:], "glare": bool(g)}, f, indent=1)
    print(f"made {fname}.jpg")


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "data/synthetic"
    os.makedirs(f"{out}/gt", exist_ok=True)
    for i, (name, cfg) in enumerate(SHOTS):
        render(name, cfg, seed=100 + i, out_dir=out)
    print(f"\n{len(SHOTS)} photos + ground truth in {out}/")
