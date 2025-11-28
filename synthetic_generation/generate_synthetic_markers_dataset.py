import os
import math
import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance
import qrcode
import cv2

# ======================================================
# Camera + scene config (approx OAK-D Lite RGB)
# ======================================================

IMG_W, IMG_H = 1280, 720
HFOV_DEG = 71.86          # approx OAK-D Lite RGB horizontal FOV
BASELINE_M = 0.075        # ~7.5 cm stereo baseline
QR_SIZE_M = 0.1          # 10 cm marker size in meters

# Derived intrinsics (pinhole model)
hfov_rad = math.radians(HFOV_DEG)
FX = (IMG_W / 2) / math.tan(hfov_rad / 2.0)
FY = FX
CX, CY = IMG_W / 2.0, IMG_H / 2.0

# Output root: synthetic_generation/datasets/synthetic_markers relative to this file
THIS_DIR = Path(__file__).resolve().parent
OUT_ROOT = (THIS_DIR / "datasets" / "synthetic_markers").resolve()
OUT_ROOT.mkdir(parents=True, exist_ok=True)


# ======================================================
# Marker generation
# ======================================================

def make_runetag(size_px: int, marker_id: int) -> Image.Image:
    """
    Simple circular marker representative:
    - outer black ring
    - inner white circle
    - four small black dots at NSEW whose presence can depend on marker_id
    """
    img = Image.new("RGB", (size_px, size_px), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    cx = cy = size_px / 2
    r_outer = size_px * 0.45
    r_inner = size_px * 0.30
    r_dot = size_px * 0.06

    # outer ring
    draw.ellipse(
        [cx - r_outer, cy - r_outer, cx + r_outer, cy + r_outer],
        outline=(0, 0, 0),
        width=max(1, int(size_px * 0.04)),
    )
    # inner white (just ensure)
    draw.ellipse(
        [cx - r_inner, cy - r_inner, cx + r_inner, cy + r_inner],
        fill=(255, 255, 255),
        outline=None,
    )

    # small dots – pattern from marker_id bits
    # bit0: north, bit1: east, bit2: south, bit3: west
    positions = {
        "N": (cx, cy - r_inner * 0.9),
        "E": (cx + r_inner * 0.9, cy),
        "S": (cx, cy + r_inner * 0.9),
        "W": (cx - r_inner * 0.9, cy),
    }
    bits = [(marker_id >> b) & 1 for b in range(4)]
    if bits[0]:
        x, y = positions["N"]
        draw.ellipse([x - r_dot, y - r_dot, x + r_dot, y + r_dot], fill=(0, 0, 0))
    if bits[1]:
        x, y = positions["E"]
        draw.ellipse([x - r_dot, y - r_dot, x + r_dot, y + r_dot], fill=(0, 0, 0))
    if bits[2]:
        x, y = positions["S"]
        draw.ellipse([x - r_dot, y - r_dot, x + r_dot, y + r_dot], fill=(0, 0, 0))
    if bits[3]:
        x, y = positions["W"]
        draw.ellipse([x - r_dot, y - r_dot, x + r_dot, y + r_dot], fill=(0, 0, 0))

    return img


def make_chromatag(size_px: int, marker_id: int) -> Image.Image:
    """
    Simple color-based representative:
    - outer black square border
    - inner 2x2 color blocks (RGBY)
    - colors permuted based on marker_id
    """
    img = Image.new("RGB", (size_px, size_px), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    border_w = max(2, size_px // 20)
    draw.rectangle(
        [border_w // 2, border_w // 2, size_px - border_w // 2, size_px - border_w // 2],
        outline=(0, 0, 0),
        width=border_w,
    )

    inner_margin = size_px * 0.15
    x0 = inner_margin
    y0 = inner_margin
    x1 = size_px - inner_margin
    y1 = size_px - inner_margin

    cx = (x0 + x1) / 2
    cy = (y0 + y1) / 2

    # base colors
    colors = [
        (255, 0, 0),    # red
        (0, 255, 0),    # green
        (0, 0, 255),    # blue
        (255, 255, 0),  # yellow
    ]

    # permute colors based on marker_id (just to get variety)
    k = marker_id % len(colors)
    colors = colors[k:] + colors[:k]

    # quadrants: TL, TR, BR, BL
    quads = [
        (x0, y0, cx, cy),
        (cx, y0, x1, cy),
        (cx, cy, x1, y1),
        (x0, cy, cx, y1),
    ]
    for quad, col in zip(quads, colors):
        draw.rectangle(quad, fill=col)

    return img


def make_coppertag(size_px: int, marker_id: int) -> Image.Image:
    """
    Simple 'industrial' style representative:
    - 7x7 binary grid with black outer border always 1
    - inner cells derived from marker_id bits
    """
    grid_size = 7
    img = Image.new("RGB", (size_px, size_px), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    cell = size_px / grid_size

    # border all 1 (black)
    grid = np.zeros((grid_size, grid_size), dtype=np.uint8)
    grid[0, :] = 1
    grid[-1, :] = 1
    grid[:, 0] = 1
    grid[:, -1] = 1

    # inner pattern from marker_id
    rng = np.random.default_rng(marker_id)
    grid[1:-1, 1:-1] = rng.integers(0, 2, size=(grid_size - 2, grid_size - 2), dtype=np.uint8)

    for r in range(grid_size):
        for c in range(grid_size):
            x0 = int(c * cell)
            y0 = int(r * cell)
            x1 = int((c + 1) * cell)
            y1 = int((r + 1) * cell)
            col = (0, 0, 0) if grid[r, c] == 1 else (255, 255, 255)
            draw.rectangle([x0, y0, x1, y1], fill=col)

    return img


def make_marker(marker_type: str,
                marker_id: int,
                size_px: int,
                qr_text: str = "hello") -> Image.Image:
    """
    marker_type: one of
      - "qr"
      - "aruco_4x4_50"
      - "aruco_6x6_250"
      - "apriltag_36h11"
      - "runetag"
      - "chromatag"
      - "coppertag"
    """
    marker_type = marker_type.lower()

    if marker_type == "qr":
        qr = qrcode.QRCode(
            version=None,
            error_correction=qrcode.constants.ERROR_CORRECT_M,
            box_size=10,
            border=4,
        )
        qr.add_data(qr_text)
        qr.make(fit=True)
        img = qr.make_image(fill_color="black", back_color="white").convert("RGB")
        return img.resize((size_px, size_px), Image.NEAREST)

    if marker_type == "runetag":
        return make_runetag(size_px, marker_id)

    if marker_type == "chromatag":
        return make_chromatag(size_px, marker_id)

    if marker_type == "coppertag":
        return make_coppertag(size_px, marker_id)

    # --- ArUco / AprilTag dictionaries ---
    if marker_type == "aruco_4x4_50":
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    elif marker_type == "aruco_6x6_250":
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    elif marker_type == "apriltag_36h11":
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    else:
        raise ValueError(f"Unsupported marker_type: {marker_type}")

    marker = cv2.aruco.generateImageMarker(aruco_dict, marker_id, size_px)
    marker_rgb = cv2.cvtColor(marker, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(marker_rgb)


# ======================================================
# Occlusion & lighting helpers
# ======================================================

def apply_occlusion(img: Image.Image, bbox, occlusion_pct: float) -> Image.Image:
    """
    bbox: (x0, y0, x1, y1) bounding box of marker
    occlusion_pct: fraction (0..0.5) of marker width to occlude with a vertical stripe
    """
    if occlusion_pct <= 0.0 or bbox is None:
        return img

    x0, y0, x1, y1 = bbox
    w = x1 - x0
    stripe_w = int(max(1, w * occlusion_pct))

    stripe_x0 = x0 + (w - stripe_w) // 2
    stripe_x1 = stripe_x0 + stripe_w

    draw = ImageDraw.Draw(img)
    draw.rectangle([stripe_x0, y0, stripe_x1, y1], fill=(200, 200, 200))
    return img


def apply_lighting(img: Image.Image, lighting: str) -> Image.Image:
    """
    lighting: "bright", "normal", "dim", "shadow"
    """
    lighting = lighting.lower()

    if lighting == "bright":
        img = ImageEnhance.Brightness(img).enhance(1.5)
    elif lighting == "dim":
        img = ImageEnhance.Brightness(img).enhance(0.5)
    elif lighting == "shadow":
        # Darken left side -> right side gradient
        arr = np.array(img).astype(np.float32) / 255.0
        x_ramp = np.linspace(0.3, 1.0, arr.shape[1])[None, :, None]
        arr *= x_ramp
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)
    # "normal" -> no change
    return img


# ======================================================
# Core renderer: one scene & all modalities
# ======================================================

def _rotation_matrices(x_deg: float, y_deg: float) -> np.ndarray:
    """Build R = Ry * Rx, with x=pitch, y=yaw (degrees)."""
    tx = math.radians(x_deg)
    ty = math.radians(y_deg)

    Rx = np.array([
        [1, 0, 0],
        [0, math.cos(tx), -math.sin(tx)],
        [0, math.sin(tx),  math.cos(tx)],
    ], dtype=np.float32)

    Ry = np.array([
        [ math.cos(ty), 0, math.sin(ty)],
        [0,            1, 0           ],
        [-math.sin(ty), 0, math.cos(ty)],
    ], dtype=np.float32)

    return Ry @ Rx


def render_scene(distance_m: float,
                 x_deg: float,
                 y_deg: float,
                 occlusion_pct: float,
                 lighting: str,
                 marker_type: str,
                 marker_id: int,
                 marker_text: str,
                 out_dir: Path,
                 sample_id: str):
    """
    Generate one sample: RGB, depth_mm, disparity, segmentation
    and save to out_dir with sample_id in filenames.

    Rotations are around X and Y only (no Z/in-plane rotation).
    The entire marker including its outer border is tilted in 3D.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- 1) marker size in pixels (fronto-parallel approx) ---
    marker_px = int(round(FX * QR_SIZE_M / distance_m))
    marker_img = make_marker(marker_type, marker_id, marker_px, qr_text=marker_text)
    marker_arr = np.array(marker_img)  # RGB

    # --- 2) base scene: wall + floor (PIL then to OpenCV) ---
    base_rgb = Image.new("RGB", (IMG_W, IMG_H), (255, 255, 255))
    draw = ImageDraw.Draw(base_rgb)

    horizon_y = int(IMG_H * 0.6)
    wall_color = (235, 235, 235)
    floor_color = (200, 195, 190)

    draw.rectangle([0, 0, IMG_W, horizon_y], fill=wall_color)
    draw.rectangle([0, horizon_y, IMG_W, IMG_H], fill=floor_color)

    # simple floor structure
    for x in range(0, IMG_W, 80):
        draw.line([(x, horizon_y), (x + 40, IMG_H)], fill=(180, 175, 170), width=1)

    scene_bgr = cv2.cvtColor(np.array(base_rgb), cv2.COLOR_RGB2BGR)

    # --- 3) 3D pose & projection of marker corners ---
    s = QR_SIZE_M / 2.0
    # Local marker coordinates (square in Z=0 plane)
    # order: TL, TR, BR, BL
    corners_local = np.array([
        [-s, -s, 0.0],
        [ s, -s, 0.0],
        [ s,  s, 0.0],
        [-s,  s, 0.0],
    ], dtype=np.float32)

    R = _rotation_matrices(x_deg, y_deg)

    # Rotate then translate so the marker center is at (0,0,distance_m)
    corners_cam = (R @ corners_local.T).T
    corners_cam[:, 2] += distance_m

    # Project to image plane
    pts_img = np.zeros((4, 2), dtype=np.float32)
    pts_img[:, 0] = FX * corners_cam[:, 0] / corners_cam[:, 2] + CX
    pts_img[:, 1] = FY * corners_cam[:, 1] / corners_cam[:, 2] + CY

    # Source points in marker image (full square, so border rotates too)
    src_pts = np.array([
        [0.0, 0.0],
        [marker_px - 1.0, 0.0],
        [marker_px - 1.0, marker_px - 1.0],
        [0.0, marker_px - 1.0],
    ], dtype=np.float32)

    H = cv2.getPerspectiveTransform(src_pts, pts_img)

    # Warp marker onto the scene
    marker_bgr = cv2.cvtColor(marker_arr, cv2.COLOR_RGB2BGR)
    warped_marker = cv2.warpPerspective(marker_bgr, H, (IMG_W, IMG_H))

    # Warp a mask to know where the marker is
    mask_src = np.ones((marker_px, marker_px), dtype=np.uint8) * 255
    warped_mask = cv2.warpPerspective(mask_src, H, (IMG_W, IMG_H))
    mask_bool = warped_mask > 0

    scene_bgr[mask_bool] = warped_marker[mask_bool]

    # --- 4) segmentation: 0 bg, 1 wall, 2 floor, 3 marker ---
    seg = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
    seg[0:horizon_y, :] = 1
    seg[horizon_y:IMG_H, :] = 2
    seg[mask_bool] = 3

    # --- 5) occlusion (in image space, vertical stripe over marker bbox) ---
    bbox = None
    if mask_bool.any():
        ys, xs = np.where(mask_bool)
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        bbox = (x0, y0, x1, y1)

    rgb = Image.fromarray(cv2.cvtColor(scene_bgr, cv2.COLOR_BGR2RGB))
    rgb = apply_occlusion(rgb, bbox, occlusion_pct)

    # --- 6) lighting ---
    rgb = apply_lighting(rgb, lighting)

    # --- 7) depth map in mm (simple wall/floor model) ---
    depth_mm = np.zeros((IMG_H, IMG_W), dtype=np.uint16)
    wall_mm = int(round(distance_m * 1000.0))
    floor_mm = int(round(distance_m * 1200.0))  # floor slightly further

    depth_mm[0:horizon_y, :] = wall_mm
    depth_mm[horizon_y:IMG_H, :] = floor_mm

    # --- 8) disparity in 1/16 px units (OAK-style) ---
    depth_m = depth_mm.astype(np.float32) / 1000.0
    disp_px = np.zeros_like(depth_m)
    valid = depth_m > 0
    disp_px[valid] = FX * BASELINE_M / depth_m[valid]
    disp_16 = (disp_px * 16.0).astype(np.uint16)

    # --- 9) save files ---
    rgb_path = out_dir / f"rgb_{sample_id}.png"
    seg_path = out_dir / f"seg_{sample_id}.png"
    depth_path = out_dir / f"depth_mm_{sample_id}.png"
    disp_path = out_dir / f"disp_{sample_id}.png"

    rgb.save(rgb_path)
    Image.fromarray(seg).save(seg_path)
    Image.fromarray(depth_mm, mode="I;16").save(depth_path)
    Image.fromarray(disp_16, mode="I;16").save(disp_path)

    return rgb_path, seg_path, depth_path, disp_path


# ======================================================
# Main: generate test sets
# ======================================================

def main():
    # You can tweak these test grids to match config/test_configurations.py
    distances = [0.3, 0.6, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
    x_rotations = [-60, -40, -20, 0, 20, 40, 60]   # pitch
    y_rotations = [-60, -40, -20, 0, 20, 40, 60]   # yaw
    occlusions = [0.05, 0.10, 0.15, 0.20]
    lightings = ["bright", "normal", "dim", "shadow"]

    # Marker families: (marker_type, marker_id, marker_text)
    marker_families = [
        ("qr",              0,  "qr_10cm"),
        ("aruco_4x4_50",   23,  "aruco_4x4_50_id23"),
        ("aruco_6x6_250",   5,  "aruco_6x6_250_id5"),
        ("apriltag_36h11",  0,  "apriltag_36h11_id0"),  # disable if cv2 lacks it
        ("runetag",         3,  "runetag_id3"),
        ("chromatag",       1,  "chromatag_id1"),
        ("coppertag",       7,  "coppertag_id7"),
    ]

    csv_path = OUT_ROOT / "annotations.csv"
    csv_file = open(csv_path, "w", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=[
        "sample_id",
        "split",
        "marker_type",
        "marker_id",
        "marker_text",
        "distance_m",
        "x_deg",
        "y_deg",
        "occlusion_pct",
        "lighting",
        "qr_size_m",
        "rgb_path",
        "depth_path",
        "disp_path",
        "seg_path",
        "fx", "fy", "cx", "cy",
        "baseline_m",
        "img_width",
        "img_height",
    ])
    writer.writeheader()

    for marker_type, marker_id, marker_text in marker_families:
        # ----- Test set: distance -----
        split = f"{marker_type}_distance"
        out_dir = OUT_ROOT / split
        out_dir.mkdir(parents=True, exist_ok=True)

        for d in distances:
            sample_id = f"{split}_d{int(d * 100):03d}"
            rgb_path, seg_path, depth_path, disp_path = render_scene(
                distance_m=d,
                x_deg=0.0,
                y_deg=0.0,
                occlusion_pct=0.0,
                lighting="normal",
                marker_type=marker_type,
                marker_id=marker_id,
                marker_text=marker_text,
                out_dir=out_dir,
                sample_id=sample_id,
            )
            writer.writerow(dict(
                sample_id=sample_id,
                split=split,
                marker_type=marker_type,
                marker_id=marker_id,
                marker_text=marker_text,
                distance_m=d,
                x_deg=0.0,
                y_deg=0.0,
                occlusion_pct=0.0,
                lighting="normal",
                qr_size_m=QR_SIZE_M,
                rgb_path=os.path.relpath(rgb_path, OUT_ROOT),
                depth_path=os.path.relpath(depth_path, OUT_ROOT),
                disp_path=os.path.relpath(disp_path, OUT_ROOT),
                seg_path=os.path.relpath(seg_path, OUT_ROOT),
                fx=FX, fy=FY, cx=CX, cy=CY,
                baseline_m=BASELINE_M,
                img_width=IMG_W,
                img_height=IMG_H,
            ))

        # ----- Test set: X-rotation (pitch) -----
        split = f"{marker_type}_x_rotation"
        out_dir = OUT_ROOT / split
        out_dir.mkdir(parents=True, exist_ok=True)

        for x in x_rotations:
            sample_id = f"{split}_x{int(x):+03d}".replace("+", "p").replace("-", "m")
            rgb_path, seg_path, depth_path, disp_path = render_scene(
                distance_m=0.6,
                x_deg=x,
                y_deg=0.0,
                occlusion_pct=0.0,
                lighting="normal",
                marker_type=marker_type,
                marker_id=marker_id,
                marker_text=marker_text,
                out_dir=out_dir,
                sample_id=sample_id,
            )
            writer.writerow(dict(
                sample_id=sample_id,
                split=split,
                marker_type=marker_type,
                marker_id=marker_id,
                marker_text=marker_text,
                distance_m=0.6,
                x_deg=x,
                y_deg=0.0,
                occlusion_pct=0.0,
                lighting="normal",
                qr_size_m=QR_SIZE_M,
                rgb_path=os.path.relpath(rgb_path, OUT_ROOT),
                depth_path=os.path.relpath(depth_path, OUT_ROOT),
                disp_path=os.path.relpath(disp_path, OUT_ROOT),
                seg_path=os.path.relpath(seg_path, OUT_ROOT),
                fx=FX, fy=FY, cx=CX, cy=CY,
                baseline_m=BASELINE_M,
                img_width=IMG_W,
                img_height=IMG_H,
            ))

        # ----- Test set: Y-rotation (yaw) -----
        split = f"{marker_type}_y_rotation"
        out_dir = OUT_ROOT / split
        out_dir.mkdir(parents=True, exist_ok=True)

        for y in y_rotations:
            sample_id = f"{split}_y{int(y):+03d}".replace("+", "p").replace("-", "m")
            rgb_path, seg_path, depth_path, disp_path = render_scene(
                distance_m=0.6,
                x_deg=0.0,
                y_deg=y,
                occlusion_pct=0.0,
                lighting="normal",
                marker_type=marker_type,
                marker_id=marker_id,
                marker_text=marker_text,
                out_dir=out_dir,
                sample_id=sample_id,
            )
            writer.writerow(dict(
                sample_id=sample_id,
                split=split,
                marker_type=marker_type,
                marker_id=marker_id,
                marker_text=marker_text,
                distance_m=0.6,
                x_deg=0.0,
                y_deg=y,
                occlusion_pct=0.0,
                lighting="normal",
                qr_size_m=QR_SIZE_M,
                rgb_path=os.path.relpath(rgb_path, OUT_ROOT),
                depth_path=os.path.relpath(depth_path, OUT_ROOT),
                disp_path=os.path.relpath(disp_path, OUT_ROOT),
                seg_path=os.path.relpath(seg_path, OUT_ROOT),
                fx=FX, fy=FY, cx=CX, cy=CY,
                baseline_m=BASELINE_M,
                img_width=IMG_W,
                img_height=IMG_H,
            ))

        # ----- Test set: occlusion -----
        split = f"{marker_type}_occlusion"
        out_dir = OUT_ROOT / split
        out_dir.mkdir(parents=True, exist_ok=True)

        for occ in occlusions:
            sample_id = f"{split}_occ{int(occ * 100):02d}"
            rgb_path, seg_path, depth_path, disp_path = render_scene(
                distance_m=0.6,
                x_deg=0.0,
                y_deg=0.0,
                occlusion_pct=occ,
                lighting="normal",
                marker_type=marker_type,
                marker_id=marker_id,
                marker_text=marker_text,
                out_dir=out_dir,
                sample_id=sample_id,
            )
            writer.writerow(dict(
                sample_id=sample_id,
                split=split,
                marker_type=marker_type,
                marker_id=marker_id,
                marker_text=marker_text,
                distance_m=0.6,
                x_deg=0.0,
                y_deg=0.0,
                occlusion_pct=occ,
                lighting="normal",
                qr_size_m=QR_SIZE_M,
                rgb_path=os.path.relpath(rgb_path, OUT_ROOT),
                depth_path=os.path.relpath(depth_path, OUT_ROOT),
                disp_path=os.path.relpath(disp_path, OUT_ROOT),
                seg_path=os.path.relpath(seg_path, OUT_ROOT),
                fx=FX, fy=FY, cx=CX, cy=CY,
                baseline_m=BASELINE_M,
                img_width=IMG_W,
                img_height=IMG_H,
            ))

        # ----- Test set: lighting -----
        split = f"{marker_type}_lighting"
        out_dir = OUT_ROOT / split
        out_dir.mkdir(parents=True, exist_ok=True)

        for lighting in lightings:
            sample_id = f"{split}_{lighting}"
            rgb_path, seg_path, depth_path, disp_path = render_scene(
                distance_m=0.6,
                x_deg=0.0,
                y_deg=0.0,
                occlusion_pct=0.0,
                lighting=lighting,
                marker_type=marker_type,
                marker_id=marker_id,
                marker_text=marker_text,
                out_dir=out_dir,
                sample_id=sample_id,
            )
            writer.writerow(dict(
                sample_id=sample_id,
                split=split,
                marker_type=marker_type,
                marker_id=marker_id,
                marker_text=marker_text,
                distance_m=0.6,
                x_deg=0.0,
                y_deg=0.0,
                occlusion_pct=0.0,
                lighting=lighting,
                qr_size_m=QR_SIZE_M,
                rgb_path=os.path.relpath(rgb_path, OUT_ROOT),
                depth_path=os.path.relpath(depth_path, OUT_ROOT),
                disp_path=os.path.relpath(disp_path, OUT_ROOT),
                seg_path=os.path.relpath(seg_path, OUT_ROOT),
                fx=FX, fy=FY, cx=CX, cy=CY,
                baseline_m=BASELINE_M,
                img_width=IMG_W,
                img_height=IMG_H,
            ))

    csv_file.close()
    print("Done. Dataset written to:", OUT_ROOT)
    print("Annotations:", csv_path)


if __name__ == "__main__":
    main()
