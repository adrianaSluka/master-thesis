"""
Merge two partial BOP scene_gt JSONs (labeled every ~5 frames),
interpolate missing frames (slerp for R, lerp for t),
compute scene_gt_info from the reprojected model bbox,
and create a reprojection debug-visualization folder.

Usage:
    python interpolate_and_visualize.py \
        --gt1 path/to/scene_gt_part1.json \
        --gt2 path/to/scene_gt_part2.json \
        --camera path/to/scene_camera.json \
        --rgb-dir path/to/rgb \
        --output-dir path/to/output \
        --num-frames 2000 \
        --bbox-corners path/to/bbox_corners.json
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


# =============================================================================
# Rotation helpers (slerp via quaternions)
# =============================================================================

def rot_to_quat(R):
    """3x3 rotation matrix -> unit quaternion [w, x, y, z]."""
    R = np.asarray(R, dtype=np.float64)
    tr = np.trace(R)
    if tr > 0:
        s = 0.5 / math.sqrt(tr + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def quat_to_rot(q):
    """Unit quaternion [w, x, y, z] -> 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ])


def slerp_quat(q0, q1, t):
    """Spherical linear interpolation between two quaternions."""
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    dot = np.dot(q0, q1)
    if dot < 0:
        q1 = -q1
        dot = -dot
    dot = np.clip(dot, -1.0, 1.0)
    if dot > 0.9995:
        result = q0 + t * (q1 - q0)
        return result / np.linalg.norm(result)
    theta_0 = math.acos(dot)
    theta = theta_0 * t
    sin_theta = math.sin(theta)
    sin_theta_0 = math.sin(theta_0)
    s0 = math.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    result = s0 * q0 + s1 * q1
    return result / np.linalg.norm(result)


def slerp_rotation(R0, R1, t):
    """Slerp between two 3x3 rotation matrices."""
    return quat_to_rot(slerp_quat(rot_to_quat(R0), rot_to_quat(R1), t))


# =============================================================================
# BOP helpers (matching your pipeline's conventions)
# =============================================================================

def project_point(K, R, t, X):
    """Project 3-D point -> pixel.  Same sign convention as your generation code."""
    X_c = R @ X + t
    fx, fy = K[0], K[4]
    cx, cy = K[2], K[5]
    u = fx * X_c[0] / X_c[2] + cx
    v = fy * X_c[1] / X_c[2] + cy
    return float(u), float(v)


def flatten_R(R):
    return [float(x) for x in np.asarray(R).reshape(-1)]

def flatten_t(t):
    return [float(x) for x in np.asarray(t).reshape(-1)]


def dump_flat_json_per_row(data: dict, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        f.write("{\n")
        items = list(data.items())
        for idx, (k, v) in enumerate(items):
            line = json.dumps(k, ensure_ascii=False) + ": " + json.dumps(v, ensure_ascii=False)
            if idx < len(items) - 1:
                line += ","
            f.write("  " + line + "\n")
        f.write("}\n")


# =============================================================================
# Merge + Interpolate scene_gt
# =============================================================================

def load_json(path):
    with open(path) as f:
        return json.load(f)


def merge_gt_dicts(d1, d2, d3):
    merged = {}
    merged.update(d1)
    merged.update(d2)
    merged.update(d3)
    return merged


def get_keyframes_sorted(gt_dict):
    return sorted(int(k) for k in gt_dict.keys())


# def interpolate_gt(gt_merged):
#     """
#     Slerp rotation + lerp translation between labeled keyframes.
#     Only produces frames in [min_keyframe .. max_keyframe].
#     Returns (full_gt, frame_min, frame_max).
#     """
#     keyframes = get_keyframes_sorted(gt_merged)
#     if len(keyframes) < 2:
#         raise ValueError("Need at least 2 keyframes to interpolate")

#     frame_min = keyframes[0]
#     frame_max = keyframes[-1]

#     kf_data = {}
#     for k in keyframes:
#         entry = gt_merged[str(k)][0]
#         R = np.array(entry["cam_R_m2c"]).reshape(3, 3)
#         t = np.array(entry["cam_t_m2c"])
#         obj_id = entry.get("obj_id", 1)
#         T = np.array(entry["t_m2c"]).reshape(4, 4) if "t_m2c" in entry else None
#         kf_data[k] = {"R": R, "t": t, "obj_id": obj_id, "T": T}

#     full_gt = {}

#     for fi in range(frame_min, frame_max + 1):
#         if fi in kf_data:
#             d = kf_data[fi]
#             T_m2c = d["T"] if d["T"] is not None else np.eye(4)
#             full_gt[str(fi)] = [{
#                 "t_m2c": flatten_R(T_m2c),
#                 "cam_R_m2c": flatten_R(d["R"]),
#                 "cam_t_m2c": flatten_t(d["t"]),
#                 "obj_id": d["obj_id"],
#             }]
#             continue

#         # Find bracketing keyframes
#         prev_kf = None
#         next_kf = None
#         for k in keyframes:
#             if k <= fi:
#                 prev_kf = k
#             if k > fi and next_kf is None:
#                 next_kf = k
#         if prev_kf is None:
#             prev_kf = next_kf
#         if next_kf is None:
#             next_kf = prev_kf

#         if prev_kf == next_kf:
#             d = kf_data[prev_kf]
#             T_m2c = d["T"] if d["T"] is not None else np.eye(4)
#             full_gt[str(fi)] = [{
#                 "t_m2c": flatten_R(T_m2c),
#                 "cam_R_m2c": flatten_R(d["R"]),
#                 "cam_t_m2c": flatten_t(d["t"]),
#                 "obj_id": d["obj_id"],
#             }]
#             continue

#         alpha = (fi - prev_kf) / (next_kf - prev_kf)
#         R_interp = slerp_rotation(kf_data[prev_kf]["R"], kf_data[next_kf]["R"], alpha)
#         t_interp = (1 - alpha) * kf_data[prev_kf]["t"] + alpha * kf_data[next_kf]["t"]

#         T_interp = np.eye(4)
#         T_interp[:3, :3] = R_interp
#         T_interp[:3, 3] = t_interp

#         full_gt[str(fi)] = [{
#             "t_m2c": flatten_R(T_interp),
#             "cam_R_m2c": flatten_R(R_interp),
#             "cam_t_m2c": flatten_t(t_interp),
#             "obj_id": kf_data[prev_kf]["obj_id"],
#         }]

#     return full_gt, frame_min, frame_max


def broadcast_camera(cam_dict, frame_min, frame_max):
    """Camera is typically static. Nearest-neighbour fill for missing frames."""
    keyframes = get_keyframes_sorted(cam_dict)
    if len(keyframes) == 1:
        entry = cam_dict[str(keyframes[0])]
        return {str(i): entry for i in range(frame_min, frame_max + 1)}

    full_cam = {}
    for fi in range(frame_min, frame_max + 1):
        if str(fi) in cam_dict:
            full_cam[str(fi)] = cam_dict[str(fi)]
        else:
            closest = min(keyframes, key=lambda k: abs(k - fi))
            full_cam[str(fi)] = cam_dict[str(closest)]
    return full_cam


# =============================================================================
# Compute scene_gt_info from reprojected bbox convex hull
# =============================================================================

def _convex_hull(points):
    """Graham scan for a small set of 2-D points."""
    pts = sorted(set((round(x, 6), round(y, 6)) for x, y in points))
    if len(pts) < 3:
        return pts

    def cross(O, A, B):
        return (A[0] - O[0]) * (B[1] - O[1]) - (A[1] - O[1]) * (B[0] - O[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def convex_hull_mask(pts_2d, width, height):
    """Rasterise the convex hull of projected 2-D points into a binary mask."""
    hull = _convex_hull(pts_2d)
    if len(hull) < 3:
        return np.zeros((height, width), dtype=np.uint8)
    img = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(img)
    draw.polygon(hull, fill=255)
    return np.array(img)


def bbox_from_mask(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return [0, 0, 0, 0]
    x_min, y_min = int(xs.min()), int(ys.min())
    x_max, y_max = int(xs.max()), int(ys.max())
    return [x_min, y_min, x_max - x_min + 1, y_max - y_min + 1]


def compute_gt_info_for_frame(cam_k, R_m2c, t_m2c, bbox_corners, width, height):
    """
    Project the 8 bbox corners + origin, rasterise the convex hull,
    and derive BOP-format gt_info fields from the resulting mask.
    """
    K = np.array(cam_k)
    R = np.array(R_m2c).reshape(3, 3)
    t = np.array(t_m2c)

    pts_2d = []
    for corner in bbox_corners:
        u, v = project_point(K, R, t, np.array(corner))
        pts_2d.append((u, v))
    u0, v0 = project_point(K, R, t, np.zeros(3))
    pts_2d.append((u0, v0))

    mask = convex_hull_mask(pts_2d, width, height)
    bbox = bbox_from_mask(mask)
    px_count = int((mask > 0).sum())

    return {
        "bbox_obj": bbox,
        "bbox_visib": bbox,
        "px_count_all": px_count,
        "px_count_valid": px_count,
        "px_count_visib": px_count,
        "visib_fract": 1.0,
    }


# =============================================================================
# Visualisation
# =============================================================================

BOX_EDGES = [
    (0, 1), (0, 2), (0, 4),
    (1, 3), (1, 5),
    (2, 3), (2, 6),
    (3, 7),
    (4, 5), (4, 6),
    (5, 7), (6, 7),
]

EDGE_COLORS = [
    "red", "green", "blue", "cyan", "magenta", "yellow",
    "orange", "white", "red", "green", "blue", "cyan",
]


def draw_reprojection(img, cam_k, R_m2c, t_m2c, bbox_corners, is_keyframe):
    """Draw wireframe bbox + origin + KF/INTERP label onto an image."""
    draw = ImageDraw.Draw(img)
    K = np.array(cam_k)
    R = np.array(R_m2c).reshape(3, 3)
    t = np.array(t_m2c)

    pts_2d = []
    for corner in bbox_corners:
        u, v = project_point(K, R, t, np.array(corner))
        pts_2d.append((u, v))

    # Edges
    for idx, (i, j) in enumerate(BOX_EDGES):
        color = EDGE_COLORS[idx % len(EDGE_COLORS)]
        draw.line([pts_2d[i], pts_2d[j]], fill=color, width=2)

    # Corner dots
    for u, v in pts_2d:
        r = 4
        draw.ellipse((u - r, v - r, u + r, v + r), fill="red", outline="white")

    # Origin
    ou, ov = project_point(K, R, t, np.zeros(3))
    r = 6
    draw.ellipse((ou - r, ov - r, ou + r, ov + r), fill="lime", outline="white", width=2)
    draw.line([(ou - 12, ov), (ou + 12, ov)], fill="lime", width=1)
    draw.line([(ou, ov - 12), (ou, ov + 12)], fill="lime", width=1)

    # Label
    label = "KEYFRAME" if is_keyframe else "INTERP"
    color = "lime" if is_keyframe else "yellow"
    draw.text((10, 10), label, fill=color)

    return img


def find_rgb(rgb_dir, frame_idx):
    print(rgb_dir)
    print(frame_idx)
    for pat in [f"rgb_{frame_idx:06d}.jpg", f"{frame_idx:06d}.jpg", f"frame_{frame_idx:06d}.jpg",
                f"rgb_{frame_idx:06d}.png", f"{frame_idx:06d}.png"]:
        p = rgb_dir / pat
        if p.exists():
            return p
    return None


# =============================================================================
# Main
# =============================================================================

def main():
    # parser = argparse.ArgumentParser(
    #     description="Merge + interpolate scene_gt, compute gt_info from reprojection, visualise."
    # )
    # parser.add_argument("--gt1", required=True, help="First partial scene_gt JSON")
    # parser.add_argument("--gt2", required=True, help="Second partial scene_gt JSON")
    # parser.add_argument("--camera", required=True, help="scene_camera.json")
    # parser.add_argument("--rgb-dir", required=True, help="RGB image directory")
    # parser.add_argument("--output-dir", required=True, help="Output directory")
    # parser.add_argument("--bbox-corners", required=True,
    #                     help="JSON file: [[x,y,z], ...] -- 8 model bbox corners")
    # parser.add_argument("--width", type=int, default=640)
    # parser.add_argument("--height", type=int, default=480)
    # args = parser.parse_args()

    W, H = 960, 540#args.width, args.height
    output_dir = Path("/home/user/Desktop/isaac_project/real_images")#Path(args.output_dir)
    viz_dir = output_dir / "reprojection_viz2"
    test_dir = output_dir / "frames_960_5fps"
    rgb_dir = Path('/home/user/Desktop/isaac_project/real_images/frames_960_5fps_new')#Path(args.rgb_dir)
    for d in [output_dir, viz_dir, test_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # ---- Load & merge ----
    print("Loading scene_gt files...")
    # gt1 = load_json('/home/user/Desktop/isaac_project/real_images/annotations/scene_gt.json')#load_json(args.gt1)
    # gt2 = load_json('/home/user/Desktop/isaac_project/real_images/annotations/scene_gt2.json')#load_json(args.gt2)
    # gt3 = load_json('/home/user/Desktop/isaac_project/real_images/annotations/scene_gt3.json')#load_json(args.gt2)
    #gt_merged = merge_gt_dicts(gt1, gt2, gt3)
    gt_merged = load_json('/home/user/Desktop/isaac_project/real_images/frames_960_5fps_new/scene_gt.json')
    keyframe_set = set(str(k) for k in get_keyframes_sorted(gt_merged))
    print(f"  Merged keyframes: {len(keyframe_set)}")

    print("Loading scene_camera...")
    # cam1 = load_json('/home/user/Desktop/isaac_project/real_images/annotations/scene_camera.json')
    # cam2 = load_json("/home/user/Desktop/isaac_project/real_images/annotations/scene_camera2.json")
    # cam3 = load_json("/home/user/Desktop/isaac_project/real_images/annotations/scene_camera3.json")
    # cam = merge_gt_dicts(cam1, cam2, cam3)
    cam = load_json('/home/user/Desktop/isaac_project/real_images/frames_960_5fps_new/scene_camera.json')
    #cam = load_json(args.camera)

    print("Loading bbox corners...")
    with open('/home/user/Desktop/isaac_project/real_images/bbox_corners.json') as f:#args.bbox_corners) as f:
        bbox_corners = [np.array(c) for c in json.load(f)]
    print(f"  {len(bbox_corners)} corners")

    # ---- Interpolate scene_gt ----
    #print("Interpolating scene_gt (slerp R + lerp t)...")
    #full_gt, frame_min, frame_max = interpolate_gt(gt_merged)
    #print(f"  {len(full_gt)} frames  (range {frame_min}..{frame_max})")

    # ---- Broadcast camera ----
    #full_cam = broadcast_camera(cam, frame_min, frame_max)

    # ---- Compute scene_gt_info + visualisation ----
    print("Computing scene_gt_info from reprojection + saving visualisations...")
    scene_gt_info = {}
    keyframes = get_keyframes_sorted(gt_merged)
    if len(keyframes) < 2:
        raise ValueError("Need at least 2 keyframes to interpolate")

    frame_min = keyframes[0]
    frame_max = keyframes[-1]
    keys = [int(key) for key, val in gt_merged.items()]
    #print(keys)

    for fi in keys:#range(frame_min, frame_max + 1, 5):
        key = str(fi)
        # if key not in full_gt or key not in full_cam:
        #     continue

        # gt_entry = full_gt[key][0]
        # cam_entry = full_cam[key]
        print(key)
        gt_entry = gt_merged[key][0]
        cam_entry = cam[key]

        info = compute_gt_info_for_frame(
            cam_k=cam_entry["cam_K"],
            R_m2c=gt_entry["cam_R_m2c"],
            t_m2c=gt_entry["cam_t_m2c"],
            bbox_corners=bbox_corners,
            width=W, height=H,
        )
        scene_gt_info[key] = [info]

        # Visualise if RGB exists
        img_path = find_rgb(rgb_dir, fi)
        print(img_path)
        if img_path is not None:
            img = Image.open(img_path).convert("RGB")
            #img.save(test_dir / f"{fi:06d}.jpg", quality=95)
            is_kf = key in keyframe_set
            img = draw_reprojection(
                img,
                cam_k=cam_entry["cam_K"],
                R_m2c=gt_entry["cam_R_m2c"],
                t_m2c=gt_entry["cam_t_m2c"],
                bbox_corners=bbox_corners,
                is_keyframe=is_kf,
            )
            suffix = "_KF" if is_kf else "_INTERP"
            img.save(viz_dir / f"{fi:06d}{suffix}.jpg", quality=95)

        if fi % 200 == 0:
            print(f"  frame {fi}/{frame_max}")

    # ---- Save ----
    print("Writing JSONs...")
    # dump_flat_json_per_row(gt_merged, output_dir / "scene_gt.json")
    # dump_flat_json_per_row(cam, output_dir / "scene_camera.json")
    # dump_flat_json_per_row(scene_gt_info, output_dir / "scene_gt_info.json")

    print(f"\nDone!")
    print(f"  JSONs          -> {output_dir}")
    print(f"  Visualisations -> {viz_dir}")


if __name__ == "__main__":
    main()