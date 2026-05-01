from isaacsim import SimulationApp

simulation_app = SimulationApp({
    "headless": True,
    "renderer": "RayTracedLighting",
    "useExtension": True,
})


import json
import math
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import random
import time

import omni.usd
import omni.replicator.core as rep
import omni.kit.app
from pxr import UsdLux, UsdGeom, Gf, UsdShade, Sdf, Usd
from omni.isaac.core.utils.transformations import get_relative_transform
import carb.settings
import omni.graph.core as og
import sys

carb.settings.get_settings().set("/omni/replicator/RTSubframes", 16)


kit = omni.kit.app.get_app()
kit.get_extension_manager().set_extension_enabled_immediate("omni.replicator.core", True)


# =========================
# CONFIG
# =========================

#DRONE_ANCHOR_PATH = "/Replicator/Ref_Xform/Ref/skywalker"
DRONE_ANCHOR_PATH = "/Replicator/Ref_Xform/Ref/skywalker_x8_new_model"
#DRONE_PIVOT_PATH = "/Replicator/Ref_Xform/Ref/skywalker"
DRONE_PIVOT_PATH = "/Replicator/Ref_Xform/Ref/skywalker_x8_new_model"
CAMERA_PRIM_PATH = "/Replicator/Camera_Xform"

DOME_LIGHT_PATH = "/Replicator/DomeLight_Xform/DomeLight"
SPHERE_LIGHT_PATH = "/Replicator/SphereLight_Xform/SphereLight"
DISTANT_LIGHT_PATH = "/Replicator/DistantLight_Xform/DistantLight"

OUTPUT_DIR = Path("/home/user/Desktop/isaac_project/debug_output_C")
OUTPUT_DIR2 = Path("/home/user/Desktop/isaac_project/debug_output/annotation")

BOP_DIR = OUTPUT_DIR / "train_pbr" / "0000xx"



N = 2 #number of images per 00000x
M = 2 #number of backround crops


RGB_DIR = BOP_DIR / "rgb"
DEPTH_DIR = BOP_DIR / "depth"
MASK_DIR = BOP_DIR / "mask"
MASK_VISIB_DIR = BOP_DIR / "mask_visib"

for d in [RGB_DIR, DEPTH_DIR, MASK_DIR, MASK_VISIB_DIR, OUTPUT_DIR2]:
    d.mkdir(parents=True, exist_ok=True)

DRONE_USD_PATH = "/home/user/Desktop/skywalker/skywalker_x8_new_model_2.usd"
WORLD_USD_PATH = "/home/user/Desktop/worlds_usdz/SnowyMountainValleyLandscape.usdz"

# Render resolution
WIDTH = 960
HEIGHT = 540
TARGET_FX = 759.0
TARGET_FY = 758.0

# Scene settings
CAMERA_POS = (0.0, 0.0, 5)
CAMERA_LOOK_AT = (0.0, 0.0, 0.3)

LIGHT_POS = (2.0, 3.0, 4.0)
LIGHT_INTENSITY = 3000.0

OBJECT_POS = (0.0, 0.0, 0.0)
OBJECT_ROT_EULER_DEG = (0.0, 0.0, 0.0)
OBJECT_SCALE = (0.01, 0.01, 0.01)

SEMANTIC_CLASS = "drone"

BG_IMAGE = '/home/user/Desktop/backgrounds/backgrounds/bennoptic-Ikp24xI1l1s-unsplash.jpg'
BG_IMAGE = '/home/user/Desktop/backgrounds/backgrounds/alexander-wark-feeney-WmRmuQ5xC8I-unsplash.jpg'
#ant-rozetsky-AGMSNedogCE-unsplash.jpg
BG_IMAGE = '/home/user/Desktop/backgrounds/backgrounds/ant-rozetsky-AGMSNedogCE-unsplash.jpg'
BG_DIR = Path("/home/user/Desktop/backgrounds/backgrounds") 
TMP_IMAGE_1 = "/tmp/bg_current_1.jpg"
TMP_IMAGE_2 = "/tmp/bg_current_2.jpg"


# =========================
# HELPERS
# =========================


def get_or_create_translate_op(xformable):
    for op in xformable.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            return op
    return xformable.AddTranslateOp()


def get_or_create_rotate_xyz_op(xformable):
    for op in xformable.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeRotateXYZ:
            return op
    return xformable.AddRotateXYZOp()


def save_random_crop(bg_source, output_path):
    src = Image.open(bg_source)
    src_w, src_h = src.size
    left = random.randint(0, src_w - WIDTH)
    top = random.randint(0, src_h - HEIGHT)
    crop = src.crop((left, top, left + WIDTH, top + HEIGHT))
    crop.save(output_path, format="JPEG", quality=95)

def gfmatrix4d_to_numpy(mat: Gf.Matrix4d) -> np.ndarray:
    arr = np.array(mat)
    return arr.reshape(4, 4)

def extract_cam_R_t_m2c(stage, model_prim_path: str, camera_prim_path: str):

    model_prim = stage.GetPrimAtPath(model_prim_path)
    camera_prim = stage.GetPrimAtPath(camera_prim_path)

    if not model_prim.IsValid():
        raise ValueError(f"Invalid model prim: {model_prim_path}")
    if not camera_prim.IsValid():
        raise ValueError(f"Invalid camera prim: {camera_prim_path}")

    # source -> target, so this is model-to-camera
    T_gf = get_relative_transform(model_prim, camera_prim)
    T_m2c = gfmatrix4d_to_numpy(T_gf)

    R_m2c = T_m2c[:3, :3].copy()
    t_m2c = T_m2c[:3, 3].copy()

    return T_m2c, R_m2c, t_m2c

def save_png_u8(path: Path, arr: np.ndarray):
    Image.fromarray(arr.astype(np.uint8)).save(path)

def save_png_u16(path: Path, arr: np.ndarray):
    Image.fromarray(arr.astype(np.uint16)).save(path)

def setup_render_pipeline(camera, width, height, rgb_dir):
    render_product = rep.create.render_product(camera, (width, height))

    writer = rep.WriterRegistry.get("BasicWriter")
    writer.initialize(
        output_dir=str(rgb_dir),
        image_output_format="jpg",
        rgb=True,
        frame_padding=6,
    )

    segmentation_anno = rep.AnnotatorRegistry.get_annotator(
        "instance_segmentation",
        init_params={"colorize": False},
    )
    depth_anno = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
    cam_params_anno = rep.AnnotatorRegistry.get_annotator("camera_params")

    #writer.attach([render_product])
    segmentation_anno.attach(render_product)
    depth_anno.attach(render_product)
    cam_params_anno.attach(render_product)
    rep.orchestrator.step()
    simulation_app.update()

    cam_params = cam_params_anno.get_data()
    writer.attach([render_product])

    return render_product, writer, segmentation_anno, depth_anno, cam_params_anno, cam_params

def extract_shader(stage):
    plane_prim = stage.GetPrimAtPath('/Replicator/Plane_Xform/Plane')
    bound_mat, binding_rel = UsdShade.MaterialBindingAPI(plane_prim).ComputeBoundMaterial()
    material = UsdShade.Material(bound_mat)
    shader = None
    for child in material.GetPrim().GetChildren():
        candidate = UsdShade.Shader(child)
        if candidate:
            shader = candidate
            break

    if not shader:
        raise RuntimeError("Could not find shader under bound material")


    return shader.GetInput("diffuse_texture")

def get_bop_camera_from_cam_params(cam_params):

    fx = cam_params['cameraFocalLength'] / cam_params['cameraAperture'][0] * WIDTH
    fy = cam_params['cameraFocalLength'] / cam_params['cameraAperture'][1] * HEIGHT
    assert abs(fx - TARGET_FX) < 1.0, f"fx mismatch: {fx} vs {TARGET_FX}"
    assert abs(fy - TARGET_FY) < 1.0, f"fy mismatch: {fy} vs {TARGET_FY}"
    cx = WIDTH / 2
    cy = HEIGHT / 2

    cam_k = [float(fx), 0, float(cx),
            0, float(fy), float(cy),
            0, 0, 1.0]
    view = np.array(cam_params["cameraViewTransform"], dtype=np.float64).reshape(4, 4).T
    R_w2c = view[:3, :3].T.copy()
    t_w2c = view[:3, 3].copy()

    dict_form  = {
        "cam_K": cam_k,
        "cam_R_w2c": [float(x) for x in R_w2c.reshape(-1)],
        "cam_t_w2c": [float(x) for x in t_w2c.reshape(-1)],
        "depth_scale": 0.1,
    }

    return cam_k, R_w2c, t_w2c, dict_form



def dump_flat_json_per_row(data: dict, dir: Path, name: str):
    path = dir / name
    with open(path, "w", encoding="utf-8") as f:
        f.write("{\n")
        items = list(data.items())
        for idx, (k, v) in enumerate(items):
            line = json.dumps(k, ensure_ascii=False) + ": " + json.dumps(v, ensure_ascii=False)
            if idx < len(items) - 1:
                line += ","
            f.write("  " + line + "\n")
        f.write("}\n")

def flatten_R(R: np.ndarray):
    return [float(x) for x in R.reshape(-1)]

def flatten_t(t: np.ndarray):
    return [float(x) for x in t.reshape(-1)]

def binary_mask_to_bbox(mask: np.ndarray):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return [0, 0, 0, 0]
    x_min = int(xs.min())
    y_min = int(ys.min())
    x_max = int(xs.max())
    y_max = int(ys.max())
    return [x_min, y_min, int(x_max - x_min + 1), int(y_max - y_min + 1)]


def project_point(K, R, t, X):
    X_c = R @ X + t
    #if X_c[2] <= 0:
    #    return None
    fx, fy = K[0], K[4]
    cx, cy = K[2], K[5]
    u = fx * X_c[0] / -X_c[2] + cx
    v = fy * X_c[1] / X_c[2] + cy
    return float(u), float(v)

def create_material_schedule(num_frames, materials, min_frames=5, max_frames=30):
    """Create a schedule with random-length material runs."""
    schedule = []
    i = 0
    while i < num_frames:
        # Pick a random material
        mat = random.choice(materials)
        # Pick a random duration
        duration = random.randint(min_frames, max_frames)
        # Fill schedule
        for _ in range(duration):
            if i < num_frames:
                schedule.append(mat)
                i += 1
    return schedule


def set_prim_position(stage, prim_path, position):
    """Set prim world position."""
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return
    xformable = UsdGeom.Xformable(prim)
    
    translate_op = None
    for op in xformable.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            translate_op = op
            break
    if translate_op is None:
        translate_op = xformable.AddTranslateOp()
    
    translate_op.Set(Gf.Vec3d(*position))


def set_light_color(stage, light_path, color):
    """Set light color directly."""
    prim = stage.GetPrimAtPath(light_path)
    if prim.IsValid():
        prim.GetAttribute("inputs:color").Set(Gf.Vec3f(*color))

def is_point_inside(pt, W, H, margin=0):
    u, v = pt
    return int((margin <= u < W - margin) and (margin <= v < H - margin))

def drone_is_visible(cam_k, R_m2c, t_m2c, image_size, model_points):
    pts = [project_point(
            np.array(cam_k), 
            np.array(R_m2c).reshape((3, 3)), 
            np.array(t_m2c), point
            ) for point in model_points]
    pts.append(project_point(np.array(cam_k), 
            np.array(R_m2c).reshape((3, 3)), 
            np.array(t_m2c), np.zeros(3)
            ))
    inside_flags = [is_point_inside(pt, WIDTH, HEIGHT, 0) for pt in pts]
    n_inside = sum(inside_flags)
    return n_inside

def get_path_for_frame(pool, frame_idx, change_every=10):
        """Same background for N consecutive frames."""
        pool_size = len(pool)
        pool_idx = (frame_idx // change_every) % pool_size
        return pool[pool_idx]


# =========================
# MAIN
# =========================
def main():
    start = time.time()

    scene_camera = {}
    scene_gt = {}
    scene_gt_info = {}


    camera = rep.create.camera(
        position=(0, 0, 0),  # Pull back and raise the camera
        rotation=(0, 0, 0),      # Aim directly at the cube/sphere
    )
    

    rep.create.light(
        light_type="Dome",
        intensity=200.0,
    )


    distant_light = rep.create.light(
        light_type="Distant",
        position=LIGHT_POS,
        rotation=(0, 45, 90),
        intensity=rep.distribution.uniform(100, 400)
    )


    tx = 0
    ty = 0
    tz = 0
            
    save_random_crop(BG_IMAGE, TMP_IMAGE_1)

    bg_mat = rep.create.material_omnipbr(
        diffuse_texture=TMP_IMAGE_1,
        roughness=1.0,
        metallic=0.0
    )
    img = Image.open(TMP_IMAGE_1)
    width, height = img.size
    aspect = width / height
    plane_height = 750
    plane_width = plane_height * aspect

    bg_plane = rep.create.plane(
        position=(0, 0, -1000),
        scale=(plane_width, plane_height, 1),
        material=bg_mat,
        visible=True,
    )
    
    stage = omni.usd.get_context().get_stage()
    for prim in stage.Traverse():
       print(prim.GetPath())


    cam_prim = stage.GetPrimAtPath('/Replicator/Camera_Xform/Camera')
    camera_usd = UsdGeom.Camera(cam_prim)

    horizontal_aperture = camera_usd.GetHorizontalApertureAttr().Get()  # keep whatever it is
    vertical_aperture = horizontal_aperture / (WIDTH / HEIGHT)           # fix vertical to match screen
    print('horizontal_aperture', horizontal_aperture)
    print('vertical_aperture', vertical_aperture)
    camera_usd.GetVerticalApertureAttr().Set(vertical_aperture)
    focal_length_mm = TARGET_FX * horizontal_aperture / WIDTH
    print('focal_length_mm', focal_length_mm)
    camera_usd.GetFocalLengthAttr().Set(focal_length_mm)

    render_product, writer, segmentation_anno, depth_anno, cam_params_anno, cam_params = setup_render_pipeline('/Replicator/Camera_Xform/Camera', WIDTH, HEIGHT, RGB_DIR)
    
    simulation_app.update()
    rep.orchestrator.run()
    cam_k, R_w2c, t_w2c, scene_cam_entry = get_bop_camera_from_cam_params(cam_params)#(cam_params_anno.get_data())

    look_at_val = (tx, ty, tz)
    cx = 10
    cy = 10
    cz = 10

    with camera:
            rep.modify.pose(
                position=(cx, cy, cz),      # Explicit tuple, not rep.distribution
                look_at=(tx, ty, tz),       # Explicit tuple
            )

    direction = np.array(look_at_val) - np.array((cx, cy, cz))

    direction_normalized = direction/np.linalg.norm(direction)
    bg_pos = look_at_val + direction_normalized*1000
    

    with bg_plane:
        rep.modify.pose(
            position=bg_pos,   
            scale=(plane_width, plane_height, 1),
            look_at=camera
        )
         

    camera_path = '/Replicator/Camera_Xform'

    distant_light_path = None
    for prim in stage.Traverse():
        path_str = str(prim.GetPath())
        if "DistantLight" in path_str and prim.IsA(UsdLux.DistantLight):
            distant_light_path = path_str
            break

    print(f"Camera: {camera_path}")
    print(f"Distant light: {distant_light_path}")


    texture_input = extract_shader(stage)
    tmp_paths = [TMP_IMAGE_1, TMP_IMAGE_2]

    print('before while loop')


    print('after while loop, before background generation')
    bg_files = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
        bg_files.extend(BG_DIR.glob(ext))
        
    bg_pool = []


    print('before bg pool generation')
    for i in range(M):#200):#CHANGE
        src_img = random.choice(bg_files)   # pick random source image
        out_path = f"/tmp/bg_{i}.jpg"       # cropped output
        save_random_crop(str(src_img), out_path)
        bg_pool.append(out_path)

    end = time.time()
    print('init time', end-start)
    start = time.time()
    for i in range(N):#2000): #CHANGE
        print('generation', i)
        dl_x = tx + np.random.uniform(-15, 15)
        dl_y = ty + np.random.uniform(-15, 15)
        dl_z = tz + np.random.uniform(-15, 15)
        set_prim_position(stage, distant_light_path, (dl_x, dl_y, dl_z))

        if i % 10 == 0:
            current_bg = get_path_for_frame(bg_pool, i, change_every=10)
            texture_input.Set(Sdf.AssetPath(current_bg))
 
        rep.orchestrator.step()
        #rep.orchestrator.step()
        simulation_app.update()
        print('b')

        scene_gt[str(i)] = []
        image_stem = f"{i:06d}"   

        depth = depth_anno.get_data()
        depth_mm = np.round(depth * 1000.0).astype(np.uint16)
        save_png_u16(DEPTH_DIR / f"{image_stem}.png", depth_mm)

        inst_data = segmentation_anno.get_data()
        info = inst_data["info"]["idToSemantics"]
        seg = inst_data["data"] 
        obj_stem = f"{i:06d}_000000"
        print('a')

        for inst_id, cls in info.items():
            if cls['class'] == 'drone':
                drone_inst_id = int(inst_id)
                break
        print('c')


        scene_gt_info[str(i)] = []

        cam_params = cam_params_anno.get_data()
        cam_k, R_w2c, t_w2c, scene_camera[str(i)] = get_bop_camera_from_cam_params(cam_params)
        if i % 10 == 0:
            # Count graph nodes
            graphs = og.get_all_graphs()
            total_nodes = sum(len(g.get_nodes()) for g in graphs)
            print(f"Frame {i}: {total_nodes} graph nodes")

    
      
    graphs = og.get_all_graphs()
    total_nodes = sum(len(g.get_nodes()) for g in graphs)
    print(f"Frame 20: {total_nodes} graph nodes")
    end = time.time()
    gen_time = end - start
    start = time.time()
    dump_flat_json_per_row(scene_camera, BOP_DIR, 'scene_camera.json')
    dump_flat_json_per_row(scene_gt, BOP_DIR, 'scene_gt.json')
    dump_flat_json_per_row(scene_gt_info, BOP_DIR, 'scene_gt_info.json')
    end = time.time()
    writing_time = end-start

    print("=" * 60)
    print("DEBUG SINGLE SCENE SUMMARY")
    print("generation time", gen_time)
    print("writing time", writing_time)
    print("=" * 60)



if __name__ == "__main__":
    try:
        main()
    finally:
        print("fuck")
        simulation_app.close()