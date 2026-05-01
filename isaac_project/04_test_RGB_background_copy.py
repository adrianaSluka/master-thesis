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

carb.settings.get_settings().set("/omni/replicator/RTSubframes", 32)


kit = omni.kit.app.get_app()
kit.get_extension_manager().set_extension_enabled_immediate("omni.replicator.core", True)


# =========================
# CONFIG
# =========================

DRONE_ANCHOR_PATH = "/Replicator/Ref_Xform/Ref/skywalker"
DRONE_PIVOT_PATH = "/Replicator/Ref_Xform/Ref/skywalker"
CAMERA_PRIM_PATH = "/Replicator/Camera_Xform"

DOME_LIGHT_PATH = "/Replicator/DomeLight_Xform/DomeLight"
SPHERE_LIGHT_PATH = "/Replicator/SphereLight_Xform/SphereLight"
DISTANT_LIGHT_PATH = "/Replicator/DistantLight_Xform/DistantLight"

OUTPUT_DIR = Path("/home/user/Desktop/isaac_project/debug_output")
OUTPUT_DIR2 = Path("/home/user/Desktop/isaac_project/debug_output/annotation")

BOP_DIR = OUTPUT_DIR / "train_pbr" / "000001"
RGB_DIR = BOP_DIR / "rgb"
DEPTH_DIR = BOP_DIR / "depth"
MASK_DIR = BOP_DIR / "mask"
MASK_VISIB_DIR = BOP_DIR / "mask_visib"

for d in [RGB_DIR, DEPTH_DIR, MASK_DIR, MASK_VISIB_DIR, OUTPUT_DIR2]:
    d.mkdir(parents=True, exist_ok=True)

# Replace this with your actual drone USD path
DRONE_USD_PATH = "/home/user/Desktop/skywalker/skywalker.usd"
WORLD_USD_PATH = "/home/user/Desktop/worlds_usdz/SnowyMountainValleyLandscape.usdz"

# Render resolution
WIDTH = 1280
HEIGHT = 720

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
#alessandro-brunello-29Z2onJVypk-unsplash.jpg
#BG_IMAGE = '/home/user/Desktop/backgrounds/alessandro-brunello-29Z2onJVypk-unsplash.jpg'
TMP_IMAGE_1 = "/tmp/bg_current_1.jpg"
TMP_IMAGE_2 = "/tmp/bg_current_2.jpg"


# =========================
# HELPERS
# =========================
def ensure_file_exists(path: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Object USD not found: {path}")


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

def get_or_create_scale_op(xformable):
    for op in xformable.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeScale:
            return op
    return xformable.AddScaleOp()


def random_translation(drone_translation, min_val, max_val):
    x = drone_translation[0] + np.random.uniform(min_val, max_val)
    y = drone_translation[1] + np.random.uniform(min_val, max_val)
    z = drone_translation[2] + np.random.uniform(min_val, max_val)
    return x, y, z

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

    T_gf = get_relative_transform(model_prim, camera_prim)
    T_m2c = gfmatrix4d_to_numpy(T_gf)

    R_m2c = T_m2c[:3, :3].copy()
    t_m2c = T_m2c[:3, 3].copy()

    return T_m2c, R_m2c, t_m2c


def setup_render_pipeline(camera, width, height, rgb_dir):
    render_product = rep.create.render_product(camera, (width, height))

    writer = rep.WriterRegistry.get("BasicWriter")
    writer.initialize(
        output_dir=str(rgb_dir),
        rgb=True,
        frame_padding=6,
    )

    segmentation_anno = rep.AnnotatorRegistry.get_annotator(
        "instance_segmentation",
        init_params={"colorize": False},
    )
    depth_anno = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
    cam_params_anno = rep.AnnotatorRegistry.get_annotator("camera_params")

    writer.attach([render_product])
    segmentation_anno.attach(render_product)
    depth_anno.attach(render_product)
    cam_params_anno.attach(render_product)

    return render_product, writer, segmentation_anno, depth_anno, cam_params_anno

def get_bop_camera_from_cam_params(cam_params, stage, camera_prim_path):
    #print('cameraAperture', cam_params['cameraAperture'])
    
    fx = cam_params['cameraFocalLength'] / cam_params['cameraAperture'][0] * WIDTH
    fy = cam_params['cameraFocalLength'] / cam_params['cameraAperture'][1] * HEIGHT
    print("fx:", fx, "fy:", fy, "should be equal")
    cx = WIDTH / 2
    cy = HEIGHT / 2

    cam_k = [float(fx), 0, float(cx),
            0, float(fy), float(cy),
            0, 0, 1.0]
    view = np.array(cam_params["cameraViewTransform"], dtype=np.float64).reshape(4, 4)
    R_w2c = view[:3, :3].T
    t_w2c = view[3, :3].copy()
    #T_cam_world = get_world_transform_np(stage, camera_prim_path)  # camera-to-world, row-vector
    #print("T_cam_world", T_cam_world)

    # Row-vector convention: R in upper-left, t in last ROW
    #R_w2c = T_cam_world[:3, :3].copy()
    #cam_world_pos = T_cam_world[3, :3].copy()
    #t_w2c = -R_w2c @ cam_world_pos
    #print('t_w2c', t_w2c)

    # Verify immediately
    #cam_pos_recovered = -R_w2c.T @ t_w2c
    #print("Recovered cam pos:", cam_pos_recovered)

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


def project_point(K, R, t, X):
    X_c = R @ X + t
    #if X_c[2] <= 0:
    #    return None
    fx, fy = K[0], K[4]
    cx, cy = K[2], K[5]
    u = fx * X_c[0] / -X_c[2] + cx
    v = fy * X_c[1] / X_c[2] + cy
    return float(u), float(v)

def get_world_transform_np(stage, prim_path: str) -> np.ndarray:
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Invalid prim: {prim_path}")
    xf = UsdGeom.Xformable(prim)
    timeline = omni.timeline.get_timeline_interface()
    current_time = timeline.get_current_time()
    timecode = Usd.TimeCode(current_time)
    T = xf.ComputeLocalToWorldTransform(timecode)#(Usd.TimeCode.Default())
    return gfmatrix4d_to_numpy(T)


# =========================
# MAIN
# =========================
def main():

    scene_camera = {}
    scene_gt = {}
    scene_gt_info = {}

   
    camera = rep.create.camera(
        position=(0, 0, 0),  
        rotation=(0, 0, 0),      
    )

    rep.create.light(
        light_type="Dome",
        intensity=200.0,
    )

    # rep.create.cube(
    #     position=(1, 1, 1), 
    #     scale = 1,
    # )

    drone = rep.create.from_usd(
        DRONE_USD_PATH,
        semantics=[("class", SEMANTIC_CLASS)],
    )
    
    
    stage = omni.usd.get_context().get_stage()
    # print("camera data", get_world_transform_np(stage, CAMERA_PRIM_PATH))
    # print("camera 2 data", get_world_transform_np(stage, '/Replicator/Camera_Xform/Camera'))
    # print("drone data", get_world_transform_np(stage, '/Replicator/Ref_Xform/Ref/skywalker'))
    # print("drone 2 data", get_world_transform_np(stage, '/Replicator/Ref_Xform/Ref/skywalker/obj_000001'))
    # for prim in stage.Traverse():
    #    print(prim.GetPath())
    
    anchor_prim = stage.GetPrimAtPath(DRONE_ANCHOR_PATH)
    pivot_prim = stage.GetPrimAtPath(DRONE_PIVOT_PATH)
    anchor_xf = UsdGeom.Xformable(anchor_prim)
    pivot_xf = UsdGeom.Xformable(pivot_prim)


    translate_op = get_or_create_translate_op(anchor_xf)
    rotate_op = get_or_create_rotate_xyz_op(pivot_xf)
    cam_prim = stage.GetPrimAtPath('/Replicator/Camera_Xform/Camera')
    camera_usd = UsdGeom.Camera(cam_prim)

    horizontal_aperture = camera_usd.GetHorizontalApertureAttr().Get()  # keep whatever it is
    vertical_aperture = horizontal_aperture / (WIDTH / HEIGHT)           # fix vertical to match screen

    camera_usd.GetVerticalApertureAttr().Set(vertical_aperture)

    print("horizontal_aperture:", horizontal_aperture)
    print("vertical_aperture fixed to:", vertical_aperture)

    render_product, writer, segmentation_anno, depth_anno, cam_params_anno = setup_render_pipeline('/Replicator/Camera_Xform/Camera', WIDTH, HEIGHT, RGB_DIR)

    

    simulation_app.update()
    rep.orchestrator.run()

    start = time.time()

    tx = 0
    ty = 0
    tz = 0
    translate_op.Set(Gf.Vec3d(tx, ty, tz))

    for i in range(5):

        rx = 45
        ry = 45
        rz = 45

        translate_op.Set(Gf.Vec3d(tx, ty, tz))
        rotate_op.Set(Gf.Vec3f(rx, ry, rz))

        cx, cy, cz = random_translation((tx, ty, tz), -10.0, 10.0)
        look_at_val = (0, 0, 0)

        with camera:
            rep.modify.pose(
                position=(cx, cy, cz),
                look_at=look_at_val,
            )
        # cam_prim = stage.GetPrimAtPath('/Replicator/Camera_Xform/Camera')
        # camera_usd = UsdGeom.Camera(cam_prim)

        # print("horizontalApertureOffset:", camera_usd.GetHorizontalApertureOffsetAttr().Get())
        # print("verticalApertureOffset:  ", camera_usd.GetVerticalApertureOffsetAttr().Get())
        # print("horizontalAperture:      ", camera_usd.GetHorizontalApertureAttr().Get())
        # print("verticalAperture:        ", camera_usd.GetVerticalApertureAttr().Get())

        # cam_data = camera_usd.GetCamera(Usd.TimeCode.Default())

        # fovH = cam_data.GetFieldOfView(Gf.Camera.FOVHorizontal)
        # fovV = cam_data.GetFieldOfView(Gf.Camera.FOVVertical)
        # print("fovH:", fovH, "fovV:", fovV)

        # fx = (WIDTH  / 2) / np.tan(np.radians(fovH / 2))
        # fy = (HEIGHT / 2) / np.tan(np.radians(fovV / 2))
        # print("fx:", fx, "fy:", fy)

        # Verify aspect
        # print("fx/fy (should be ~1.778):", fx/fy)
        # fx = (WIDTH / 2) / np.tan(np.radians(fovH / 2))
        # fy = fx   # square pixels — do NOT derive fy from fovV

        # cx = WIDTH  / 2   # 640
        # cy = HEIGHT / 2   # 360

        # print("fx=fy:", fx)
        # Should be ~1465
        # Verify effective vertical FOV matches what renderer actually uses:
        # fovV_effective = 2 * np.degrees(np.arctan(HEIGHT / (2 * fx)))
        # print("effective fovV in render:", fovV_effective)  # ~27.7°, not 35.34°

        distance_to_camera = np.linalg.norm((tx-cx, ty-cy, tz-cz))
        dist_to_origin = np.linalg.norm((0-cx, 0-cy, 0-cz))

        print(
        f"frame {i:02d} | ", "\n",
        f"drone position t=({tx:.2f}, {ty:.2f}, {tz:.2f}), dist: {distance_to_camera}, {dist_to_origin}", "\n",
        f"camera position t=({cx:.2f}, {cy:.2f}, {cz:.2f})", "\n",
        )

        rep.orchestrator.step()
        simulation_app.update()
        simulation_app.update()
        simulation_app.update()



        T_m2c, R_m2c, t_m2c = extract_cam_R_t_m2c(
            stage=stage,
            model_prim_path='/Replicator/Cube_Xform',
            camera_prim_path='/Replicator/Camera_Xform/Camera',
        )

        scene_gt[str(i)] = [{
            "t_m2c": flatten_R(T_m2c),
            "cam_R_m2c": flatten_R(R_m2c),
            "cam_t_m2c": flatten_t(t_m2c),
            "obj_id": 1,
        }]


        cam_params = cam_params_anno.get_data()
        # print("obj data", get_world_transform_np(stage, DRONE_ANCHOR_PATH))
        # print("camera data", get_world_transform_np(stage, CAMERA_PRIM_PATH))
        # print("camera 2 data", get_world_transform_np(stage, '/Replicator/Camera_Xform/Camera'))
        cam_k, R_w2c, t_w2c, scene_camera[str(i)] = get_bop_camera_from_cam_params(cam_params, stage, '/Replicator/Camera_Xform/Camera')
    
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

    cube_prim = stage.GetPrimAtPath('/Replicator/Cube_Xform/Cube')

    # Get the world-space bounding box — no guessing
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default', 'render'])
    bbox = bbox_cache.ComputeWorldBound(cube_prim)
    bbox_range = bbox.GetRange()

    bmin = np.array(bbox_range.GetMin())
    bmax = np.array(bbox_range.GetMax())

    cube_corners = [
    np.array([x, y, z])
        for x in [bmin[0], bmax[0]]
        for y in [bmin[1], bmax[1]]
        for z in [bmin[2], bmax[2]]
    ]
    for i in range(5):
        points = []
        for j in cube_corners:
            pt = project_point(np.array(scene_camera[str(i)]['cam_K']), np.array(scene_gt[str(i)][0]['cam_R_m2c']).reshape((3, 3)), np.array(scene_gt[str(i)][0]['cam_t_m2c']), j)
            #pt = project_point(np.array(scene_camera[str(i)]['cam_K']), np.array(scene_camera[str(i)]['cam_R_w2c']).reshape((3, 3)), np.array(scene_camera[str(i)]['cam_t_w2c']), j)
            points.append(pt)
            
            #cam_pos_recovered = -np.array(scene_camera[str(i)]['cam_R_w2c']).reshape((3, 3)).T @ np.array(scene_camera[str(i)]['cam_t_w2c'])


        if pt is not None:
            print('f')
            img_path = RGB_DIR / f"rgb_{i:06d}.png"
            img = Image.open(img_path)
            draw = ImageDraw.Draw(img)
            for a in points:
                u, v = a
                r = 5
                draw.ellipse((u-r, v-r, u+r, v+r), outline="red", width=2)
            img.save(RGB_DIR / f"{i:06d}_debug.png")



if __name__ == "__main__":
    try:
        main()
    finally:
        print("fuck")
        simulation_app.close()