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

def save_random_crop(bg_source, output_path):
    src = Image.open(bg_source)
    src_w, src_h = src.size
    left = random.randint(0, src_w - WIDTH)
    top = random.randint(0, src_h - HEIGHT)
    crop = src.crop((left, top, left + WIDTH, top + HEIGHT))
    crop.save(output_path, format="JPEG", quality=95)


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

    # source -> target, so this is model-to-camera
    T_gf = get_relative_transform(model_prim, camera_prim)
    T_m2c = gfmatrix4d_to_numpy(T_gf)

    R_m2c = T_m2c[:3, :3].copy()
    t_m2c = T_m2c[:3, 3].copy()

    return T_m2c, R_m2c, t_m2c

def save_pose_txt(txt_path: Path, R_m2c: np.ndarray, t_m2c: np.ndarray):
    with open(txt_path, "w") as f:
        f.write("\nR_m2c:\n")
        np.savetxt(f, R_m2c, fmt="%.8f")
        f.write("\nt_m2c:\n")
        np.savetxt(f, t_m2c.reshape(1, 3), fmt="%.8f")


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

    writer.attach([render_product])
    segmentation_anno.attach(render_product)
    depth_anno.attach(render_product)
    cam_params_anno.attach(render_product)

    return render_product, writer, segmentation_anno, depth_anno, cam_params_anno

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
    cx = WIDTH / 2
    cy = HEIGHT / 2
    # fx = cam_params['cameraFocalLength'] / cam_params['cameraAperture'][0] * WIDTH
    # fy = cam_params['cameraFocalLength'] / cam_params['cameraAperture'][1] * HEIGHT
    # cx = WIDTH / 2
    # cy = HEIGHT / 2

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

def create_material_schedule(num_frames, materials, change_every=20):
        """Create a schedule of which material to use for each frame."""
        schedule = []
        for i in range(num_frames):
            if i % change_every == 0:
                schedule.append(random.choice(materials))
            else:
                schedule.append(schedule[-1])  # Same as previous
        return schedule



# =========================
# MAIN
# =========================
def main():

    scene_camera = {}
    scene_gt = {}
    scene_gt_info = {}

    scene_gt_coco = {
        "info": {
            "description": "drone_train",
            "url": "",
            "version": "1.0",
            "year": 2026,
            "contributor": "",
            "date_created": "",
        },
        "licenses": [],
        "categories": [
            {"id": 1, "name": "drone", "supercategory": "drone"}
        ],
        "images": [],
        "annotations": [],
    }

    coco_ann_id = 1

    camera = rep.create.camera(
        position=(0, 0, 0),  # Pull back and raise the camera
        rotation=(0, 0, 0),      # Aim directly at the cube/sphere
    )
    

    rep.create.light(
        light_type="Dome",
        intensity=200.0,
    )

    sphere_lights = rep.create.light(
        light_type="Sphere",
        position=LIGHT_POS,
        count = 2,
        intensity=rep.distribution.uniform(80000, 100000),
        scale=rep.distribution.uniform(0.5, 1),
    )

    distant_light = rep.create.light(
        light_type="Distant",
        position=LIGHT_POS,
        rotation=(0, 45, 90),
        intensity=300
    )

    drone = rep.create.from_usd(
        DRONE_USD_PATH,
        semantics=[("class", SEMANTIC_CLASS)],
    )
    


    save_random_crop(BG_IMAGE, TMP_IMAGE_1)

    bg_mat = rep.create.material_omnipbr(
        diffuse_texture=TMP_IMAGE_1,
        roughness=1.0,
        metallic=0.0
    )
    img = Image.open(TMP_IMAGE_1)
    width, height = img.size
    aspect = width / height
    plane_height = 700
    plane_width = plane_height * aspect

    bg_plane = rep.create.plane(
        position=(0, 0, -1000),
        scale=(plane_width, plane_height, 1),
        material=bg_mat,
        visible=True,
    )
    
    stage = omni.usd.get_context().get_stage()
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

    render_product, writer, segmentation_anno, depth_anno, cam_params_anno = setup_render_pipeline('/Replicator/Camera_Xform/Camera', WIDTH, HEIGHT, RGB_DIR)
    

    simulation_app.update()
    rep.orchestrator.run()
    start = time.time()

    texture_input = extract_shader(stage)
    tmp_paths = [TMP_IMAGE_1, TMP_IMAGE_2]

    tx = 0
    ty = 0
    tz = 0
    translate_op.Set(Gf.Vec3d(tx, ty, tz))
    colors = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 1.0, 0.0), (1.0, 0.5, 0.0),(0.6, 0.3, 0.0), (0.5, 0.5, 0.5),
              (0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (0.15, 0.15, 0.15), (0.25, 0.25, 0.25), (0.35, 0.35, 0.35), (0.30, 0.32, 0.28),
              (0.85, 0.85, 0.85), (0.28, 0.33, 0.38), (0.30, 0.26, 0.22), (0.55, 0.52, 0.45), (0.22, 0.24, 0.20), (0.60, 0.58, 0.55), (0.60, 1.00, 0.60),
              (0.50, 0.00, 0.00), (1.00, 0.00, 0.50), (0.00, 0.50, 1.00), (0.00, 1.00, 1.00), (0.50, 1.00, 0.00)]
    
    # mamterials = [rep.create.material_omnipbr(
    #         diffuse=color,
    #         roughness=0.4,
    #         metallic=0.0
    #     ) for color in colors
    # ]


    #material_schedule = create_material_schedule(20, materials, change_every=10)

    for i in range(20):
        # Only apply material when it changes
        # if i == 0 or material_schedule[i] != material_schedule[i-1]:
        #     with drone:
        #         rep.modify.attribute("material", material_schedule[i])

        # with drone:
        #     rep.randomizer.materials(materials)

        rx = np.random.uniform(0, 360.0)
        ry = np.random.uniform(0, 360.0)
        rz = np.random.uniform(0.0, 360.0)
        translate_op.Set(Gf.Vec3d(tx, ty, tz))
        rotate_op.Set(Gf.Vec3f(rx, ry, rz))
        #scale_op.Set(Gf.Vec3f(0.002, 0.002, 0.002))

        cx, cy, cz = random_translation((tx, ty, tz), 3.0, 10.0)
        #look_at_val = np.random.uniform((-1, -1, -1), (1, 1, 1))
        look_at_val = (1, 1, 1)

        with camera:
            rep.modify.pose(
                position=(cx, cy, cz),
                look_at=look_at_val,
            )

        with sphere_lights:
            rep.modify.pose(
                position = rep.distribution.uniform((tx-3, ty-3, tz-3), (tx+3, ty+3, tz+3))
            )
            rep.modify.attribute(
                "color",
                rep.distribution.uniform(
                    (0, 0, 0),
                    (1.0, 1.0, 1.0)
            )
        )
        dl_x, dl_y, dl_z = random_translation((tx, ty, tz), -15.0, 15.0)
            
        with distant_light:
            rep.modify.pose(
                position = (dl_x, dl_y, dl_z),
                look_at=drone
            )
        #     rep.modify.attribute(
        #         "color",
        #         rep.distribution.uniform(
        #             (0.2, 0.2, 0.2),
        #             (1.0, 1.0, 1.0)
        #     )
        # )
        direction = np.array(look_at_val) - np.array((cx, cy, cz))


        direction_normalized = direction/np.linalg.norm(direction)
        bg_pos = look_at_val + direction_normalized*1000

        with bg_plane:
            rep.modify.pose(
                position=bg_pos,   
                scale=(plane_width, plane_height, 1),
                look_at=camera
            )

        current_path = tmp_paths[(i+1) % len(tmp_paths)]
        save_random_crop(BG_IMAGE, current_path)
        texture_input.Set(Sdf.AssetPath(current_path))
        

        distance_to_camera = np.linalg.norm((tx-cx, ty-cy, tz-cz))
        dist_to_origin = np.linalg.norm((0-cx, 0-cy, 0-cz))

        print(
        f"frame {i:02d} | ", "\n",
        f"drone position t=({tx:.2f}, {ty:.2f}, {tz:.2f}), dist: {distance_to_camera}, {dist_to_origin}", "\n",
        f"camera position t=({cx:.2f}, {cy:.2f}, {cz:.2f})", "\n",
        f"bg position t=({bg_pos})", "\n",
        f"distant light position t=({dl_x:.2f}, {dl_y:.2f}, {dl_z:.2f})"
        )

        rep.orchestrator.step()
        #rep.orchestrator.step()
        simulation_app.update()
        simulation_app.update()

        T_m2c, R_m2c, t_m2c = extract_cam_R_t_m2c(
            stage=stage,
            model_prim_path=DRONE_PIVOT_PATH,
            #camera_prim_path='/Replicator/Camera_Xform',
            camera_prim_path='/Replicator/Camera_Xform/Camera',
        )

        scene_gt[str(i)] = [{
            "t_m2c": flatten_R(T_m2c),
            "cam_R_m2c": flatten_R(R_m2c),
            "cam_t_m2c": flatten_t(t_m2c),
            "obj_id": 1,
        }]
        image_stem = f"{i:06d}"   
        txt_path = OUTPUT_DIR2 / f"{image_stem}.txt"
        save_pose_txt(txt_path, R_m2c, t_m2c)

        depth = depth_anno.get_data()
        depth_mm = np.round(depth * 1000.0).astype(np.uint16)
        save_png_u16(DEPTH_DIR / f"{image_stem}.png", depth_mm)

        inst_data = segmentation_anno.get_data()
        info = inst_data["info"]["idToSemantics"]
        seg = inst_data["data"] 
        obj_stem = f"{i:06d}_000000"

        for inst_id, cls in info.items():
            if cls['class'] == 'drone':
                drone_inst_id = int(inst_id)
                break
        mask_visib = (seg == drone_inst_id).astype(np.uint8) * 255
        mask_full = mask_visib.copy()
        save_png_u8(MASK_VISIB_DIR / f"{obj_stem}.png", mask_visib)
        save_png_u8(MASK_DIR / f"{obj_stem}.png", mask_full)
        bbox_obj = binary_mask_to_bbox(mask_full)
        #bbox_visib = binary_mask_to_bbox(mask_visib)
        bbox_visib = bbox_obj.copy()
        px_count_all = int((mask_full > 0).sum())
        #px_count_visib = int((mask_visib > 0).sum())
        px_count_valid = px_count_all if px_count_all > 0 else 0

        visib_fract = 1.0
        #if px_count_all > 0:
        #    visib_fract = float(px_count_visib / px_count_all)

        scene_gt_info[str(i)] = [{
            "bbox_obj": bbox_obj,
            "bbox_visib": bbox_visib,
            "px_count_all": px_count_all,
            "px_count_valid": px_count_valid,
            "px_count_visib": px_count_all,
            "visib_fract": visib_fract,
        }]


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
    #cube_prim = stage.GetPrimAtPath('/Replicator/Cube_Xform/Cube')

    # Get the world-space bounding box — no guessing
    cube_prim = stage.GetPrimAtPath(DRONE_ANCHOR_PATH)
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
        # print(np.array(scene_gt[str(i)][0]['cam_R_m2c']).reshape((3, 3)))
        # print(scene_gt[str(i)][0]['cam_t_m2c'])
        points = []
        for j in cube_corners:
            pt = project_point(np.array(scene_camera[str(i)]['cam_K']), np.array(scene_gt[str(i)][0]['cam_R_m2c']).reshape((3, 3)), np.array(scene_gt[str(i)][0]['cam_t_m2c']), j)
            points.append(pt)
        if pt is not None:
            img_path = RGB_DIR / f"rgb_{i:06d}.jpg"
            img = Image.open(img_path)
            draw = ImageDraw.Draw(img)
            for a in points:
                u, v = a
                r = 5
                draw.ellipse((u-r, v-r, u+r, v+r), outline="red", width=2)
            img.save(RGB_DIR / f"{i:06d}_debug.jpg")



if __name__ == "__main__":
    try:
        main()
    finally:
        print("fuck")
        simulation_app.close()