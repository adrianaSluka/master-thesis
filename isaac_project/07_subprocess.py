# run_scenes.py
import subprocess

NUM_SCENES = 23
PYTHON = "/home/user/isaacsim/isaac-sim-standalone-5.1.0-linux-x86_64/python.sh"  
SCRIPT = "/home/user/Desktop/isaac_project/05_speed_up_copy_2.py"

for scene_idx in range(23, NUM_SCENES + 1):
    scene_name = f"{scene_idx:06d}"
    print(f"\n{'='*60}")
    print(f"Starting scene {scene_name} ({scene_idx}/{NUM_SCENES})")
    print(f"{'='*60}\n")

    result = subprocess.run([PYTHON, SCRIPT, scene_name,
                             "--/renderer/multiGpu/enabled=false",
                            "--/renderer/multiGpu/autoEnable=false",
                            "--/renderer/multiGpu/maxGpuCount=1",
                            "--/renderer/activeGpu=0",
                            "--/physics/cudaDevice=0",])

    if result.returncode != 0:
        print(f"Scene {scene_name} FAILED (code {result.returncode})")
        continue

    print(f"Scene {scene_name} done")