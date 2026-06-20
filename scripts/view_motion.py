"""View simple walking motion on G1 robot — kinematic replay.
Usage: uv run python scripts/view_motion.py
"""

import mujoco
import mujoco.viewer as viewer
import numpy as np
from pathlib import Path
import time

base = Path(__file__).parent.parent

# Load raw G1 model (no actuators needed for kinematics)
g1_xml = base / "src/mjlab/asset_zoo/robots/unitree_g1/xmls/g1.xml"
model = mujoco.MjModel.from_xml_path(str(g1_xml))
data = mujoco.MjData(model)

# Load motion
motion = np.load(base / "videos/simple_walk.npz")
jp = motion["joint_pos"]
nframes = len(jp)

# Timing
FRAME_RATE = 50  # original data at 50Hz
FRAME_TIME = 1.0 / FRAME_RATE

# Move robot sideways so camera has a good angle
data.qpos[1] = -0.5

print(f"🦿 G1 robot — kinematic replay")
print(f"   {nframes} frames = {nframes/50:.1f}s")
print(f"   Playback @ {FRAME_RATE}Hz (real-time)")
print(f"   Close viewer to exit")

with viewer.launch_passive(model, data) as v:
    v.cam.distance = 3.0
    v.cam.elevation = -15
    v.cam.azimuth = 90

    frame = 0
    last_time = time.time()
    while v.is_running():
        now = time.time()
        if now - last_time < FRAME_TIME:
            time.sleep(0.001)
            v.sync()
            continue
        last_time += FRAME_TIME

        i = frame % nframes
        # Freejoint: forward movement + standing height
        data.qpos[0] = i * 0.01
        data.qpos[2] = 1.15
        # Joint positions
        data.qpos[7:7+29] = jp[i]
        mujoco.mj_forward(model, data)

        v.cam.lookat[:] = [data.qpos[0], -0.5, 1.0]
        v.sync()
        frame += 1
