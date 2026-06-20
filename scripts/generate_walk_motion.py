"""Generate a simple walking motion file for G1 tracking training."""
import numpy as np

duration = 8.0
dt = 0.02
num_steps = int(duration / dt)

freq = 0.6  # Hz
speed = 0.3  # m/s
t = np.linspace(0, duration, num_steps)

phase_right = 2 * np.pi * freq * t
phase_left = phase_right + np.pi

# === Joint positions (29 DOF) ===
joint_pos = np.zeros((num_steps, 29))

# Legs
hip_pitch_amp = 0.35
joint_pos[:, 0] = hip_pitch_amp * np.sin(phase_left)    # left_hip_pitch
joint_pos[:, 1] = 0.06 * np.sin(phase_left)              # left_hip_roll
joint_pos[:, 3] = 0.6 * np.maximum(0, np.sin(phase_left))  # left_knee
joint_pos[:, 4] = 0.25 * np.maximum(0, -np.sin(phase_left))  # left_ankle_pitch

joint_pos[:, 6] = hip_pitch_amp * np.sin(phase_right)    # right_hip_pitch
joint_pos[:, 7] = 0.06 * np.sin(phase_right)              # right_hip_roll
joint_pos[:, 9] = 0.6 * np.maximum(0, np.sin(phase_right))  # right_knee
joint_pos[:, 10] = 0.25 * np.maximum(0, -np.sin(phase_right))  # right_ankle_pitch

# Waist
joint_pos[:, 14] = -0.05  # slight forward lean

# Arms (opposite to legs)
joint_pos[:, 15] = 0.3 * np.sin(phase_right)  # left_shoulder_pitch
joint_pos[:, 22] = 0.3 * np.sin(phase_left)   # right_shoulder_pitch
joint_pos[:, 16] = 0.1                           # left_shoulder_roll
joint_pos[:, 23] = -0.1                          # right_shoulder_roll
joint_pos[:, 18] = 0.3                           # left_elbow
joint_pos[:, 25] = 0.3                           # right_elbow

joint_vel = np.gradient(joint_pos, dt, axis=0)

# === Body tracking (14 bodies) ===
num_bodies = 14
body_pos_w = np.zeros((num_steps, num_bodies, 3))

# Root position
body_pos_w[:, 0, 0] = t * speed
body_pos_w[:, 0, 2] = 1.15 + 0.015 * np.sin(2 * np.pi * freq * 2 * t)

# Body offset vectors for each tracked link
offsets = np.array([
    [0, 0, 0],           # 0: pelvis
    [0, 0.07, -0.12],    # 1: left_hip_roll_link
    [0, 0.07, -0.42],    # 2: left_knee_link
    [0, 0.07, -0.75],    # 3: left_ankle_roll_link
    [0, -0.07, -0.12],   # 4: right_hip_roll_link
    [0, -0.07, -0.42],   # 5: right_knee_link
    [0, -0.07, -0.75],   # 6: right_ankle_roll_link
    [0.12, 0, 0.05],     # 7: torso_link
    [0.18, 0.15, 0.1],   # 8: left_shoulder_roll_link
    [0.18, 0.15, -0.15],  # 9: left_elbow_link
    [0.18, 0.15, -0.4],  # 10: left_wrist_yaw_link
    [0.18, -0.15, 0.1],  # 11: right_shoulder_roll_link
    [0.18, -0.15, -0.15], # 12: right_elbow_link
    [0.18, -0.15, -0.4],  # 13: right_wrist_yaw_link
])
for i, off in enumerate(offsets):
    body_pos_w[:, i, :] = body_pos_w[:, 0, :] + off

body_quat_w = np.zeros((num_steps, num_bodies, 4))
body_quat_w[:, :, 0] = 1.0

body_lin_vel_w = np.gradient(body_pos_w, dt, axis=0)
body_ang_vel_w = np.zeros((num_steps, num_bodies, 3))

out = "/home/dzh/mjlab/videos/simple_walk.npz"
np.savez(out,
    joint_pos=joint_pos,
    joint_vel=joint_vel,
    body_pos_w=body_pos_w,
    body_quat_w=body_quat_w,
    body_lin_vel_w=body_lin_vel_w,
    body_ang_vel_w=body_ang_vel_w,
)
print(f"✅ {out}")
print(f"   {num_steps} frames x {joint_pos.shape[1]} joints x {num_bodies} bodies")
print(f"   Duration: {duration}s @ {1/dt:.0f}Hz @ {speed} m/s")
