import numpy as np

def estimate_pose_bins_from_5pts(kps5: np.ndarray):
    """
    Ước lượng hướng mặt từ 5 landmarks (InsightFace):
    Trả về (yaw_score, pitch_score) tương đối để phân loại góc.
    """
    le, re, nose, lm, rm = kps5.astype(np.float32)

    eye_mid = (le + re) / 2.0
    mouth_mid = (lm + rm) / 2.0

    eye_dist = np.linalg.norm(re - le) + 1e-6
    yaw_score = (nose[0] - eye_mid[0]) / eye_dist

    up_dist = (nose[1] - eye_mid[1])
    down_dist = (mouth_mid[1] - nose[1]) + 1e-6
    pitch_ratio = up_dist / down_dist
    pitch_score = (pitch_ratio - 1.0)

    return float(yaw_score), float(pitch_score)

POSE_BINS = [
    ("FRONT", "Nhìn thẳng",        lambda yaw, pitch: abs(yaw) < 0.18 and abs(pitch) < 0.25),
    ("LEFT",  "Quay mặt sang TRÁI",  lambda yaw, pitch: yaw < -0.18),
    ("RIGHT", "Quay mặt sang PHẢI", lambda yaw, pitch: yaw > 0.18),
    ("UP",    "Ngẩng mặt LÊN",      lambda yaw, pitch: pitch < -0.18),
    ("DOWN",  "Cúi mặt XUỐNG",      lambda yaw, pitch: pitch > 0.18),
]