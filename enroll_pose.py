import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from insightface.app import FaceAnalysis
from PIL import ImageFont, ImageDraw, Image

# Font hỗ trợ tiếng Việt
VI_FONT_PATH = r"C:\Windows\Fonts\arial.ttf"

def put_vietnamese_text(img_bgr, text, org, font_size=28, color=(255, 255, 255), thickness=2):
    """
    Vẽ text tiếng Việt lên ảnh OpenCV bằng PIL.
    img_bgr: ảnh OpenCV (BGR)
    org: (x, y) góc trên-trái của text
    color: BGR
    """
    x, y = org
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)

    try:
        font = ImageFont.truetype(VI_FONT_PATH, font_size)
    except Exception:
        font = ImageFont.load_default()

    rgb = (color[2], color[1], color[0])

    # viền chữ
    if thickness > 0:
        outline_rgb = (0, 0, 0)
        for dx in range(-thickness, thickness + 1):
            for dy in range(-thickness, thickness + 1):
                if dx == 0 and dy == 0:
                    continue
                draw.text((x + dx, y + dy), text, font=font, fill=outline_rgb)

    draw.text((x, y), text, font=font, fill=rgb)

    out = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return out

DB_DIR = Path("db")
DB_DIR.mkdir(parents=True, exist_ok=True)

def l2_normalize(v: np.ndarray) -> np.ndarray:
    v = v.astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-9)

def draw_progress_circle(frame, progress01: float, center=None, radius=45, thickness=10):
    """
    Vẽ vòng tròn tiến độ + % ở giữa. (return frame)
    progress01: 0..1
    """
    h, w = frame.shape[:2]
    if center is None:
        center = (w - 70, 70)  # góc phải trên

    p = float(np.clip(progress01, 0.0, 1.0))
    pct = int(round(p * 100))

    # vòng nền
    cv2.circle(frame, center, radius, (70, 70, 70), thickness)

    # vòng tiến độ
    angle = int(360 * p)
    cv2.ellipse(frame, center, (radius, radius), -90, 0, angle, (0, 255, 255), thickness)

    # text % (PIL)
    frame = put_vietnamese_text(
        frame, f"{pct}%", (center[0] - 22, center[1] - 18),
        font_size=26, color=(255, 255, 255), thickness=2
    )
    return frame

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

def main(person_id: str, name: str, cam_index: int = 0):
    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=0, det_size=(320, 320))

    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print("Khong mo duoc camera")
        return

    NEED_PER_BIN = 1
    collected = {k: [] for k, _, _ in POSE_BINS}

    MIN_FACE_SIZE = 120
    MIN_DET_SCORE = 0.6

    last_take_t = {k: 0.0 for k, _, _ in POSE_BINS}
    TAKE_COOLDOWN = 0.08
    tick = cv2.getTickFrequency()

    total_need = len(POSE_BINS) * NEED_PER_BIN

    print(f"[ENROLL] ID={person_id} | NAME={name}")
    print("Nhìn theo hướng dẫn trên màn hình. Bấm Q để thoát.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        faces = app.get(frame)

        face = None
        if faces:
            face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))

        # progress
        current_count = sum(len(collected[k]) for k, _, _ in POSE_BINS)
        progress01 = current_count / float(total_need)
        frame = draw_progress_circle(frame, progress01)

        if face is None:
            frame = put_vietnamese_text(frame, "Không thấy khuôn mặt - Hãy đứng gần camera", (20, 20),
                                        font_size=28, color=(0, 0, 255), thickness=2)
        else:
            x1, y1, x2, y2 = face.bbox.astype(int)
            fw, fh = x2 - x1, y2 - y1
            det_score = float(getattr(face, "det_score", 1.0))

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            if fw < MIN_FACE_SIZE or fh < MIN_FACE_SIZE:
                frame = put_vietnamese_text(frame, "Lại gần camera hơn (mặt quá nhỏ)", (20, 20),
                                            font_size=28, color=(0, 0, 255), thickness=2)
            elif det_score < MIN_DET_SCORE:
                frame = put_vietnamese_text(frame, "Ảnh mờ / mặt chưa rõ (det_score thấp)", (20, 20),
                                            font_size=28, color=(0, 0, 255), thickness=2)
            elif not hasattr(face, "kps") or face.kps is None:
                frame = put_vietnamese_text(frame, "Không lấy được landmarks (kps)", (20, 20),
                                            font_size=28, color=(0, 0, 255), thickness=2)
            else:
                yaw_score, pitch_score = estimate_pose_bins_from_5pts(face.kps)

                matched_bin = None
                matched_hint = None
                for b, hint, rule in POSE_BINS:
                    if rule(yaw_score, pitch_score):
                        matched_bin = b
                        matched_hint = hint
                        break

                missing_bins = [b for b, _, _ in POSE_BINS if len(collected[b]) < NEED_PER_BIN]
                target_bin = missing_bins[0] if missing_bins else None
                target_hint = next((hint for b, hint, _ in POSE_BINS if b == target_bin), None)

                frame = put_vietnamese_text(frame, f"yaw={yaw_score:+.2f}  pitch={pitch_score:+.2f}",
                                            (20, 10), font_size=22, color=(255, 255, 255), thickness=1)

                progress_text = " | ".join([f"{b}:{len(collected[b])}/{NEED_PER_BIN}" for b, _, _ in POSE_BINS])
                frame = put_vietnamese_text(frame, f"Tiến độ: {progress_text}",
                                            (20, 40), font_size=22, color=(255, 255, 255), thickness=1)

                if target_hint:
                    frame = put_vietnamese_text(frame, f"Hãy: {target_hint}",
                                                (20, 70), font_size=28, color=(0, 255, 255), thickness=2)

                if matched_hint:
                    frame = put_vietnamese_text(frame, f"Bạn đang: {matched_hint}",
                                                (20, 105), font_size=24, color=(200, 255, 200), thickness=2)

                if target_bin and matched_bin == target_bin and len(collected[target_bin]) < NEED_PER_BIN:
                    now_t = cv2.getTickCount() / tick
                    if now_t - last_take_t[target_bin] >= TAKE_COOLDOWN:
                        emb = l2_normalize(face.embedding)
                        collected[target_bin].append(emb)
                        last_take_t[target_bin] = now_t

        # done
        current_count = sum(len(collected[k]) for k, _, _ in POSE_BINS)
        if current_count >= total_need:
            all_embs = []
            for b, _, _ in POSE_BINS:
                all_embs.extend(collected[b])

            mean_emb = l2_normalize(np.mean(np.stack(all_embs, axis=0), axis=0))

            out_path = DB_DIR / f"{person_id}.npz"
            np.savez_compressed(
                out_path,
                person_id=str(person_id),
                name=str(name),
                embedding=mean_emb.astype(np.float32),
                created_at=datetime.now().isoformat(timespec="seconds"),
                embeddings=np.stack(all_embs, axis=0).astype(np.float32),
            )

            frame = draw_progress_circle(frame, 1.0)
            frame = put_vietnamese_text(frame, "ENROLL 100% - ĐÃ LƯU db/", (20, 150),
                                        font_size=30, color=(0, 255, 0), thickness=2)
            cv2.imshow("Enroll (FaceID style)", frame)
            cv2.waitKey(900)
            break

        frame = put_vietnamese_text(frame, "Q=Thoát", (20, h - 40),
                                    font_size=24, color=(255, 255, 255), thickness=1)

        cv2.imshow("Enroll (FaceID style)", frame)
        if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    person_id = input("Nhap ma NV (ID): ").strip()
    name = input("Nhap ten: ").strip()
    cam_in = input("Nhap camera index (Enter=0): ").strip()
    cam = int(cam_in) if cam_in else 0

    if not person_id or not name:
        print("Thieu ID hoac Ten. Thoat.")
    else:
        main(person_id, name, cam)