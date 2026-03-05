import cv2
import sqlite3
import pickle
import numpy as np
from pathlib import Path
from datetime import datetime
from insightface.app import FaceAnalysis

from utils.text_draw import put_vietnamese_text, draw_progress_circle
from utils.pose import POSE_BINS, estimate_pose_bins_from_5pts
from utils.face_db import find_duplicate_identity, load_db_identities, l2_normalize

# =========================
# CONFIG / STORAGE
# =========================
DB_DIR = Path("db")
DB_PATH = "data/attendance.db"
DB_DIR.mkdir(parents=True, exist_ok=True)

NEED_PER_BIN = 1

MIN_FACE_SIZE = 120
MIN_DET_SCORE = 0.60

TAKE_COOLDOWN = 0.08
DET_SIZE = (320, 320)

DUPLICATE_THRESHOLD = 0.60  # tăng 0.65 nếu báo nhầm nhiều


# ===== TOAST MESSAGE =====
toast_text = ""
toast_until = 0.0

def set_toast(text: str, seconds: float = 1.6):
    """Set thông báo nổi trong vài giây."""
    global toast_text, toast_until
    toast_text = text
    toast_until = cv2.getTickCount() / cv2.getTickFrequency() + seconds

def draw_toast(frame):
    """Vẽ toast lên frame nếu còn hạn."""
    now_t = cv2.getTickCount() / cv2.getTickFrequency()
    if not toast_text or now_t > toast_until:
        return frame

    h, w = frame.shape[:2]
    # nền tối (rectangle)
    cv2.rectangle(frame, (10, h - 90), (w - 10, h - 20), (0, 0, 0), -1)

    # vẽ text tiếng Việt bằng PIL util
    frame = put_vietnamese_text(
        frame,
        toast_text,
        (20, h - 82),
        font_size=26,
        color=(0, 255, 255),
        thickness=2
    )
    return frame

# =========================
# HELPERS (gọn code main)
# =========================
def pick_largest_face(faces):
    """Chọn khuôn mặt lớn nhất trong khung hình (ưu tiên mặt gần camera)."""
    if not faces:
        return None
    return max(
        faces,
        key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
    )


def draw_status_texts(frame, yaw, pitch, collected):
    """Vẽ thông tin yaw/pitch + tiến độ các pose + hướng dẫn/nhận dạng pose hiện tại."""
    frame = put_vietnamese_text(
        frame, f"yaw={yaw:+.2f}  pitch={pitch:+.2f}",
        (20, 10), font_size=22, color=(255, 255, 255), thickness=1
    )

    progress_text = " | ".join([f"{b}:{len(collected[b])}/{NEED_PER_BIN}" for b, _, _ in POSE_BINS])
    frame = put_vietnamese_text(
        frame, f"Tiến độ: {progress_text}",
        (20, 40), font_size=22, color=(255, 255, 255), thickness=1
    )
    return frame


def show_duplicate_dialog(frame, dup_pid, dup_name, dup_sim):
    """
    Hiển thị cảnh báo trùng khuôn mặt và chờ phím:
    - Y: vẫn lưu
    - R: enroll lại
    - Q/ESC: thoát
    Trả về một trong: "save" | "retry" | "quit"
    """
    frame = draw_progress_circle(frame, 1.0)
    frame = put_vietnamese_text(
        frame, "CẢNH BÁO: Có thể TRÙNG khuôn mặt!",
        (20, 140), font_size=30, color=(0, 0, 255), thickness=2
    )
    frame = put_vietnamese_text(
        frame, f"Trùng nhất: ID={dup_pid} | Tên={dup_name} | sim={dup_sim:.3f}",
        (20, 180), font_size=24, color=(255, 255, 255), thickness=2
    )
    frame = put_vietnamese_text(
        frame, "Y=Vẫn lưu | R=Enroll lại | Q=Thoát",
        (20, 220), font_size=24, color=(0, 255, 255), thickness=2
    )

    cv2.imshow("Enroll (FaceID style)", frame)

    while True:
        k = cv2.waitKey(0) & 0xFF
        if k in (ord("q"), ord("Q"), 27):
            return "quit"
        if k in (ord("r"), ord("R")):
            return "retry"
        if k in (ord("y"), ord("Y")):
            return "save"


# =========================
# MAIN FLOW
# =========================
def main(person_id: str, name: str, cam_index: int = 0):
    """
    Chức năng chính:
    1) Mở camera + detect khuôn mặt
    2) Ước lượng pose (FRONT/LEFT/RIGHT/UP/DOWN) từ 5 landmarks
    3) Thu thập embedding theo từng pose (mỗi pose NEED_PER_BIN mẫu)
    4) Khi đủ: tính embedding trung bình, kiểm tra trùng DB, rồi lưu db/{person_id}.npz
    """
    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=0, det_size=DET_SIZE)

    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print("Khong mo duoc camera")
        return

    collected = {k: [] for k, _, _ in POSE_BINS}
    last_take_t = {k: 0.0 for k, _, _ in POSE_BINS}
    tick = cv2.getTickFrequency()

    total_need = len(POSE_BINS) * NEED_PER_BIN

    print(f"[ENROLL] ID={person_id} | NAME={name}")
    print("Nhìn theo hướng dẫn trên màn hình. Bấm Q để thoát.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]

        # 1) Detect face
        faces = app.get(frame)
        face = pick_largest_face(faces)

        # 2) Progress UI
        current_count = sum(len(collected[k]) for k, _, _ in POSE_BINS)
        progress01 = current_count / float(total_need)
        frame = draw_progress_circle(frame, progress01)

        if face is None:
            # Không thấy mặt
            frame = put_vietnamese_text(
                frame, "Không thấy khuôn mặt - Hãy đứng gần camera",
                (20, 20), font_size=28, color=(0, 0, 255), thickness=2
            )
            set_toast("Thất bại: Không thấy khuôn mặt", 1.2)
        else:
            # 3) Kiểm tra chất lượng mặt + landmarks
            x1, y1, x2, y2 = face.bbox.astype(int)
            fw, fh = x2 - x1, y2 - y1
            det_score = float(getattr(face, "det_score", 1.0))

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            if fw < MIN_FACE_SIZE or fh < MIN_FACE_SIZE:
                frame = put_vietnamese_text(
                    frame, "Lại gần camera hơn (mặt quá nhỏ)",
                    (20, 20), font_size=28, color=(0, 0, 255), thickness=2
                )
                set_toast("Thất bại: Mặt quá nhỏ - lại gần camera", 1.2)

            elif det_score < MIN_DET_SCORE:
                frame = put_vietnamese_text(
                    frame, "Ảnh mờ / mặt chưa rõ (det_score thấp)",
                    (20, 20), font_size=28, color=(0, 0, 255), thickness=2
                )
                set_toast("Thất bại: Ảnh mờ / det_score thấp", 1.2)

            elif not hasattr(face, "kps") or face.kps is None:
                frame = put_vietnamese_text(
                    frame, "Không lấy được landmarks (kps)",
                    (20, 20), font_size=28, color=(0, 0, 255), thickness=2
                )
                set_toast("Thất bại: Không lấy được landmarks", 1.2)
            else:
                # 4) Pose estimation + hướng dẫn pose còn thiếu
                yaw_score, pitch_score = estimate_pose_bins_from_5pts(face.kps)

                matched_bin, matched_hint = None, None
                for b, hint, rule in POSE_BINS:
                    if rule(yaw_score, pitch_score):
                        matched_bin, matched_hint = b, hint
                        break

                missing_bins = [b for b, _, _ in POSE_BINS if len(collected[b]) < NEED_PER_BIN]
                target_bin = missing_bins[0] if missing_bins else None
                target_hint = next((hint for b, hint, _ in POSE_BINS if b == target_bin), None)

                frame = draw_status_texts(frame, yaw_score, pitch_score, collected)

                if target_hint:
                    frame = put_vietnamese_text(
                        frame, f"Hãy: {target_hint}",
                        (20, 70), font_size=28, color=(0, 255, 255), thickness=2
                    )
                if matched_hint:
                    frame = put_vietnamese_text(
                        frame, f"Bạn đang: {matched_hint}",
                        (20, 105), font_size=24, color=(200, 255, 200), thickness=2
                    )

                # 5) Auto-take embedding khi đúng pose mục tiêu + cooldown
                if target_bin and matched_bin == target_bin and len(collected[target_bin]) < NEED_PER_BIN:
                    now_t = cv2.getTickCount() / tick
                    if now_t - last_take_t[target_bin] >= TAKE_COOLDOWN:
                        collected[target_bin].append(l2_normalize(face.embedding))
                        last_take_t[target_bin] = now_t
                        set_toast(f"Thành công: Đã lấy mẫu {target_bin}", 1.2)

        # 6) Nếu đủ pose -> tính mean embedding -> check trùng -> lưu
        current_count = sum(len(collected[k]) for k, _, _ in POSE_BINS)
        if current_count >= total_need:
            all_embs = []
            for b, _, _ in POSE_BINS:
                all_embs.extend(collected[b])

            mean_emb = l2_normalize(np.mean(np.stack(all_embs, axis=0), axis=0))

            # Check duplicate with existing DB
            db_items = load_db_identities(DB_PATH, exclude_person_id=person_id)
            is_dup, dup_pid, dup_name, dup_sim = find_duplicate_identity(
                mean_emb, db_items, threshold=DUPLICATE_THRESHOLD
            )

            if is_dup:
                set_toast(f"Cảnh báo: Có thể trùng với {dup_name} (sim={dup_sim:.2f})", 2.5)
                action = show_duplicate_dialog(frame.copy(), dup_pid, dup_name, dup_sim)
                if action == "quit":
                    cap.release()
                    cv2.destroyAllWindows()
                    return
                if action == "retry":
                    collected = {k2: [] for k2, _, _ in POSE_BINS}
                    last_take_t = {k2: 0.0 for k2, _, _ in POSE_BINS}
                    continue
                # action == "save" -> tiếp tục lưu

            # ===== SAVE TO SQLITE =====
            conn = sqlite3.connect("data/attendance.db")

            embedding_blob = pickle.dumps(mean_emb)

            conn.execute("""
            INSERT OR REPLACE INTO face_embeddings (person_id, embedding)
            VALUES (?, ?)
            """, (person_id, embedding_blob))

            conn.commit()
            set_toast("Thành công: Đã lưu dữ liệu khuôn mặt", 2.0)
            conn.close()
            # ===== END SAVE TO SQLITE =====

            frame = draw_progress_circle(frame, 1.0)
            frame = put_vietnamese_text(
                frame, "ENROLL 100% - ĐÃ LƯU db/",
                (20, 150), font_size=30, color=(0, 255, 0), thickness=2
            )
            frame = draw_toast(frame)
            cv2.imshow("Enroll (FaceID style)", frame)
            cv2.waitKey(900)
            break

        # Footer + exit
        frame = put_vietnamese_text(
            frame, "Q=Thoát",
            (20, h - 40), font_size=24, color=(255, 255, 255), thickness=1
        )

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