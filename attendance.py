import cv2
import sqlite3
import pickle
import numpy as np
from datetime import datetime
from insightface.app import FaceAnalysis
from time import time

from utils.text_draw import put_vietnamese_text

# ===== CONFIG =====
DB_PATH = "data/attendance.db"
CAM_INDEX = 0
DET_SIZE = (320, 320)

TH = 0.60          # ngưỡng cosine
COOLDOWN_SEC = 10  # chống spam điểm danh 1 người liên tục
last_mark_time = {}  # person_id -> last timestamp seconds

msg_text = ""
msg_until = 0.0


# ===== DATA LOAD =====
def load_people_from_sqlite(db_path=DB_PATH):
    """
    Load (person_id, name, embedding) từ SQLite.
    - face_embeddings.embedding: BLOB pickle(np.ndarray)
    - persons.name: lấy tên hiển thị (nếu có)
    """
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute("""
            SELECT fe.person_id,
                   COALESCE(p.name, fe.person_id) AS name,
                   fe.embedding
            FROM face_embeddings fe
            LEFT JOIN persons p ON p.person_id = fe.person_id
        """).fetchall()

        people = []
        for person_id, name, emb_blob in rows:
            if emb_blob is None:
                continue
            emb = pickle.loads(emb_blob)
            emb = np.asarray(emb, dtype=np.float32)
            people.append({
                "person_id": str(person_id),
                "name": str(name),
                "emb": emb,
            })
        return people
    finally:
        conn.close()


# ===== MATH =====
def cosine(a, b):
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    return float(np.dot(a, b))


# ===== TOAST MESSAGE =====
def set_msg(text, seconds=2):
    global msg_text, msg_until
    msg_text = text
    msg_until = time() + seconds


def draw_msg(frame):
    """Vẽ toast tiếng Việt bằng put_vietnamese_text"""
    if time() > msg_until or not msg_text:
        return frame

    h, w = frame.shape[:2]
    cv2.rectangle(frame, (10, 10), (w - 10, 70), (0, 0, 0), -1)
    frame = put_vietnamese_text(
        frame,
        msg_text,
        (20, 20),
        font_size=26,
        color=(0, 255, 255),
        thickness=2
    )
    return frame


def can_mark_now(person_id: str) -> bool:
    now = time()
    last = last_mark_time.get(person_id, 0.0)
    if now - last < COOLDOWN_SEC:
        return False
    last_mark_time[person_id] = now
    return True


# ===== OPTIONAL: Ghi log vào SQLite (chuẩn hệ thống cuộc họp) =====
def mark_sqlite(meeting_id: str, person_id: str, score: float, camera_id="cam0", db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("""
            INSERT INTO attendance_events(meeting_id, person_id, event_type, ts, confidence, camera_id)
            VALUES (?, ?, 'checkin', ?, ?, ?)
        """, (meeting_id, person_id, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), float(score), camera_id))
        conn.commit()
    finally:
        conn.close()


def main():
    people = load_people_from_sqlite(DB_PATH)
    if not people:
        print("Chưa có embedding trong SQLite. Hãy enroll trước (lưu vào face_embeddings).")
        return

    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=0, det_size=DET_SIZE)

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print("Không mở được camera")
        return

    MEETING_ID = "mt001"  # đổi theo cuộc họp đang mở

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = app.get(frame)

        for face in faces:
            x1, y1, x2, y2 = face.bbox.astype(int)
            emb = face.embedding.astype(np.float32)

            best = None
            best_score = -1.0

            for p in people:
                s = cosine(emb, p["emb"])
                if s > best_score:
                    best_score = s
                    best = p

            sim_percent = max(0.0, min(100.0, best_score * 100.0))

            if best is not None and best_score >= TH:
                label = f"{best['name']} | ID: {best['person_id']} | {sim_percent:.1f}%"
                color = (0, 255, 0)

                # thông báo thành công (TV)
                set_msg(f"Thành công: {best['name']} điểm danh OK", 2)

                if can_mark_now(best["person_id"]):
                    # Bật nếu muốn ghi log meeting
                    # mark_sqlite(MEETING_ID, best["person_id"], best_score, camera_id=f"cam{CAM_INDEX}")
                    pass
            else:
                label = f"Không xác định | {sim_percent:.1f}%"
                color = (0, 0, 255)

                # tránh spam toast liên tục: chỉ báo khi gần ngưỡng
                if best_score >= TH - 0.05:
                    set_msg("Thất bại: Độ giống chưa đủ ngưỡng", 1.2)

            # Vẽ khung
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Vẽ label tiếng Việt (thay cv2.putText)
            frame = put_vietnamese_text(
                frame,
                label,
                (x1, max(10, y1 - 32)),
                font_size=24,
                color=color,
                thickness=2
            )

        # Footer tiếng Việt
        frame = put_vietnamese_text(
            frame,
            f"Q=Thoát | Ngưỡng={TH} | Cooldown={COOLDOWN_SEC}s",
            (20, frame.shape[0] - 45),
            font_size=22,
            color=(255, 255, 255),
            thickness=1
        )

        # Toast
        frame = draw_msg(frame)

        cv2.imshow("Diem danh (InsightFace)", frame)
        if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()