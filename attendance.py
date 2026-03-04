import cv2
import csv
import numpy as np
from datetime import datetime
from pathlib import Path
from insightface.app import FaceAnalysis
from time import time
last_mark_time = {}   # person_id -> last timestamp seconds
COOLDOWN_SEC = 10     # ví dụ 10 giây

DB_DIR = Path("db")
CSV_PATH = Path("attendance.csv")

#Đọc data
def load_db():
    people = []
    for f in DB_DIR.glob("*.npz"):
        data = np.load(f, allow_pickle=True)
        people.append({
            "person_id": str(data["person_id"]),
            "name": str(data["name"]),
            "emb": data["embedding"].astype(np.float32),
        })
    return people

# Tính độ giống nhau giữa 2 khuôn mặt
def cosine(a, b):
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    return float(np.dot(a, b))

#Kiểm tra hôm nay đã chấm chưa
def marked_today(person_id, date_str):
    if not CSV_PATH.exists():
        return False
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            if row["person_id"] == person_id and row["date"] == date_str:
                return True
    return False

# Ghi log vào CSV
def mark(person_id, name, score):
    now = datetime.now()
    ts = now.isoformat(timespec="seconds")
    date_str = now.strftime("%Y-%m-%d")

    if marked_today(person_id, date_str):
        return False, "Da diem danh hom nay"

    new_file = not CSV_PATH.exists()
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["person_id", "name", "timestamp", "date", "score"])
        w.writerow([person_id, name, ts, date_str, f"{score:.3f}"])
    return True, "Diem danh OK"

def main():
    people = load_db()
    if not people:
        print("Chua co du lieu trong db/. Hay chay enroll.py truoc.")
        return

    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=0, det_size=(320, 320))
    # Mở camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Khong mo duoc camera")
        return

    # Ngưỡng cosine: 0.35~0.6 tùy dữ liệu; start 0.45
    TH = 0.45

    # msg, msg_time = "", 0
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

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
            print("so diem giong:",  best)

            msg = f"{best['name']}: {best_score:.2f}"

            sim_percent = max(0.0, min(100.0, best_score * 100.0))
            if best is not None and best_score >= TH:
                label = f"{best['name']} | ID: {best['person_id']} | {sim_percent:.1f}%"
                color = (0, 255, 0)

                # Ghi dữ liệu vào csv:
                # ok, m = mark(best["person_id"], best["name"], best_score)
                # if ok:
                #     msg, msg_time = f"{best['name']}: {m}", frame_idx
                # else:
                #     # chỉ hiện nhẹ, tránh spam
                #     msg, msg_time = f"{best['name']}: {m}", frame_idx

            else:
                # label = f"Unknown {best_score:.2f}"
                label = f"Unknown | {sim_percent:.1f}%"
                color = (0, 0, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # if msg and (frame_idx - msg_time) < 60:
        #     cv2.putText(frame, msg, (20, 40),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

        cv2.putText(frame, "Q=Quit | TH=0.45 (edit in code)", (20, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        cv2.imshow("Attendance (InsightFace)", frame)
        if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()