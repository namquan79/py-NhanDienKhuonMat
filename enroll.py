import cv2
import numpy as np
from pathlib import Path
from insightface.app import FaceAnalysis

DB_DIR = Path("db")
DB_DIR.mkdir(exist_ok=True)

def main():
    person_id = input("Nhap ID (vd: NV001): ").strip()
    name = input("Nhap Ten (vd: Nam Quan): ").strip()

    if not person_id:
        print("ID khong duoc rong")
        return

    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=0, det_size=(640, 640))

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Khong mo duoc camera")
        return

    embeddings = []
    target_samples = 20

    print("\nHuong dan:")
    print("- Nhin thang camera, doi anh sang tot")
    print("- Quay nhe trai/phai, len/xuong de lay nhieu goc")
    print("- Bam SPACE de chup, ESC de thoat\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = app.get(frame)
        # chọn mặt lớn nhất (đỡ bắt nhầm người phía sau)
        face = None
        if faces:
            face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
            x1, y1, x2, y2 = face.bbox.astype(int)
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

        cv2.putText(frame, f"Enroll: {person_id} - {name}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.putText(frame, f"Samples: {len(embeddings)}/{target_samples} (SPACE=cap, ESC=quit)", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        cv2.imshow("Enroll (InsightFace)", frame)
        k = cv2.waitKey(1) & 0xFF

        if k == 27:  # ESC
            break

        if k == 32:  # SPACE
            if face is None:
                print("Chua thay khuon mat - thu lai")
                continue
            # quality check nhẹ: score detection
            if getattr(face, "det_score", 1.0) < 0.6:
                print("Mat bi mo/khong ro - thu lai")
                continue

            embeddings.append(face.embedding.astype(np.float32))
            print(f"Captured {len(embeddings)}/{target_samples}")

            if len(embeddings) >= target_samples:
                break

    cap.release()
    cv2.destroyAllWindows()

    if len(embeddings) < 5:
        print("Qua it mau, huy enroll")
        return

    emb = np.mean(np.vstack(embeddings), axis=0).astype(np.float32)

    out_path = DB_DIR / f"{person_id}.npz"
    np.savez_compressed(out_path, person_id=person_id, name=name, embedding=emb)
    print(f"\nDa luu: {out_path}")

if __name__ == "__main__":
    main()