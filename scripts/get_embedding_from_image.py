import cv2
import numpy as np
from insightface.app import FaceAnalysis
from tkinter import Tk, filedialog

def imread_unicode(path: str):
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)

def main():
    Tk().withdraw()
    img_path = filedialog.askopenfilename(
        title="Chọn ảnh khuôn mặt",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
    )
    if not img_path:
        print("Không chọn ảnh")
        return

    print("Ảnh đã chọn:", img_path)

    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=0, det_size=(320, 320))

    img = imread_unicode(img_path)
    if img is None:
        print("Không đọc được ảnh:", img_path)
        return

    faces = app.get(img)
    if not faces:
        print("Không phát hiện khuôn mặt trong ảnh.")
        return

    face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
    emb = face.embedding.astype(np.float32)

    print("Embedding shape:", emb.shape)
    print("Embedding (first 10):", emb[:10])

if __name__ == "__main__":
    main()