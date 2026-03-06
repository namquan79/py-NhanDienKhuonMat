import cv2
import sqlite3
import pickle
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
from insightface.app import FaceAnalysis

from utils.text_draw import put_vietnamese_text, draw_progress_circle
from utils.pose import POSE_BINS, estimate_pose_bins_from_5pts
from utils.face_db import find_duplicate_identity, load_db_identities, l2_normalize


# =========================
# CẤU HÌNH HỆ THỐNG
# =========================
PROJECT_ROOT = Path(__file__).resolve().parent
DB_PATH = PROJECT_ROOT / "data" / "attendance.db"

NEED_PER_BIN = 1
MIN_FACE_SIZE = 120
MIN_DET_SCORE = 0.60
TAKE_COOLDOWN = 0.08
DET_SIZE = (320, 320)
DUPLICATE_THRESHOLD = 0.60


# =========================
# TOAST MESSAGE TRÊN KHUNG CAMERA
# =========================
toast_text = ""
toast_until = 0.0


def set_toast(text: str, seconds: float = 1.6):
    """
    Đặt thông báo tạm thời hiển thị trên khung camera.
    """
    global toast_text, toast_until
    toast_text = text
    toast_until = cv2.getTickCount() / cv2.getTickFrequency() + seconds


def draw_toast(frame):
    """
    Vẽ thông báo toast lên frame nếu còn thời gian hiển thị.
    """
    now_t = cv2.getTickCount() / cv2.getTickFrequency()
    if not toast_text or now_t > toast_until:
        return frame

    h, w = frame.shape[:2]
    cv2.rectangle(frame, (10, h - 90), (w - 10, h - 20), (0, 0, 0), -1)

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
# HÀM HỖ TRỢ XỬ LÝ KHUÔN MẶT
# =========================
def pick_largest_face(faces):
    """
    Chọn khuôn mặt lớn nhất trong khung hình.
    Ưu tiên mặt gần camera để ổn định khi enroll.
    """
    if not faces:
        return None

    return max(
        faces,
        key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
    )


def draw_status_texts(frame, yaw, pitch, collected):
    """
    Hiển thị thông tin yaw/pitch và tiến độ thu mẫu khuôn mặt.
    """
    frame = put_vietnamese_text(
        frame,
        f"yaw={yaw:+.2f}  pitch={pitch:+.2f}",
        (20, 10),
        font_size=22,
        color=(255, 255, 255),
        thickness=1
    )

    progress_text = " | ".join(
        [f"{b}:{len(collected[b])}/{NEED_PER_BIN}" for b, _, _ in POSE_BINS]
    )
    frame = put_vietnamese_text(
        frame,
        f"Tiến độ: {progress_text}",
        (20, 40),
        font_size=22,
        color=(255, 255, 255),
        thickness=1
    )
    return frame


def show_duplicate_dialog(frame, dup_pid, dup_name, dup_sim):
    """
    Hiển thị cảnh báo trùng khuôn mặt.
    Người dùng có thể chọn:
    - Y: vẫn lưu
    - R: enroll lại
    - Q / ESC: thoát
    """
    frame = draw_progress_circle(frame, 1.0)
    frame = put_vietnamese_text(
        frame,
        "CẢNH BÁO: Có thể TRÙNG khuôn mặt!",
        (20, 140),
        font_size=30,
        color=(0, 0, 255),
        thickness=2
    )
    frame = put_vietnamese_text(
        frame,
        f"Trùng nhất: ID={dup_pid} | Tên={dup_name} | sim={dup_sim:.3f}",
        (20, 180),
        font_size=24,
        color=(255, 255, 255),
        thickness=2
    )
    frame = put_vietnamese_text(
        frame,
        "Y=Vẫn lưu | R=Enroll lại | Q=Thoát",
        (20, 220),
        font_size=24,
        color=(0, 255, 255),
        thickness=2
    )

    cv2.imshow("Enroll (FaceID style)", frame)

    while True:
        key = cv2.waitKey(0) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            return "quit"
        if key in (ord("r"), ord("R")):
            return "retry"
        if key in (ord("y"), ord("Y")):
            return "save"


def save_person_and_embedding(person_id: str, name: str, department: str, mean_emb: np.ndarray):
    """
    Lưu thông tin nhân viên và embedding vào SQLite.
    - persons
    - face_embeddings
    """
    conn = sqlite3.connect(str(DB_PATH))
    try:
        conn.execute("""
            INSERT OR REPLACE INTO persons (person_id, name, department)
            VALUES (?, ?, ?)
        """, (person_id, name, department))

        embedding_blob = pickle.dumps(mean_emb.astype(np.float32))
        conn.execute("""
            INSERT OR REPLACE INTO face_embeddings (person_id, embedding)
            VALUES (?, ?)
        """, (person_id, embedding_blob))

        conn.commit()
    finally:
        conn.close()


# =========================
# LUỒNG ENROLL TỪ CAMERA
# =========================
def run_enroll(person_id: str, name: str, department: str, cam_index: int = 0):
    """
    Luồng đăng ký khuôn mặt:
    1. Mở camera
    2. Phát hiện khuôn mặt
    3. Hướng dẫn người dùng quay theo các pose
    4. Thu embedding theo nhiều góc
    5. Tính mean embedding
    6. Kiểm tra trùng khuôn mặt
    7. Lưu dữ liệu vào SQLite

    Trả về:
    - True nếu đăng ký thành công
    - False nếu thất bại / thoát giữa chừng
    """
    face_app = FaceAnalysis(name="buffalo_l")
    face_app.prepare(ctx_id=0, det_size=DET_SIZE)

    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        messagebox.showerror("Lỗi", f"Không mở được camera {cam_index}")
        return False

    collected = {key: [] for key, _, _ in POSE_BINS}
    last_take_t = {key: 0.0 for key, _, _ in POSE_BINS}
    tick = cv2.getTickFrequency()
    total_need = len(POSE_BINS) * NEED_PER_BIN

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        faces = face_app.get(frame)
        face = pick_largest_face(faces)

        # Vòng tròn tiến độ
        current_count = sum(len(collected[key]) for key, _, _ in POSE_BINS)
        progress01 = current_count / float(total_need)
        frame = draw_progress_circle(frame, progress01)

        # Thông tin người đang đăng ký
        frame = put_vietnamese_text(
            frame,
            f"ID: {person_id} | Tên: {name}",
            (20, h - 75),
            font_size=22,
            color=(255, 255, 255),
            thickness=1
        )

        if face is None:
            frame = put_vietnamese_text(
                frame,
                "Không thấy khuôn mặt - Hãy đứng gần camera",
                (20, 20),
                font_size=28,
                color=(0, 0, 255),
                thickness=2
            )
            set_toast("Thất bại: Không thấy khuôn mặt", 1.2)
        else:
            x1, y1, x2, y2 = face.bbox.astype(int)
            face_w, face_h = x2 - x1, y2 - y1
            det_score = float(getattr(face, "det_score", 1.0))

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            if face_w < MIN_FACE_SIZE or face_h < MIN_FACE_SIZE:
                frame = put_vietnamese_text(
                    frame,
                    "Lại gần camera hơn (mặt quá nhỏ)",
                    (20, 20),
                    font_size=28,
                    color=(0, 0, 255),
                    thickness=2
                )
                set_toast("Thất bại: Mặt quá nhỏ - lại gần camera", 1.2)

            elif det_score < MIN_DET_SCORE:
                frame = put_vietnamese_text(
                    frame,
                    "Ảnh mờ / mặt chưa rõ (det_score thấp)",
                    (20, 20),
                    font_size=28,
                    color=(0, 0, 255),
                    thickness=2
                )
                set_toast("Thất bại: Ảnh mờ / det_score thấp", 1.2)

            elif not hasattr(face, "kps") or face.kps is None:
                frame = put_vietnamese_text(
                    frame,
                    "Không lấy được landmarks (kps)",
                    (20, 20),
                    font_size=28,
                    color=(0, 0, 255),
                    thickness=2
                )
                set_toast("Thất bại: Không lấy được landmarks", 1.2)

            else:
                yaw_score, pitch_score = estimate_pose_bins_from_5pts(face.kps)

                matched_bin = None
                matched_hint = None
                for pose_key, hint, rule in POSE_BINS:
                    if rule(yaw_score, pitch_score):
                        matched_bin = pose_key
                        matched_hint = hint
                        break

                missing_bins = [
                    pose_key for pose_key, _, _ in POSE_BINS
                    if len(collected[pose_key]) < NEED_PER_BIN
                ]
                target_bin = missing_bins[0] if missing_bins else None
                target_hint = next(
                    (hint for pose_key, hint, _ in POSE_BINS if pose_key == target_bin),
                    None
                )

                frame = draw_status_texts(frame, yaw_score, pitch_score, collected)

                if target_hint:
                    frame = put_vietnamese_text(
                        frame,
                        f"Hãy: {target_hint}",
                        (20, 70),
                        font_size=28,
                        color=(0, 255, 255),
                        thickness=2
                    )

                if matched_hint:
                    frame = put_vietnamese_text(
                        frame,
                        f"Bạn đang: {matched_hint}",
                        (20, 105),
                        font_size=24,
                        color=(200, 255, 200),
                        thickness=2
                    )

                # Thu mẫu đúng pose
                if target_bin and matched_bin == target_bin and len(collected[target_bin]) < NEED_PER_BIN:
                    now_t = cv2.getTickCount() / tick
                    if now_t - last_take_t[target_bin] >= TAKE_COOLDOWN:
                        collected[target_bin].append(l2_normalize(face.embedding))
                        last_take_t[target_bin] = now_t
                        set_toast(f"Thành công: Đã lấy mẫu {target_bin}", 1.2)

        # Kiểm tra đã đủ mẫu chưa
        current_count = sum(len(collected[key]) for key, _, _ in POSE_BINS)
        if current_count >= total_need:
            all_embs = []
            for pose_key, _, _ in POSE_BINS:
                all_embs.extend(collected[pose_key])

            mean_emb = l2_normalize(np.mean(np.stack(all_embs, axis=0), axis=0))

            # Kiểm tra trùng khuôn mặt
            db_items = load_db_identities(str(DB_PATH), exclude_person_id=person_id)
            is_dup, dup_pid, dup_name, dup_sim = find_duplicate_identity(
                mean_emb, db_items, threshold=DUPLICATE_THRESHOLD
            )

            if is_dup:
                set_toast(f"Cảnh báo: Có thể trùng với {dup_name} (sim={dup_sim:.2f})", 2.5)
                action = show_duplicate_dialog(frame.copy(), dup_pid, dup_name, dup_sim)

                if action == "quit":
                    cap.release()
                    cv2.destroyAllWindows()
                    return False

                if action == "retry":
                    collected = {key: [] for key, _, _ in POSE_BINS}
                    last_take_t = {key: 0.0 for key, _, _ in POSE_BINS}
                    continue

            # Lưu dữ liệu
            save_person_and_embedding(person_id, name, department, mean_emb)
            set_toast("Thành công: Đã lưu dữ liệu khuôn mặt", 2.0)

            frame = draw_progress_circle(frame, 1.0)
            frame = put_vietnamese_text(
                frame,
                "ENROLL 100% - ĐÃ LƯU DATABASE",
                (20, 150),
                font_size=30,
                color=(0, 255, 0),
                thickness=2
            )
            frame = draw_toast(frame)
            cv2.imshow("Enroll (FaceID style)", frame)
            cv2.waitKey(800)

            cap.release()
            cv2.destroyAllWindows()
            return True

        # Footer
        frame = put_vietnamese_text(
            frame,
            "Q=Thoát",
            (20, h - 40),
            font_size=24,
            color=(255, 255, 255),
            thickness=1
        )

        frame = draw_toast(frame)
        cv2.imshow("Enroll (FaceID style)", frame)

        if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
            break

    cap.release()
    cv2.destroyAllWindows()
    return False


# =========================
# FORM NHẬP THÔNG TIN ENROLL
# =========================
class EnrollForm:
    def __init__(self, root):
        """
        Giao diện form nhập:
        - Mã nhân viên
        - Họ tên
        - Phòng ban
        - Camera

        Nút Bắt đầu đăng ký chỉ được bật khi nhập đủ 3 trường bắt buộc.
        """
        self.root = root
        self.root.title("Đăng ký khuôn mặt từ camera")
        self.root.geometry("620x720")
        self.root.minsize(560, 620)
        self.root.configure(bg="#eef3f8")

        self.build_ui()
        self.bind_events()
        self.update_start_button_state()

    # =========================
    # DỰNG GIAO DIỆN FORM
    # =========================
    def build_ui(self):
        header = tk.Frame(self.root, bg="#173b63", height=86)
        header.pack(fill="x")
        header.pack_propagate(False)

        tk.Label(
            header,
            text="ĐĂNG KÝ KHUÔN MẶT TỪ CAMERA",
            font=("Arial", 20, "bold"),
            fg="white",
            bg="#173b63"
        ).pack(expand=True)

        outer = tk.Frame(self.root, bg="#eef3f8")
        outer.pack(fill="both", expand=True, padx=18, pady=18)

        canvas = tk.Canvas(outer, bg="#eef3f8", highlightthickness=0)
        scrollbar = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        self.scrollable_frame = tk.Frame(canvas, bg="white", bd=1, relief="solid")

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        def on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        canvas.bind_all("<MouseWheel>", on_mousewheel)

        body = self.scrollable_frame

        tk.Label(
            body,
            text="Thông tin đăng ký",
            font=("Arial", 15, "bold"),
            fg="#1f2937",
            bg="white"
        ).pack(anchor="w", padx=20, pady=(20, 15))

        form = tk.Frame(body, bg="white")
        form.pack(fill="x", padx=20)

        # Mã nhân viên
        tk.Label(form, text="Mã nhân viên", font=("Arial", 11), bg="white").pack(anchor="w")
        self.entry_person_id = tk.Entry(
            form,
            font=("Arial", 12),
            relief="flat",
            highlightthickness=1,
            highlightbackground="#cfd4dc",
            highlightcolor="#0d6efd"
        )
        self.entry_person_id.pack(fill="x", ipady=8, pady=(5, 14))

        # Họ tên
        tk.Label(form, text="Họ và tên", font=("Arial", 11), bg="white").pack(anchor="w")
        self.entry_name = tk.Entry(
            form,
            font=("Arial", 12),
            relief="flat",
            highlightthickness=1,
            highlightbackground="#cfd4dc",
            highlightcolor="#0d6efd"
        )
        self.entry_name.pack(fill="x", ipady=8, pady=(5, 14))

        # Phòng ban
        tk.Label(form, text="Phòng ban", font=("Arial", 11), bg="white").pack(anchor="w")
        self.entry_department = tk.Entry(
            form,
            font=("Arial", 12),
            relief="flat",
            highlightthickness=1,
            highlightbackground="#cfd4dc",
            highlightcolor="#0d6efd"
        )
        self.entry_department.pack(fill="x", ipady=8, pady=(5, 14))

        # Camera
        tk.Label(form, text="Camera", font=("Arial", 11), bg="white").pack(anchor="w")
        self.camera_var = tk.StringVar(value="0")
        self.combo_camera = ttk.Combobox(
            form,
            textvariable=self.camera_var,
            values=["0", "1", "2", "3"],
            font=("Arial", 11),
            state="readonly"
        )
        self.combo_camera.pack(fill="x", ipady=5, pady=(5, 20))

        # Hướng dẫn
        info_text = (
            "Hướng dẫn:\n"
            "- Nhập đầy đủ Mã NV, Họ tên, Phòng ban.\n"
            "- Chọn camera cần sử dụng.\n"
            "- Khi nhập đủ thông tin, nút 'Bắt đầu đăng ký' sẽ được kích hoạt.\n"
            "- Hệ thống sẽ mở camera để thu mẫu khuôn mặt theo nhiều góc.\n"
            "- Sau khi hoàn tất, dữ liệu sẽ được lưu vào persons và face_embeddings."
        )

        tk.Label(
            body,
            text=info_text,
            justify="left",
            font=("Arial", 10),
            fg="#475569",
            bg="#f8fafc",
            bd=1,
            relief="solid",
            padx=12,
            pady=12,
            wraplength=520
        ).pack(fill="x", padx=20, pady=(0, 18))

        # Nút thao tác
        btn_frame = tk.Frame(body, bg="white")
        btn_frame.pack(fill="x", padx=20, pady=(0, 20))

        self.btn_start = tk.Button(
            btn_frame,
            text="Bắt đầu đăng ký",
            font=("Arial", 11, "bold"),
            bg="#198754",
            fg="white",
            activebackground="#157347",
            activeforeground="white",
            disabledforeground="white",
            bd=0,
            padx=14,
            pady=11,
            state="disabled",
            command=self.start_enroll
        )
        self.btn_start.pack(side="left", fill="x", expand=True, padx=(0, 6))

        tk.Button(
            btn_frame,
            text="Đóng",
            font=("Arial", 11, "bold"),
            bg="#6c757d",
            fg="white",
            activebackground="#5c636a",
            activeforeground="white",
            bd=0,
            padx=14,
            pady=11,
            command=self.root.destroy
        ).pack(side="left", fill="x", expand=True, padx=(6, 0))

    # =========================
    # GẮN SỰ KIỆN CHO Ô NHẬP
    # =========================
    def bind_events(self):
        """
        Theo dõi thay đổi trên các ô nhập để bật/tắt nút Bắt đầu đăng ký.
        """
        self.entry_person_id.bind("<KeyRelease>", self.validate_form)
        self.entry_name.bind("<KeyRelease>", self.validate_form)
        self.entry_department.bind("<KeyRelease>", self.validate_form)

    # =========================
    # CẬP NHẬT TRẠNG THÁI NÚT BẮT ĐẦU
    # =========================
    def update_start_button_state(self):
        """
        Bật nút Bắt đầu đăng ký khi nhập đủ:
        - Mã nhân viên
        - Họ tên
        - Phòng ban
        """
        person_id = self.entry_person_id.get().strip()
        name = self.entry_name.get().strip()
        department = self.entry_department.get().strip()

        if person_id and name and department:
            self.btn_start.config(state="normal")
        else:
            self.btn_start.config(state="disabled")

    # =========================
    # KIỂM TRA FORM KHI NHẬP
    # =========================
    def validate_form(self, event=None):
        """
        Gọi lại kiểm tra trạng thái nút sau mỗi lần gõ phím.
        """
        self.update_start_button_state()

    # =========================
    # BẮT ĐẦU ENROLL
    # =========================
    def start_enroll(self):
        """
        Lấy dữ liệu từ form, kiểm tra hợp lệ và bắt đầu luồng enroll.
        """
        person_id = self.entry_person_id.get().strip()
        name = self.entry_name.get().strip()
        department = self.entry_department.get().strip()
        cam_index = self.camera_var.get().strip()

        if not person_id or not name or not department:
            messagebox.showwarning(
                "Thiếu thông tin",
                "Vui lòng nhập đầy đủ:\n- Mã nhân viên\n- Họ tên\n- Phòng ban"
            )
            return

        try:
            cam_index = int(cam_index)
        except ValueError:
            messagebox.showerror("Lỗi", "Camera phải là số nguyên.")
            return

        self.root.withdraw()
        try:
            result = run_enroll(person_id, name, department, cam_index)
            if result:
                self.root.destroy()
            else:
                self.root.deiconify()
        except Exception as exc:
            self.root.deiconify()
            messagebox.showerror("Lỗi", f"Không thể đăng ký khuôn mặt:\n{exc}")


# =========================
# CHẠY CHƯƠNG TRÌNH
# =========================
if __name__ == "__main__":
    root = tk.Tk()
    app = EnrollForm(root)
    root.mainloop()