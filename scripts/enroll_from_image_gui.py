import sqlite3
import pickle
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path

import cv2
import numpy as np
from insightface.app import FaceAnalysis
from PIL import Image, ImageTk


MATCH_THRESHOLD = 0.45

# =========================
# CẤU HÌNH ĐƯỜNG DẪN DATABASE
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_ROOT / "data" / "attendance.db"

if not DB_PATH.exists():
    raise FileNotFoundError(f"Không tìm thấy database: {DB_PATH}")

# =========================
# HÀM TIỆN ÍCH
# =========================
def imread_unicode(path: str):
    """
    Đọc ảnh bằng OpenCV theo cách hỗ trợ đường dẫn có dấu / Unicode.
    Trả về ảnh dạng BGR hoặc None nếu đọc thất bại.
    """
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def l2_normalize(vec):
    """
    Chuẩn hóa vector về độ dài 1 để so sánh cosine similarity ổn định hơn.
    """
    vec = np.asarray(vec, dtype=np.float32).reshape(-1)
    norm = np.linalg.norm(vec)
    if norm < 1e-12:
        return vec
    return vec / norm


# =========================
# GIAO DIỆN ĐĂNG KÝ KHUÔN MẶT TỪ ẢNH
# =========================
class EnrollFromImageApp:
    def __init__(self, root):
        """
        Khởi tạo ứng dụng:
        - thiết lập cửa sổ
        - khởi tạo model InsightFace
        - khởi tạo dữ liệu trạng thái
        - dựng giao diện
        """
        self.root = root
        self.root.title("Đăng ký khuôn mặt từ ảnh")
        self.root.geometry("1240x760")
        self.root.minsize(1180, 720)
        self.root.configure(bg="#eef2f7")

        # Dữ liệu trạng thái hiện tại
        self.image_path = None
        self.original_image = None
        self.embedding = None
        self.detected_face = None
        self.matched_person = None

        # Ảnh preview cho Tkinter
        self.preview_original_tk = None
        self.preview_detected_tk = None

        # Model nhận diện
        self.face_app = FaceAnalysis(name="buffalo_l")
        self.face_app.prepare(ctx_id=0, det_size=(320, 320))

        self._build_style()
        self._build_ui()
        self._bind_events()
        self._update_save_button_state()

    # =========================
    # STYLE GIAO DIỆN
    # =========================
    def _build_style(self):
        """
        Cấu hình style cho các widget ttk.
        """
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass

        style.configure(
            "Header.TLabel",
            background="#164c7e",
            foreground="white",
            font=("Arial", 22, "bold")
        )
        style.configure(
            "SectionTitle.TLabel",
            background="white",
            foreground="#1f2937",
            font=("Arial", 15, "bold")
        )
        style.configure(
            "Label.TLabel",
            background="white",
            foreground="#222222",
            font=("Arial", 11)
        )
        style.configure(
            "StatusBlue.TLabel",
            background="white",
            foreground="#0d6efd",
            font=("Arial", 11, "bold")
        )
        style.configure(
            "StatusGreen.TLabel",
            background="white",
            foreground="#198754",
            font=("Arial", 11, "bold")
        )
        style.configure(
            "StatusRed.TLabel",
            background="white",
            foreground="#dc3545",
            font=("Arial", 11, "bold")
        )
        style.configure(
            "Small.TLabel",
            background="white",
            foreground="#6b7280",
            font=("Arial", 10)
        )

    # =========================
    # DỰNG GIAO DIỆN TỔNG THỂ
    # =========================
    def _build_ui(self):
        """
        Tạo giao diện chính gồm:
        - header
        - panel trái xem trước ảnh
        - panel phải nhập thông tin, thao tác, nhật ký
        """
        header = tk.Frame(self.root, bg="#164c7e", height=78)
        header.pack(fill="x")
        header.pack_propagate(False)

        ttk.Label(
            header,
            text="HỆ THỐNG ĐĂNG KÝ KHUÔN MẶT TỪ ẢNH",
            style="Header.TLabel"
        ).pack(expand=True)

        main = tk.Frame(self.root, bg="#eef2f7")
        main.pack(fill="both", expand=True, padx=18, pady=18)

        self.left_card = tk.Frame(main, bg="white", bd=1, relief="solid")
        self.left_card.pack(side="left", fill="both", expand=True, padx=(0, 10))

        self.right_card = tk.Frame(main, bg="white", bd=1, relief="solid", width=420)
        self.right_card.pack(side="right", fill="y")
        self.right_card.pack_propagate(False)

        self._build_left_panel()
        self._build_right_panel()

    # =========================
    # PANEL TRÁI: ẢNH GỐC / ẢNH PHÁT HIỆN
    # =========================
    def _build_left_panel(self):
        """
        Tạo khu vực xem trước ảnh và hiển thị khuôn mặt phát hiện.
        """
        top = tk.Frame(self.left_card, bg="white")
        top.pack(fill="x", padx=18, pady=(16, 10))

        ttk.Label(
            top,
            text="Xem trước và phân tích ảnh",
            style="SectionTitle.TLabel"
        ).pack(anchor="w")

        meta = tk.Frame(self.left_card, bg="white")
        meta.pack(fill="x", padx=18, pady=(0, 8))

        self.lbl_path = ttk.Label(
            meta,
            text="Đường dẫn: Chưa chọn ảnh",
            style="Small.TLabel"
        )
        self.lbl_path.pack(anchor="w", pady=2)

        self.lbl_info = ttk.Label(
            meta,
            text="Kích thước ảnh: - | Số khuôn mặt: -",
            style="Small.TLabel"
        )
        self.lbl_info.pack(anchor="w", pady=2)

        preview_container = tk.Frame(self.left_card, bg="white")
        preview_container.pack(fill="both", expand=True, padx=18, pady=(6, 18))

        left_preview = tk.Frame(preview_container, bg="white")
        left_preview.pack(side="left", fill="both", expand=True, padx=(0, 8))

        ttk.Label(
            left_preview,
            text="Ảnh gốc",
            style="SectionTitle.TLabel"
        ).pack(anchor="w", pady=(0, 8))

        self.original_preview = tk.Label(
            left_preview,
            text="Chưa có ảnh",
            font=("Arial", 12),
            bg="#f3f4f6",
            fg="#6b7280",
            bd=1,
            relief="flat"
        )
        self.original_preview.pack(fill="both", expand=True)

        right_preview = tk.Frame(preview_container, bg="white")
        right_preview.pack(side="left", fill="both", expand=True, padx=(8, 0))

        ttk.Label(
            right_preview,
            text="Khuôn mặt phát hiện",
            style="SectionTitle.TLabel"
        ).pack(anchor="w", pady=(0, 8))

        self.detected_preview = tk.Label(
            right_preview,
            text="Chưa phân tích",
            font=("Arial", 12),
            bg="#f3f4f6",
            fg="#6b7280",
            bd=1,
            relief="flat"
        )
        self.detected_preview.pack(fill="both", expand=True)

    # =========================
    # PANEL PHẢI: FORM + NÚT + NHẬT KÝ
    # =========================
    def _build_right_panel(self):
        """
        Tạo panel phải:
        - trạng thái
        - form nhập liệu
        - nút chọn ảnh / lưu / làm mới
        - hướng dẫn
        - nhật ký thao tác
        """
        body = tk.Frame(self.right_card, bg="white")
        body.pack(fill="both", expand=True, padx=18, pady=16)

        ttk.Label(
            body,
            text="Thông tin nhân viên",
            style="SectionTitle.TLabel"
        ).pack(anchor="w", pady=(0, 8))

        self.status_var = tk.StringVar(value="Trạng thái: Chưa chọn ảnh")
        self.status_label = ttk.Label(
            body,
            textvariable=self.status_var,
            style="StatusBlue.TLabel"
        )
        self.status_label.pack(anchor="w", pady=(0, 16))

        form = tk.Frame(body, bg="white")
        form.pack(fill="x")

        ttk.Label(form, text="Mã nhân viên", style="Label.TLabel").pack(anchor="w", pady=(0, 6))
        self.entry_person_id = tk.Entry(
            form,
            font=("Arial", 12),
            relief="flat",
            highlightthickness=1,
            highlightbackground="#cfd4dc",
            highlightcolor="#0d6efd",
            bg="white"
        )
        self.entry_person_id.pack(fill="x", ipady=9, pady=(0, 14))

        ttk.Label(form, text="Họ và tên", style="Label.TLabel").pack(anchor="w", pady=(0, 6))
        self.entry_name = tk.Entry(
            form,
            font=("Arial", 12),
            relief="flat",
            highlightthickness=1,
            highlightbackground="#cfd4dc",
            highlightcolor="#0d6efd",
            bg="white"
        )
        self.entry_name.pack(fill="x", ipady=9, pady=(0, 14))

        ttk.Label(form, text="Phòng ban", style="Label.TLabel").pack(anchor="w", pady=(0, 6))
        self.entry_department = tk.Entry(
            form,
            font=("Arial", 12),
            relief="flat",
            highlightthickness=1,
            highlightbackground="#cfd4dc",
            highlightcolor="#0d6efd",
            bg="white"
        )
        self.entry_department.pack(fill="x", ipady=9, pady=(0, 18))

        # Hàng nút thao tác
        btn_row = tk.Frame(body, bg="white")
        btn_row.pack(fill="x", pady=(2, 16))

        self.btn_select = tk.Button(
            btn_row,
            text="📁  Chọn ảnh",
            font=("Arial", 11, "bold"),
            bg="#0d6efd",
            fg="white",
            activebackground="#0b5ed7",
            activeforeground="white",
            bd=0,
            padx=12,
            pady=11,
            command=self.select_image
        )
        self.btn_select.pack(side="left", fill="x", expand=True, padx=(0, 6))

        self.btn_save = tk.Button(
            btn_row,
            text="💾  Lưu dữ liệu",
            font=("Arial", 11, "bold"),
            bg="#198754",
            fg="white",
            activebackground="#157347",
            activeforeground="white",
            disabledforeground="white",
            bd=0,
            padx=12,
            pady=11,
            state="disabled",
            command=self.save_data
        )
        self.btn_save.pack(side="left", fill="x", expand=True, padx=6)

        self.btn_reset = tk.Button(
            btn_row,
            text="🔄  Làm mới",
            font=("Arial", 11, "bold"),
            bg="#6c757d",
            fg="white",
            activebackground="#5c636a",
            activeforeground="white",
            bd=0,
            padx=12,
            pady=11,
            command=self.reset_form
        )
        self.btn_reset.pack(side="left", fill="x", expand=True, padx=(6, 0))

        # Hướng dẫn
        info_box = tk.Frame(body, bg="#f8fafc", bd=1, relief="solid")
        info_box.pack(fill="x", pady=(0, 14))

        ttk.Label(
            info_box,
            text="Hướng dẫn sử dụng",
            style="SectionTitle.TLabel"
        ).pack(anchor="w", padx=12, pady=(10, 6))

        guide_text = (
            "- Chọn ảnh có khuôn mặt rõ, đủ sáng, ưu tiên chính diện.\n"
            "- Sau khi chọn ảnh, hệ thống sẽ tự động phân tích và trích xuất embedding.\n"
            "- Hệ thống tự kiểm tra xem khuôn mặt đã có trong CSDL hay chưa.\n"
            "- Nếu tìm thấy, form sẽ tự động điền thông tin nhân viên.\n"
            "- Khi đủ điều kiện, nút “Lưu dữ liệu” sẽ được kích hoạt.\n"
            "- Bấm “Lưu dữ liệu” để ghi vào bảng persons và face_embeddings."
        )

        tk.Label(
            info_box,
            text=guide_text,
            justify="left",
            anchor="w",
            font=("Arial", 10),
            bg="#f8fafc",
            fg="#374151",
            wraplength=350
        ).pack(fill="x", padx=12, pady=(0, 12))

        # Nhật ký thao tác
        log_frame = tk.Frame(body, bg="white")
        log_frame.pack(fill="both", expand=True)

        ttk.Label(
            log_frame,
            text="Nhật ký thao tác",
            style="SectionTitle.TLabel"
        ).pack(anchor="w", pady=(0, 8))

        self.log_text = tk.Text(
            log_frame,
            height=8,
            font=("Consolas", 10),
            bg="#0f172a",
            fg="#e2e8f0",
            insertbackground="white",
            relief="flat",
            wrap="word"
        )
        self.log_text.pack(fill="both", expand=True)
        self.log_text.config(state="disabled")

        self.write_log("Khởi tạo giao diện thành công.")
        self.write_log("Sẵn sàng chọn ảnh để đăng ký khuôn mặt.")

    # =========================
    # BIND SỰ KIỆN NHẬP LIỆU
    # =========================
    def _bind_events(self):
        """
        Gắn sự kiện gõ phím vào các ô nhập để cập nhật trạng thái nút lưu.
        """
        self.entry_person_id.bind("<KeyRelease>", self.validate_form)
        self.entry_name.bind("<KeyRelease>", self.validate_form)
        self.entry_department.bind("<KeyRelease>", self.validate_form)

    # =========================
    # CẬP NHẬT TRẠNG THÁI
    # =========================
    def set_status(self, text, mode="blue"):
        """
        Cập nhật dòng trạng thái:
        - blue: thông tin chung
        - green: thành công
        - red: lỗi
        """
        self.status_var.set(f"Trạng thái: {text}")
        if mode == "green":
            self.status_label.configure(style="StatusGreen.TLabel")
        elif mode == "red":
            self.status_label.configure(style="StatusRed.TLabel")
        else:
            self.status_label.configure(style="StatusBlue.TLabel")

    # =========================
    # GHI NHẬT KÝ THAO TÁC
    # =========================
    def write_log(self, text):
        """
        Ghi 1 dòng nhật ký vào ô log.
        """
        self.log_text.config(state="normal")
        self.log_text.insert("end", f"- {text}\n")
        self.log_text.see("end")
        self.log_text.config(state="disabled")

    # =========================
    # KIỂM TRA MÃ NHÂN VIÊN ĐÃ TỒN TẠI
    # =========================
    def person_exists(self, person_id):
        """
        Kiểm tra nhân viên đã tồn tại trong bảng persons hay chưa.
        """
        conn = sqlite3.connect(str(DB_PATH))
        try:
            row = conn.execute(
                "SELECT person_id FROM persons WHERE person_id = ?",
                (person_id,)
            ).fetchone()
            return row is not None
        finally:
            conn.close()

    # =========================
    # XÓA / ĐỔ DỮ LIỆU FORM
    # =========================
    def clear_person_form(self):
        """
        Xóa dữ liệu trong các ô thông tin nhân viên.
        """
        self.entry_person_id.delete(0, tk.END)
        self.entry_name.delete(0, tk.END)
        self.entry_department.delete(0, tk.END)

    def fill_person_form(self, person_id="", name="", department=""):
        """
        Tự động điền thông tin nhân viên vào form.
        """
        self.clear_person_form()
        self.entry_person_id.insert(0, person_id)
        self.entry_name.insert(0, name)
        self.entry_department.insert(0, department)

    # =========================
    # HIỂN THỊ ẢNH LÊN KHUNG PREVIEW
    # =========================
    def show_image_on_label(self, label_widget, img_bgr, max_w=420, max_h=500, is_original=True):
        """
        Resize ảnh theo tỷ lệ, căn giữa vào khung preview rồi hiển thị trên Label.
        """
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]

        scale = min(max_w / w, max_h / h)
        scale = min(scale, 1.0)

        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(img_rgb, (new_w, new_h))

        canvas = np.full((max_h, max_w, 3), 245, dtype=np.uint8)
        y_off = (max_h - new_h) // 2
        x_off = (max_w - new_w) // 2
        canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized

        pil_img = Image.fromarray(canvas)
        imgtk = ImageTk.PhotoImage(pil_img)

        label_widget.config(image=imgtk, text="")
        if is_original:
            self.preview_original_tk = imgtk
        else:
            self.preview_detected_tk = imgtk

    # =========================
    # TẠO ẢNH DEBUG CHO KHUÔN MẶT PHÁT HIỆN
    # =========================
    def create_face_debug_preview(self, img_bgr, face):
        """
        Tạo ảnh preview khuôn mặt phát hiện:
        - phóng to khuôn mặt
        - hiển thị bbox
        - hiển thị detection score
        - vẽ 5 landmark
        - bỏ embedding dim / norm / vector chart
        """
        x1, y1, x2, y2 = face.bbox.astype(int)

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img_bgr.shape[1], x2)
        y2 = min(img_bgr.shape[0], y2)

        # Nới rộng vùng crop để mặt nhìn tự nhiên hơn
        face_w = x2 - x1
        face_h = y2 - y1
        pad_x = int(face_w * 0.22)
        pad_y = int(face_h * 0.18)

        cx1 = max(0, x1 - pad_x)
        cy1 = max(0, y1 - pad_y)
        cx2 = min(img_bgr.shape[1], x2 + pad_x)
        cy2 = min(img_bgr.shape[0], y2 + pad_y)

        crop = img_bgr[cy1:cy2, cx1:cx2].copy()
        if crop.size == 0:
            crop = np.full((340, 260, 3), 30, dtype=np.uint8)

        crop_h, crop_w = crop.shape[:2]

        # Phóng to ảnh khuôn mặt hơn trước
        target_face_h = 360
        scale = target_face_h / max(1, crop_h)
        target_face_w = max(260, int(crop_w * scale))
        crop = cv2.resize(crop, (target_face_w, target_face_h))

        panel_w = max(460, target_face_w + 30)
        panel_h = 500
        canvas = np.full((panel_h, panel_w, 3), 18, dtype=np.uint8)

        # Đặt ảnh mặt
        x_off = (panel_w - target_face_w) // 2
        y_off = 14
        canvas[y_off:y_off + target_face_h, x_off:x_off + target_face_w] = crop

        # Khung viền vàng
        cv2.rectangle(
            canvas,
            (x_off, y_off),
            (x_off + target_face_w, y_off + target_face_h),
            (0, 255, 255),
            2
        )

        # Landmark nếu có
        if hasattr(face, "kps") and face.kps is not None:
            kps = np.asarray(face.kps, dtype=np.float32)
            for px, py in kps:
                # Quy đổi từ ảnh gốc -> ảnh crop đã resize
                px2 = int((px - cx1) * scale + x_off)
                py2 = int((py - cy1) * scale + y_off)
                cv2.circle(canvas, (px2, py2), 4, (0, 255, 0), -1)

        det_score = float(getattr(face, "det_score", 0.0))

        # Chỉ giữ thông tin ngắn gọn bên dưới
        info_y = y_off + target_face_h + 28
        text_color = (230, 230, 230)
        title_color = (0, 220, 255)

        cv2.putText(
            canvas,
            "FACE ANALYSIS",
            (18, info_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.78,
            title_color,
            2
        )
        info_y += 32

        cv2.putText(
            canvas,
            f"BBox: ({x1},{y1}) - ({x2},{y2})",
            (18, info_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            text_color,
            1
        )
        info_y += 28

        cv2.putText(
            canvas,
            f"Det score: {det_score:.4f}",
            (18, info_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            text_color,
            1
        )

        return canvas
    # =========================
    # ĐỌC TẤT CẢ DỮ LIỆU KHUÔN MẶT TỪ DB
    # =========================
    def fetch_registered_faces(self):
        """
        Lấy toàn bộ person + embedding từ CSDL để phục vụ so khớp.
        """
        conn = sqlite3.connect(str(DB_PATH))
        try:
            rows = conn.execute("""
                SELECT p.person_id, p.name, p.department, f.embedding
                FROM face_embeddings f
                JOIN persons p ON p.person_id = f.person_id
            """).fetchall()

            results = []
            for person_id, name, department, emb_blob in rows:
                try:
                    db_emb = pickle.loads(emb_blob)
                    db_emb = l2_normalize(np.asarray(db_emb, dtype=np.float32))
                    results.append({
                        "person_id": person_id,
                        "name": name,
                        "department": department,
                        "embedding": db_emb
                    })
                except Exception as e:
                    self.write_log(f"Bỏ qua 1 embedding lỗi của person_id={person_id}: {e}")
            return results
        finally:
            conn.close()

    # =========================
    # SO KHỚP KHUÔN MẶT VỚI CSDL
    # =========================
    def find_best_match(self, query_embedding, threshold=MATCH_THRESHOLD):
        """
        So khớp embedding hiện tại với dữ liệu trong DB bằng cosine similarity.
        Trả về:
        - dict thông tin người khớp nhất nếu vượt ngưỡng
        - None nếu chưa có trong dữ liệu
        """
        records = self.fetch_registered_faces()
        if not records:
            self.write_log("CSDL khuôn mặt đang trống hoặc chưa có embedding nào.")
            return None

        query_embedding = l2_normalize(query_embedding)

        best_score = -1.0
        best_record = None

        for rec in records:
            score = float(np.dot(query_embedding, rec["embedding"]))
            if score > best_score:
                best_score = score
                best_record = rec

        if best_record is not None:
            best_record = dict(best_record)
            best_record["score"] = best_score
            self.write_log(
                f"So khớp tốt nhất: {best_record['person_id']} - "
                f"{best_record['name']} | cosine={best_score:.4f}"
            )
            if best_score >= threshold:
                return best_record

        return None

    # =========================
    # BẬT / TẮT NÚT LƯU
    # =========================
    def _update_save_button_state(self):
        """
        Bật nút Lưu dữ liệu khi đủ điều kiện:
        - đã có embedding
        - có Mã NV
        - có Họ tên
        - có Phòng ban
        """
        person_id = self.entry_person_id.get().strip()
        name = self.entry_name.get().strip()
        department = self.entry_department.get().strip()

        should_enable = (
            self.embedding is not None
            and person_id != ""
            and name != ""
            and department != ""
        )

        if should_enable:
            self.btn_save.config(state="normal")
        else:
            self.btn_save.config(state="disabled")

    # =========================
    # KIỂM TRA FORM SAU MỖI LẦN NHẬP
    # =========================
    def validate_form(self, event=None):
        """
        Kiểm tra dữ liệu nhập và cập nhật trạng thái nút lưu.
        """
        self._update_save_button_state()

    # =========================
    # CHỌN ẢNH VÀ TỰ ĐỘNG PHÂN TÍCH
    # =========================
    def select_image(self):
        """
        Mở hộp thoại chọn ảnh.
        Sau khi chọn ảnh thành công:
        - hiển thị ảnh gốc
        - tự động phân tích ảnh
        """
        file_path = filedialog.askopenfilename(
            title="Chọn ảnh khuôn mặt",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )

        if not file_path:
            self.write_log("Người dùng đã hủy chọn ảnh.")
            return

        img = imread_unicode(file_path)
        if img is None:
            self.set_status("Không đọc được ảnh", "red")
            self.write_log(f"Lỗi đọc ảnh: {file_path}")
            messagebox.showerror("Lỗi", "Không đọc được ảnh đã chọn.")
            return

        self.image_path = file_path
        self.original_image = img
        self.embedding = None
        self.detected_face = None
        self.matched_person = None

        self.clear_person_form()

        self.show_image_on_label(self.original_preview, img, is_original=True)
        self.detected_preview.config(image="", text="Đang phân tích...", bg="#f3f4f6", fg="#6b7280")

        h, w = img.shape[:2]
        self.lbl_path.config(text=f"Đường dẫn: {file_path}")
        self.lbl_info.config(text=f"Kích thước ảnh: {w} x {h} | Số khuôn mặt: Đang phân tích")

        self.set_status("Đã chọn ảnh, đang tự động phân tích...", "blue")
        self.write_log(f"Đã chọn ảnh: {file_path}")
        self.write_log(f"Kích thước ảnh: {w} x {h}")

        self._update_save_button_state()
        self.detect_face()

    # =========================
    # PHÂN TÍCH ẢNH, TRÍCH XUẤT EMBEDDING
    # =========================
    def detect_face(self):
        """
        Phân tích ảnh đã chọn:
        - phát hiện khuôn mặt
        - chọn khuôn mặt lớn nhất
        - trích xuất embedding
        - hiển thị khuôn mặt đã detect
        - so khớp với CSDL
        """
        if self.original_image is None:
            self.set_status("Chưa chọn ảnh", "red")
            self.write_log("Phân tích thất bại: chưa chọn ảnh.")
            messagebox.showwarning("Thông báo", "Vui lòng chọn ảnh trước.")
            return

        faces = self.face_app.get(self.original_image)

        if not faces:
            self.embedding = None
            self.detected_face = None
            self.matched_person = None
            self.set_status("Không phát hiện khuôn mặt", "red")
            self.lbl_info.config(
                text=f"Kích thước ảnh: {self.original_image.shape[1]} x {self.original_image.shape[0]} | Số khuôn mặt: 0"
            )
            self.write_log("Không phát hiện khuôn mặt trong ảnh.")
            messagebox.showwarning("Thông báo", "Không phát hiện khuôn mặt trong ảnh.")
            self._update_save_button_state()
            return

        self.lbl_info.config(
            text=f"Kích thước ảnh: {self.original_image.shape[1]} x {self.original_image.shape[0]} | Số khuôn mặt: {len(faces)}"
        )

        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        self.detected_face = face
        self.embedding = np.asarray(face.embedding, dtype=np.float32).reshape(-1)

        x1, y1, x2, y2 = face.bbox.astype(int)

        # preview hiển thị "xịn hơn"
        debug_preview = self.create_face_debug_preview(self.original_image, face)
        self.show_image_on_label(
            self.detected_preview,
            debug_preview,
            max_w=460,
            max_h=540,
            is_original=False
        )

        self.write_log(f"Phát hiện {len(faces)} khuôn mặt. Chọn khuôn mặt lớn nhất.")
        self.write_log(f"BBox khuôn mặt: ({x1}, {y1}) - ({x2}, {y2})")
        self.write_log("Đã trích xuất embedding thành công.")

        # tự động so khớp với DB
        matched = self.find_best_match(self.embedding)
        self.matched_person = matched

        if matched is not None:
            self.fill_person_form(
                person_id=matched["person_id"],
                name=matched["name"],
                department=matched["department"]
            )
            self.set_status(
                f"Đã nhận diện trùng dữ liệu: "
                f"\n{matched['person_id']} | {matched['name']} (score={matched['score']*100:.2f}%)",
                "red"
            )
            self.write_log(
                f"Tìm thấy khuôn mặt đã tồn tại trong CSDL -> "
                f"\n{matched['person_id']} | {matched['name']} | {matched['department']}"
            )
        else:
            self.set_status("Phân tích ảnh thành công, chưa thấy trùng trong CSDL", "green")
            self.write_log("Không tìm thấy khuôn mặt trùng trong CSDL.")
            self.write_log("Vui lòng nhập thông tin nhân viên để lưu mới.")

        self.entry_person_id.focus_set()
        self._update_save_button_state()

    # =========================
    # LƯU DỮ LIỆU VÀO DATABASE
    # =========================
    def save_data(self):
        """
        Lưu dữ liệu vào:
        - bảng persons
        - bảng face_embeddings

        Điều kiện:
        - đã có embedding
        - đã nhập đủ Mã NV, Họ tên, Phòng ban

        Sau khi lưu thành công:
        - hiện thông báo
        - tự động đóng cửa sổ
        """
        if self.embedding is None:
            self.set_status("Chưa phân tích khuôn mặt", "red")
            self.write_log("Lưu thất bại: chưa có embedding.")
            messagebox.showwarning("Thông báo", "Vui lòng chọn ảnh hợp lệ trước khi lưu.")
            return

        person_id = self.entry_person_id.get().strip()
        name = self.entry_name.get().strip()
        department = self.entry_department.get().strip()

        if person_id == "" or name == "" or department == "":
            self.set_status("Thiếu thông tin bắt buộc", "red")
            self.write_log("Lưu thất bại: thiếu Mã NV / Họ tên / Phòng ban.")
            messagebox.showwarning(
                "Thiếu thông tin",
                "Vui lòng nhập đầy đủ các trường:\n- Mã nhân viên\n- Họ tên\n- Phòng ban"
            )
            return

        if self.person_exists(person_id):
            ask = messagebox.askyesno(
                "Xác nhận cập nhật",
                f"Mã nhân viên '{person_id}' đã tồn tại.\nBạn có muốn cập nhật lại thông tin và khuôn mặt không?"
            )
            if not ask:
                self.write_log(f"Người dùng hủy cập nhật nhân viên: {person_id}")
                return

        try:
            conn = sqlite3.connect(str(DB_PATH))
            try:
                conn.execute("""
                    INSERT OR REPLACE INTO persons (person_id, name, department)
                    VALUES (?, ?, ?)
                """, (person_id, name, department))

                emb_blob = pickle.dumps(np.asarray(self.embedding, dtype=np.float32))
                conn.execute("""
                    INSERT OR REPLACE INTO face_embeddings (person_id, embedding)
                    VALUES (?, ?)
                """, (person_id, emb_blob))

                conn.commit()
            finally:
                conn.close()

            self.set_status(f"Đã lưu thành công nhân viên {name}", "green")
            self.write_log(f"Lưu thành công nhân viên: {person_id} - {name}")
            self.write_log("Dữ liệu đã ghi vào persons và face_embeddings.")

            messagebox.showinfo(
                "Thành công",
                f"Đã lưu dữ liệu khuôn mặt thành công.\n\n"
                f"Mã NV: {person_id}\n"
                f"Họ tên: {name}\n"
                f"Phòng ban: {department}"
            )

            self.root.after(300, self.root.destroy)

        except Exception as e:
            self.set_status("Lưu dữ liệu thất bại", "red")
            self.write_log(f"Lỗi khi lưu dữ liệu: {e}")
            messagebox.showerror("Lỗi", f"Lưu dữ liệu thất bại:\n{e}")

    # =========================
    # LÀM MỚI TOÀN BỘ FORM
    # =========================
    def reset_form(self):
        """
        Reset toàn bộ giao diện về trạng thái ban đầu.
        """
        self.image_path = None
        self.original_image = None
        self.embedding = None
        self.detected_face = None
        self.matched_person = None

        self.preview_original_tk = None
        self.preview_detected_tk = None

        self.original_preview.config(image="", text="Chưa có ảnh", bg="#f3f4f6", fg="#6b7280")
        self.detected_preview.config(image="", text="Chưa phân tích", bg="#f3f4f6", fg="#6b7280")

        self.clear_person_form()

        self.lbl_path.config(text="Đường dẫn: Chưa chọn ảnh")
        self.lbl_info.config(text="Kích thước ảnh: - | Số khuôn mặt: -")

        self.set_status("Chưa chọn ảnh", "blue")
        self.write_log("Đã làm mới biểu mẫu.")

        self._update_save_button_state()


# =========================
# CHẠY CHƯƠNG TRÌNH
# =========================
if __name__ == "__main__":
    root = tk.Tk()
    app = EnrollFromImageApp(root)
    root.mainloop()