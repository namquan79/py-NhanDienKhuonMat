import sqlite3
import subprocess
import sys
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox


# =========================
# PATH CONFIG
# =========================
PROJECT_ROOT = Path(__file__).resolve().parent
DB_PATH = PROJECT_ROOT / "data" / "attendance.db"

# Các module hiện có trong project
SCRIPT_ENROLL_IMAGE = PROJECT_ROOT / "scripts" / "enroll_from_image_gui.py"
SCRIPT_ATTENDANCE = PROJECT_ROOT / "attendance.py"
SCRIPT_ENROLL_CAMERA = PROJECT_ROOT / "enroll_pose.py"


class HomePageApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Hệ thống điểm danh cuộc họp bằng khuôn mặt")
        self.root.geometry("1380x820")
        self.root.minsize(1280, 760)
        self.root.configure(bg="#eef3f8")

        self._build_style()
        self._build_ui()
        self.refresh_dashboard()

    # =========================
    # STYLE
    # =========================
    def _build_style(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass

        style.configure("Main.TFrame", background="#eef3f8")
        style.configure("Card.TFrame", background="white")
        style.configure("Header.TLabel", background="#173b63", foreground="white",
                        font=("Arial", 22, "bold"))
        style.configure("Section.TLabel", background="white", foreground="#1f2937",
                        font=("Arial", 15, "bold"))
        style.configure("Normal.TLabel", background="white", foreground="#334155",
                        font=("Arial", 11))
        style.configure("Muted.TLabel", background="white", foreground="#64748b",
                        font=("Arial", 10))
        style.configure("StatValue.TLabel", background="white", foreground="#0f172a",
                        font=("Arial", 22, "bold"))
        style.configure("StatTitle.TLabel", background="white", foreground="#475569",
                        font=("Arial", 11))
        style.configure("Treeview.Heading", font=("Arial", 10, "bold"))
        style.configure("Treeview", rowheight=28, font=("Arial", 10))

    # =========================
    # UI
    # =========================
    def _build_ui(self):
        # Header
        header = tk.Frame(self.root, bg="#173b63", height=82)
        header.pack(fill="x")
        header.pack_propagate(False)

        tk.Label(
            header,
            text="HỆ THỐNG ĐIỂM DANH CUỘC HỌP BẰNG NHẬN DIỆN KHUÔN MẶT",
            font=("Arial", 21, "bold"),
            fg="white",
            bg="#173b63"
        ).pack(side="left", padx=24, pady=20)

        right_header = tk.Frame(header, bg="#173b63")
        right_header.pack(side="right", padx=20)

        self.lbl_db_status = tk.Label(
            right_header,
            text="DB: Đang kiểm tra...",
            font=("Arial", 10, "bold"),
            fg="#dbeafe",
            bg="#173b63"
        )
        self.lbl_db_status.pack(anchor="e", pady=(16, 4))

        self.lbl_path = tk.Label(
            right_header,
            text=str(DB_PATH),
            font=("Arial", 9),
            fg="#cbd5e1",
            bg="#173b63"
        )
        self.lbl_path.pack(anchor="e")

        # Main body
        body = tk.Frame(self.root, bg="#eef3f8")
        body.pack(fill="both", expand=True, padx=18, pady=18)

        # Left side
        left = tk.Frame(body, bg="#eef3f8")
        left.pack(side="left", fill="both", expand=True, padx=(0, 10))

        # Right side
        right = tk.Frame(body, bg="#eef3f8", width=360)
        right.pack(side="right", fill="y")
        right.pack_propagate(False)

        self._build_top_stats(left)
        self._build_middle_tables(left)
        self._build_right_panel(right)

    def _build_top_stats(self, parent):
        wrap = tk.Frame(parent, bg="#eef3f8")
        wrap.pack(fill="x", pady=(0, 12))

        self.card_persons = self._create_stat_card(wrap, "Tổng nhân sự", "0", "#2563eb")
        self.card_persons.pack(side="left", fill="both", expand=True, padx=(0, 8))

        self.card_faces = self._create_stat_card(wrap, "Đã đăng ký khuôn mặt", "0", "#16a34a")
        self.card_faces.pack(side="left", fill="both", expand=True, padx=8)

        self.card_meetings = self._create_stat_card(wrap, "Tổng cuộc họp", "0", "#f59e0b")
        self.card_meetings.pack(side="left", fill="both", expand=True, padx=8)

        self.card_events = self._create_stat_card(wrap, "Lượt điểm danh", "0", "#7c3aed")
        self.card_events.pack(side="left", fill="both", expand=True, padx=(8, 0))

    def _create_stat_card(self, parent, title, value, color):
        card = tk.Frame(parent, bg="white", bd=1, relief="solid", height=120)
        card.pack_propagate(False)

        top_bar = tk.Frame(card, bg=color, height=6)
        top_bar.pack(fill="x")
        top_bar.pack_propagate(False)

        content = tk.Frame(card, bg="white")
        content.pack(fill="both", expand=True, padx=16, pady=14)

        lbl_title = tk.Label(content, text=title, font=("Arial", 11), fg="#475569", bg="white")
        lbl_title.pack(anchor="w")

        lbl_value = tk.Label(content, text=value, font=("Arial", 24, "bold"), fg="#0f172a", bg="white")
        lbl_value.pack(anchor="w", pady=(8, 0))

        setattr(card, "lbl_value", lbl_value)
        return card

    def _build_middle_tables(self, parent):
        wrap = tk.Frame(parent, bg="#eef3f8")
        wrap.pack(fill="both", expand=True)

        # Recent meetings
        meetings_card = tk.Frame(wrap, bg="white", bd=1, relief="solid")
        meetings_card.pack(fill="both", expand=True, pady=(0, 10))

        top1 = tk.Frame(meetings_card, bg="white")
        top1.pack(fill="x", padx=14, pady=(12, 8))
        tk.Label(top1, text="Cuộc họp gần đây", font=("Arial", 14, "bold"),
                 fg="#1f2937", bg="white").pack(side="left")

        self.tree_meetings = ttk.Treeview(
            meetings_card,
            columns=("meeting_id", "title", "start_time", "status"),
            show="headings",
            height=8
        )
        self.tree_meetings.heading("meeting_id", text="Mã cuộc họp")
        self.tree_meetings.heading("title", text="Tên cuộc họp")
        self.tree_meetings.heading("start_time", text="Thời gian")
        self.tree_meetings.heading("status", text="Trạng thái")

        self.tree_meetings.column("meeting_id", width=120, anchor="center")
        self.tree_meetings.column("title", width=360, anchor="w")
        self.tree_meetings.column("start_time", width=180, anchor="center")
        self.tree_meetings.column("status", width=120, anchor="center")

        self.tree_meetings.pack(fill="both", expand=True, padx=14, pady=(0, 14))

        # Recent attendance events
        events_card = tk.Frame(wrap, bg="white", bd=1, relief="solid")
        events_card.pack(fill="both", expand=True)

        top2 = tk.Frame(events_card, bg="white")
        top2.pack(fill="x", padx=14, pady=(12, 8))
        tk.Label(top2, text="Điểm danh gần đây", font=("Arial", 14, "bold"),
                 fg="#1f2937", bg="white").pack(side="left")

        self.tree_events = ttk.Treeview(
            events_card,
            columns=("person_id", "name", "meeting", "event_type", "ts"),
            show="headings",
            height=8
        )
        self.tree_events.heading("person_id", text="Mã NV")
        self.tree_events.heading("name", text="Họ tên")
        self.tree_events.heading("meeting", text="Cuộc họp")
        self.tree_events.heading("event_type", text="Loại")
        self.tree_events.heading("ts", text="Thời gian")

        self.tree_events.column("person_id", width=100, anchor="center")
        self.tree_events.column("name", width=180, anchor="w")
        self.tree_events.column("meeting", width=260, anchor="w")
        self.tree_events.column("event_type", width=100, anchor="center")
        self.tree_events.column("ts", width=170, anchor="center")

        self.tree_events.pack(fill="both", expand=True, padx=14, pady=(0, 14))

    def _build_right_panel(self, parent):
        # Quick actions
        action_card = tk.Frame(parent, bg="white", bd=1, relief="solid")
        action_card.pack(fill="x", pady=(0, 10))

        tk.Label(action_card, text="Thao tác nhanh", font=("Arial", 14, "bold"),
                 fg="#1f2937", bg="white").pack(anchor="w", padx=16, pady=(14, 12))

        btn_wrap = tk.Frame(action_card, bg="white")
        btn_wrap.pack(fill="x", padx=16, pady=(0, 16))

        self._create_action_button(btn_wrap, "📁 Đăng ký khuôn mặt từ ảnh", "#0d6efd",
                                   self.open_enroll_image).pack(fill="x", pady=5)
        self._create_action_button(btn_wrap, "📷 Đăng ký khuôn mặt từ camera", "#6f42c1",
                                   self.open_enroll_camera).pack(fill="x", pady=5)
        self._create_action_button(btn_wrap, "✅ Mở điểm danh realtime", "#198754",
                                   self.open_attendance).pack(fill="x", pady=5)
        self._create_action_button(btn_wrap, "🔄 Làm mới dữ liệu", "#6c757d",
                                   self.refresh_dashboard).pack(fill="x", pady=5)

        # Summary card
        summary_card = tk.Frame(parent, bg="white", bd=1, relief="solid")
        summary_card.pack(fill="x", pady=(0, 10))

        tk.Label(summary_card, text="Tóm tắt dữ liệu", font=("Arial", 14, "bold"),
                 fg="#1f2937", bg="white").pack(anchor="w", padx=16, pady=(14, 12))

        self.lbl_summary = tk.Label(
            summary_card,
            text="Đang tải dữ liệu...",
            justify="left",
            anchor="w",
            wraplength=310,
            font=("Arial", 10),
            fg="#374151",
            bg="white"
        )
        self.lbl_summary.pack(fill="x", padx=16, pady=(0, 16))

        # Logs card
        log_card = tk.Frame(parent, bg="white", bd=1, relief="solid")
        log_card.pack(fill="both", expand=True)

        tk.Label(log_card, text="Nhật ký hệ thống", font=("Arial", 14, "bold"),
                 fg="#1f2937", bg="white").pack(anchor="w", padx=16, pady=(14, 12))

        self.txt_log = tk.Text(
            log_card,
            font=("Consolas", 10),
            bg="#0f172a",
            fg="#e2e8f0",
            insertbackground="white",
            relief="flat",
            wrap="word"
        )
        self.txt_log.pack(fill="both", expand=True, padx=16, pady=(0, 16))
        self.txt_log.config(state="disabled")

        self.write_log("Khởi tạo màn hình trang chủ thành công.")
        self.write_log("Sẵn sàng đồng bộ dữ liệu từ SQLite.")

    def _create_action_button(self, parent, text, color, command):
        return tk.Button(
            parent,
            text=text,
            font=("Arial", 11, "bold"),
            bg=color,
            fg="white",
            activebackground=color,
            activeforeground="white",
            bd=0,
            padx=12,
            pady=12,
            command=command
        )

    # =========================
    # DB
    # =========================
    def get_connection(self):
        return sqlite3.connect(str(DB_PATH))

    def refresh_dashboard(self):
        if not DB_PATH.exists():
            self.lbl_db_status.config(text="DB: Không tìm thấy database", fg="#fecaca")
            self.write_log(f"Không tìm thấy database tại: {DB_PATH}")
            messagebox.showerror("Lỗi", f"Không tìm thấy database:\n{DB_PATH}")
            return

        self.lbl_db_status.config(text="DB: Kết nối thành công", fg="#bbf7d0")

        try:
            conn = self.get_connection()
            try:
                persons_count = conn.execute("SELECT COUNT(*) FROM persons").fetchone()[0]
                faces_count = conn.execute("SELECT COUNT(*) FROM face_embeddings").fetchone()[0]
                meetings_count = conn.execute("SELECT COUNT(*) FROM meetings").fetchone()[0]
                events_count = conn.execute("SELECT COUNT(*) FROM attendance_events").fetchone()[0]

                open_meetings = conn.execute(
                    "SELECT COUNT(*) FROM meetings WHERE status = 'OPEN'"
                ).fetchone()[0]

                attendees_count = conn.execute(
                    "SELECT COUNT(*) FROM meeting_attendees"
                ).fetchone()[0]

                # update stats
                self.card_persons.lbl_value.config(text=str(persons_count))
                self.card_faces.lbl_value.config(text=str(faces_count))
                self.card_meetings.lbl_value.config(text=str(meetings_count))
                self.card_events.lbl_value.config(text=str(events_count))

                # summary
                summary_text = (
                    f"- Tổng nhân sự trong hệ thống: {persons_count}\n"
                    f"- Số người đã đăng ký khuôn mặt: {faces_count}\n"
                    f"- Tổng cuộc họp: {meetings_count}\n"
                    f"- Cuộc họp đang mở: {open_meetings}\n"
                    f"- Tổng người tham gia các cuộc họp: {attendees_count}\n"
                    f"- Tổng log điểm danh: {events_count}"
                )
                self.lbl_summary.config(text=summary_text)

                # meetings table
                for item in self.tree_meetings.get_children():
                    self.tree_meetings.delete(item)

                meeting_rows = conn.execute("""
                    SELECT meeting_id, title, start_time, status
                    FROM meetings
                    ORDER BY start_time DESC
                    LIMIT 10
                """).fetchall()

                for row in meeting_rows:
                    self.tree_meetings.insert("", "end", values=row)

                # events table
                for item in self.tree_events.get_children():
                    self.tree_events.delete(item)

                event_rows = conn.execute("""
                    SELECT e.person_id,
                           COALESCE(p.name, e.person_id) AS name,
                           COALESCE(m.title, e.meeting_id) AS meeting_title,
                           e.event_type,
                           e.ts
                    FROM attendance_events e
                    LEFT JOIN persons p ON p.person_id = e.person_id
                    LEFT JOIN meetings m ON m.meeting_id = e.meeting_id
                    ORDER BY e.id DESC
                    LIMIT 12
                """).fetchall()

                for row in event_rows:
                    self.tree_events.insert("", "end", values=row)

            finally:
                conn.close()

            self.write_log("Làm mới dashboard thành công.")
            self.write_log("Đã tải dữ liệu từ persons, face_embeddings, meetings, meeting_attendees, attendance_events.")

        except Exception as e:
            self.write_log(f"Lỗi khi tải dashboard: {e}")
            messagebox.showerror("Lỗi", f"Không thể tải dữ liệu dashboard:\n{e}")

    # =========================
    # MODULE LAUNCHER
    # =========================
    def open_script(self, script_path: Path, module_name: str):
        if not script_path.exists():
            self.write_log(f"Không tìm thấy module: {script_path}")
            messagebox.showerror("Lỗi", f"Không tìm thấy file:\n{script_path}")
            return

        try:
            subprocess.Popen([sys.executable, str(script_path)], cwd=str(PROJECT_ROOT))
            self.write_log(f"Đã mở module: {module_name}")
        except Exception as e:
            self.write_log(f"Lỗi mở module {module_name}: {e}")
            messagebox.showerror("Lỗi", f"Không thể mở module {module_name}:\n{e}")

    def open_enroll_image(self):
        self.open_script(SCRIPT_ENROLL_IMAGE, "Đăng ký khuôn mặt từ ảnh")

    def open_enroll_camera(self):
        self.open_script(SCRIPT_ENROLL_CAMERA, "Đăng ký khuôn mặt từ camera")

    def open_attendance(self):
        self.open_script(SCRIPT_ATTENDANCE, "Điểm danh realtime")

    # =========================
    # LOG
    # =========================
    def write_log(self, text):
        self.txt_log.config(state="normal")
        self.txt_log.insert("end", f"- {text}\n")
        self.txt_log.see("end")
        self.txt_log.config(state="disabled")


if __name__ == "__main__":
    root = tk.Tk()
    app = HomePageApp(root)
    root.mainloop()