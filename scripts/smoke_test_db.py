from pathlib import Path
import sqlite3
from datetime import datetime

DB_PATH = Path("data/attendance.db")

def main():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    try:
        con.execute("PRAGMA foreign_keys=ON;")

        # Insert sample person (upsert)
        con.execute("""
            INSERT INTO persons(person_id, name, department)
            VALUES (?, ?, ?)
            ON CONFLICT(person_id) DO UPDATE SET
              name=excluded.name,
              department=excluded.department
        """, ("nv001", "Nguyễn Văn A", "Phòng CNTT"))

        # Insert sample meeting (upsert)
        meeting_id = "mt001"
        con.execute("""
            INSERT INTO meetings(meeting_id, title, start_time, late_after_minutes, location, status)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(meeting_id) DO UPDATE SET
              title=excluded.title,
              start_time=excluded.start_time,
              late_after_minutes=excluded.late_after_minutes,
              location=excluded.location,
              status=excluded.status
        """, (meeting_id, "Họp giao ban", datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 10, "Phòng họp 1", "OPEN"))

        # Add attendee
        con.execute("""
            INSERT OR IGNORE INTO meeting_attendees(meeting_id, person_id, role)
            VALUES (?, ?, ?)
        """, (meeting_id, "nv001", "attendee"))

        # Add attendance event
        con.execute("""
            INSERT INTO attendance_events(meeting_id, person_id, event_type, confidence, camera_id)
            VALUES (?, ?, ?, ?, ?)
        """, (meeting_id, "nv001", "checkin", 0.78, "cam0"))

        con.commit()

        # Query report
        rows = con.execute("""
            SELECT p.person_id, p.name, m.title, e.event_type, e.ts, e.confidence
            FROM attendance_events e
            JOIN persons p ON p.person_id = e.person_id
            JOIN meetings m ON m.meeting_id = e.meeting_id
            ORDER BY e.id DESC
            LIMIT 10
        """).fetchall()

        print("=== Latest events ===")
        for r in rows:
            print(dict(r))

        print("[OK] Smoke test done.")
    finally:
        con.close()

if __name__ == "__main__":
    main()