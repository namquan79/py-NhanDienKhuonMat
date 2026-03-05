from pathlib import Path
import sqlite3

DB_PATH = Path("data/attendance.db")
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

-- 1) Danh sách nhân sự / người tham dự
CREATE TABLE IF NOT EXISTS persons (
  person_id   TEXT PRIMARY KEY,
  name        TEXT NOT NULL,
  department  TEXT,
  created_at  TEXT NOT NULL DEFAULT (datetime('now','localtime'))
);

-- tạo bảng face_embeddings
CREATE TABLE IF NOT EXISTS face_embeddings (
  person_id TEXT PRIMARY KEY,
  embedding BLOB NOT NULL,
  created_at TEXT NOT NULL DEFAULT (datetime('now','localtime')),
  FOREIGN KEY (person_id) REFERENCES persons(person_id) ON DELETE CASCADE
);

-- 2) Cuộc họp
CREATE TABLE IF NOT EXISTS meetings (
  meeting_id          TEXT PRIMARY KEY,
  title               TEXT NOT NULL,
  start_time          TEXT NOT NULL,        -- ISO: 'YYYY-MM-DD HH:MM:SS'
  late_after_minutes  INTEGER NOT NULL DEFAULT 0,
  location            TEXT,
  status              TEXT NOT NULL DEFAULT 'DRAFT', -- DRAFT/OPEN/CLOSED
  created_at          TEXT NOT NULL DEFAULT (datetime('now','localtime'))
);

-- 3) Danh sách mời theo cuộc họp
CREATE TABLE IF NOT EXISTS meeting_attendees (
  meeting_id  TEXT NOT NULL,
  person_id   TEXT NOT NULL,
  role        TEXT NOT NULL DEFAULT 'attendee', -- attendee/host/secretary
  PRIMARY KEY (meeting_id, person_id),
  FOREIGN KEY (meeting_id) REFERENCES meetings(meeting_id) ON DELETE CASCADE,
  FOREIGN KEY (person_id)  REFERENCES persons(person_id)  ON DELETE CASCADE
);

-- 4) Sự kiện điểm danh (check-in/check-out)
CREATE TABLE IF NOT EXISTS attendance_events (
  id           INTEGER PRIMARY KEY AUTOINCREMENT,
  meeting_id   TEXT NOT NULL,
  person_id    TEXT NOT NULL,
  event_type   TEXT NOT NULL,  -- checkin/checkout
  ts           TEXT NOT NULL DEFAULT (datetime('now','localtime')),
  confidence   REAL,
  camera_id    TEXT,
  snapshot_path TEXT,
  FOREIGN KEY (meeting_id) REFERENCES meetings(meeting_id) ON DELETE CASCADE,
  FOREIGN KEY (person_id)  REFERENCES persons(person_id)  ON DELETE CASCADE
);

-- Index giúp query nhanh
CREATE INDEX IF NOT EXISTS idx_events_meeting ON attendance_events(meeting_id, ts);
CREATE INDEX IF NOT EXISTS idx_events_person  ON attendance_events(person_id, ts);
"""

def main():
    con = sqlite3.connect(DB_PATH)
    try:
        con.executescript(SCHEMA_SQL)
        con.commit()
        print(f"[OK] Initialized DB at: {DB_PATH.resolve()}")
    finally:
        con.close()

if __name__ == "__main__":
    main()