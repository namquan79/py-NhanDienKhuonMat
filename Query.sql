CREATE TABLE face_embeddings (
  person_id TEXT PRIMARY KEY,
  embedding BLOB,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP
);