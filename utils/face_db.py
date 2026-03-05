import sqlite3
import pickle
import numpy as np
from pathlib import Path

def l2_normalize(v: np.ndarray) -> np.ndarray:
    v = v.astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-9)

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = l2_normalize(a)
    b = l2_normalize(b)
    return float(np.dot(a, b))

def load_db_identities(db_path: str, exclude_person_id: str = None):
    """
    Load danh sách (pid, name, embedding, source) từ SQLite face_embeddings.
    source: chuỗi mô tả nguồn (để debug)
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

        items = []
        for pid, name, emb_blob in rows:
            pid = str(pid)
            if exclude_person_id and pid == str(exclude_person_id):
                continue
            if emb_blob is None:
                continue
            emb = pickle.loads(emb_blob)
            emb = np.asarray(emb, dtype=np.float32)
            items.append((pid, str(name), emb, f"sqlite:{db_path}"))
        return items
    finally:
        conn.close()

def find_duplicate_identity(query_emb: np.ndarray, db_items, threshold: float = 0.60):
    """
    Trả về: (is_dup, best_pid, best_name, best_sim)
    """
    best_sim, best_pid, best_name = -1.0, None, None
    for pid, name, emb, _p in db_items:
        sim = cosine_sim(query_emb, emb)
        if sim > best_sim:
            best_sim, best_pid, best_name = sim, pid, name

    if best_pid is not None and best_sim >= threshold:
        return True, best_pid, best_name, best_sim
    return False, best_pid, best_name, best_sim
