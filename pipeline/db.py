"""DATA STORE: SQLite tables mirroring the architecture's data model.

  Property | Frame + location | Prediction | Violation | Review record
"""
import json
import sqlite3
from datetime import datetime

import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY, source_name TEXT, kind TEXT, status TEXT, detector TEXT,
    started_at TEXT, finished_at TEXT, stats_json TEXT, track_json TEXT);
CREATE TABLE IF NOT EXISTS properties (
    id INTEGER PRIMARY KEY, key TEXT UNIQUE, address TEXT, lat REAL, lon REAL, method TEXT);
CREATE TABLE IF NOT EXISTS frames (
    id INTEGER PRIMARY KEY, run_id INTEGER, property_id INTEGER, source_name TEXT, frame_idx INTEGER,
    video_time_s REAL, gps_time TEXT, lat REAL, lon REAL, heading_deg REAL, look_lat REAL, look_lon REAL,
    blur REAL, building_prob REAL, image_path TEXT);
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY, frame_id INTEGER, property_id INTEGER, violation_type TEXT,
    confidence REAL, model TEXT);
CREATE TABLE IF NOT EXISTS boxes (
    id INTEGER PRIMARY KEY, frame_id INTEGER, violation_type TEXT, label TEXT, score REAL,
    x0 REAL, y0 REAL, x1 REAL, y1 REAL, model TEXT);
CREATE TABLE IF NOT EXISTS violations (
    id INTEGER PRIMARY KEY, property_id INTEGER, violation_type TEXT, confidence REAL,
    supporting_frame_id INTEGER, status TEXT, created_at TEXT, updated_at TEXT,
    UNIQUE(property_id, violation_type));
CREATE TABLE IF NOT EXISTS reviews (
    id INTEGER PRIMARY KEY, violation_id INTEGER, action TEXT, old_type TEXT, new_type TEXT,
    reviewer TEXT, note TEXT, created_at TEXT);
CREATE INDEX IF NOT EXISTS ix_frames_prop ON frames(property_id);
CREATE INDEX IF NOT EXISTS ix_pred_prop ON predictions(property_id, violation_type);
CREATE INDEX IF NOT EXISTS ix_boxes_frame ON boxes(frame_id);
"""

REVIEWED = ("confirmed", "rejected")


def now():
    return datetime.now().isoformat(timespec="seconds")


def connect():
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(config.DB_PATH, timeout=30)
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)
    return con


def start_run(con, source_name, kind, detector):
    cur = con.execute("INSERT INTO runs(source_name, kind, status, detector, started_at) VALUES (?,?,?,?,?)",
                      (source_name, kind, "running", detector, now()))
    con.commit()
    return cur.lastrowid


def finish_run(con, run_id, status, stats, track=None):
    con.execute("UPDATE runs SET status=?, finished_at=?, stats_json=?, track_json=? WHERE id=?",
                (status, now(), json.dumps(stats), json.dumps(track) if track is not None else None, run_id))
    con.commit()


def clear_source(con, source_name):
    """Remove frames/predictions from earlier runs of the same file. Returns touched property ids."""
    props = {r[0] for r in con.execute("SELECT DISTINCT property_id FROM frames WHERE source_name=?", (source_name,))}
    con.execute("DELETE FROM predictions WHERE frame_id IN (SELECT id FROM frames WHERE source_name=?)", (source_name,))
    con.execute("DELETE FROM boxes WHERE frame_id IN (SELECT id FROM frames WHERE source_name=?)", (source_name,))
    con.execute("DELETE FROM frames WHERE source_name=?", (source_name,))
    con.execute("DELETE FROM runs WHERE source_name=? AND status!='running'", (source_name,))
    con.commit()
    return props


def upsert_property(con, m):
    row = con.execute("SELECT id FROM properties WHERE key=?", (m["property_key"],)).fetchone()
    if row:
        return row[0]
    cur = con.execute("INSERT INTO properties(key, address, lat, lon, method) VALUES (?,?,?,?,?)",
                      (m["property_key"], m["address"], m["lat"], m["lon"], m["method"]))
    return cur.lastrowid


def insert_frame(con, **f):
    cols = ",".join(f)
    cur = con.execute(f"INSERT INTO frames({cols}) VALUES ({','.join('?' * len(f))})", tuple(f.values()))
    return cur.lastrowid


def status_for(conf):
    return "likely" if conf >= config.AUTO_ACCEPT_AT else "needs_review"


def recompute_violations(con, property_ids):
    """STAGE 4 -> 5 hand-off: one candidate violation per (property, type) = best frame.
    Human decisions (confirmed / rejected) are never overwritten."""
    for pid in property_ids:
        best = {r["violation_type"]: r for r in con.execute(
            """SELECT violation_type, frame_id, MAX(confidence) AS confidence FROM predictions
               WHERE property_id=? GROUP BY violation_type""", (pid,))}
        existing = {r["violation_type"]: r for r in con.execute(
            "SELECT * FROM violations WHERE property_id=?", (pid,))}
        for vtype, ex in existing.items():
            b = best.get(vtype)
            if ex["status"] in REVIEWED:
                if b and ex["supporting_frame_id"] not in _frame_ids(con, pid):
                    con.execute("UPDATE violations SET supporting_frame_id=? WHERE id=?", (b["frame_id"], ex["id"]))
                continue
            if not b or b["confidence"] < config.DISCARD_BELOW:
                con.execute("DELETE FROM violations WHERE id=?", (ex["id"],))
            else:
                con.execute("UPDATE violations SET confidence=?, supporting_frame_id=?, status=?, updated_at=? WHERE id=?",
                            (b["confidence"], b["frame_id"], status_for(b["confidence"]), now(), ex["id"]))
        for vtype, b in best.items():
            if vtype in existing or b["confidence"] < config.DISCARD_BELOW:
                continue
            con.execute("""INSERT INTO violations(property_id, violation_type, confidence, supporting_frame_id,
                           status, created_at, updated_at) VALUES (?,?,?,?,?,?,?)""",
                        (pid, vtype, b["confidence"], b["frame_id"], status_for(b["confidence"]), now(), now()))
    con.commit()


def _frame_ids(con, pid):
    return {r[0] for r in con.execute("SELECT id FROM frames WHERE property_id=?", (pid,))}


def review(con, violation_id, action, new_type=None, reviewer="reviewer", note=""):
    """STAGE 5 - REVIEW: confirm / reject / correct a candidate violation."""
    v = con.execute("SELECT * FROM violations WHERE id=?", (violation_id,)).fetchone()
    if not v:
        return
    old_type = v["violation_type"]
    if action == "confirm":
        con.execute("UPDATE violations SET status='confirmed', updated_at=? WHERE id=?", (now(), violation_id))
    elif action == "reject":
        con.execute("UPDATE violations SET status='rejected', updated_at=? WHERE id=?", (now(), violation_id))
    elif action == "correct" and new_type and new_type != old_type:
        clash = con.execute("SELECT id FROM violations WHERE property_id=? AND violation_type=?",
                            (v["property_id"], new_type)).fetchone()
        if clash:  # that type already exists for this property: confirm it, reject the wrong label
            con.execute("UPDATE violations SET status='confirmed', updated_at=? WHERE id=?", (now(), clash[0]))
            con.execute("UPDATE violations SET status='rejected', updated_at=? WHERE id=?", (now(), violation_id))
        else:
            con.execute("UPDATE violations SET violation_type=?, status='confirmed', updated_at=? WHERE id=?",
                        (new_type, now(), violation_id))
    con.execute("""INSERT INTO reviews(violation_id, action, old_type, new_type, reviewer, note, created_at)
                   VALUES (?,?,?,?,?,?,?)""", (violation_id, action, old_type, new_type, reviewer, note, now()))
    con.commit()
