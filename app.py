"""Local web app = DATA STORE / API + the three user views Salesforce would provide.

  Admin      -> /process      run the pipeline on videos / upload photos
  Inspector  -> /review       confirm / reject / correct uncertain predictions
  Public     -> /public       confirmed violations only
  API        -> /api/...      JSON for any downstream system

  python app.py   ->  http://127.0.0.1:5000
"""
import csv
import io
import json
import threading
import traceback
from datetime import datetime
from pathlib import Path

from flask import Flask, Response, abort, jsonify, redirect, render_template, request, send_from_directory, url_for
from werkzeug.utils import secure_filename

import config
from pipeline import db
from pipeline.detect import get_detector

app = Flask(__name__)
for d in (config.VIDEO_DIR, config.IMAGE_DIR, config.PARCEL_DIR, config.OUTPUT_DIR):
    d.mkdir(parents=True, exist_ok=True)

VTYPES = list(config.VIOLATIONS)
STATUS_ORDER = ["confirmed", "likely", "needs_review", "rejected"]

# ---------------------------------------------------------------------------
# Background pipeline job (one at a time)
# ---------------------------------------------------------------------------
JOB = {"running": False, "log": [], "started": None, "finished": None, "error": None}
LAST_SELECTION = {"videos": [], "images": []}  # re-ticked on the Run Pipeline page after a run
_detectors = {}


def _log(msg):
    print(msg)
    JOB["log"].append(f"[{datetime.now():%H:%M:%S}] {msg}")


def _run_job(videos, images, every_m, max_seconds, detector):
    from pipeline.runner import process_images, process_video
    try:
        if detector not in _detectors:
            _log(f"Loading detector '{detector}' (first time can take a minute to download CLIP) ...")
            _detectors[detector] = get_detector(detector)
        det = _detectors[detector]
        for v in videos:
            try:
                process_video(v, every_m=every_m, max_seconds=max_seconds, detector=det, log=_log)
            except Exception as e:
                _log(f"! {Path(v).name} failed: {e}")
                traceback.print_exc()
        if images:
            process_images(images, detector=det, log=_log)
    except Exception as e:
        JOB["error"] = str(e)
        _log(f"! job failed: {e}")
        traceback.print_exc()
    finally:
        JOB["running"] = False
        JOB["finished"] = datetime.now().isoformat(timespec="seconds")


def start_job(videos, images, every_m, max_seconds, detector):
    if JOB["running"]:
        return False
    JOB.update(running=True, log=[], started=datetime.now().isoformat(timespec="seconds"), finished=None, error=None)
    threading.Thread(target=_run_job, args=(videos, images, every_m, max_seconds, detector), daemon=True).start()
    return True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def q(sql, args=()):
    con = db.connect()
    try:
        return [dict(r) for r in con.execute(sql, args)]
    finally:
        con.close()


def nice(vtype):
    return vtype.replace("_", " ").title()


def boxes_for(frame_ids):
    """{frame_id: [box, ...]} for the given frames, highest score first."""
    ids = [i for i in set(frame_ids) if i is not None]
    out = {}
    for i in range(0, len(ids), 500):
        chunk = ids[i:i + 500]
        for b in q(f"SELECT * FROM boxes WHERE frame_id IN ({','.join('?' * len(chunk))}) ORDER BY score DESC", chunk):
            out.setdefault(b["frame_id"], []).append(b)
    return out


app.jinja_env.filters["nice"] = nice


def property_summaries(public=False):
    props = q("""SELECT p.*, COUNT(DISTINCT f.id) AS n_frames FROM properties p
                 LEFT JOIN frames f ON f.property_id = p.id GROUP BY p.id HAVING n_frames > 0""")
    viols = q("SELECT * FROM violations")
    by_prop = {}
    for v in viols:
        if public and v["status"] != "confirmed":
            continue
        by_prop.setdefault(v["property_id"], []).append(v)
    for p in props:
        vs = by_prop.get(p["id"], [])
        p["violations"] = sorted(vs, key=lambda v: -v["confidence"])
        active = [v for v in vs if v["status"] != "rejected"]
        p["status"] = next((s for s in STATUS_ORDER if any(v["status"] == s for v in active)), "clear")
        p["max_conf"] = max((v["confidence"] for v in active), default=0)
    return props


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
@app.route("/")
def dashboard():
    runs = q("SELECT * FROM runs ORDER BY id DESC")
    for r in runs:
        r["stats"] = json.loads(r["stats_json"] or "{}")
        r["track"] = json.loads(r["track_json"] or "null")
    counts = {r["status"]: r["n"] for r in q("SELECT status, COUNT(*) AS n FROM violations GROUP BY status")}
    by_type = q("""SELECT violation_type, COUNT(*) AS n FROM violations WHERE status != 'rejected'
                   GROUP BY violation_type ORDER BY n DESC""")
    totals = dict(
        videos=len([r for r in runs if r["kind"] == "video"]),
        gps=sum(r["stats"].get("raw_points", 0) for r in runs),
        sampled=sum(r["stats"].get("frames_distance_sampled", 0) for r in runs),
        frames=q("SELECT COUNT(*) AS n FROM frames")[0]["n"],
        properties=q("SELECT COUNT(DISTINCT property_id) AS n FROM frames")[0]["n"],
        violations=sum(v for k, v in counts.items() if k != "rejected"),
        frames_total=sum(r["stats"].get("frames_total", 0) for r in runs),
    )
    return render_template("dashboard.html", runs=runs, counts=counts, by_type=by_type, totals=totals,
                           props=property_summaries(), tracks=[r["track"] for r in runs if r["track"]])


@app.route("/process", methods=["GET", "POST"])
def process():
    if request.method == "POST":
        videos = [config.VIDEO_DIR / secure_filename(n) for n in request.form.getlist("videos")]
        images = [config.IMAGE_DIR / secure_filename(n) for n in request.form.getlist("images")]
        for f in request.files.getlist("upload"):
            if f and f.filename and Path(f.filename).suffix.lower() in config.IMAGE_EXTS:
                dest = config.IMAGE_DIR / secure_filename(f.filename)
                f.save(dest)
                images.append(dest)
        videos = [v for v in videos if v.exists()]
        images = [i for i in images if i.exists()]
        LAST_SELECTION.update(videos=[v.name for v in videos], images=[i.name for i in images])
        every_m = float(request.form.get("every_m") or config.SAMPLE_EVERY_M)
        max_s = float(request.form["max_seconds"]) if request.form.get("max_seconds") else None
        detector = request.form.get("detector", config.DETECTOR)
        LAST_SELECTION.update(every_m=every_m, max_seconds=max_s, detector=detector)
        if videos or images:
            start_job(videos, images, every_m, max_s, detector)
        return redirect(url_for("process"))
    videos = sorted(p for p in config.VIDEO_DIR.iterdir() if p.suffix.lower() in config.VIDEO_EXTS)
    images = sorted(p for p in config.IMAGE_DIR.iterdir() if p.suffix.lower() in config.IMAGE_EXTS)
    done = {r["source_name"] for r in q("SELECT source_name FROM runs WHERE status='done'")}
    return render_template("process.html", videos=videos, images=images, done=done, job=JOB, cfg=config,
                           sel=LAST_SELECTION, video_dir=config.VIDEO_DIR, image_dir=config.IMAGE_DIR)


@app.route("/api/job")
def api_job():
    return jsonify(JOB)


@app.route("/properties")
def properties():
    props = sorted(property_summaries(), key=lambda p: (STATUS_ORDER + ["clear"]).index(p["status"]))
    return render_template("properties.html", props=props)


@app.route("/property/<int:pid>")
def property_detail(pid):
    p = q("SELECT * FROM properties WHERE id=?", (pid,))
    if not p:
        abort(404)
    frames = q("SELECT * FROM frames WHERE property_id=? ORDER BY source_name, frame_idx", (pid,))
    preds = q("SELECT * FROM predictions WHERE property_id=?", (pid,))
    by_frame = {}
    for pr in preds:
        by_frame.setdefault(pr["frame_id"], {})[pr["violation_type"]] = pr["confidence"]
    boxes = boxes_for([f["id"] for f in frames])
    for f in frames:
        f["scores"] = by_frame.get(f["id"], {})
        f["boxes"] = boxes.get(f["id"], [])
    viols = q("""SELECT v.*, f.image_path FROM violations v LEFT JOIN frames f ON f.id = v.supporting_frame_id
                 WHERE v.property_id=? ORDER BY v.confidence DESC""", (pid,))
    for v in viols:
        v["boxes"] = [b for b in boxes.get(v["supporting_frame_id"], []) if b["violation_type"] == v["violation_type"]]
    reviews = q("""SELECT r.* FROM reviews r JOIN violations v ON v.id = r.violation_id
                   WHERE v.property_id=? ORDER BY r.id DESC""", (pid,))
    return render_template("property.html", p=p[0], frames=frames, viols=viols, reviews=reviews, vtypes=VTYPES)


@app.route("/review")
def review_queue():
    show = request.args.get("show", "needs_review")
    statuses = {"needs_review": ("needs_review",), "likely": ("likely",),
                "all": ("needs_review", "likely")}.get(show, ("needs_review",))
    items = q(f"""SELECT v.*, f.image_path, f.source_name, f.video_time_s, f.lat AS cam_lat, f.lon AS cam_lon,
                        p.address FROM violations v
                 JOIN frames f ON f.id = v.supporting_frame_id JOIN properties p ON p.id = v.property_id
                 WHERE v.status IN ({','.join('?' * len(statuses))}) ORDER BY v.confidence DESC""", statuses)
    boxes = boxes_for([v["supporting_frame_id"] for v in items])
    for v in items:
        v["all_boxes"] = boxes.get(v["supporting_frame_id"], [])
        v["boxes"] = [b for b in v["all_boxes"] if b["violation_type"] == v["violation_type"]]
    counts = {r["status"]: r["n"] for r in q("SELECT status, COUNT(*) AS n FROM violations GROUP BY status")}
    return render_template("review.html", items=items, vtypes=VTYPES, show=show, counts=counts, cfg=config)


@app.route("/review/<int:vid>", methods=["POST"])
def do_review(vid):
    con = db.connect()
    db.review(con, vid, request.form["action"], request.form.get("new_type"),
              request.form.get("reviewer") or "inspector", request.form.get("note", ""))
    con.close()
    return redirect(request.form.get("next") or url_for("review_queue"))


@app.route("/public")
def public():
    props = [p for p in property_summaries(public=True) if p["violations"]]
    return render_template("public.html", props=props)


@app.route("/metrics")
def metrics():
    rows = q("""SELECT violation_type,
                       SUM(status='confirmed') AS confirmed, SUM(status='rejected') AS rejected,
                       SUM(status='needs_review') AS pending, SUM(status='likely') AS likely,
                       AVG(confidence) AS avg_conf
                FROM violations GROUP BY violation_type""")
    corrected = q("SELECT old_type, new_type, COUNT(*) AS n FROM reviews WHERE action='correct' GROUP BY 1, 2")
    for r in rows:
        reviewed = r["confirmed"] + r["rejected"]
        r["precision"] = r["confirmed"] / reviewed if reviewed else None
    return render_template("metrics.html", rows=rows, corrected=corrected)


@app.route("/export/<what>.csv")
def export_csv(what):
    sql = {
        "violations": """SELECT v.id, p.address, p.lat, p.lon, v.violation_type, ROUND(v.confidence, 3) AS confidence,
                         v.status, f.source_name, f.video_time_s, f.gps_time, f.image_path, v.updated_at
                         FROM violations v JOIN properties p ON p.id = v.property_id
                         LEFT JOIN frames f ON f.id = v.supporting_frame_id ORDER BY v.id""",
        "frames": """SELECT f.*, p.address FROM frames f JOIN properties p ON p.id = f.property_id ORDER BY f.id""",
        "reviews": "SELECT * FROM reviews ORDER BY id",
        # box coordinates are fractions of image width/height (multiply by the JPG size for pixels)
        "boxes": """SELECT b.id, f.image_path, b.violation_type, b.label, ROUND(b.score, 3) AS score,
                    ROUND(b.x0, 4) AS x0, ROUND(b.y0, 4) AS y0, ROUND(b.x1, 4) AS x1, ROUND(b.y1, 4) AS y1,
                    b.model, v.status AS violation_status
                    FROM boxes b JOIN frames f ON f.id = b.frame_id
                    LEFT JOIN violations v ON v.supporting_frame_id = b.frame_id AND v.violation_type = b.violation_type
                    ORDER BY b.id""",
    }.get(what)
    if not sql:
        abort(404)
    rows = q(sql)
    buf = io.StringIO()
    if rows:
        w = csv.DictWriter(buf, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return Response(buf.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": f"attachment; filename={what}.csv"})


# ---------------------------------------------------------------------------
# JSON API (what a Salesforce integration would consume)
# ---------------------------------------------------------------------------
@app.route("/api/properties")
def api_properties():
    return jsonify(property_summaries(public=request.args.get("public") == "1"))


@app.route("/api/violations")
def api_violations():
    status = request.args.get("status")
    sql = """SELECT v.*, p.address, p.lat, p.lon, f.image_path FROM violations v
             JOIN properties p ON p.id = v.property_id LEFT JOIN frames f ON f.id = v.supporting_frame_id"""
    return jsonify(q(sql + " WHERE v.status=?", (status,)) if status else q(sql))


@app.route("/media/<path:p>")
def media(p):
    return send_from_directory(config.OUTPUT_DIR, p)


@app.route("/reset", methods=["POST"])
def reset():
    if not JOB["running"] and config.DB_PATH.exists():
        config.DB_PATH.unlink()
    return redirect(url_for("dashboard"))


if __name__ == "__main__":
    app.run(debug=False, port=5000, threaded=True)
