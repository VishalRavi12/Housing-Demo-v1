"""Runs the five stages end-to-end for a GoPro video or a batch of user photos."""
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import ExifTags, Image

import config
from . import db
from .detect import get_detector
from .geo import destination
from .locate import clean_track, load_telemetry, sync_frames
from .match import ParcelIndex, look_point, match_property
from .select_frames import blur_score, dhash, distance_targets, hamming, read_frames, resize_for_save

BATCH = 16


def _rel(p):
    return Path(p).resolve().relative_to(config.OUTPUT_DIR.resolve()).as_posix()


def process_video(video_path, every_m=None, max_seconds=None, detector=None, log=print):
    video_path = Path(video_path)
    every_m = every_m or config.SAMPLE_EVERY_M
    name = video_path.name
    t_start = time.time()
    con = db.connect()
    touched = db.clear_source(con, name)
    stats = {}

    log(f"=== Processing video: {name} ===")
    log("STAGE 1 - LOCATE: extracting GPS telemetry ...")
    raw = load_telemetry(video_path)
    tel_dir = config.OUTPUT_DIR / "telemetry"
    tel_dir.mkdir(parents=True, exist_ok=True)
    raw.to_csv(tel_dir / f"{video_path.stem}_gps_raw.csv", index=False)
    src = raw["source"].iloc[0]
    hz = len(raw) / max(raw["video_time_s"].max(), 1e-6)
    log(f"   {len(raw)} GPS samples from {src} (~{hz:.1f} Hz)")

    track, rep = clean_track(raw)
    track.to_csv(tel_dir / f"{video_path.stem}_gps_clean.csv", index=False)
    log(f"   cleaning: {rep['raw_points']} raw -> {rep['after_fix_dop']} after fix/DOP "
        f"-> {rep['after_jump_removal']} after jump removal")
    log(f"   route: {rep['route_length_m']} m over {rep['duration_s']} s")
    stats.update(gps_source=src, gps_hz=round(hz, 1), **rep)

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if n_frames <= 0:
        raise ValueError("OpenCV could not read video frames (codec not supported?).")
    if max_seconds:
        n_frames = min(n_frames, int(max_seconds * fps))
    frames = sync_frames(track, fps, n_frames)
    log(f"   video {w}x{h} @ {fps:.2f} fps, {n_frames} frames synced to GPS"
        + (f" (limited to first {max_seconds}s)" if max_seconds else ""))
    stats.update(fps=round(fps, 2), resolution=f"{w}x{h}", frames_total=n_frames)

    log(f"STAGE 2 - SELECT: one frame every {every_m} m, blur filter, near-duplicate removal ...")
    targets = distance_targets(frames, every_m)
    log(f"   distance sampling: {n_frames} -> {len(targets)} candidate frames")
    frame_dir = config.OUTPUT_DIR / "frames" / video_path.stem
    frame_dir.mkdir(parents=True, exist_ok=True)
    for old in frame_dir.glob("*.jpg"):
        old.unlink()

    by_idx = targets.set_index("frame_idx")
    kept, n_blur, n_dup, last_hash = [], 0, 0, None
    next_report = 0.25
    for n, (idx, img) in enumerate(read_frames(video_path, targets["frame_idx"].values), 1):
        if n / len(targets) >= next_report:
            log(f"   decoding frames ... {n}/{len(targets)}")
            next_report += 0.25
        b = blur_score(img)
        if b < config.BLUR_THRESHOLD:
            n_blur += 1
            continue
        hsh = dhash(img)
        if last_hash is not None and hamming(hsh, last_hash) <= config.DUP_HASH_DISTANCE:
            n_dup += 1
            continue
        last_hash = hsh
        r = by_idx.loc[idx]
        path = frame_dir / f"f_{idx:06d}.jpg"
        small = resize_for_save(img)
        cv2.imwrite(str(path), small, [cv2.IMWRITE_JPEG_QUALITY, 88])
        kept.append(dict(frame_idx=int(idx), video_time_s=float(r["video_time_s"]), lat=float(r["lat"]),
                         lon=float(r["lon"]), heading_deg=float(r["heading_deg"]), blur=b, path=path, img=small))
    log(f"   blur filter removed {n_blur}, near-duplicate removal removed {n_dup} -> {len(kept)} frames kept")
    stats.update(frames_distance_sampled=len(targets), frames_blurry=n_blur, frames_duplicate=n_dup,
                 frames_kept=len(kept))

    det = detector or get_detector()
    run_id = db.start_run(con, name, "video", det.name)
    raw_sorted = raw.sort_values("video_time_s")
    raw_t, raw_gps_time = raw_sorted["video_time_s"].values, raw_sorted["gps_time"].values
    items = []
    for k in kept:
        k["look"] = look_point(k["lat"], k["lon"], k["heading_deg"])
        j = min(int(np.searchsorted(raw_t, k["video_time_s"])), len(raw_t) - 1)
        k["gps_time"] = raw_gps_time[j] if isinstance(raw_gps_time[j], str) else None
        items.append(k)

    track_ds = track.iloc[:: max(1, len(track) // 1500)][["lat", "lon"]].round(6).values.tolist()
    _match_detect_store(con, run_id, name, items, det, touched, stats, log)
    stats["seconds"] = round(time.time() - t_start, 1)
    db.finish_run(con, run_id, "done", stats, track_ds)
    log(f"=== Done in {stats['seconds']} s ===")
    con.close()
    return stats


def process_images(image_paths, detector=None, log=print):
    """User images already carry GPS + timestamp -> skip Stages 1 and 2."""
    con = db.connect()
    stats, items, skipped = {}, [], []
    log(f"=== Processing {len(image_paths)} user image(s) - Stages 1 & 2 skipped ===")
    out_dir = config.OUTPUT_DIR / "frames" / "user_images"
    out_dir.mkdir(parents=True, exist_ok=True)
    touched = set()
    for p in map(Path, image_paths):
        touched |= db.clear_source(con, p.name)
        meta = read_exif_gps(p)
        if not meta:
            skipped.append(p.name)
            continue
        img = cv2.imread(str(p))
        if img is None:
            skipped.append(p.name)
            continue
        small = resize_for_save(img)
        dest = out_dir / (p.stem + ".jpg")
        cv2.imwrite(str(dest), small, [cv2.IMWRITE_JPEG_QUALITY, 88])
        lat, lon, direction = meta["lat"], meta["lon"], meta.get("direction")
        # photo taken towards the house: look point is ahead of the camera if direction is known
        look = destination(lat, lon, direction, config.LOOK_OFFSET_M / 2) if direction is not None else (lat, lon)
        items.append(dict(frame_idx=0, video_time_s=0.0, lat=lat, lon=lon, heading_deg=direction or 0.0,
                          blur=blur_score(img), path=dest, img=small, look=look, gps_time=meta.get("time"),
                          source=p.name))
    if skipped:
        log(f"   ! skipped (no EXIF GPS): {', '.join(skipped)}")
    stats.update(images_in=len(image_paths), images_with_gps=len(items), images_skipped=len(skipped))
    if not items:
        log("   nothing to process")
        return stats
    det = detector or get_detector()
    run_id = db.start_run(con, f"{len(items)} user image(s)", "images", det.name)
    _match_detect_store(con, run_id, None, items, det, touched, stats, log)
    db.finish_run(con, run_id, "done", stats)
    log("=== Done ===")
    con.close()
    return stats


def _match_detect_store(con, run_id, source_name, items, det, touched, stats, log):
    log("STAGE 3 - MATCH: GPS + direction of travel -> property ...")
    parcels = ParcelIndex()
    log(f"   {len(parcels)} parcels loaded" if len(parcels)
        else f"   no parcel file in data/parcels -> using {config.LOT_SIZE_M:.0f} m virtual lots")
    for it in items:
        it["match"] = match_property(*it["look"], parcels)
    props = {it["match"]["property_key"] for it in items}
    log(f"   {len(items)} frames grouped into {len(props)} properties")
    stats["properties"] = len(props)

    log(f"STAGE 4 - DETECT: vision-language model ({det.name}) on {len(items)} frames ...")
    preds = []
    for i in range(0, len(items), BATCH):
        preds += det.predict([it["img"] for it in items[i:i + BATCH]])
        log(f"   {min(i + BATCH, len(items))}/{len(items)} frames scored")

    n_pred = n_gate = n_box = 0
    for it, pr in zip(items, preds):
        pid = db.upsert_property(con, it["match"])
        touched.add(pid)
        fid = db.insert_frame(
            con, run_id=run_id, property_id=pid, source_name=source_name or it.get("source"),
            frame_idx=it["frame_idx"], video_time_s=it["video_time_s"], gps_time=it.get("gps_time"),
            lat=it["lat"], lon=it["lon"], heading_deg=it["heading_deg"], look_lat=it["look"][0],
            look_lon=it["look"][1], blur=it["blur"], building_prob=pr["building"], image_path=_rel(it["path"]))
        if pr["building"] < config.HOUSE_GATE:
            n_gate += 1
            continue
        for vtype, conf in pr["scores"].items():
            con.execute("INSERT INTO predictions(frame_id, property_id, violation_type, confidence, model) "
                        "VALUES (?,?,?,?,?)", (fid, pid, vtype, conf, det.name))
            n_pred += 1
        for b in pr.get("boxes", []):
            con.execute("INSERT INTO boxes(frame_id, violation_type, label, score, x0, y0, x1, y1, model) "
                        "VALUES (?,?,?,?,?,?,?,?,?)", (fid, b["violation_type"], b["label"], b["score"],
                                                       b["x0"], b["y0"], b["x1"], b["y1"], det.name))
            n_box += 1
    con.commit()
    log(f"   {n_gate} frames had no building in view; {n_pred} predictions and {n_box} boxes stored")
    stats["boxes"] = n_box

    log("STAGE 5 - REVIEW: routing results ...")
    db.recompute_violations(con, touched)
    counts = {r[0]: r[1] for r in con.execute(
        f"SELECT status, COUNT(*) FROM violations WHERE property_id IN ({','.join('?' * len(touched))}) "
        "GROUP BY status", tuple(touched))} if touched else {}
    log(f"   needs review: {counts.get('needs_review', 0)} | likely: {counts.get('likely', 0)} | "
        f"confirmed: {counts.get('confirmed', 0)} | rejected: {counts.get('rejected', 0)}")
    stats.update(predictions=n_pred, frames_no_building=n_gate, **{f"violations_{k}": v for k, v in counts.items()})


def read_exif_gps(path):
    """Return dict(lat, lon, direction?, time?) from a photo's EXIF, or None."""
    try:
        exif = Image.open(path).getexif()
    except Exception:
        return None
    gps = exif.get_ifd(0x8825)
    if not gps or 2 not in gps or 4 not in gps:
        return None

    def dms(v):
        d, m, s = (float(x) for x in v)
        return d + m / 60 + s / 3600

    lat = dms(gps[2]) * (-1 if gps.get(1) in ("S", b"S") else 1)
    lon = dms(gps[4]) * (-1 if gps.get(3) in ("W", b"W") else 1)
    out = dict(lat=lat, lon=lon)
    if 17 in gps:
        out["direction"] = float(gps[17])
    sub = exif.get_ifd(0x8769)
    t = sub.get(0x9003) or exif.get(ExifTags.Base.DateTime)
    if t:
        out["time"] = str(t)
    return out
