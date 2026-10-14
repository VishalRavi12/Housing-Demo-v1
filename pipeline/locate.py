"""STAGE 1 - LOCATE: Where was the camera?

  telemetry extraction -> GPS cleaning (fix + DOP + jumps) -> smoothing
  -> direction of travel -> frame-to-GPS sync
"""
from pathlib import Path

import numpy as np
import pandas as pd

import config
from .geo import bearing, haversine
from .gpmf import extract_gps


def load_telemetry(video_path):
    """GPS9/GPS5 from the GoPro file, or a sidecar <video>.csv as a fallback.

    Sidecar CSV columns: video_time_s, lat, lon  (optional: alt, speed_2d, dop, fix, gps_time)
    """
    video_path = Path(video_path)
    rows = extract_gps(video_path)
    if rows:
        return pd.DataFrame(rows)
    sidecar = video_path.with_suffix(".csv")
    if sidecar.exists():
        df = pd.read_csv(sidecar)
        for col, default in [("alt", 0.0), ("speed_2d", np.nan), ("speed_3d", np.nan),
                             ("dop", 1.0), ("fix", 3), ("gps_time", None)]:
            if col not in df:
                df[col] = default
        df["source"] = "sidecar_csv"
        return df
    raise ValueError(f"No GPMF GPS track in {video_path.name} and no sidecar {sidecar.name} found.")


def clean_track(raw):
    """Drop bad fixes / high DOP / jumps, then smooth. Returns (clean_df, report)."""
    df = raw.sort_values("video_time_s").reset_index(drop=True).copy()
    report = {"raw_points": len(df)}

    df = df[(df["fix"] >= config.MIN_GPS_FIX) & (df["dop"] <= config.MAX_DOP)]
    df = df[(df["lat"].abs() > 0.0001) | (df["lon"].abs() > 0.0001)]  # (0,0) = no fix
    report["after_fix_dop"] = len(df)

    # remove jumps: iteratively drop points implying an impossible speed
    for _ in range(5):
        if len(df) < 3:
            break
        d = haversine(df["lat"].values[:-1], df["lon"].values[:-1], df["lat"].values[1:], df["lon"].values[1:])
        dt = np.diff(df["video_time_s"].values)
        speed = d / np.maximum(dt, 1e-3)
        bad = np.concatenate([[False], speed > config.MAX_JUMP_SPEED_MS])
        if not bad.any():
            break
        df = df[~bad]
    report["after_jump_removal"] = len(df)

    if len(df) < 2:
        raise ValueError("Not enough good GPS points after cleaning (is the GoPro getting a GPS lock?).")

    w = config.SMOOTH_WINDOW
    df["lat"] = df["lat"].rolling(w, center=True, min_periods=1).median()
    df["lon"] = df["lon"].rolling(w, center=True, min_periods=1).median()

    # cumulative distance along the route
    seg = haversine(df["lat"].values[:-1], df["lon"].values[:-1], df["lat"].values[1:], df["lon"].values[1:])
    df["cum_dist_m"] = np.concatenate([[0.0], np.cumsum(seg)])

    df["heading_deg"] = _heading(df)
    report["route_length_m"] = round(float(df["cum_dist_m"].iloc[-1]), 1)
    report["duration_s"] = round(float(df["video_time_s"].iloc[-1] - df["video_time_s"].iloc[0]), 1)
    return df.reset_index(drop=True), report


def _heading(df, min_move_m=3.0):
    """Direction of travel from points ~min_move_m apart; held constant while stopped."""
    lat, lon, cd = df["lat"].values, df["lon"].values, df["cum_dist_m"].values
    out = np.full(len(df), np.nan)
    j = 0
    for i in range(len(df)):
        j = max(j, i)
        while j < len(df) - 1 and cd[j] - cd[i] < min_move_m:
            j += 1
        if cd[j] - cd[i] >= min_move_m:
            out[i] = bearing(lat[i], lon[i], lat[j], lon[j])
    s = pd.Series(out).ffill().bfill().fillna(0.0)
    return s.values


def sync_frames(track, fps, n_frames):
    """Interpolate a position, heading and distance for every video frame timestamp."""
    t = np.arange(n_frames) / fps
    tt = track["video_time_s"].values
    rad = np.radians(track["heading_deg"].values)
    frames = pd.DataFrame({
        "frame_idx": np.arange(n_frames),
        "video_time_s": t,
        "lat": np.interp(t, tt, track["lat"].values),
        "lon": np.interp(t, tt, track["lon"].values),
        "cum_dist_m": np.interp(t, tt, track["cum_dist_m"].values),
        "heading_deg": (np.degrees(np.arctan2(np.interp(t, tt, np.sin(rad)),
                                              np.interp(t, tt, np.cos(rad)))) + 360) % 360,
    })
    # frames outside the GPS time range have no trustworthy position
    frames["has_gps"] = (t >= tt[0] - 0.5) & (t <= tt[-1] + 0.5)
    return frames
