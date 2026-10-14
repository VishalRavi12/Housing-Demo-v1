"""STAGE 1a - Telemetry extraction.

Pure-Python reader for GoPro GPMF telemetry inside an MP4 file. No ffmpeg needed.

  MP4 file -> find the 'gpmd' metadata track -> read each ~1 s payload
  -> walk the GPMF KLV tree -> decode GPS9 (HERO11+) or GPS5 (older models).

Every GPS sample comes back with `video_time_s` (seconds from start of video)
so frames can later be synced to positions.
"""
import struct
from datetime import datetime, timedelta, timezone

# ---------------------------------------------------------------------------
# MP4 box walking
# ---------------------------------------------------------------------------
_CONTAINERS = {"moov", "trak", "mdia", "minf", "stbl", "edts", "udta"}


def _boxes(f, start, end):
    """Yield (type, payload_start, box_end) for boxes in [start, end)."""
    pos = start
    while pos + 8 <= end:
        f.seek(pos)
        size, typ = struct.unpack(">I4s", f.read(8))
        header = 8
        if size == 1:
            size = struct.unpack(">Q", f.read(8))[0]
            header = 16
        elif size == 0:
            size = end - pos
        if size < header:
            break
        yield typ.decode("latin-1"), pos + header, pos + size
        pos += size


def _find(f, start, end, typ):
    for t, s, e in _boxes(f, start, end):
        if t == typ:
            return s, e
    return None


def _read(f, start, end):
    f.seek(start)
    return f.read(end - start)


def _gpmd_samples(f, file_size):
    """Return (list of (offset, size, start_s, duration_s)) for the GPMF track."""
    moov = _find(f, 0, file_size, "moov")
    if not moov:
        return []
    for t, ts, te in _boxes(f, *moov):
        if t != "trak":
            continue
        mdia = _find(f, ts, te, "mdia")
        if not mdia:
            continue
        minf = _find(f, *mdia, "minf")
        stbl = _find(f, *minf, "stbl") if minf else None
        if not stbl:
            continue
        stsd = _read(f, *_find(f, *stbl, "stsd"))
        if len(stsd) < 16 or stsd[12:16] != b"gpmd":
            continue

        # timescale from mdhd
        mdhd = _read(f, *_find(f, *mdia, "mdhd"))
        timescale = struct.unpack(">I", mdhd[20:24] if mdhd[0] == 1 else mdhd[12:16])[0]

        # sample sizes
        stsz = _read(f, *_find(f, *stbl, "stsz"))
        fixed, count = struct.unpack(">II", stsz[4:12])
        sizes = [fixed] * count if fixed else list(struct.unpack(f">{count}I", stsz[12:12 + 4 * count]))

        # chunk offsets
        co = _find(f, *stbl, "stco")
        if co:
            b = _read(f, *co)
            n = struct.unpack(">I", b[4:8])[0]
            chunks = list(struct.unpack(f">{n}I", b[8:8 + 4 * n]))
        else:
            b = _read(f, *_find(f, *stbl, "co64"))
            n = struct.unpack(">I", b[4:8])[0]
            chunks = list(struct.unpack(f">{n}Q", b[8:8 + 8 * n]))

        # samples per chunk
        b = _read(f, *_find(f, *stbl, "stsc"))
        n = struct.unpack(">I", b[4:8])[0]
        runs = [struct.unpack(">III", b[8 + 12 * i:20 + 12 * i]) for i in range(n)]
        per_chunk = []
        for i, (first, spc, _) in enumerate(runs):
            last = runs[i + 1][0] - 1 if i + 1 < len(runs) else len(chunks)
            per_chunk += [spc] * (last - first + 1)

        offsets = []
        for chunk_off, spc in zip(chunks, per_chunk):
            o = chunk_off
            for _ in range(spc):
                if len(offsets) == len(sizes):
                    break
                offsets.append(o)
                o += sizes[len(offsets) - 1]

        # sample durations
        b = _read(f, *_find(f, *stbl, "stts"))
        n = struct.unpack(">I", b[4:8])[0]
        durations = []
        for i in range(n):
            c, d = struct.unpack(">II", b[8 + 8 * i:16 + 8 * i])
            durations += [d] * c

        out, t = [], 0
        for i, (o, s) in enumerate(zip(offsets, sizes)):
            d = durations[i] if i < len(durations) else (durations[-1] if durations else timescale)
            out.append((o, s, t / timescale, d / timescale))
            t += d
        return out
    return []


# ---------------------------------------------------------------------------
# GPMF KLV decoding
# ---------------------------------------------------------------------------
_FMT = {"b": "b", "B": "B", "s": "h", "S": "H", "l": "i", "L": "I",
        "f": "f", "d": "d", "j": "q", "J": "Q", "c": "c", "F": "4s"}


def _klv(buf, start, end):
    pos = start
    while pos + 8 <= end:
        key = buf[pos:pos + 4]
        if key == b"\0\0\0\0":
            break
        typ = chr(buf[pos + 4])
        ssize = buf[pos + 5]
        repeat = struct.unpack(">H", buf[pos + 6:pos + 8])[0]
        dstart = pos + 8
        dlen = ssize * repeat
        yield key.decode("latin-1"), typ, ssize, repeat, dstart, dlen
        pos = dstart + ((dlen + 3) & ~3)


def _values(buf, typ, ssize, repeat, start):
    """Decode a simple-typed KLV into a flat list of numbers."""
    if typ not in _FMT or typ in "cF":
        return []
    width = struct.calcsize(">" + _FMT[typ])
    n = (ssize * repeat) // width
    return list(struct.unpack(f">{n}{_FMT[typ]}", buf[start:start + n * width]))


def _walk_streams(buf, start, end, emit):
    """Recurse DEVC/STRM containers; call emit(stream_items) for each STRM."""
    items = list(_klv(buf, start, end))
    if any(k == "STRM" for k, *_ in items) or any(k == "DEVC" for k, *_ in items):
        for key, typ, ssize, rep, ds, dl in items:
            if typ == "\0":
                _walk_streams(buf, ds, ds + dl, emit)
    else:
        emit(buf, items)


def _parse_payload(buf, t0, dur, out):
    def emit(b, items):
        scal, type_str, fix, dop, utc = None, None, None, None, None
        for key, typ, ssize, rep, ds, dl in items:
            if key == "SCAL":
                scal = _values(b, typ, ssize, rep, ds)
            elif key == "TYPE":
                type_str = b[ds:ds + dl].split(b"\0")[0].decode("latin-1")
            elif key == "GPSF":
                fix = _values(b, typ, ssize, rep, ds)[0]
            elif key == "GPSP":
                dop = _values(b, typ, ssize, rep, ds)[0] / 100.0
            elif key == "GPSU":
                utc = b[ds:ds + dl].decode("latin-1", "ignore")
            elif key == "GPS9" and rep:
                fmt = ">" + "".join(_FMT[c] for c in (type_str or "lllllllSS"))
                scale = scal or [1e7, 1e7, 1e3, 1e3, 1e2, 1, 1e3, 1e2, 1]
                for i in range(rep):
                    raw = struct.unpack(fmt, b[ds + i * ssize: ds + i * ssize + struct.calcsize(fmt)])
                    v = [r / (scale[j] if j < len(scale) and scale[j] else 1) for j, r in enumerate(raw)]
                    gps_time = (datetime(2000, 1, 1, tzinfo=timezone.utc)
                                + timedelta(days=v[5], seconds=v[6]))
                    out.append(dict(video_time_s=t0 + dur * i / rep, lat=v[0], lon=v[1], alt=v[2],
                                    speed_2d=v[3], speed_3d=v[4], gps_time=gps_time.isoformat(),
                                    dop=v[7], fix=int(v[8]), source="GPS9"))
            elif key == "GPS5" and rep:
                vals = _values(b, typ, ssize, rep, ds)
                scale = scal or [1e7, 1e7, 1e3, 1e3, 1e2]
                if len(scale) == 1:
                    scale = scale * 5
                for i in range(rep):
                    v = [vals[i * 5 + j] / scale[j] for j in range(5)]
                    out.append(dict(video_time_s=t0 + dur * i / rep, lat=v[0], lon=v[1], alt=v[2],
                                    speed_2d=v[3], speed_3d=v[4], gps_time=_gpsu(utc, dur * i / rep),
                                    dop=dop if dop is not None else 1.0,
                                    fix=int(fix) if fix is not None else 3, source="GPS5"))

    _walk_streams(buf, 0, len(buf), emit)


def _gpsu(utc, offset_s):
    try:
        t = datetime.strptime(utc[:16], "%y%m%d%H%M%S.%f").replace(tzinfo=timezone.utc)
        return (t + timedelta(seconds=offset_s)).isoformat()
    except Exception:
        return None


def extract_gps(path):
    """Return a list of GPS sample dicts from a GoPro MP4 (empty list if none)."""
    import os
    rows = []
    with open(path, "rb") as f:
        for off, size, t0, dur in _gpmd_samples(f, os.path.getsize(path)):
            f.seek(off)
            payload = f.read(size)
            n_before = len(rows)
            _parse_payload(payload, t0, dur, rows)
            # if a model writes both GPS5 and GPS9, keep only GPS9
            new = rows[n_before:]
            if any(r["source"] == "GPS9" for r in new) and any(r["source"] == "GPS5" for r in new):
                rows[n_before:] = [r for r in new if r["source"] == "GPS9"]
    return rows
