"""Builds a tiny MP4 with a GoPro-style 'gpmd' GPS9 track and checks the parser reads it back.

  python -m pytest tests      (or)      python tests/test_gpmf.py
"""
import struct
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.gpmf import extract_gps  # noqa: E402


def klv(key, typ, ssize, repeat, data):
    data += b"\0" * ((4 - len(data) % 4) % 4)
    return key.encode() + bytes([ord(typ), ssize]) + struct.pack(">H", repeat) + data


def nest(key, *children):
    body = b"".join(children)
    return key.encode() + b"\0\x01" + struct.pack(">H", len(body)) + body  # ssize 1 x repeat bytes


def box(typ, *children):
    body = b"".join(children)
    return struct.pack(">I", 8 + len(body)) + typ.encode() + body


def gps9_payload(sec, lat0, lon0):
    scal = [10000000, 10000000, 1000, 1000, 100, 1, 1000, 100, 1]
    samples = b""
    for i in range(10):
        lat, lon = lat0 + (sec * 10 + i) * 1e-5, lon0
        samples += struct.pack(">iiiiiiiHH", int(lat * 1e7), int(lon * 1e7), 180000, 8000, 800,
                               9600, (43200 + sec) * 1000 + i * 100, 150, 3)
    strm = nest("STRM",
                klv("STNM", "c", 1, 3, b"GPS"),
                klv("SCAL", "l", 4, 9, struct.pack(">9i", *scal)),
                klv("TYPE", "c", 1, 9, b"lllllllSS"),
                klv("GPS9", "?", 32, 10, samples))
    return nest("DEVC", klv("DVNM", "c", 1, 6, b"HERO12"), strm)


def build_mp4(path, seconds=3):
    payloads = [gps9_payload(s, 42.85, -78.82) for s in range(seconds)]
    ftyp = box("ftyp", b"mp41", b"\0\0\0\0", b"mp41")
    mdat_start = len(ftyp) + 8
    offsets, o = [], mdat_start
    for p in payloads:
        offsets.append(o)
        o += len(p)
    mdat = box("mdat", *payloads)
    fullbox = b"\0\0\0\0"
    mdhd = box("mdhd", fullbox, struct.pack(">IIII", 0, 0, 1000, 1000 * seconds), b"\0\0\0\0")
    hdlr = box("hdlr", fullbox, b"\0\0\0\0meta", b"\0" * 12, b"GoPro MET\0")
    stsd = box("stsd", fullbox, struct.pack(">I", 1), box("gpmd", b"\0" * 6, struct.pack(">H", 1)))
    stts = box("stts", fullbox, struct.pack(">III", 1, seconds, 1001))
    stsc = box("stsc", fullbox, struct.pack(">IIII", 1, 1, 1, 1))
    stsz = box("stsz", fullbox, struct.pack(">II", 0, seconds), *[struct.pack(">I", len(p)) for p in payloads])
    stco = box("stco", fullbox, struct.pack(">I", seconds), *[struct.pack(">I", x) for x in offsets])
    trak = box("trak", box("mdia", mdhd, hdlr, box("minf", box("stbl", stsd, stts, stsc, stsz, stco))))
    Path(path).write_bytes(ftyp + mdat + box("moov", trak))


def test_gps9_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "GX010001.MP4"
        build_mp4(p, seconds=3)
        rows = extract_gps(p)
    assert len(rows) == 30, len(rows)
    r0, r29 = rows[0], rows[29]
    assert abs(r0["lat"] - 42.85) < 1e-6 and abs(r0["lon"] + 78.82) < 1e-6
    assert abs(r29["lat"] - (42.85 + 29e-5)) < 1e-6
    assert r0["fix"] == 3 and abs(r0["dop"] - 1.5) < 1e-9 and abs(r0["alt"] - 180) < 1e-9
    assert abs(r0["speed_2d"] - 8.0) < 1e-9
    assert abs(rows[10]["video_time_s"] - 1.001) < 1e-9          # 2nd payload starts at 1.001 s
    assert r0["gps_time"].startswith("2026-04-14T12:00:00")       # day 9600 since 2000-01-01
    print("GPS9 parser OK:", r0)


if __name__ == "__main__":
    test_gps9_roundtrip()
