"""STAGE 3 - MATCH: Which property is it?

  GPS position + direction of travel -> "look point" on the facade side of the road
  -> parcel polygon that contains it (if a parcel GeoJSON is provided)
  -> otherwise a "virtual lot" (fixed-size grid cell), so the demo works with no parcel data.

Grouping frames by property means repeat drives (e.g. March 30 and April 17) land on
the same property record.
"""
import json
import math

import config
from .geo import destination, haversine, to_local_m

_ID_KEYS = ["SBL", "sbl", "PARCELID", "parcel_id", "PRINTKEY", "print_key", "PIN", "id", "ID", "OBJECTID"]
_ADDR_KEYS = ["ADDRESS", "address", "FULL_ADDRESS", "full_address", "PROP_ADDR", "LOCATION", "situs_address"]


class ParcelIndex:
    def __init__(self):
        self.parcels = []  # dicts: id, address, rings [[(lon,lat),...]], bbox, centroid
        for path in sorted(config.PARCEL_DIR.glob("*.geojson")) + sorted(config.PARCEL_DIR.glob("*.json")):
            self._load(path)

    def _load(self, path):
        gj = json.loads(path.read_text(encoding="utf-8"))
        for i, feat in enumerate(gj.get("features", [])):
            geom, props = feat.get("geometry") or {}, feat.get("properties") or {}
            if geom.get("type") == "Polygon":
                polys = [geom["coordinates"]]
            elif geom.get("type") == "MultiPolygon":
                polys = geom["coordinates"]
            else:
                continue
            rings = [p[0] for p in polys if p]
            pts = [pt for r in rings for pt in r]
            if not pts:
                continue
            pid = next((str(props[k]) for k in _ID_KEYS if props.get(k) not in (None, "")), f"{path.stem}-{i}")
            addr = next((str(props[k]) for k in _ADDR_KEYS if props.get(k)), None)
            self.parcels.append(dict(
                id=pid, address=addr, rings=rings,
                bbox=(min(p[0] for p in pts), min(p[1] for p in pts), max(p[0] for p in pts), max(p[1] for p in pts)),
                centroid=(sum(p[1] for p in pts) / len(pts), sum(p[0] for p in pts) / len(pts)),
            ))

    def __len__(self):
        return len(self.parcels)

    def lookup(self, lat, lon):
        for p in self.parcels:
            x0, y0, x1, y1 = p["bbox"]
            if x0 <= lon <= x1 and y0 <= lat <= y1 and any(_inside(lon, lat, r) for r in p["rings"]):
                return p
        best, best_d = None, config.PARCEL_MAX_DIST_M
        for p in self.parcels:
            d = haversine(lat, lon, *p["centroid"])
            if d < best_d:
                best, best_d = p, d
        return best


def _inside(x, y, ring):
    """Ray-casting point-in-polygon."""
    inside = False
    j = len(ring) - 1
    for i in range(len(ring)):
        xi, yi = ring[i][0], ring[i][1]
        xj, yj = ring[j][0], ring[j][1]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi + 1e-15) + xi:
            inside = not inside
        j = i
    return inside


def look_point(lat, lon, heading):
    side = 90 if config.CAMERA_SIDE == "right" else -90
    return destination(lat, lon, (heading + side) % 360, config.LOOK_OFFSET_M)


def match_property(llat, llon, parcels):
    """Map a look point to a property. Returns dict(property_key, address, lat, lon, look_lat, look_lon, method)."""
    if parcels is not None and len(parcels):
        p = parcels.lookup(llat, llon)
        if p:
            return dict(property_key=f"parcel:{p['id']}", address=p["address"] or f"Parcel {p['id']}",
                        lat=p["centroid"][0], lon=p["centroid"][1], look_lat=llat, look_lon=llon, method="parcel")
    # virtual lot: snap the look point to a LOT_SIZE_M grid (lat0 fixed so cells are stable across runs)
    lat0 = round(llat, 1)
    x, y = to_local_m(llat, llon, lat0)
    cx, cy = math.floor(x / config.LOT_SIZE_M), math.floor(y / config.LOT_SIZE_M)
    # centre of the cell back in degrees
    clat = math.degrees((cy + 0.5) * config.LOT_SIZE_M / 6371000.0)
    clon = math.degrees((cx + 0.5) * config.LOT_SIZE_M / (6371000.0 * math.cos(math.radians(lat0))))
    return dict(property_key=f"lot:{lat0}:{cx}:{cy}", address=f"Lot near {clat:.5f}, {clon:.5f}",
                lat=clat, lon=clon, look_lat=llat, look_lon=llon, method="virtual_lot")
