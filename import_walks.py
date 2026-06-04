"""
import_walks.py
--------------
Scans the gpx/ folder for new walk files, parses each one,
and writes a summary row into the SQLite database (walks.db).
Safe to run multiple times — already-imported files are skipped.

Usage:
    python import_walks.py
"""

import os
import json
import math
import sqlite3
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

from rdp import rdp  # pip install rdp

# ── Configuration ────────────────────────────────────────────────────────────

GPX_FOLDER = "gpx"          # folder containing your .gpx files
DB_PATH    = "walks.db"     # SQLite database file

# RDP tolerance: higher = more thinning. 0.0001 degrees ≈ ~10 metres.
# At this tolerance a 2700-point walk typically thins to ~80–150 points.
RDP_TOLERANCE = 0.0001

# GPS outlier detection. A real signal dropout is a LARGE distance jump over
# a short time (the tracker lost GPS and reconnected far away). We require BOTH
# conditions before splitting a route:
#   1. the step covers at least MIN_JUMP_METRES, AND
#   2. it implies a speed above MAX_SPEED_KMH (or happens in zero time)
# This distinction matters: ordinary GPS jitter wobbles a few metres between
# readings, which over a 1-second interval can compute to a high "speed" even
# though almost no distance was covered. Judging by speed alone wrongly splits
# on that jitter; requiring a real distance jump as well only catches genuine
# dropouts (like a tracker that teleported hundreds of metres).
MAX_SPEED_KMH = 20.0     # speed ceiling for a normal-time step
MIN_JUMP_METRES = 150.0  # a step must cover at least this far to count as a jump

# GPX namespace used by the HealthExport app
NS = {"gpx": "http://www.topografix.com/GPX/1/1"}


# ── Database setup ────────────────────────────────────────────────────────────

def init_db(conn):
    """Create tables if they don't already exist."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS walks (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            filename        TEXT UNIQUE NOT NULL,
            date            TEXT NOT NULL,       -- ISO date, e.g. 2026-06-03
            start_time      TEXT NOT NULL,       -- ISO datetime UTC
            end_time        TEXT NOT NULL,
            duration_min    REAL NOT NULL,
            distance_km     REAL NOT NULL,
            elevation_gain_m REAL NOT NULL,
            elevation_loss_m REAL NOT NULL,
            avg_pace_min_km REAL NOT NULL,
            start_lat       REAL NOT NULL,
            start_lon       REAL NOT NULL,
            centroid_lat    REAL NOT NULL,
            centroid_lon    REAL NOT NULL,
            route_json      TEXT NOT NULL,       -- thinned [[lat,lon], ...] as JSON
            cluster_id      INTEGER DEFAULT -1,  -- assigned later by clustering step
            route_name      TEXT DEFAULT NULL    -- assigned later manually or by you
        );

        CREATE TABLE IF NOT EXISTS imported_files (
            filename    TEXT PRIMARY KEY,
            imported_at TEXT NOT NULL
        );
    """)
    conn.commit()


# ── GPX parsing ───────────────────────────────────────────────────────────────

def parse_gpx(filepath):
    """
    Parse a GPX file and return a dict of walk statistics.
    Returns None if the file has no track points.
    """
    tree = ET.parse(filepath)
    root = tree.getroot()

    points = root.findall(".//gpx:trkpt", NS)
    if not points:
        print(f"  ⚠️  No track points found in {filepath}, skipping.")
        return None

    lats, lons, eles, times = [], [], [], []

    for p in points:
        lats.append(float(p.get("lat")))
        lons.append(float(p.get("lon")))

        ele_el = p.find("gpx:ele", NS)
        eles.append(float(ele_el.text) if ele_el is not None else 0.0)

        time_el = p.find("gpx:time", NS)
        if time_el is not None:
            times.append(_parse_time(time_el.text))

    if len(times) < 2:
        print(f"  ⚠️  Not enough timestamps in {filepath}, skipping.")
        return None

    # ── Core metrics ──────────────────────────────────────────────────────────

    # ── Split the track at GPS dropouts ──────────────────────────────────────
    # This removes impossible jumps from the distance total and lets us draw
    # each clean segment separately (no straight line across signal gaps).

    segments, n_jumps = _split_into_segments(
        lats, lons, eles, times, MAX_SPEED_KMH
    )

    # ── Core metrics (computed within segments only) ─────────────────────────

    duration_min = (times[-1] - times[0]).total_seconds() / 60.0

    # Distance: sum within each segment, skipping the jumps between them.
    distance_km = sum(
        _total_distance_km(seg["lats"], seg["lons"])
        for seg in segments if len(seg["lats"]) >= 2
    )

    # Elevation: also computed within segments, to avoid a fake spike at a jump.
    elev_gain = 0.0
    elev_loss = 0.0
    for seg in segments:
        e = seg["eles"]
        elev_gain += sum(max(0.0, e[i] - e[i-1]) for i in range(1, len(e)))
        elev_loss += sum(max(0.0, e[i-1] - e[i]) for i in range(1, len(e)))

    avg_pace = duration_min / distance_km if distance_km > 0 else 0.0

    # Centroid uses all points — a few outliers barely shift the average, and
    # it keeps clustering stable.
    centroid_lat = sum(lats) / len(lats)
    centroid_lon = sum(lons) / len(lons)

    # ── Thin each segment for map display ─────────────────────────────────────
    # route_json is now a LIST OF SEGMENTS: [ [[lat,lon],...], [[lat,lon],...] ]
    # Each segment is thinned independently and drawn as its own polyline.

    thinned_segments = []
    for seg in segments:
        coords = list(zip(seg["lats"], seg["lons"]))
        if len(coords) < 2:
            continue
        thinned = rdp(coords, epsilon=RDP_TOLERANCE)
        thinned_segments.append(
            [[round(lat, 6), round(lon, 6)] for lat, lon in thinned]
        )

    route_json = json.dumps(thinned_segments)

    if n_jumps:
        print(f"(cleaned {n_jumps} GPS jump{'s' if n_jumps > 1 else ''}) ", end="")

    return {
        "date":             times[0].strftime("%Y-%m-%d"),
        "start_time":       times[0].isoformat(),
        "end_time":         times[-1].isoformat(),
        "duration_min":     round(duration_min, 2),
        "distance_km":      round(distance_km, 4),
        "elevation_gain_m": round(elev_gain, 1),
        "elevation_loss_m": round(elev_loss, 1),
        "avg_pace_min_km":  round(avg_pace, 2),
        "start_lat":        round(lats[0], 6),
        "start_lon":        round(lons[0], 6),
        "centroid_lat":     round(centroid_lat, 6),
        "centroid_lon":     round(centroid_lon, 6),
        "route_json":       route_json,
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_time(text):
    """Parse a GPX timestamp string to a timezone-aware datetime."""
    text = text.strip()
    # Handle both 'Z' suffix and '+00:00' style
    text = text.replace("Z", "+00:00")
    return datetime.fromisoformat(text)


def _haversine_km(lat1, lon1, lat2, lon2):
    """Return the great-circle distance in kilometres between two points."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _total_distance_km(lats, lons):
    """Sum haversine distances along the track."""
    return sum(
        _haversine_km(lats[i-1], lons[i-1], lats[i], lons[i])
        for i in range(1, len(lats))
    )


def _split_into_segments(lats, lons, eles, times, max_speed_kmh):
    """Split a track into clean segments at GPS dropouts.

    A real dropout is a LARGE distance jump over a short time. The crucial
    point is that ordinary GPS jitter wobbles only a few metres between
    readings — but over a 1-second interval that small wobble can still
    compute to a high "speed". So speed alone produces false splits on jitter.

    We therefore require a genuine distance jump (>= MIN_JUMP_METRES) AS WELL
    AS impossibility (too fast, or a leap in zero time) before splitting. A
    6-metre wobble never splits; a 300-metre teleport does.

    Returns a list of segments (each a dict of lats/lons/eles/times) and the
    number of jumps found. The bogus jump between segments is excluded from
    distance and never drawn on the map.
    """
    min_jump_km = MIN_JUMP_METRES / 1000.0

    segments = []
    seg_lats, seg_lons, seg_eles, seg_times = [lats[0]], [lons[0]], [eles[0]], [times[0]]
    n_jumps = 0

    for i in range(1, len(lats)):
        gap_km = _haversine_km(lats[i-1], lons[i-1], lats[i], lons[i])
        gap_hr = (times[i] - times[i-1]).total_seconds() / 3600.0

        # A jump must FIRST be a real distance leap. Small wobble is never a
        # jump, regardless of the implied speed.
        if gap_km < min_jump_km:
            is_jump = False
        elif gap_hr > 0:
            is_jump = (gap_km / gap_hr) > max_speed_kmh   # far AND fast
        else:
            is_jump = True                                # far AND zero-time

        if is_jump:
            segments.append({
                "lats": seg_lats, "lons": seg_lons,
                "eles": seg_eles, "times": seg_times,
            })
            seg_lats, seg_lons, seg_eles, seg_times = [], [], [], []
            n_jumps += 1

        seg_lats.append(lats[i])
        seg_lons.append(lons[i])
        seg_eles.append(eles[i])
        seg_times.append(times[i])

    segments.append({
        "lats": seg_lats, "lons": seg_lons,
        "eles": seg_eles, "times": seg_times,
    })

    return segments, n_jumps


# ── Main import loop ──────────────────────────────────────────────────────────

def already_imported(conn, filename):
    row = conn.execute(
        "SELECT 1 FROM imported_files WHERE filename = ?", (filename,)
    ).fetchone()
    return row is not None


def mark_imported(conn, filename):
    conn.execute(
        "INSERT INTO imported_files (filename, imported_at) VALUES (?, ?)",
        (filename, datetime.now(timezone.utc).isoformat()),
    )


def insert_walk(conn, filename, stats):
    conn.execute("""
        INSERT INTO walks (
            filename, date, start_time, end_time,
            duration_min, distance_km, elevation_gain_m, elevation_loss_m,
            avg_pace_min_km, start_lat, start_lon,
            centroid_lat, centroid_lon, route_json
        ) VALUES (
            :filename, :date, :start_time, :end_time,
            :duration_min, :distance_km, :elevation_gain_m, :elevation_loss_m,
            :avg_pace_min_km, :start_lat, :start_lon,
            :centroid_lat, :centroid_lon, :route_json
        )
    """, {"filename": filename, **stats})


def run_import():
    if not os.path.isdir(GPX_FOLDER):
        print(f"❌  GPX folder '{GPX_FOLDER}' not found.")
        print("    Create a 'gpx/' folder next to this script and drop your files in.")
        return

    gpx_files = sorted(
        f for f in os.listdir(GPX_FOLDER) if f.lower().endswith(".gpx")
    )

    if not gpx_files:
        print(f"No .gpx files found in '{GPX_FOLDER}/'.")
        return

    conn = sqlite3.connect(DB_PATH)
    init_db(conn)

    new_count  = 0
    skip_count = 0

    for filename in gpx_files:
        if already_imported(conn, filename):
            skip_count += 1
            continue

        filepath = os.path.join(GPX_FOLDER, filename)
        print(f"  Importing {filename} ...", end=" ")

        stats = parse_gpx(filepath)
        if stats is None:
            skip_count += 1
            continue

        insert_walk(conn, filename, stats)
        mark_imported(conn, filename)
        conn.commit()

        print(
            f"✅  {stats['date']}  "
            f"{stats['distance_km']:.2f} km  "
            f"{stats['duration_min']:.0f} min  "
            f"route: {len(json.loads(stats['route_json']))} points"
        )
        new_count += 1

    conn.close()

    print(f"\nDone — {new_count} new walk(s) imported, {skip_count} skipped.")
    if new_count > 0:
        print("Next step: run cluster_walks.py to assign route categories.")


if __name__ == "__main__":
    run_import()
