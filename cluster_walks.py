"""
cluster_walks.py
----------------
Groups walks into route categories based on WHERE they happen, using DBSCAN
with a haversine (real-world distance) metric on each walk's centroid.

Re-clusters ALL walks from scratch each run, so it's safe to run again
after importing new walks. Writes the cluster label back to each walk's
`cluster_id` column in walks.db.

Usage:
    python cluster_walks.py
"""

import math
import sqlite3
import numpy as np
from sklearn.cluster import DBSCAN

# ── Configuration ────────────────────────────────────────────────────────────

DB_PATH = "walks.db"

# How close two walks' centroids must be (in metres) to be considered the
# same route area. Walks within this distance of each other get grouped.
# Start at 300m; tune up if separate routes get merged, down if one route
# splits into several clusters.
EPS_METRES = 300

# Minimum number of walks needed to form a cluster. With min_samples=2,
# any pair of nearby walks forms a route; a one-off walk stays "noise" (-1).
MIN_SAMPLES = 2

# How strongly walk DIRECTION influences clustering, relative to location.
# Walks from the same spot heading opposite ways should separate, while
# location still dominates overall. This is the "equivalent distance" in
# metres assigned to two walks heading in fully opposite directions (180°
# apart) from the same point. Larger = direction matters more.
#   - Set comparable to EPS_METRES so opposite directions roughly equal
#     "one cluster-width apart".
#   - Set to 0 to ignore direction entirely (old centroid-only behaviour).
DIRECTION_WEIGHT_METRES = 350

EARTH_RADIUS_M = 6_371_000  # for converting the metre threshold to radians

# When reattaching saved names after clustering, a cluster is matched to a
# saved name if its centroid is within this many metres of the name's stored
# centroid. Yellowknife routes are geographically distinct, so this can be
# generous without causing mismatches.
NAME_MATCH_METRES = 400


# ── Clustering ───────────────────────────────────────────────────────────────

def _haversine_m(lat1, lon1, lat2, lon2):
    """Great-circle distance in metres between two lat/lon points."""
    R = EARTH_RADIUS_M
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _bearing_rad(lat1, lon1, lat2, lon2):
    """Compass bearing (in radians, 0 = north) from point 1 to point 2.

    If the two points coincide (a walk whose centroid equals its start, e.g.
    a tight loop), bearing is undefined; we return 0.0, which simply means
    direction contributes nothing for that walk.
    """
    if lat1 == lat2 and lon1 == lon2:
        return 0.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dlam = math.radians(lon2 - lon1)
    x = math.sin(dlam) * math.cos(phi2)
    y = (math.cos(phi1) * math.sin(phi2)
         - math.sin(phi1) * math.cos(phi2) * math.cos(dlam))
    return math.atan2(x, y)


def ensure_names_table(conn):
    """Create the route_names table if it doesn't exist.

    Names are keyed to a geographic centroid (where the route is), NOT to a
    cluster_id (which changes every re-cluster). This is what lets names
    survive re-clustering.
    """
    conn.execute("""
        CREATE TABLE IF NOT EXISTS route_names (
            name         TEXT PRIMARY KEY,
            centroid_lat REAL NOT NULL,
            centroid_lon REAL NOT NULL
        )
    """)
    conn.commit()


def reattach_names(conn, cluster_centroids):
    """Match saved names to current clusters by nearest centroid.

    cluster_centroids: dict of {cluster_id: (lat, lon)} for real clusters.

    For each cluster, find the closest saved name within NAME_MATCH_METRES and
    write it onto that cluster's walks. Clears route_name first so stale names
    never linger. Returns a list of saved names that found NO matching cluster,
    so the caller can warn about them.
    """
    # Start clean: no walk keeps a name unless it's reassigned this run.
    conn.execute("UPDATE walks SET route_name = NULL")

    saved = conn.execute(
        "SELECT name, centroid_lat, centroid_lon FROM route_names"
    ).fetchall()

    if not saved:
        conn.commit()
        return []

    matched_names = set()

    # For each cluster, find the nearest saved name within the threshold.
    for cid, (clat, clon) in cluster_centroids.items():
        best_name, best_dist = None, float("inf")
        for s in saved:
            d = _haversine_m(clat, clon, s["centroid_lat"], s["centroid_lon"])
            if d < best_dist:
                best_name, best_dist = s["name"], d

        if best_name is not None and best_dist <= NAME_MATCH_METRES:
            conn.execute(
                "UPDATE walks SET route_name = ? WHERE cluster_id = ?",
                (best_name, int(cid)),
            )
            matched_names.add(best_name)

    conn.commit()

    # Any saved name not matched to a cluster this run.
    all_names = {s["name"] for s in saved}
    return sorted(all_names - matched_names)


def cluster_walks():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    ensure_names_table(conn)

    rows = conn.execute(
        "SELECT id, start_lat, start_lon, centroid_lat, centroid_lon FROM walks"
    ).fetchall()

    if not rows:
        print("No walks found in the database. Run import_walks.py first.")
        conn.close()
        return

    if len(rows) < MIN_SAMPLES:
        print(f"Only {len(rows)} walk(s) in the database — need at least "
              f"{MIN_SAMPLES} to cluster. Skipping.")
        conn.close()
        return

    # ── Build a feature vector per walk: location + weighted direction ────────
    #
    # Location: we can't feed raw lat/lon into a Euclidean metric (degrees of
    # longitude shrink as you go north). So we project to local metres on a
    # flat plane centred on the data — fine for a city-sized area. X is east,
    # Y is north, both in metres.
    #
    # Direction: the bearing from each walk's START to its CENTROID captures
    # which way the walk headed away from its origin. We encode it as
    # (sin, cos) so the 0°/360° wrap is smooth, then scale it so that two
    # fully-opposite walks sit DIRECTION_WEIGHT_METRES apart in feature space.

    lat0 = float(np.mean([r["centroid_lat"] for r in rows]))   # projection origin
    lon0 = float(np.mean([r["centroid_lon"] for r in rows]))
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = 111_320.0 * math.cos(math.radians(lat0))

    # The (sin, cos) direction vector has length 1; two opposite directions are
    # 2 units apart. We want that to equal DIRECTION_WEIGHT_METRES, so scale by
    # half the weight.
    dir_scale = DIRECTION_WEIGHT_METRES / 2.0

    features = []
    for r in rows:
        # Location in local metres
        x = (r["centroid_lon"] - lon0) * m_per_deg_lon
        y = (r["centroid_lat"] - lat0) * m_per_deg_lat

        # Direction of travel: bearing from start -> centroid
        brg = _bearing_rad(
            r["start_lat"], r["start_lon"],
            r["centroid_lat"], r["centroid_lon"],
        )
        dx = math.sin(brg) * dir_scale
        dy = math.cos(brg) * dir_scale

        features.append([x, y, dx, dy])

    X = np.array(features)

    # Now everything is in metres, so plain Euclidean distance with eps in
    # metres does what we want: walks cluster when they're close in BOTH
    # location and direction.
    db = DBSCAN(
        eps=EPS_METRES,
        min_samples=MIN_SAMPLES,
        metric="euclidean",
    )
    labels = db.fit_predict(X)

    # ── Write labels back to the database ────────────────────────────────────

    ids = [r["id"] for r in rows]
    for walk_id, label in zip(ids, labels):
        conn.execute(
            "UPDATE walks SET cluster_id = ? WHERE id = ?",
            (int(label), walk_id),
        )
    conn.commit()

    # ── Report ────────────────────────────────────────────────────────────────

    n_clusters = len(set(labels) - {-1})
    n_noise = int(np.sum(labels == -1))

    print(f"Clustered {len(rows)} walks into {n_clusters} route group(s).")
    if n_noise:
        print(f"  {n_noise} walk(s) didn't fit any group (cluster_id = -1, "
              f"one-off locations).")

    # Show how many walks fall into each cluster, with a representative centroid
    cluster_centroids = {}   # {cluster_id: (lat, lon)} for real clusters
    print("\nCluster summary:")
    for label in sorted(set(labels)):
        member_rows = [
            (r["centroid_lat"], r["centroid_lon"])
            for r, l in zip(rows, labels) if l == label
        ]
        count = len(member_rows)
        avg_lat = sum(m[0] for m in member_rows) / count
        avg_lon = sum(m[1] for m in member_rows) / count
        name = "noise / one-offs" if label == -1 else f"cluster {label}"
        print(f"  {name:<18} {count:>3} walk(s)   "
              f"~({avg_lat:.4f}, {avg_lon:.4f})")
        if label != -1:
            cluster_centroids[int(label)] = (avg_lat, avg_lon)

    # ── Reattach saved names by location ──────────────────────────────────────
    unmatched = reattach_names(conn, cluster_centroids)
    if unmatched:
        print("\n⚠️  These saved route names found NO matching cluster this run:")
        for nm in unmatched:
            print(f"     • {nm}")
        print("     (The route may have dropped below min_samples, or your")
        print("      EPS_METRES changed. The name is kept but unused for now.)")

    conn.close()

    print("\nTip: open the dashboard to see these on a map. Named routes keep")
    print("their names automatically; new routes appear unnamed for you to label.")


if __name__ == "__main__":
    cluster_walks()
