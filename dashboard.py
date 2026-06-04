"""
dashboard.py — Layer 1 + interaction
------------------------------------
A Streamlit dashboard showing walks as colour-coded polylines, grouped by
their DBSCAN cluster (route category).

Features:
  - A dropdown to focus on a single cluster (or view all at once)
  - A stats panel: walk count, average distance/pace/duration, date range
  - A seasonality breakdown so you can see if a route is year-round or seasonal
  - A simple naming box to label the selected cluster (writes to the DB)

Cluster colours are generated dynamically, so any number of clusters gets a
distinct colour with no collisions.

Run with:
    streamlit run dashboard.py
"""

import json
import colorsys
import sqlite3

import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium

DB_PATH = "walks.db"

NOISE_COLOR = "#B0B0B0"  # grey for one-off (unclustered) walks


# ── Data loading ─────────────────────────────────────────────────────────────

@st.cache_data
def load_walks():
    """Load all walks from the database into a DataFrame.

    Cached so Streamlit doesn't re-query on every widget interaction.
    The route_json column is parsed from a JSON string into a list of
    segments here, and a few helper columns (season, month) are derived.
    """
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM walks ORDER BY start_time", conn)
    conn.close()

    if df.empty:
        return df

    # route is now a list of segments: [ [[lat,lon],...], [[lat,lon],...] ]
    df["route"] = df["route_json"].apply(json.loads)

    # Derive datetime parts for the monthly-rhythm analysis
    df["dt"] = pd.to_datetime(df["start_time"])
    df["month"] = df["dt"].dt.month
    df["year"] = df["dt"].dt.year

    return df


MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def monthly_average(sub, full_df):
    """Average walks-per-month for a subset of walks, corrected for partial
    years so the result is a fair 'typical year' rhythm.

    For each calendar month (1–12) we count this subset's walks in that month,
    then divide by how many years that month was actually *observed* in the
    whole dataset — not by a flat year count. This stops months that only have
    one year of data from looking artificially low next to months with several.

    Returns a pandas Series indexed by month name (Jan…Dec).
    """
    import pandas as pd

    # Years observed per calendar month, across ALL walks (the data's coverage).
    observed = full_df.groupby("month")["dt"].apply(
        lambda s: s.dt.year.nunique()
    )

    counts = sub.groupby("month").size()

    avg = {}
    for m in range(1, 13):
        total = int(counts.get(m, 0))
        years = int(observed.get(m, 0))
        avg[MONTH_NAMES[m - 1]] = (total / years) if years > 0 else 0.0

    return pd.Series(avg).reindex(MONTH_NAMES)


# ── Colour generation ────────────────────────────────────────────────────────

def make_cluster_colors(cluster_ids):
    """Generate a distinct colour for each cluster by spacing hues evenly
    around the colour wheel. Returns a dict mapping cluster_id -> hex colour.

    Scales to any number of clusters with no collisions. Saturation and
    value are kept moderate to keep the palette muted and earthy rather
    than neon. Noise (-1) is always grey.
    """
    real_clusters = sorted(c for c in cluster_ids if c != -1)
    n = len(real_clusters)

    colors = {-1: NOISE_COLOR}

    for i, cid in enumerate(real_clusters):
        hue = i / max(n, 1)                  # evenly spaced around wheel (0–1)
        r, g, b = colorsys.hsv_to_rgb(hue, 0.45, 0.65)   # muted, earthy
        colors[cid] = f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"

    return colors


# ── Cluster naming (writes to DB) ────────────────────────────────────────────

def get_cluster_name(df, cluster_id):
    """Return the stored route_name for a cluster, if any walk in it has one."""
    names = df.loc[df["cluster_id"] == cluster_id, "route_name"].dropna()
    names = names[names.astype(str).str.len() > 0]
    return names.iloc[0] if len(names) else None


def save_cluster_name(df, cluster_id, name):
    """Save a route name, keyed to the cluster's geographic centroid.

    The name is stored in the route_names table (name + centroid), and also
    written onto the current cluster's walks for immediate display. Because
    it's keyed to location, cluster_walks.py can reattach it after any future
    re-cluster even when the cluster_id changes.

    Passing an empty name removes the name (both the saved entry nearest this
    cluster and the live label).
    """
    sub = df[df["cluster_id"] == cluster_id]
    clat = sub["centroid_lat"].mean()
    clon = sub["centroid_lon"].mean()

    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS route_names (
            name         TEXT PRIMARY KEY,
            centroid_lat REAL NOT NULL,
            centroid_lon REAL NOT NULL
        )
    """)

    # Remove any previous name currently on this cluster (so renaming replaces
    # rather than accumulates).
    old = get_cluster_name(df, cluster_id)
    if old:
        conn.execute("DELETE FROM route_names WHERE name = ?", (old,))

    name = (name or "").strip()
    if name:
        # Upsert the new name with this cluster's centroid.
        conn.execute(
            "INSERT OR REPLACE INTO route_names (name, centroid_lat, centroid_lon) "
            "VALUES (?, ?, ?)",
            (name, float(clat), float(clon)),
        )
        conn.execute(
            "UPDATE walks SET route_name = ? WHERE cluster_id = ?",
            (name, int(cluster_id)),
        )
    else:
        # Clearing the name.
        conn.execute(
            "UPDATE walks SET route_name = NULL WHERE cluster_id = ?",
            (int(cluster_id),),
        )

    conn.commit()
    conn.close()


def cluster_label(df, cluster_id, colors):
    """A human-readable label for the dropdown: name if set, else 'cluster N'."""
    if cluster_id == -1:
        return "One-off walks"
    name = get_cluster_name(df, cluster_id)
    base = name if name else f"Cluster {cluster_id}"
    count = int((df["cluster_id"] == cluster_id).sum())
    return f"{base} ({count})"


# ── Map building ─────────────────────────────────────────────────────────────

def build_map(df, cluster_colors, dim_others=None):
    """Build a Folium map. If dim_others is set to a cluster_id, walks NOT in
    that cluster are drawn faint so the selected route stands out, and the map
    centers/zooms on that cluster. Otherwise it centers on Yellowknife."""

    # Yellowknife town centre — a sensible default view for "All routes".
    YK_LAT, YK_LON = 62.4540, -114.3718
    YK_ZOOM = 12

    if dim_others is not None:
        # Center on the selected cluster's walks, not the global average.
        sel = df[df["cluster_id"] == dim_others]
        center_lat = sel["centroid_lat"].mean()
        center_lon = sel["centroid_lon"].mean()
        zoom = 14   # closer in, since a single route covers a small area
    else:
        center_lat, center_lon, zoom = YK_LAT, YK_LON, YK_ZOOM

    fmap = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom,
        tiles="CartoDB positron",
    )

    for _, walk in df.iterrows():
        segments = walk["route"]
        if not segments:
            continue

        cid = walk["cluster_id"]
        color = cluster_colors[cid]

        # Dimming logic for focus mode
        if dim_others is not None and cid != dim_others:
            weight, opacity = 1.5, 0.15
        else:
            weight, opacity = 3, 0.75

        label = (
            f"{walk['date']} · {walk['distance_km']:.2f} km · "
            f"{walk['duration_min']:.0f} min · cluster {cid}"
        )

        for seg in segments:
            if not seg or len(seg) < 2:
                continue
            folium.PolyLine(
                locations=seg,
                color=color,
                weight=weight,
                opacity=opacity,
                tooltip=label,
            ).add_to(fmap)

    return fmap


# ── Stats panel ──────────────────────────────────────────────────────────────

def render_stats_panel(df, cluster_id, colors):
    """Render the right-hand stats panel for the selected cluster."""
    sub = df[df["cluster_id"] == cluster_id]

    if cluster_id == -1:
        st.subheader("One-off walks")
    else:
        name = get_cluster_name(df, cluster_id)
        title = name if name else f"Cluster {cluster_id}"
        swatch = colors[cluster_id]
        st.markdown(
            f"<h3><span style='display:inline-block;width:14px;height:14px;"
            f"background:{swatch};border-radius:3px;margin-right:6px;'></span>"
            f"{title}</h3>",
            unsafe_allow_html=True,
        )

    # ── Core stats ────────────────────────────────────────────────────────────
    st.metric("Walks in this route", len(sub))

    c1, c2 = st.columns(2)
    c1.metric("Avg distance", f"{sub['distance_km'].mean():.2f} km")
    c2.metric("Avg duration", f"{sub['duration_min'].mean():.0f} min")

    c3, c4 = st.columns(2)
    c3.metric("Avg pace", f"{sub['avg_pace_min_km'].mean():.1f} min/km")
    c4.metric("Total distance", f"{sub['distance_km'].sum():.1f} km")

    st.caption(
        f"From {sub['date'].min()} to {sub['date'].max()}"
    )

    # ── Monthly rhythm ───────────────────────────────────────────────────────
    st.write("**Monthly rhythm** (avg walks per month, typical year)")
    monthly = monthly_average(sub, df)
    st.bar_chart(monthly)

    # A plain-language read on when this route is walked
    walked_months = [MONTH_NAMES[m-1] for m in range(1, 13)
                     if (sub["month"] == m).any()]
    if len(walked_months) == 12:
        st.caption("Walked in every month of the year.")
    elif len(walked_months) <= 3:
        st.caption("Only walked in: " + ", ".join(walked_months) + ".")
    else:
        st.caption("Walked mainly " + walked_months[0] + "–" + walked_months[-1] + ".")

    # ── Naming box (only for real clusters) ──────────────────────────────────
    if cluster_id != -1:
        st.write("**Name this route**")
        current = get_cluster_name(df, cluster_id) or ""
        new_name = st.text_input(
            "Route name", value=current,
            label_visibility="collapsed",
            placeholder="e.g. Frame Lake loop",
            key=f"name_{cluster_id}",
        )
        if st.button("Save name", key=f"save_{cluster_id}"):
            save_cluster_name(df, cluster_id, new_name.strip())
            st.cache_data.clear()   # so the new name shows everywhere
            st.rerun()


# ── Page ─────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(page_title="George's Walk Log", page_icon="🐕",
                       layout="wide")

    st.title("George's Walk Log 🐕")
    st.caption("Walks coloured by route cluster. Pick a route to focus on it.")

    df = load_walks()

    if df.empty:
        st.warning("No walks found in the database. "
                   "Run import_walks.py and cluster_walks.py first.")
        return

    cluster_colors = make_cluster_colors(df["cluster_id"].unique())

    # ── Summary strip ─────────────────────────────────────────────────────────
    n_walks = len(df)
    n_clusters = df.loc[df["cluster_id"] != -1, "cluster_id"].nunique()
    n_noise = int((df["cluster_id"] == -1).sum())
    c1, c2, c3 = st.columns(3)
    c1.metric("Total walks", n_walks)
    c2.metric("Route clusters", n_clusters)
    c3.metric("One-off walks", n_noise)

    # ── Dropdown to choose focus ──────────────────────────────────────────────
    # Build options: "All" plus each cluster (real clusters first, noise last)
    cluster_ids = sorted(
        df["cluster_id"].unique(),
        key=lambda c: (c == -1, c),   # push -1 to the end
    )
    options = ["All routes"] + [
        cluster_label(df, cid, cluster_colors) for cid in cluster_ids
    ]
    # Map the label back to its cluster_id
    label_to_id = {"All routes": None}
    for cid in cluster_ids:
        label_to_id[cluster_label(df, cid, cluster_colors)] = cid

    choice = st.selectbox("Focus on a route", options)
    selected_id = label_to_id[choice]

    # ── Two-column layout: map left, stats right ──────────────────────────────
    map_col, stats_col = st.columns([2, 1])

    with map_col:
        fmap = build_map(df, cluster_colors, dim_others=selected_id)
        st_folium(fmap, width=None, height=600, returned_objects=[])

    with stats_col:
        if selected_id is None:
            st.subheader("All routes")
            st.metric("Total distance walked", f"{df['distance_km'].sum():.1f} km")
            st.metric("Total time", f"{df['duration_min'].sum()/60:.1f} hours")
            st.caption("Pick a specific route from the dropdown to see its "
                       "stats and seasonality, and to name it.")
            st.write("**Monthly rhythm (all walks)**")
            st.caption("Average walks per month in a typical year")
            st.bar_chart(monthly_average(df, df))
        else:
            render_stats_panel(df, selected_id, cluster_colors)


if __name__ == "__main__":
    main()
