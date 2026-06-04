# George's Walk Log 🐕

A personal dashboard for exploring GPS walk data — built around walks with my dog George in Northwest Territories.

GPX files from Apple Health exported from a third-party [Apple Health export app](https://healthexport.app) are parsed into a local SQLite database, walks are clustered into route categories by location, and everything is displayed in an interactive Streamlit dashboard with maps, stats, and trends.

## AI Disclosure

This project is done using Claude AI. All code is reviewed, and edited where required by author, but is largely AI-driven. This is a learning project to enhance python and sqlite skills, as well as an introduction to Streamlit. This is a first foray into using AI to generate a personalized app! 


## Pipeline

```
GPX files  →  import_walks.py  →  walks.db  →  cluster_walks.py  →  dashboard.py
  (raw)         (parse + thin)    (SQLite)      (DBSCAN routes)      (Streamlit)
```

## Setup

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

1. **Add your GPX files** to the `gpx/` folder.
2. **Import them** into the database:
   ```bash
   python import_walks.py
   ```
   Safe to re-run — already-imported files are skipped.
3. **Cluster walks** into route categories by location:
   ```bash
   python cluster_walks.py
   ```
4. **Launch the dashboard:**
   ```bash
   streamlit run dashboard.py
   ```

## Database schema

**`walks`** — one row per walk, with computed stats, a thinned route (stored as JSON), and a cluster label.

**`imported_files`** — tracks which GPX files have already been processed, so imports are idempotent.

## Notes

- Raw GPX files and the database are **git-ignored** — they contain home coordinates and are kept private.
- Routes are thinned with the Ramer–Douglas–Peucker algorithm before storage to keep the database small while preserving route shape on the map.
- Clustering uses DBSCAN with a Haversine distance metric, so walks are grouped by real-world location.

## Tech

Python · pandas · SQLite · scikit-learn · Streamlit · Folium · rdp



