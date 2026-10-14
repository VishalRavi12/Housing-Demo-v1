# Housing Code Violation Detection: Local Demo (Team Debuggers)

A working, fully local version of the architecture in the project overview:
GoPro video → **Locate → Select → Match → Detect → Review** → data store/API → user views.
Salesforce is replaced by a local web app with the same three audiences (Admin, Inspector, Public).

## Quick start

```powershell
cd housing_demo
pip install -r requirements.txt          # already done on this machine
# 1. copy GoPro .MP4 files into  data\videos\
# 2. start the dashboard
python app.py                            # or double-click start_demo.bat
# 3. open http://127.0.0.1:5000  ->  "Run Pipeline"  ->  ▶ Run
```

Or run the pipeline from the terminal, which shows a stage-by-stage log you can present:

```powershell
python run_pipeline.py                              # all videos in data\videos + photos in data\images
python run_pipeline.py --video data\videos\GX010001.MP4 --max-seconds 120
python run_pipeline.py --detector clip              # CLIP scores only, no boxes (faster)
python run_pipeline.py --detector mock              # no ML model, instant (for testing)
python tests\test_gpmf.py                           # GPS9 parser self-test
```

The first run downloads CLIP (~1.7 GB) and OWLv2 (~620 MB) once, then works offline. Only the map tiles need internet.

## How the code maps to the architecture

| Stage | Question | File | What it does |
|---|---|---|---|
| 1 Locate | Where was the camera? | `pipeline/gpmf.py`, `pipeline/locate.py` | Reads the GPMF track straight from the MP4 (no ffmpeg), decodes **GPS9** (lat, lon, alt, speed, GPS time, DOP, fix), or GPS5 on older GoPros. Drops points with a poor fix, high DOP, or impossible jumps, smooths the track, computes direction of travel, and interpolates a position for every frame. Saves `outputs/telemetry/*_gps_raw.csv` and `*_gps_clean.csv`. |
| 2 Select | Is this frame worth keeping? | `pipeline/select_frames.py` | Keeps one frame every *N* metres, so a stop at a traffic light adds no frames. Then applies a blur filter (variance of Laplacian) and near-duplicate removal (dHash). |
| 3 Match | Which property is it? | `pipeline/match.py` | Projects a "look point" from the camera position, perpendicular to the direction of travel, and finds the parcel polygon containing it. Without parcel data it uses 25 m virtual lots. Frames are grouped by property, so both drives (Mar 30 and Apr 17) land on the same record. |
| 4 Detect | What looks wrong? | `pipeline/detect.py` | Two zero-shot models on the GPU. **CLIP ViT-L/14** decides *how likely* each violation is: it scores the whole frame against a "problem" vs. "normal" sentence, and a building-in-view gate skips empty road. **OWLv2** finds *where*: short text queries such as "a boarded up window" return bounding boxes, which are drawn on the Review and Property pages. Output: violation type, confidence, supporting frame, boxes. |
| 5 Review | Can we trust it? | `pipeline/db.py`, `/review` page | Confidence < 55 % is discarded, 55–90 % goes to **Needs review**, ≥ 90 % is **Likely**. An inspector can confirm, reject, or correct, and re-running a video never overwrites a human decision. |
| Data store / API | | `pipeline/db.py`, `app.py` | SQLite tables: Property, Frame + location, Prediction, Violation, Review. JSON at `/api/properties` and `/api/violations`, CSV export on the Metrics page. |
| "Salesforce" | | `app.py` + `templates/` | **Admin**: Run Pipeline. **Inspector**: Review. **Public**: confirmed violations only. |

User photos with EXIF GPS (from `data\images` or uploaded on the Run Pipeline page) skip Stages 1 and 2, as in the diagram.

## Folders

```
data/videos/    <- GoPro .MP4 files (originals, so the GPS track is intact)
data/images/    <- optional user photos with GPS
data/parcels/   <- optional parcel GeoJSON (address matching)
outputs/        <- housing.db, telemetry CSVs, extracted frame JPGs (safe to delete to start over)
config.py       <- all thresholds and the violation prompts
```

## Demo tips

- **Process before the presentation.** 4K/5.3K HEVC decoding is the slow part. Run the full videos beforehand, then show the dashboard and do a short live run with "Only first 60 seconds".
- Suggested walkthrough: Run Pipeline (live log, all 5 stages) → Dashboard (route map, funnel of frames kept) → a property (evidence frames, line-of-sight map) → Review (confirm one, reject one, correct one) → Public view (only the confirmed one appears) → Metrics (precision from reviews, CSV/JSON export).
- **Tuning** (`config.py`): too many or too few frames → `SAMPLE_EVERY_M`. Good frames dropped as blurry → lower `BLUR_THRESHOLD`. Review queue flooded → raise `DISCARD_BELOW`. Prompts are in `VIOLATIONS` and are the easiest way to improve accuracy. Too many or too few boxes → `BOX_THRESHOLD`; what OWLv2 looks for → `BOX_QUERIES`.
- **Boxes as training data:** Metrics → `boxes.csv` lists every box (image path, type, label, score, coordinates as fractions of the image) together with the inspector's decision on that violation. This is the starting point for labelling a YOLO training set.
- `CAMERA_SIDE` / `LOOK_OFFSET_M` control which side of the street a frame is attributed to.
- The video must be the camera original. Re-encoded copies usually lose the GPS track. As a fallback, a `<video>.csv` sidecar with `video_time_s,lat,lon` columns works.
