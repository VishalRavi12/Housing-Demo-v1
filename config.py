"""Central settings for the Housing Violation demo. Tweak these for your footage."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ---- Folders --------------------------------------------------------------
DATA_DIR = ROOT / "data"
VIDEO_DIR = DATA_DIR / "videos"      # <- put GoPro .MP4 files here
IMAGE_DIR = DATA_DIR / "images"      # <- put user-uploaded photos (with EXIF GPS) here
PARCEL_DIR = DATA_DIR / "parcels"    # <- optional parcel boundaries (.geojson)
OUTPUT_DIR = ROOT / "outputs"        # telemetry CSVs, frame JPGs, etc.
DB_PATH = OUTPUT_DIR / "housing.db"

VIDEO_EXTS = {".mp4", ".mov"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

# ---- Stage 1: LOCATE (GPS cleaning) ----------------------------------------
MIN_GPS_FIX = 2          # 0 = no fix, 2 = 2D fix, 3 = 3D fix
MAX_DOP = 5.0            # dilution of precision; lower is better (<2 is very good)
MAX_JUMP_SPEED_MS = 60   # points implying > 60 m/s (216 km/h) are treated as GPS jumps
SMOOTH_WINDOW = 5        # rolling-median window (samples, ~0.5 s at 10 Hz)

# ---- Stage 2: SELECT -------------------------------------------------------
SAMPLE_EVERY_M = 8.0     # keep one frame every N metres travelled
BLUR_THRESHOLD = 40.0    # variance of Laplacian; lower = blurrier
DUP_HASH_DISTANCE = 6    # dHash Hamming distance; <= this means near-duplicate
FRAME_SAVE_WIDTH = 1280  # saved JPG width (px)

# ---- Stage 3: MATCH --------------------------------------------------------
CAMERA_SIDE = "right"    # which side of the road the property is assumed to be on
LOOK_OFFSET_M = 18.0     # distance from the road centre to the house facade (m)
LOT_SIZE_M = 25.0        # grid size for "virtual lots" when no parcel file is present
PARCEL_MAX_DIST_M = 40.0 # nearest-parcel fallback radius

# ---- Stage 4: DETECT -------------------------------------------------------
# "clip+boxes" = CLIP scores the frame + OWLv2 draws boxes where it finds each problem
# "clip"       = CLIP scores only (faster, no boxes)
# "mock"       = no ML, fake scores and boxes (for testing the app)
DETECTOR = "clip+boxes"
CLIP_MODEL = "openai/clip-vit-large-patch14"   # ~1.7 GB, downloaded once on first run
HOUSE_GATE = 0.50        # min probability that a building is in view

BOX_MODEL = "google/owlv2-base-patch16-ensemble"  # ~620 MB open-vocabulary detector
BOX_THRESHOLD = 0.12     # OWLv2 scores run low; 0.1-0.3 is a typical useful range
BOX_MAX_PER_TYPE = 3     # keep at most N boxes per violation type per frame
BOX_MAX_AREA = 0.85      # ignore boxes covering more than 85% of the frame
# What OWLv2 looks for. Several phrasings per type; the label shown is the phrase that matched.
BOX_QUERIES = {
    "broken_window": ["a broken window", "a boarded up window", "a window covered with plywood", "shattered window glass"],
    "debris": ["a pile of trash", "garbage bags", "junk and debris", "discarded furniture", "a tire on the ground"],
    "roof_damage": ["a damaged roof", "missing roof shingles", "a blue tarp on a roof", "a hole in a roof"],
    "structural_damage": ["a collapsed porch", "a sagging roof", "a damaged wall", "broken porch steps"],
    "peeling_paint": ["peeling paint", "flaking paint on siding", "weathered bare wood siding"],
}

# Each violation type: (what the problem looks like, what "normal" looks like).
# CLIP scores the frame against both prompts; the softmax gives a confidence.
VIOLATIONS = {
    "broken_window": (
        "a photo of a house with a broken or boarded-up window",
        "a photo of a house with intact clean windows",
    ),
    "debris": (
        "a photo of a yard full of trash, junk and debris",
        "a photo of a clean tidy front yard",
    ),
    "roof_damage": (
        "a photo of a house with a damaged roof with missing shingles or a tarp",
        "a photo of a house with a roof in good condition",
    ),
    "structural_damage": (
        "a photo of a house with a collapsing porch or sagging structure",
        "a photo of a structurally sound house",
    ),
    "peeling_paint": (
        "a photo of a house with peeling, flaking paint on the siding",
        "a photo of a house with fresh, well-maintained paint",
    ),
}

# ---- Stage 5: REVIEW -------------------------------------------------------
DISCARD_BELOW = 0.55     # predictions below this are not recorded as possible violations
AUTO_ACCEPT_AT = 0.90    # at or above this they are "likely" (still reviewable)
                         # everything in between goes to the human review queue
