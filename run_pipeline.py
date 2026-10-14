"""Command-line runner.

  python run_pipeline.py                      # every video in data/videos + photos in data/images
  python run_pipeline.py --video data/videos/GX010001.MP4 --max-seconds 120
  python run_pipeline.py --images             # only the photos in data/images
  python run_pipeline.py --detector clip      # CLIP scores only, no boxes (faster)
  python run_pipeline.py --detector mock      # skip the ML models (fast, fake scores)
"""
import argparse
from pathlib import Path

import config
from pipeline.detect import get_detector
from pipeline.runner import process_images, process_video


def main():
    ap = argparse.ArgumentParser(description="Housing code violation detection pipeline")
    ap.add_argument("--video", action="append", help="video file(s) to process (default: all in data/videos)")
    ap.add_argument("--images", action="store_true", help="process only photos in data/images")
    ap.add_argument("--every-m", type=float, default=config.SAMPLE_EVERY_M, help="sample one frame every N metres")
    ap.add_argument("--max-seconds", type=float, help="only process the first N seconds of each video")
    ap.add_argument("--detector", choices=["clip+boxes", "clip", "mock"], default=config.DETECTOR)
    a = ap.parse_args()

    for d in (config.VIDEO_DIR, config.IMAGE_DIR, config.PARCEL_DIR):
        d.mkdir(parents=True, exist_ok=True)

    videos = [Path(v) for v in a.video] if a.video else (
        [] if a.images else sorted(p for p in config.VIDEO_DIR.iterdir() if p.suffix.lower() in config.VIDEO_EXTS))
    images = [] if a.video else sorted(p for p in config.IMAGE_DIR.iterdir() if p.suffix.lower() in config.IMAGE_EXTS)
    if not videos and not images:
        print(f"Nothing to do. Put GoPro videos in {config.VIDEO_DIR} or photos in {config.IMAGE_DIR}.")
        return

    print(f"Loading detector '{a.detector}' ...")
    det = get_detector(a.detector)
    for v in videos:
        try:
            process_video(v, every_m=a.every_m, max_seconds=a.max_seconds, detector=det)
        except Exception as e:
            print(f"   ! {v.name} failed: {e}")
    if images:
        process_images(images, detector=det)
    print("\nOpen the dashboard with:  python app.py   ->  http://127.0.0.1:5000")


if __name__ == "__main__":
    main()
