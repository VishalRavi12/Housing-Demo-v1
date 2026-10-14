"""STAGE 2 - SELECT: Is this frame worth keeping?

  distance-based sampling -> blur filter -> near-duplicate removal
"""
import cv2
import numpy as np

import config


def distance_targets(frames, every_m):
    """Pick the first frame each time the vehicle has travelled another `every_m` metres.
    Stops at traffic lights produce no extra frames because distance doesn't grow."""
    f = frames[frames["has_gps"]]
    bucket = np.floor(f["cum_dist_m"].values / every_m).astype(int)
    first = np.concatenate([[True], bucket[1:] != bucket[:-1]])
    return f[first].copy()


def blur_score(img):
    """Variance of the Laplacian - sharp images have strong edges, so higher = sharper."""
    gray = cv2.cvtColor(cv2.resize(img, (640, int(640 * img.shape[0] / img.shape[1]))), cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def dhash(img, size=8):
    """64-bit difference hash; similar images have a small Hamming distance."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (size + 1, size), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def hamming(a, b):
    return bin(a ^ b).count("1")


SEEK_GAP = 150  # frames; jump ahead (e.g. past a long stop) instead of decoding every frame


def read_frames(video_path, wanted_idx):
    """Yield (idx, image) for the wanted frame indices. Short gaps are decoded sequentially
    with grab() (skips colour conversion); long gaps use a seek."""
    cap = cv2.VideoCapture(str(video_path))
    pos = 0  # index of the frame the next grab() returns
    try:
        for t in sorted(set(int(i) for i in wanted_idx)):
            if t - pos > SEEK_GAP:
                cap.set(cv2.CAP_PROP_POS_FRAMES, t)
                pos = t
            while pos < t:
                if not cap.grab():
                    return
                pos += 1
            ok, img = cap.read()
            pos += 1
            if not ok:
                return
            yield t, img
    finally:
        cap.release()


def resize_for_save(img):
    h, w = img.shape[:2]
    if w <= config.FRAME_SAVE_WIDTH:
        return img
    s = config.FRAME_SAVE_WIDTH / w
    return cv2.resize(img, (config.FRAME_SAVE_WIDTH, int(h * s)), interpolation=cv2.INTER_AREA)
