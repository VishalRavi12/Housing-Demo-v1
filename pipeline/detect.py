"""STAGE 4 - DETECT: What looks wrong?

Two zero-shot models:
  * CLIP (vision-language) decides HOW LIKELY each violation is. For every type it compares
    the frame with a "problem" sentence and a "normal" sentence; the softmax of the two
    similarities is the confidence. A "building in view" gate skips frames of empty road.
  * OWLv2 (open-vocabulary detector) finds WHERE: it takes short text queries such as
    "a boarded up window" and returns bounding boxes with a score.

Output per frame: dict(building=p, scores={type: conf}, boxes=[{violation_type, label, score, x0..y1}])
Box coordinates are fractions (0-1) of image width/height, so they work at any image size.
"""
import hashlib

import cv2
import numpy as np

import config

_GATE = ("a photo of the front of a house or building", "a photo of an empty road, trees or sky with no buildings")


def _device(torch):
    return "cuda" if torch.cuda.is_available() else "cpu"


class ClipDetector:
    name = "clip"

    def __init__(self):
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self.torch = torch
        self.device = _device(torch)
        self.model = CLIPModel.from_pretrained(config.CLIP_MODEL, use_safetensors=True).to(self.device).eval()
        self.proc = CLIPProcessor.from_pretrained(config.CLIP_MODEL)
        self.pairs = [_GATE] + list(config.VIOLATIONS.values())
        prompts = [p for pair in self.pairs for p in pair]
        with torch.no_grad():
            tok = self.proc(text=prompts, return_tensors="pt", padding=True).to(self.device)
            t = self.model.get_text_features(**tok)
            self.text = t / t.norm(dim=-1, keepdim=True)

    def predict(self, images_bgr):
        torch = self.torch
        rgb = [cv2.cvtColor(im, cv2.COLOR_BGR2RGB) for im in images_bgr]
        with torch.no_grad():
            px = self.proc(images=rgb, return_tensors="pt").to(self.device)
            v = self.model.get_image_features(**px)
            v = v / v.norm(dim=-1, keepdim=True)
            logits = self.model.logit_scale.exp() * v @ self.text.T  # [N, 2*pairs]
            logits = logits.view(len(rgb), len(self.pairs), 2)
            probs = logits.softmax(dim=-1)[..., 0].cpu().numpy()  # P(first prompt) per pair
        out = []
        for row in probs:
            out.append(dict(building=float(row[0]), boxes=[],
                            scores={k: float(s) for k, s in zip(config.VIOLATIONS, row[1:])}))
        return out


class OwlBoxDetector:
    """Open-vocabulary object detector: text queries in, bounding boxes out."""

    def __init__(self):
        import torch
        from transformers import Owlv2ForObjectDetection, Owlv2Processor

        self.torch = torch
        self.device = _device(torch)
        self.model = Owlv2ForObjectDetection.from_pretrained(config.BOX_MODEL, use_safetensors=True).to(self.device).eval()
        self.proc = Owlv2Processor.from_pretrained(config.BOX_MODEL)
        self.queries = [(vtype, q) for vtype, qs in config.BOX_QUERIES.items() for q in qs]

    BATCH = 2  # OWLv2 runs at 960x960 with large attention maps; small batches fit an 8 GB GPU

    def boxes(self, images_bgr):
        out = []
        for i in range(0, len(images_bgr), self.BATCH):
            out += self._boxes(images_bgr[i:i + self.BATCH])
        return out

    def _boxes(self, images_bgr):
        from torchvision.ops import nms
        torch = self.torch
        rgb = [cv2.cvtColor(im, cv2.COLOR_BGR2RGB) for im in images_bgr]
        texts = [[q for _, q in self.queries]] * len(rgb)
        with torch.no_grad():
            inputs = self.proc(text=texts, images=rgb, return_tensors="pt").to(self.device)
            out = self.model(**inputs)
            # OWLv2 pads every image to a square (bottom/right), so boxes come back in the
            # coordinates of a side x side square; dividing by the real w/h gives fractions.
            sides = [max(im.shape[:2]) for im in rgb]
            res = self.proc.image_processor.post_process_object_detection(
                outputs=out, threshold=config.BOX_THRESHOLD,
                target_sizes=torch.tensor([[s, s] for s in sides], device=self.device))

        all_boxes = []
        for im, r in zip(rgb, res):
            h, w = im.shape[:2]
            found = []
            types = np.array([self.queries[int(i)][0] for i in r["labels"]])
            for vtype in config.BOX_QUERIES:
                idx = np.where(types == vtype)[0]
                if not len(idx):
                    continue
                b, s = r["boxes"][idx], r["scores"][idx]
                keep = nms(b.float(), s.float(), 0.3)[: config.BOX_MAX_PER_TYPE]
                for k in keep.tolist():
                    x0, y0, x1, y1 = b[k].tolist()
                    x0, x1 = max(0.0, x0 / w), min(1.0, x1 / w)
                    y0, y1 = max(0.0, y0 / h), min(1.0, y1 / h)
                    if x1 <= x0 or y1 <= y0 or (x1 - x0) * (y1 - y0) > config.BOX_MAX_AREA:
                        continue
                    found.append(dict(violation_type=vtype, label=self.queries[int(r["labels"][idx[k]])][1],
                                      score=float(s[k]), x0=x0, y0=y0, x1=x1, y1=y1))
            all_boxes.append(found)
        return all_boxes


class ClipWithBoxes:
    """CLIP decides the confidence, OWLv2 shows where."""
    name = "clip+owlv2"

    def __init__(self):
        self.clip = ClipDetector()
        self.owl = OwlBoxDetector()

    def predict(self, images_bgr):
        preds = self.clip.predict(images_bgr)
        for p, b in zip(preds, self.owl.boxes(images_bgr)):
            p["boxes"] = b
        return preds


class MockDetector:
    """No-ML stand-in so the app can be demoed without downloading a model.
    Scores and boxes are deterministic pseudo-random values derived from the image content."""
    name = "mock"

    def predict(self, images_bgr):
        out = []
        for im in images_bgr:
            h = hashlib.md5(cv2.resize(im, (16, 16)).tobytes()).digest()
            rnd = np.frombuffer(h, dtype=np.uint8) / 255.0
            scores = {k: float(0.25 + 0.75 * rnd[i] ** 2) for i, k in enumerate(config.VIOLATIONS)}
            boxes = []
            for i, (k, s) in enumerate(scores.items()):
                if s > 0.55:
                    x, y = 0.1 + 0.6 * rnd[i + 5], 0.2 + 0.5 * rnd[i + 10]
                    boxes.append(dict(violation_type=k, label=config.BOX_QUERIES[k][0], score=round(s / 3, 3),
                                      x0=x, y0=y, x1=min(1.0, x + 0.2), y1=min(1.0, y + 0.2)))
            out.append(dict(building=float(0.4 + 0.6 * rnd[-1]), scores=scores, boxes=boxes))
        return out


def get_detector(name=None):
    name = name or config.DETECTOR
    try:
        if name == "clip+boxes":
            return ClipWithBoxes()
        if name == "clip":
            return ClipDetector()
    except Exception as e:  # missing package / no internet for first download
        print(f"   ! {name} unavailable ({e.__class__.__name__}: {e}); falling back to mock detector")
    return MockDetector()
