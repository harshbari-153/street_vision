import time
import torch
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
import config

device = "cuda" if torch.cuda.is_available() else "cpu"


class Detector:
    def __init__(self, owl_path, dino_path):
        from transformers import (
            Owlv2Processor,
            Owlv2ForObjectDetection,
            AutoProcessor,
            AutoModelForZeroShotObjectDetection,
        )

        self.owl_proc = Owlv2Processor.from_pretrained(owl_path)
        self.owl_model = Owlv2ForObjectDetection.from_pretrained(owl_path).to(device).eval()

        self.dino_proc = AutoProcessor.from_pretrained(dino_path)
        self.dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(dino_path).to(device).eval()

        self.executor = ThreadPoolExecutor(max_workers=2)

    def _run_owl(self, image, phrases):
        inputs = self.owl_proc(text=[phrases], images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = self.owl_model(**inputs)
        target_sizes = torch.tensor([image.size[::-1]]).to(device)
        results = self.owl_proc.post_process_grounded_object_detection(
            outputs=outputs, target_sizes=target_sizes, threshold=config.DETECTION_SCORE_THRESHOLD
        )[0]
        boxes = results["boxes"].cpu().numpy().tolist()
        scores = results["scores"].cpu().numpy().tolist()
        labels_idx = results["labels"].cpu().numpy().tolist()
        labels = [phrases[i] for i in labels_idx]
        return list(zip(boxes, scores, labels))

    def _run_dino(self, image, phrases):
        text = ". ".join(phrases) + "."
        inputs = self.dino_proc(images=image, text=text, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = self.dino_model(**inputs)
        results = self.dino_proc.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=config.DETECTION_SCORE_THRESHOLD,
            text_threshold=config.DETECTION_SCORE_THRESHOLD,
            target_sizes=[image.size[::-1]],
        )[0]
        boxes = results["boxes"].cpu().numpy().tolist()
        scores = results["scores"].cpu().numpy().tolist()
        labels = results["labels"]
        return list(zip(boxes, scores, labels))

    @staticmethod
    def _iou(b1, b2):
        x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
        x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
        a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
        union = a1 + a2 - inter
        return inter / union if union > 0 else 0

    def _dedupe(self, dets_a, dets_b):
        merged = list(dets_a)
        for box_b, score_b, label_b in dets_b:
            duplicate = False
            for box_a, score_a, label_a in dets_a:
                if (
                    label_a.strip().lower() == label_b.strip().lower()
                    and self._iou(box_a, box_b) > config.IOU_DEDUPE_THRESHOLD
                ):
                    duplicate = True
                    break
            if not duplicate:
                merged.append((box_b, score_b, label_b))
        return merged

    def detect(self, frame_bgr, phrases):
        image = Image.fromarray(frame_bgr[:, :, ::-1])
        start = time.time()
        fut_owl = self.executor.submit(self._run_owl, image, phrases)
        fut_dino = self.executor.submit(self._run_dino, image, phrases)
        dets_owl = fut_owl.result()
        dets_dino = fut_dino.result()
        merged = self._dedupe(dets_owl, dets_dino)
        elapsed = time.time() - start
        return merged, elapsed
