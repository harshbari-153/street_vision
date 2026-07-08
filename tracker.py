import cv2
import config


class TrackedObject:
    def __init__(self, tr, color, init_area):
        self.tracker = tr
        self.color = color
        self.init_area = init_area


class ObjectTracker:
    def __init__(self):
        self.objects = []

    def reset(self):
        self.objects = []

    def add(self, frame, box, color):
        x1, y1, x2, y2 = box
        bbox = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
        tr = cv2.TrackerCSRT_create()
        tr.init(frame, bbox)
        init_area = max(bbox[2] * bbox[3], 1)
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), color, 2)
        self.objects.append(TrackedObject(tr, color, init_area))

    def update(self, frame):
        alive = []
        for obj in self.objects:
            ok, bbox = obj.tracker.update(frame)
            if not ok:
                continue
            x, y, bw, bh = bbox
            if (bw * bh) / obj.init_area < config.MIN_TRACK_AREA_RATIO:
                continue
            cv2.rectangle(frame, (int(x), int(y)), (int(x + bw), int(y + bh)), obj.color, 2)
            alive.append(obj)
        self.objects = alive
        return frame
