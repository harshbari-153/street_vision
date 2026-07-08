import os
import time
from collections import deque
from threading import Thread
import cv2
import config

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "headers;Referer: https://www.earthcam.com/\r\n"


class StreamBuffer:
    def __init__(self, url):
        self.url = url
        self.buffer = deque(maxlen=config.BUFFER_SIZE)
        self.stop_flag = False
        self.thread = Thread(target=self._capture, daemon=True)

    def start(self):
        self.thread.start()

    def _capture(self):
        cap = cv2.VideoCapture(self.url)
        fail_count = 0
        while not self.stop_flag:
            ret, frame = cap.read()
            if not ret:
                fail_count += 1
                if fail_count > 50:
                    cap.release()
                    cap = cv2.VideoCapture(self.url)
                    fail_count = 0
                time.sleep(0.05)
                continue
            fail_count = 0
            frame = cv2.resize(frame, (config.WIDTH, config.HEIGHT))
            self.buffer.append(frame)
        cap.release()

    def is_full(self):
        return len(self.buffer) >= config.BUFFER_SIZE

    def get_frame(self):
        if self.buffer:
            return self.buffer.popleft()
        return None

    def stop(self):
        self.stop_flag = True
