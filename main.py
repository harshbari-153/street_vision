import sys
import time
import threading
import cv2

import config
from setup_models import first_run_setup
from stream import StreamBuffer
from detector import Detector
from tracker import ObjectTracker

# Update this whenever the EarthCam token expires
STREAM_URL = "https://videos-3.earthcam.com/fecnetwork/24322.flv/playlist.m3u8?t=qAP3aum0UbcBtTuO%2Fx%2F7Lz9UytxcCWnrPDJyjgaIxerCQuFS76zjEHDlTZ2IQvp0&td=202607080209"

command_lock = threading.Lock()
pending_command = None


def input_listener():
    global pending_command
    while True:
        try:
            cmd = input().strip().lower()
        except EOFError:
            break
        with command_lock:
            pending_command = cmd
        if cmd == "q":
            break


def parse_query():
    with open(config.QUERY_FILE, "r") as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
    return lines


def main():
    owl_path, dino_path = first_run_setup()

    print("Loading models...")
    detector = Detector(owl_path, dino_path)

    stream = StreamBuffer(STREAM_URL)
    stream.start()
    print("Buffering...")
    while not stream.is_full():
        print(f"Buffered {len(stream.buffer)}/{config.BUFFER_SIZE}", end="\r")
        time.sleep(0.5)
    print("Buffer full. Starting playback.")

    tracker = ObjectTracker()
    tracking_enabled = False
    avg_time_text = ""

    cv2.namedWindow("Stream", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Stream", config.WIN_WIDTH, config.WIN_HEIGHT)

    threading.Thread(target=input_listener, daemon=True).start()

    delay = 1.0 / config.FPS
    global pending_command
    running = True

    while running:
        loop_start = time.time()
        frame = stream.get_frame()

        if frame is not None:
            display = frame.copy()

            with command_lock:
                cmd = pending_command
                pending_command = None

            if cmd == "q":
                running = False
                break

            if cmd == "p":
                lines = parse_query()

                if len(lines) == 0:
                    tracker.reset()
                    tracking_enabled = False
                    avg_time_text = ""

                elif len(lines) >= 3:
                    print("Error: query.txt must contain at most 2 lines.")

                else:
                    tracker.reset()
                    all_phrases = []
                    line_map = []
                    for idx, line in enumerate(lines):
                        for p in [x.strip() for x in line.split(",") if x.strip()]:
                            all_phrases.append(p)
                            line_map.append(idx)

                    dets, elapsed = detector.detect(display, all_phrases)
                    avg_time_text = f"Detection time: {elapsed:.2f}s"

                    for box, score, label in dets:
                        line_idx = 0
                        for i, phrase in enumerate(all_phrases):
                            if phrase.lower() in label.lower() or label.lower() in phrase.lower():
                                line_idx = line_map[i]
                                break
                        color = (0, 255, 0) if line_idx == 0 else (0, 0, 255)
                        tracker.add(display, box, color)

                    tracking_enabled = True

            if tracking_enabled:
                display = tracker.update(display)

            if avg_time_text:
                cv2.putText(display, avg_time_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow("Stream", display)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            running = False

        elapsed_loop = time.time() - loop_start
        if elapsed_loop < delay:
            time.sleep(delay - elapsed_loop)

    stream.stop()
    cv2.destroyAllWindows()
    sys.exit(0)


if __name__ == "__main__":
    main()
