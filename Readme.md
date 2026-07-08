# Street Vision

Street Vision is a real-time computer vision pipeline that watches live public EarthCam streams, buffers frames for smooth playback, and lets you detect and track any object you describe in plain text — using OWLv2 and Grounding DINO in parallel, with lightweight continuous visual tracking.

**Demo video:** https://youtu.be/oZiTN4s_GmU

## Features
- Live EarthCam HLS stream ingestion (any public EarthCam feed)
- Frame buffering for smooth, jitter-free playback
- Dual zero-shot object detectors (OWLv2 + Grounding DINO) running in parallel
- Duplicate detection merging via IoU
- Continuous lightweight object tracking (OpenCV CSRT)
- Text-based object queries — just describe what to track
- Fully local/offline inference after first-time model download

## Requirements
- Ubuntu 20.04+ (tested on 24.04)
- NVIDIA GPU (CUDA) recommended
- Miniconda / Anaconda

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/harshbari-153/street_vision.git
cd street_vision
```

### 2. Create a conda environment
```bash
conda create -n street_vision python=3.10 -y
conda activate street_vision
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

## Running the Program

### 1. Get a live stream URL
Open any EarthCam page (see `live camera streams.txt` for tested links), open browser DevTools → Network tab → filter `m3u8` → copy the `playlist.m3u8` link (with its token).

### 2. Set the stream URL
Open `main.py` and set:
```python
STREAM_URL = "your_m3u8_link_here"
```

### 3. First run
```bash
python main.py
```
On first run, it will automatically create a `model/` folder and download OWLv2 + Grounding DINO (one-time, requires internet). It will also create an empty `query.txt`.

### 4. Set what to detect
Edit `query.txt` with comma-separated objects (see `sample_query_text.txt` for examples), e.g.:
human with hat, green car, dog
street light, red color human

- Line 1 → tracked with a **green** box
- Line 2 (optional) → tracked with a **red** box

### 5. Controls
Type these in the terminal (not the video window), followed by Enter:
| Key | Action |
|---|---|
| `p` | Run detection on the current frame and start tracking |
| `q` | Stop and exit |

## Notes
- EarthCam stream tokens expire — refresh the URL periodically.
- Detection runs once per `p` press; tracking then continues live until the object shrinks out of frame.
