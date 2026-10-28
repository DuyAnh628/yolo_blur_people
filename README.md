# YoloBlur

Small computer-vision scripts for face matching, person tracking and blurring, and hand detection.

## Run a script

Run from the project root:

```powershell
python detect_people.py
python detect_faces.py
python hand_tracking.py
```

The Python implementations are in these root scripts. `detect_people.py` uses `media/vlog_japan.mp4`, `models/yolo26n.pt`, and `assets/faces/face_1.png` by default. The sample video is kept locally because it exceeds GitHub's 100 MB file limit; after cloning, place it at `media/vlog_japan.mp4`. The local depth script is not included in Git.

## Project layout

```text
assets/
  faces/                 # face reference images
  images/                # sample images
media/                   # local-only sample video (not tracked)
models/                  # YOLO weights and OpenVINO model
datasets/                # local training datasets
runs/                    # YOLO training outputs
docs/                    # historical change summary and performance notes
uniface/                 # face detection and recognition library
detect_people.py         # person tracking, identity decisions, and blur
detect_faces.py          # face detection and embedding extraction
hand_tracking.py         # standalone MediaPipe hand example
```

`hand_tracking.py` is a standalone MediaPipe example. The main person-blurring pipeline does not call it.
