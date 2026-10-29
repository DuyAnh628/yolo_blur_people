import threading
from pathlib import Path
import cv2
import torch
from ultralytics import YOLO
from detect_faces import detect_faces
from uniface.face_utils import compute_similarity

device = "cuda" if torch.cuda.is_available() else "cpu"

# Each usable image in assets/faces is a separate person to keep unblurred.
# Multiple selected people are accepted independently: matching any one of
# their reference images is enough to leave that tracked face clear.
reference_face_embeddings = {}
for reference_path in sorted(Path("assets/faces").glob("face_*")):
    if reference_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
        continue
    embedding = detect_faces(str(reference_path))
    if embedding is not None:
        reference_face_embeddings[reference_path.stem] = embedding

FACE_SIMILARITY_THRESHOLD = 0.15

# Cache: track_id -> status and face-check scheduling data
identity_cache = {}
identity_cache_lock = threading.Lock()
FACE_RECHECK_INTERVAL = 3

def thread_track(track_id, crop):
    face_box = None
    try:
        # A face detection is required before identity can be decided.
        emb, face_box = detect_faces(crop, return_bbox=True)
        if emb is None:
            status = "no_face"
        elif reference_face_embeddings:
            similarities = [
                compute_similarity(emb, reference, normalized=True)
                for reference in reference_face_embeddings.values()
            ]
            # Any selected person can match. The references are alternatives
            # for the keep-clear set, not photos that one face must match all of.
            status = "match" if max(similarities) > FACE_SIMILARITY_THRESHOLD else "no_match"
        else:
            # With no configured target face, a visible face cannot be matched.
            status = "no_match"
    except Exception as exc:
        print(f"[WARN] Face check failed for track {track_id}: {exc}")
        status = "no_face"

    with identity_cache_lock:
        state = identity_cache.get(track_id)
        if state is not None:
            state["status"] = status
            state["checking"] = False
            if face_box is not None:
                h, w = crop.shape[:2]
                state["face_box"] = (
                    face_box[0] / w, face_box[1] / h,
                    face_box[2] / w, face_box[3] / h,
                )
            else:
                state["face_box"] = None

def detect_person_yolov8(video_path):

    model = YOLO("models/yolo26n.pt").to(device)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Video couldn't be opened.")
        return

    print("[INFO] Running YOLOv8 Person Detection on Video. Press 'q' to quit.")

    frame_index = 0
    while True:
        ret, frame = cap.read()
        if not ret:  # End of video
            break
        frame_index += 1

        # Run YOLOv8 inference on GPU
        results = model.track(frame, device=device, persist=True,
                              classes=[0], conf=0.2, verbose=False)
        active_ids = set()
        for r in results:
            if r.boxes.id is None:
                continue
            for box, track_id_t in zip(r.boxes, r.boxes.id):
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                # Only keep class 'person' (COCO class id 0)
                if cls == 0 and conf > 0.2:
                    track_id = int(track_id_t)
                    active_ids.add(track_id)
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                    if x2 <= x1 or y2 <= y1:
                        continue

                    with identity_cache_lock:
                        state = identity_cache.get(track_id)
                        if state is None:
                            state = {"status": "pending", "checking": False,
                                 "last_check": -FACE_RECHECK_INTERVAL,
                                 "face_box": None}
                            identity_cache[track_id] = state
                        should_check = (
                            not state["checking"] and
                            frame_index - state["last_check"] >= FACE_RECHECK_INTERVAL
                        )
                        if should_check:
                            state["checking"] = True
                            state["last_check"] = frame_index
                        label = state["status"]
                        face_box = state["face_box"]

                    if should_check:
                        crop = frame[y1:y2, x1:x2].copy()
                        threading.Thread(
                            target=thread_track, args=(track_id, crop), daemon=True
                        ).start()

                    color = (0, 255, 0) if label == "match" else (0, 0, 255)
                    display_label = {
                        "match": "target",
                        "no_match": "stranger",
                        "no_face": "no visible face",
                        "pending": "checking face",
                    }.get(label, "unknown")

                    # Blur only the detected face of a stranger.
                    if label == "no_match" and face_box is not None:
                        person_w, person_h = x2 - x1, y2 - y1
                        face_x1 = x1 + int(face_box[0] * person_w)
                        face_y1 = y1 + int(face_box[1] * person_h)
                        face_x2 = x1 + int(face_box[2] * person_w)
                        face_y2 = y1 + int(face_box[3] * person_h)
                        pad_x = int((face_x2 - face_x1) * 0.3)
                        pad_y = int((face_y2 - face_y1) * 0.3)
                        fx1 = max(x1, face_x1 - pad_x)
                        fy1 = max(y1, face_y1 - pad_y)
                        fx2 = min(x2, face_x2 + pad_x)
                        fy2 = min(y2, face_y2 + pad_y)
                        if fx2 > fx1 and fy2 > fy1:
                            roi = frame[fy1:fy2, fx1:fx2]
                            frame[fy1:fy2, fx1:fx2] = cv2.blur(roi, (31, 31))

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, display_label, (x1, max(20, y1 - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        # Remove track_ids no longer in the current frame from cache
        # If the person reappears in subsequent frames, they will be re-added and re-identified
        with identity_cache_lock:
            for tid in list(identity_cache.keys()):
                if tid not in active_ids:
                    del identity_cache[tid]

        cv2.imshow("YOLOv8 Person Detection - Video", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Detection complete.")

if __name__ == "__main__":
    detect_person_yolov8("media/vlog_warehouse.mp4")
