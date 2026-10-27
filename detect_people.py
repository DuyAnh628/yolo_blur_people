import threading
import cv2
import torch
from ultralytics import YOLO
from detect_faces import detect_faces
from uniface.face_utils import compute_similarity
# Note: Đã loại bỏ module nhận diện tay (hand_tracking) để tối ưu CPU và đơn giản hóa pipeline

device = "cuda" if torch.cuda.is_available() else "cpu"

# Target face profile
target_face_embedding = detect_faces("face_1.png")

# Cache: track_id -> status and face-check scheduling data
identity_cache = {}
identity_cache_lock = threading.Lock()
FACE_RECHECK_INTERVAL = 3

def thread_track(track_id, crop):
    try:
        # A face detection is required before identity can be decided.
        emb = detect_faces(crop)
        if emb is None:
            status = "no_face"
        elif target_face_embedding is not None:
            similarity = compute_similarity(emb, target_face_embedding, normalized=True)
            status = "match" if similarity > 0.2 else "no_match"
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

def detect_person_yolov8(video_path):

    model = YOLO("yolo26n.pt").to(device)
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
                              classes=[0], conf=0.15, verbose=False)
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
                                     "last_check": -FACE_RECHECK_INTERVAL}
                            identity_cache[track_id] = state
                        should_check = (
                            not state["checking"] and
                            frame_index - state["last_check"] >= FACE_RECHECK_INTERVAL
                        )
                        if should_check:
                            state["checking"] = True
                            state["last_check"] = frame_index
                        label = state["status"]

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

                    # Blur the whole person only while a visible face identifies them as a stranger.
                    if label == "no_match":
                        roi = frame[y1:y2, x1:x2]
                        blurred = cv2.blur(roi, (31, 31))
                        frame[y1:y2, x1:x2] = blurred

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
    detect_person_yolov8("vlog_japan.mp4")
