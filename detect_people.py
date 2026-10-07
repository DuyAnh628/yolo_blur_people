import threading
import cv2
import torch
from ultralytics import YOLO
from detect_faces import detect_faces
from uniface.face_utils import compute_similarity

# Tự động chọn GPU (CUDA) nếu có, ngược lại dùng CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# Target face embedding to compare (extract once outside the loop)
target_embedding = detect_faces("face_1.png")

# Cache: track_id -> "match" | "no match" | None
identity_cache = {}

def thread_track(track_id, crop):
    emb = detect_faces(crop)
    if emb is not None:
        s = compute_similarity(emb, target_embedding, normalized=True)
        identity_cache[track_id] = "match" if s > 0.175 else "no match"
    else:
        identity_cache[track_id] = None

def detect_person_yolov8(video_path):

    # Load pretrained YOLOv8n model và nạp lên GPU
    model = YOLO("yolo26n.pt").to(device)

    # Open video file instead of webcam
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Video couldn't be opened.")
        return

    print("[INFO] Running YOLOv8 Person Detection on Video. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:  # End of video
            break

        # Run YOLOv8 inference trên GPU
        results = model.track(frame, device=device, persist=True,
                              classes=[0], conf=0.25, verbose=False)

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

                    # Nhận diện chỉ khi chưa có trong cache
                    if track_id not in identity_cache:
                        identity_cache[track_id] = "pending" 
                        crop = frame[y1:y2, x1:x2].copy()
                        threading.Thread(target=thread_track, args=(track_id, crop), daemon=True).start()

                    label = identity_cache.get(track_id) or "unknown"
                    color = (0, 255, 0) if label == "match" else (0, 0, 255)

                    # Blur người không match hoặc chưa nhận diện được
                    if label != "match":
                        roi = frame[y1:y2, x1:x2]
                        blurred = cv2.GaussianBlur(roi, (51, 51), 0)
                        frame[y1:y2, x1:x2] = blurred

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Xóa track_id không còn trong frame
        for sid in set(identity_cache) - active_ids:
            del identity_cache[sid]

        cv2.imshow("YOLOv8 Person Detection - Video", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Detection complete.")

if __name__ == "__main__":
    detect_person_yolov8("vlog_japan.mp4")
