import glob
import threading
import cv2
import torch
from ultralytics import YOLO
from detect_faces import detect_faces
from uniface.face_utils import compute_similarity
# Note: Đã loại bỏ module nhận diện tay (hand_tracking) để tối ưu CPU và đơn giản hóa pipeline

device = "cuda" if torch.cuda.is_available() else "cpu"

# Target profiles: face and body (loaded once)
target_face_embedding = detect_faces("face_1.png")
# Extract color histogram from body_1.png (simple, lightweight, and fast)
def extract_body_hist(image):
    if isinstance(image, str):
        image = cv2.imread(image)
    if image is None or image.size == 0:
        return None
    # Use HSV color space to represent clothing/body color distribution
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
    cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return hist

target_body_hists = []
for file in sorted(glob.glob("body_*.png")):
    hist = extract_body_hist(file)
    if hist is not None:
        target_body_hists.append(hist)

# Cache: track_id -> "match" | "no match" | "pending"
identity_cache = {}

def thread_track(track_id, crop):
    # 1. Match against face embedding first
    emb = detect_faces(crop)
    if emb is not None and target_face_embedding is not None:
        s = compute_similarity(emb, target_face_embedding, normalized=True)
        if s > 0.2:
            if track_id in identity_cache:
                identity_cache[track_id] = "match"
            return

    # 2. Fallback: match body/clothing histogram if face is occluded or turned away
    if target_body_hists:
        crop_hist = extract_body_hist(crop)
        if crop_hist is not None:
            matched = any(cv2.compareHist(h, crop_hist, cv2.HISTCMP_CORREL) > 0.75 for h in target_body_hists)
            if matched:
                if track_id in identity_cache:
                    identity_cache[track_id] = "match"
                return

    if track_id in identity_cache:
        identity_cache[track_id] = "no match"

def detect_person_yolov8(video_path):

    model = YOLO("yolo26n.pt").to(device)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Video couldn't be opened.")
        return

    print("[INFO] Running YOLOv8 Person Detection on Video. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:  # End of video
            break

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

                    # Detect face if not in cache
                    if track_id not in identity_cache:
                        identity_cache[track_id] = "pending" 
                        crop = frame[y1:y2, x1:x2].copy()
                        threading.Thread(target=thread_track, args=(track_id, crop), daemon=True).start()

                    label = identity_cache.get(track_id) or "unknown"
                    color = (0, 255, 0) if label == "match" else (0, 0, 255)
                    
                    # Only blur when confirmed as "no match"
                    # Ignore during "pending" to prevent initial frame flickering for the target person
                    # Note: Làm mờ trực tiếp người không khớp (bỏ kiểm tra tay để tránh flicker và nghẽn CPU)
                    if label == "no match":
                        roi = frame[y1:y2, x1:x2]
                        blurred = cv2.blur(roi, (31, 31))
                        frame[y1:y2, x1:x2] = blurred

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Remove track_ids no longer in the current frame from cache
        # If the person reappears in subsequent frames, they will be re-added and re-identified
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
