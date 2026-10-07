import os
import time
import cv2
import torch
from ultralytics import YOLO
from ultralytics.utils.plotting import colorize_depth

def display_depth_video(
    video_path="vlog_japan.mp4",
    model_path="yolo26n-depth.pt",  # Nano depth model for 2x+ FPS over small model
    imgsz=480,                      # 480 or 320 for much faster processing
    half=None
):
    if not os.path.exists(video_path):
            print(f"[ERROR] Video file '{video_path}' not found.")
            return

    # Check GPU availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if half is None:
        half = (device == "cuda")

    print(f"[INFO] Using Device: {device.upper()} | Half Precision (FP16): {half} | ImgSz: {imgsz}")

    model = YOLO(model_path)
    model.to(device)

    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video '{video_path}'.")
        return

    prev_time = time.time()
    fps_history = []
    max_history = 30
    current_fps = 0.0

    with torch.inference_mode():
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[INFO] End of video reached.")
                break

            # Predict depth with imgsz and half-precision
            result = model.predict(frame, imgsz=imgsz, half=half, device=device, verbose=False)[0]

            # Extract metric depth (must transfer tensor to CPU for NumPy/OpenCV)
            depth = result.depth.data.cpu().numpy()

            # Colorize depth map (near objects are warm/red, far objects are cool/blue)
            colored_depth = colorize_depth(depth, cmap="spectral", mode="disparity")

            # Show only the colorized depth map
            cv2.imshow("YOLO Depth Map Video", colored_depth)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[INFO] Quit key pressed.")
                break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Playback finished.")

if __name__ == "__main__":
    display_depth_video("vlog_japan.mp4", model_path="yolo26n-depth.pt", imgsz=480)