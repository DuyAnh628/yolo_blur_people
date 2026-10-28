# Video Smoothness Analysis, Bottlenecks & Optimization Notes

## 1. Root Causes of Stutter / Lag and Their Solutions

Even though the GPU handles YOLO inference efficiently, stutter and lack of smoothness in the video playback stemmed from the following bottlenecks:

### 1.1 Repeated Initialization of MediaPipe Hand Tracking on CPU
- **Issue:** Inside `hand_tracking.py`, every time `detect_hands()` was invoked, it re-initialized the full graph model `mp_hands.Hands(static_image_mode=True)` from scratch, drew landmarks onto a temporary image copy, and discarded it. This heavily consumed CPU cycles and created contention over Python's GIL (Global Interpreter Lock) with the video rendering loop.
- **Solution:** Reuse a single persistent `Hands` detector instance configured with tracking mode (`static_image_mode=False`), and downscale the input frame before inference so that hand detection finishes in ~5–10 ms without stalling the main loop.

### 1.2 Bounding Box Jitter (Coordinate Shaking)
- **Issue:** Raw bounding box coordinates `(x1, y1, x2, y2)` from YOLO fluctuate by several pixels across consecutive frames, causing the boundary rectangles and blur regions to visually vibrate or jitter.
- **Solution:** Apply an Exponential Moving Average (EMA) smoothing filter per `track_id` so bounding boxes and blur areas glide smoothly between frames.

### 1.3 Blur Flickering (Hand Detection Drops)
- **Issue:** Hand detection runs asynchronously; if a hand is momentarily occluded or dropped for a single frame, the blur rapidly flashes on and off.
- **Solution:** Implement a hysteresis / debounce buffer that retains the hand detection state for a few frames after detection to suppress rapid blinking.

### 1.4 Blur Filter Overhead & Playback FPS Synchronization
- **Issue:** Computing `cv2.GaussianBlur((51, 51))` on CPU for large bounding boxes costs ~15–25 ms per person. Additionally, fixed `cv2.waitKey(1)` delays do not synchronize with the video's actual frame rate, leading to irregular playback speeds.
- **Solution:** Use simpler or downscaled blur techniques (`cv2.blur` or dual-pass downsample-upsample) and dynamically calculate the `cv2.waitKey()` delay to maintain stable 30 FPS playback.

---

## 2. Blur Types Comparison (Speed vs Aesthetics)

| Blur Type | FPS Impact | Processing Time | Characteristics |
| :--- | :--- | :--- | :--- |
| **Mosaic / Pixelation** | Minimal (~0% CPU) | ~0.08 ms | Industry standard for YouTube street vlogs (Japan/Korea) & news reports. |
| **Downsample Linear Blur** | Very Light | ~0.35 ms | Cinematic frosted-glass effect; soft look without heavy computation. |
| **`cv2.blur` (Box Filter)** | Light | ~1.2 ms | Basic spatial average; very clean 1-line implementation, much faster than Gaussian. |
| **`cv2.GaussianBlur` (51x51)** | Heavy (Causes FPS drops) | ~5.5–20 ms | High CPU math overhead for large bounding boxes. |

---

## 3. Video Display / Player Performance

- **Standard `cv2.imshow` bottleneck:** Relies on CPU-based GDI drawing on Windows, requiring memory copies every frame. `cv2.waitKey(1)` on Windows often has an actual OS timer jitter of 5–15 ms.
- **Higher FPS alternatives:**
  1. *Threaded Display Queue:* Separate inference from rendering using a dedicated display thread with `queue(maxsize=1)`.
  2. *Hardware-accelerated Window:* Initialize OpenCV with OpenGL context (`cv2.WINDOW_OPENGL`).
  3. *Subprocess Pipe to `ffplay`:* Stream raw bytes to `ffplay` over stdout/stdin using hardware rendering (Direct3D/SDL2) for tear-free 60+ FPS.

---

## 4. Track ID Cleanup & Re-addition Mechanism

- **Cleanup of Stale IDs:** At the end of each frame, any `track_id` not present in `active_ids` is pruned from `identity_cache` to release memory.
- **Re-detection After 1 Frame:** If a person reappears in subsequent frames, the system automatically detects that `track_id not in identity_cache`, marks the status as `pending`, and triggers a new `thread_track` to re-verify identity.
- **Thread Safety:** Added `if track_id in identity_cache:` guards inside `thread_track` to prevent background worker threads from restoring deleted IDs after a person has left the frame.