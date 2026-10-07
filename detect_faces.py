from ultralytics import YOLO
import cv2
from uniface import RetinaFace, AdaFace
from uniface.landmark import Landmark106
from uniface.face_utils import compute_similarity
import os

# def detect_landmarks_1(image):
#     detector = RetinaFace()
#     landmarker = Landmark106()
#     faces = detector.detect(image)
#     if faces:
#         landmarks = landmarker.get_landmarks(image, faces[0].bbox)
#         for points in landmarks:
#             x, y = points
#             cv2.circle(image, (int(x), int(y)), 2, (0, 255, 0), -1)

#     return cv2.imshow("image", image) , cv2.waitKey(0)


def detect_landmarks_2(image):
    detector = RetinaFace()
    landmarker = Landmark106()

    faces = detector.detect(image)
    
    if faces:
        landmarks = landmarker.get_landmarks(image, faces[0].bbox)
        # Face contour
        contour = landmarks[0:33]
        # Left eyebrow
        left_eyebrow = landmarks[33:42]
        # Right eyebrow
        right_eyebrow = landmarks[42:51]
        # Nose
        nose = landmarks[51:63]
        # Left eye
        left_eye = landmarks[63:72]
        # Right eye
        right_eye = landmarks[76:84]
        # Mouth
        mouth = landmarks[87:106]

    landmarks_dict = {
        "contour": contour,
        "left_eyebrow": left_eyebrow,
        "right_eyebrow": right_eyebrow,
        "nose": nose,
        "left_eye": left_eye,
        "right_eye": right_eye,
        "mouth": mouth
    }
    return landmarks_dict

detector, recognizer = RetinaFace(), AdaFace()

def detect_faces(image, max_dim=480):
    if isinstance(image, str):
        image = cv2.imread(image)
    if image is None:
        return None

    # Tự động scale ảnh nếu quá lớn để detect cực nhanh
    if max_dim is not None:
        h, w = image.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            image = cv2.resize(image, (int(w * scale), int(h * scale)))

    faces = detector.detect(image)
    if not faces:
        return None

    best_face = max(faces, key=lambda f: f.confidence)
    embedding = recognizer.get_normalized_embedding(image, best_face.landmarks)
    return embedding

if __name__ == "__main__":
    emb1 = detect_faces("face_1.png")
    emb2 = detect_faces("face_2.png")
    if emb1 is not None and emb2 is not None:
        score = compute_similarity(emb1, emb2, normalized=True)
        print(f"{score:+.3f}", "match" if score > 0.4 else "no match")