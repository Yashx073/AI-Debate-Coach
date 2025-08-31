import cv2
import numpy as np
import mediapipe as mp
from typing import Dict


def _iter_frames(video_path: str, max_frames: int = 600, step: int = 3):
    """Yield frames from a video in RGB format, skipping by step."""
    cap = cv2.VideoCapture(video_path)
    i = 0
    while cap.isOpened() and i < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if i % step == 0:
            yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        i += 1
    cap.release()


def analyze_posture(video_path: str) -> Dict:
    """
    Analyze posture from a video.
    Returns:
        - facing_camera_ratio: proportion of frames with detected face
        - avg_head_tilt_deg: average head tilt (eye alignment) in degrees
    """
    facing_count = 0
    total = 0
    tilts = []

    mp_face = mp.solutions.face_mesh

    with mp_face.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True
    ) as fm:
        for frame in _iter_frames(video_path):
            total += 1
            res = fm.process(frame)

            if res.multi_face_landmarks:
                facing_count += 1

                # Approx head tilt using eye corners (landmarks 33 = right eye, 263 = left eye)
                h, w, _ = frame.shape
                lms = res.multi_face_landmarks[0].landmark
                try:
                    r_eye = lms[33]
                    l_eye = lms[263]
                    p1 = np.array([r_eye.x * w, r_eye.y * h])
                    p2 = np.array([l_eye.x * w, l_eye.y * h])
                    dy = p2[1] - p1[1]
                    dx = p2[0] - p1[0]
                    angle = np.degrees(np.arctan2(dy, dx))
                    tilts.append(angle)
                except Exception:
                    pass

    facing_ratio = float(facing_count / total) if total else 0.0
    avg_tilt = float(np.mean(tilts)) if tilts else 0.0

    return {
        "facing_camera_ratio": facing_ratio,
        "avg_head_tilt_deg": avg_tilt,
    }
