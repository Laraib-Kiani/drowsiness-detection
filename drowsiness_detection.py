"""
Real-Time Drowsiness Detection
------------------------------
Detects driver/user drowsiness from a webcam or video file using
MediaPipe Face Mesh and the Eye Aspect Ratio (EAR).

Usage:
    python drowsiness_detection.py                 # default webcam
    python drowsiness_detection.py --video my.mp4  # video file
    python drowsiness_detection.py --ear-threshold 0.22 --frames 20

Press 'q' to quit.
"""
import argparse
import sys
import time

import cv2
import mediapipe as mp
import numpy as np

# MediaPipe Face Mesh landmark indices for each eye, ordered as
# [p1 (outer corner), p2, p3 (upper lid), p4 (inner corner), p5, p6 (lower lid)]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
LEFT_EYE = [362, 385, 387, 263, 373, 380]


def eye_aspect_ratio(points: np.ndarray) -> float:
    """EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|). Drops toward 0 as the eye closes."""
    p1, p2, p3, p4, p5, p6 = points
    vertical = np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)
    horizontal = np.linalg.norm(p1 - p4)
    return float(vertical / (2.0 * horizontal + 1e-6))


def landmarks_to_pixels(face_landmarks, indices, width, height) -> np.ndarray:
    return np.array(
        [(face_landmarks.landmark[i].x * width, face_landmarks.landmark[i].y * height) for i in indices]
    )


def sound_alarm() -> None:
    """Beep on Windows; fall back to the terminal bell on other systems."""
    try:
        import winsound
        winsound.Beep(1200, 400)
    except ImportError:
        sys.stdout.write("\a")
        sys.stdout.flush()


def parse_args():
    parser = argparse.ArgumentParser(description="Real-time drowsiness detection")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index (default 0)")
    parser.add_argument("--video", type=str, default=None, help="Path to a video file instead of a webcam")
    parser.add_argument("--ear-threshold", type=float, default=0.21,
                        help="EAR below this value counts as a closed eye (default 0.21)")
    parser.add_argument("--frames", type=int, default=20,
                        help="Consecutive closed-eye frames before the alert (default 20)")
    return parser.parse_args()


def main():
    args = parse_args()
    cap = cv2.VideoCapture(args.video if args.video else args.camera)
    if not cap.isOpened():
        sys.exit("Could not open the video source. Check the camera index or file path.")

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1, refine_landmarks=True,
        min_detection_confidence=0.5, min_tracking_confidence=0.5,
    )

    closed_frames = 0
    drowsy_events = 0
    alert_active = False
    prev_time = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        ear = None
        if results.multi_face_landmarks:
            face = results.multi_face_landmarks[0]
            right = landmarks_to_pixels(face, RIGHT_EYE, w, h)
            left = landmarks_to_pixels(face, LEFT_EYE, w, h)
            ear = (eye_aspect_ratio(right) + eye_aspect_ratio(left)) / 2.0

            for eye in (right, left):
                cv2.polylines(frame, [eye.astype(np.int32)], True, (0, 255, 0), 1)

            if ear < args.ear_threshold:
                closed_frames += 1
            else:
                closed_frames = 0
                alert_active = False

            if closed_frames >= args.frames:
                if not alert_active:
                    drowsy_events += 1
                    alert_active = True
                cv2.putText(frame, "DROWSINESS ALERT!", (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                sound_alarm()
        else:
            closed_frames = 0
            cv2.putText(frame, "No face detected", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 2)

        now = time.time()
        fps = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now

        cv2.putText(frame, f"EAR: {ear:.2f}" if ear is not None else "EAR: --", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"Alerts: {drowsy_events}  FPS: {fps:.0f}", (20, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow("Drowsiness Detection (press q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()
    print(f"Session finished. Drowsiness alerts triggered: {drowsy_events}")


if __name__ == "__main__":
    main()
