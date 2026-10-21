# Real-Time Drowsiness Detection

A computer vision system that monitors a person's eyes through a webcam or video file and raises an alert when they stay closed for too long, a typical sign of drowsiness. Built with **OpenCV** and **MediaPipe Face Mesh**.

> Re-implementation of my original 2024 drowsiness detection project, rebuilt with cleaner code, command-line options, and documentation.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Laraib-Kiani/drowsiness-detection/blob/main/drowsiness_detection_colab.ipynb)

<!-- Add a screenshot or GIF of the demo here, for example: ![Demo](demo.gif) -->

## How It Works

1. Each frame is read from the webcam (or a video file) with OpenCV.
2. MediaPipe Face Mesh locates 468+ facial landmarks in real time.
3. Six landmarks around each eye are used to compute the **Eye Aspect Ratio (EAR)**:

   `EAR = (|p2 - p6| + |p3 - p5|) / (2 * |p1 - p4|)`

   EAR stays around 0.25-0.35 when the eye is open and drops close to 0 when it closes.
4. If the average EAR of both eyes stays below a threshold for a number of consecutive frames, a visual and audio **drowsiness alert** is triggered.

## Features

- Real-time detection on a standard CPU, with no GPU needed
- Works with a webcam or a recorded video file
- Adjustable sensitivity through command-line options
- On-screen EAR value, FPS counter, and alert count
- Audio alarm (Windows beep, terminal bell on other systems)

## Installation

Requires Python 3.9-3.12.

```bash
git clone https://github.com/Laraib-Kiani/drowsiness-detection.git
cd drowsiness-detection
python -m venv venv
venv\Scripts\activate        # Windows (on Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt
```

## Usage

```bash
python drowsiness_detection.py                       # default webcam
python drowsiness_detection.py --camera 1            # a different camera
python drowsiness_detection.py --video sample.mp4    # a video file
python drowsiness_detection.py --ear-threshold 0.22 --frames 25
```

| Option | Default | Meaning |
|---|---|---|
| `--camera` | 0 | Webcam index |
| `--video` | none | Path to a video file |
| `--ear-threshold` | 0.21 | EAR below this value counts as a closed eye |
| `--frames` | 20 | Consecutive closed-eye frames needed to trigger the alert |

Press **q** to quit.

## Run in Google Colab (no installation)

Click the **Open in Colab** badge above, run the cells in order, and upload a short video of your face. The notebook marks every eye closure, shows the annotated video, and plots the Eye Aspect Ratio over time. This version analyzes a recorded video, because live webcam access is limited in Colab.

## Tech Stack

Python, OpenCV, MediaPipe, NumPy

## Limitations

- The EAR threshold varies between people, so it may need adjusting for your face.
- Performance drops in very low light, with strong glare on glasses, or when the face is turned far from the camera.
- This is a learning and demonstration project, not a certified safety system.

## Future Improvements

- Yawn detection using the Mouth Aspect Ratio (MAR)
- Automatic per-user EAR calibration
- Head-pose estimation to detect nodding
- Logging alerts to a file with timestamps

## Author

**Laraib Zahid Kiani**: AI & Computer Vision Engineer
[LinkedIn](https://linkedin.com/in/laraib-zahid-kiani-7604602a4) | [GitHub](https://github.com/Laraib-Kiani)
