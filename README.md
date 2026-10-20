# Air Canvas (Virtual Paint)

Draw in the air with just your webcam and your fingers. Built with **Python, OpenCV and MediaPipe**: real-time hand tracking turns your fingertips into a brush, with a clickable toolbar for colours, eraser, undo and save, and no physical input device needed.

## How to use
| Gesture | Action |
|---|---|
| Touch **index + middle fingertips together** and move | **Draw** |
| Separate the two fingers, or show only the index finger | **Move** the cursor (nothing is drawn) |
| Point your index finger at a toolbar button and hold ~0.7 s | **Click** it (a ring shows progress) |
| Click a button with the **mouse** | **Click** it |

**Toolbar:** Blue, Green, Red, Yellow, Purple, White, **Eraser**, **Undo**, **Clear**, **Save**, Size -, Size +
**Keys:** `s` save | `c` clear | `u` undo | `e` eraser on/off | `+` / `-` brush size | `q` quit

## Features
- Real-time hand landmark tracking (MediaPipe Hands, 21 landmarks)
- Touch-to-draw gesture that works at any distance from the camera (the fingertip gap is measured relative to palm size)
- Hysteresis and smoothing so strokes don't flicker or jitter
- Eraser, multi-step undo (one step per stroke), clear and PNG export
- Dwell-to-click toolbar with progress ring, plus mouse support
- Drawing logic is separate from the camera, so it is unit-tested without a webcam

## Run it
```bash
pip install -r requirements.txt
python air_canvas.py
```
> This project uses MediaPipe's `mp.solutions` API, which newer MediaPipe releases removed. `requirements.txt` pins `mediapipe==0.10.14` (Python 3.8-3.12).

Tips: use good lighting, keep your palm facing the camera, and tune `TOUCH_START` / `TOUCH_STOP` at the top of `air_canvas.py` if drawing starts too early or too late.

## Tests
```bash
pip install pytest
python -m pytest -q
```

## How it works
1. OpenCV captures and mirrors each webcam frame.
2. MediaPipe returns the hand landmarks; `hand_state()` finds which fingers are up and how close the index and middle fingertips are.
3. When the fingertips touch, strokes are drawn on a separate canvas layer; otherwise the index fingertip is a pointer that can click toolbar buttons.
4. The canvas is merged onto the live video and the toolbar is rendered on top.

## Tech stack
Python · OpenCV · MediaPipe · NumPy

## Demo
_Add a screenshot or GIF here (e.g. `docs/demo.gif`)._
