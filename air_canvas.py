"""Air Canvas - draw in the air with your fingers (OpenCV + MediaPipe).

HOW TO USE
  DRAW   : touch your INDEX and MIDDLE fingertips together and move your hand.
  MOVE   : separate the two fingers (or show only the index finger) - nothing is drawn.
  CLICK  : point your index finger at a button in the top bar and hold for a moment
           (a ring fills up). You can also click the buttons with the mouse.
Buttons : 6 colours, Eraser, Undo, Clear, Save, Size -, Size +
Keys    : s save | c clear | u undo | e eraser on/off | + / - brush size | q quit
"""
import math
import os
import time
from collections import deque

import cv2
import numpy as np

WIDTH, HEIGHT = 1280, 720
HEADER_H = 90
DWELL_SECONDS = 0.7        # hold a finger over a button this long to click it
TOUCH_START = 0.30         # fingertip gap (as a fraction of palm size) that starts drawing
TOUCH_STOP = 0.45          # gap that stops drawing (bigger than start = no flicker)
SMOOTHING = 0.5            # 0 = no smoothing, closer to 1 = smoother but laggier
MIN_BRUSH, MAX_BRUSH = 2, 40
ERASER_FACTOR = 4          # eraser is this many times thicker than the brush
HISTORY = 30               # number of undo steps

# (label, BGR colour, action)
BUTTONS = [
    ("Blue", (255, 100, 0), "colour"),
    ("Green", (0, 200, 0), "colour"),
    ("Red", (0, 0, 230), "colour"),
    ("Yellow", (0, 220, 255), "colour"),
    ("Purple", (200, 60, 160), "colour"),
    ("White", (255, 255, 255), "colour"),
    ("Eraser", (70, 70, 70), "eraser"),
    ("Undo", (110, 110, 110), "undo"),
    ("Clear", (110, 110, 110), "clear"),
    ("Save", (110, 110, 110), "save"),
    ("Size -", (110, 110, 110), "smaller"),
    ("Size +", (110, 110, 110), "bigger"),
]
BUTTON_W = WIDTH // len(BUTTONS)


def hand_state(lm):
    """From 21 normalised landmarks return (index_tip, middle_tip, gap, index_up, middle_up).

    gap = distance between the two fingertips divided by palm size, so it works at any
    distance from the camera.
    """
    def px(i):
        return lm[i].x * WIDTH, lm[i].y * HEIGHT

    palm = max(math.dist(px(0), px(9)), 1.0)          # wrist -> middle-finger knuckle
    index_tip, middle_tip = px(8), px(12)
    gap = math.dist(index_tip, middle_tip) / palm
    index_up = lm[8].y < lm[6].y                      # tip above the middle joint
    middle_up = lm[12].y < lm[10].y
    return index_tip, middle_tip, gap, index_up, middle_up


class AirCanvas:
    """All drawing logic. It needs no camera, so it can be unit-tested."""

    def __init__(self, save_dir="."):
        self.canvas = np.zeros((HEIGHT, WIDTH, 3), np.uint8)   # black = transparent
        self.save_dir = save_dir
        self.colour, self.colour_name = BUTTONS[0][1], BUTTONS[0][0]
        self.eraser = False
        self.brush = 8
        self.history = deque(maxlen=HISTORY)
        self.drawing = False
        self.prev = None
        self.smooth = None
        self.hover = None
        self.hover_since = 0.0
        self.hover_fired = False
        self.pointer = None
        self.message, self.message_until = "", 0.0

    # ------------------------------------------------------------ buttons
    @staticmethod
    def button_at(x, y):
        if y < 0 or y >= HEADER_H or x < 0:
            return None
        return min(int(x // BUTTON_W), len(BUTTONS) - 1)

    def press(self, index, now=None):
        label, colour, action = BUTTONS[index]
        if action == "colour":
            self.colour, self.colour_name, self.eraser = colour, label, False
            self.flash(f"Colour: {label}", now)
        elif action == "eraser":
            self.eraser = True
            self.flash("Eraser on", now)
        elif action == "undo":
            if self.history:
                self.canvas[:] = self.history.pop()
                self.flash("Undo", now)
            else:
                self.flash("Nothing to undo", now)
        elif action == "clear":
            self._snapshot()
            self.canvas[:] = 0
            self.flash("Canvas cleared", now)
        elif action == "save":
            self.flash(f"Saved {self.save_png()}", now)
        elif action == "smaller":
            self.brush = max(MIN_BRUSH, self.brush - 2)
            self.flash(f"Brush size {self.brush}", now)
        elif action == "bigger":
            self.brush = min(MAX_BRUSH, self.brush + 2)
            self.flash(f"Brush size {self.brush}", now)

    def toggle_eraser(self):
        self.eraser = not self.eraser
        self.flash("Eraser on" if self.eraser else f"Colour: {self.colour_name}")

    def flash(self, text, now=None):
        self.message = text
        self.message_until = (time.time() if now is None else now) + 1.8

    # ------------------------------------------------------------ drawing
    def _snapshot(self):
        self.history.append(self.canvas.copy())

    def _size(self):
        return self.brush * ERASER_FACTOR if self.eraser else self.brush

    def _stroke(self, point):
        x, y = int(point[0]), int(point[1])
        colour = (0, 0, 0) if self.eraser else self.colour
        if self.prev is None:
            self._snapshot()                                    # one undo step per stroke
            cv2.circle(self.canvas, (x, y), self._size() // 2, colour, cv2.FILLED)
        else:
            cv2.line(self.canvas, self.prev, (x, y), colour, self._size(), cv2.LINE_AA)
        self.prev = (x, y)
        self.drawing = True

    def _end_stroke(self):
        self.drawing = False
        self.prev = None
        self.smooth = None

    def _smoothed(self, point):
        if self.smooth is None:
            self.smooth = point
        else:
            self.smooth = (SMOOTHING * self.smooth[0] + (1 - SMOOTHING) * point[0],
                           SMOOTHING * self.smooth[1] + (1 - SMOOTHING) * point[1])
        return self.smooth

    def _update_hover(self, pointer, now):
        index = self.button_at(*pointer)
        if index != self.hover:
            self.hover, self.hover_since, self.hover_fired = index, now, False
        elif index is not None and not self.hover_fired and now - self.hover_since >= DWELL_SECONDS:
            self.hover_fired = True
            self.press(index, now)

    def update(self, lm, now=None):
        """Feed one frame's hand landmarks (or None when no hand is visible)."""
        now = time.time() if now is None else now
        if lm is None:
            self._end_stroke()
            self.hover, self.pointer = None, None
            return
        index_tip, middle_tip, gap, index_up, middle_up = hand_state(lm)
        limit = TOUCH_STOP if self.drawing else TOUCH_START
        touching = index_up and middle_up and gap < limit
        self.pointer = index_tip

        if touching:
            point = self._smoothed(((index_tip[0] + middle_tip[0]) / 2, (index_tip[1] + middle_tip[1]) / 2))
            self.pointer = point
            if point[1] < HEADER_H:                             # never draw on the toolbar
                self._end_stroke()
            else:
                self.hover = None
                self._stroke(point)
        else:
            self._end_stroke()
            if index_up:
                self._update_hover(index_tip, now)
            else:
                self.hover = None

    # ------------------------------------------------------------ output
    def save_png(self):
        mask = self._mask()
        sheet = np.full_like(self.canvas, 255)
        sheet[mask > 0] = self.canvas[mask > 0]
        name = os.path.join(self.save_dir, time.strftime("air_canvas_%Y%m%d_%H%M%S.png"))
        cv2.imwrite(name, sheet)
        return os.path.basename(name)

    def _mask(self):
        gray = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
        return cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)[1]

    def render(self, frame, now=None):
        """Draw the canvas, toolbar and cursor on top of a webcam frame (in place)."""
        now = time.time() if now is None else now
        frame[self._mask() > 0] = self.canvas[self._mask() > 0]

        for i, (label, colour, action) in enumerate(BUTTONS):
            x1, x2 = i * BUTTON_W, (i + 1) * BUTTON_W
            cv2.rectangle(frame, (x1, 0), (x2, HEADER_H), colour, cv2.FILLED)
            active = (action == "colour" and not self.eraser and label == self.colour_name) or \
                     (action == "eraser" and self.eraser)
            cv2.rectangle(frame, (x1, 0), (x2, HEADER_H), (255, 255, 255) if active else (30, 30, 30),
                          5 if active else 1)
            brightness = sum(colour) / 3
            text_colour = (0, 0, 0) if brightness > 170 else (255, 255, 255)
            size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.putText(frame, label, (x1 + (BUTTON_W - size[0]) // 2, HEADER_H // 2 + 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_colour, 2, cv2.LINE_AA)

        if self.pointer is not None:
            x, y = int(self.pointer[0]), int(self.pointer[1])
            if self.drawing:
                colour = (255, 255, 255) if self.eraser else self.colour
                cv2.circle(frame, (x, y), self._size() // 2 + 2, colour, 2)
            else:
                cv2.circle(frame, (x, y), 12, (255, 255, 255), 2)
            if self.hover is not None and not self.hover_fired:      # dwell progress ring
                progress = min((now - self.hover_since) / DWELL_SECONDS, 1.0)
                cv2.ellipse(frame, (x, y), (24, 24), -90, 0, 360 * progress, (0, 255, 255), 4)

        mode = "DRAWING" if self.drawing else "MOVE"
        label = "ERASER" if self.eraser else self.colour_name
        cv2.putText(frame, f"{mode} | {label} | size {self.brush}", (15, HEIGHT - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "Touch index+middle fingertips = draw | separate = move | hover on a button = click | q = quit",
                    (15, HEIGHT - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
        if now < self.message_until:
            cv2.putText(frame, self.message, (15, HEADER_H + 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (0, 255, 255), 2, cv2.LINE_AA)
        return frame


def main():
    import mediapipe as mp

    if not hasattr(mp, "solutions"):
        raise SystemExit("This MediaPipe version has no 'solutions' API. Run: pip install mediapipe==0.10.14")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("Could not open the webcam. Close other apps that use the camera and try again.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

    app = AirCanvas()
    window = "Air Canvas"
    cv2.namedWindow(window)

    def on_mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            index = app.button_at(x, y)
            if index is not None:
                app.press(index)

    cv2.setMouseCallback(window, on_mouse)
    mp_hands, mp_draw = mp.solutions.hands, mp.solutions.drawing_utils

    with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.6) as hands:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.resize(cv2.flip(frame, 1), (WIDTH, HEIGHT))
            result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if result.multi_hand_landmarks:
                hand = result.multi_hand_landmarks[0]
                mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
                app.update(hand.landmark)
            else:
                app.update(None)
            cv2.imshow(window, app.render(frame))

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
            elif key == ord("c"):
                app.press(8)
            elif key == ord("s"):
                app.press(9)
            elif key == ord("u"):
                app.press(7)
            elif key == ord("e"):
                app.toggle_eraser()
            elif key in (ord("+"), ord("=")):
                app.press(11)
            elif key in (ord("-"), ord("_")):
                app.press(10)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
