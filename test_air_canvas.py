"""Logic tests - no camera or MediaPipe needed.  Run: python -m pytest -q"""
import os
import sys
from types import SimpleNamespace as P

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np

import air_canvas as ac


def hand(index=(0.5, 0.5), middle=(0.5, 0.5), index_up=True, middle_up=True):
    """Fake 21 landmarks. Wrist at (0.5, 0.9), middle knuckle at (0.5, 0.7)."""
    lm = [P(x=0.5, y=0.8) for _ in range(21)]
    lm[0], lm[9] = P(x=0.5, y=0.9), P(x=0.5, y=0.7)
    lm[8], lm[12] = P(x=index[0], y=index[1]), P(x=middle[0], y=middle[1])
    lm[6] = P(x=index[0], y=index[1] + (0.1 if index_up else -0.1))
    lm[10] = P(x=middle[0], y=middle[1] + (0.1 if middle_up else -0.1))
    return lm


def touching_at(x, y):
    return hand(index=(x, y), middle=(x + 0.005, y))


def apart_at(x, y):
    return hand(index=(x, y), middle=(x + 0.12, y))


def ink(app):
    return int(np.count_nonzero(app.canvas))


def test_touching_fingers_draw():
    app = ac.AirCanvas()
    app.update(touching_at(0.4, 0.5), now=0)
    app.update(touching_at(0.6, 0.5), now=0.1)
    assert app.drawing and ink(app) > 0


def test_separated_fingers_do_not_draw():
    app = ac.AirCanvas()
    app.update(apart_at(0.4, 0.5), now=0)
    app.update(apart_at(0.6, 0.5), now=0.1)
    assert not app.drawing and ink(app) == 0


def test_only_index_finger_does_not_draw():
    app = ac.AirCanvas()
    app.update(hand(index=(0.5, 0.5), middle=(0.505, 0.5), middle_up=False), now=0)
    assert ink(app) == 0


def test_no_drawing_on_toolbar():
    app = ac.AirCanvas()
    app.update(touching_at(0.5, 0.05), now=0)
    assert ink(app) == 0


def test_hover_dwell_changes_colour():
    app = ac.AirCanvas()
    x = (1 * ac.BUTTON_W + 40) / ac.WIDTH            # the "Green" button
    pointer = hand(index=(x, 0.05), middle=(x + 0.12, 0.05))
    app.update(pointer, now=0.0)
    assert app.colour_name == "Blue"
    app.update(pointer, now=ac.DWELL_SECONDS + 0.1)
    assert app.colour_name == "Green"


def test_draw_in_selected_colour():
    app = ac.AirCanvas()
    app.press(2)                                      # Red
    app.update(touching_at(0.4, 0.5), now=0)
    app.update(touching_at(0.5, 0.5), now=0.1)
    ys, xs = np.nonzero(app.canvas[:, :, 2])
    assert len(xs) > 0 and np.all(app.canvas[ys, xs, 0] == 0)    # red only


def test_eraser_removes_ink():
    app = ac.AirCanvas()
    for step in range(5):
        app.update(touching_at(0.4 + step * 0.04, 0.5), now=step * 0.1)
    app.update(None)
    before = ink(app)
    app.press(6)                                      # Eraser
    for step in range(5):
        app.update(touching_at(0.4 + step * 0.04, 0.5), now=1 + step * 0.1)
    assert app.eraser and ink(app) < before


def test_undo_and_clear():
    app = ac.AirCanvas()
    for step in range(4):
        app.update(touching_at(0.4 + step * 0.05, 0.5), now=step * 0.1)
    app.update(None)
    assert ink(app) > 0
    app.press(7)                                      # Undo
    assert ink(app) == 0
    for step in range(4):
        app.update(touching_at(0.4 + step * 0.05, 0.5), now=2 + step * 0.1)
    app.update(None)
    app.press(8)                                      # Clear
    assert ink(app) == 0


def test_brush_size_limits():
    app = ac.AirCanvas()
    for _ in range(50):
        app.press(11)
    assert app.brush == ac.MAX_BRUSH
    for _ in range(50):
        app.press(10)
    assert app.brush == ac.MIN_BRUSH


def test_save_and_render(tmp_path):
    app = ac.AirCanvas(save_dir=str(tmp_path))
    app.update(touching_at(0.4, 0.5), now=0)
    app.update(touching_at(0.6, 0.5), now=0.1)
    name = app.save_png()
    assert (tmp_path / name).exists()
    frame = np.zeros((ac.HEIGHT, ac.WIDTH, 3), np.uint8)
    assert app.render(frame).shape == (ac.HEIGHT, ac.WIDTH, 3)


def test_mouse_click_button_index():
    assert ac.AirCanvas.button_at(10, 10) == 0
    assert ac.AirCanvas.button_at(ac.WIDTH - 1, 10) == len(ac.BUTTONS) - 1
    assert ac.AirCanvas.button_at(100, ac.HEADER_H + 5) is None
