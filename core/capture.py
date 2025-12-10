# core/capture.py
import time
from typing import Optional

import mss
import mss.tools
import numpy as np
import win32gui

from config import CAPTURE_FPS, GAME_WINDOW_TITLE


def _get_window_rect(title: str) -> Optional[tuple]:
    hwnd = win32gui.FindWindow(None, title)
    if not hwnd:
        return None
    rect = win32gui.GetWindowRect(hwnd)
    return rect  # (left, top, right, bottom)


class WindowCapture:
    def __init__(self, window_title: str = GAME_WINDOW_TITLE, fps: int = CAPTURE_FPS):
        self.window_title = window_title
        self.fps = fps
        self.sct = mss.mss()

    def capture_once(self) -> Optional[np.ndarray]:
        rect = _get_window_rect(self.window_title)
        if rect is None:
            return None

        left, top, right, bottom = rect
        monitor = {"left": left, "top": top, "width": right - left, "height": bottom - top}
        img = self.sct.grab(monitor)
        frame = np.array(img)[:, :, :3]  # BGR
        return frame

    def capture_loop(self):
        delay = 1.0 / self.fps
        while True:
            frame = self.capture_once()
            yield frame
            time.sleep(delay)
