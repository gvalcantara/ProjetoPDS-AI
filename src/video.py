"""Read video frames using one-based frame numbers."""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class MetadadosVideo:
    largura: int
    altura: int
    fps: float
    total_quadros: int

    @property
    def duracao_segundos(self):
        return self.total_quadros / self.fps


class LeitorVideo:
    """Context-managed video reader. Frame numbers start at one."""

    def __init__(self, caminho):
        self.caminho = Path(caminho)
        self._capture = None

    def abrir(self):
        if self._capture is not None:
            return self
        capture = cv2.VideoCapture(str(self.caminho))
        if not capture.isOpened():
            capture.release()
            raise ValueError(f"Could not open video: {self.caminho}")
        self._capture = capture
        return self

    def fechar(self):
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def __enter__(self):
        return self.abrir()

    def __exit__(self, exc_type, exc, traceback):
        self.fechar()

    def _aberto(self):
        if self._capture is None:
            raise RuntimeError("Video is not open; call abrir() or use a context manager.")
        return self._capture

    def metadados(self):
        capture = self._aberto()
        width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = capture.get(cv2.CAP_PROP_FPS)
        count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        if not np.isfinite([width, height, fps]).all() or min(width, height, fps) <= 0:
            raise ValueError("Video metadata is invalid.")
        # Live-recorded WebM files often have no duration or frame count.
        total = int(count) if np.isfinite(count) and count > 0 else 0
        return MetadadosVideo(int(width), int(height), float(fps), total)

    def ler_quadro(self, numero):
        if isinstance(numero, bool) or not isinstance(numero, int) or numero < 1:
            raise ValueError("Frame number must be a positive integer, starting at 1.")
        capture = self._aberto()
        if not capture.set(cv2.CAP_PROP_POS_FRAMES, numero - 1):
            raise ValueError(f"Could not seek to frame {numero}.")
        ok, frame = capture.read()
        if not ok:
            raise ValueError(f"Could not read frame {numero} of the video.")
        return frame

    def ler_quadros(self):
        """Yield (one-based frame number, seconds, frame) from the beginning."""
        capture = self._aberto()
        fps = self.metadados().fps
        if not capture.set(cv2.CAP_PROP_POS_FRAMES, 0):
            raise ValueError("Could not seek to the beginning of the video.")
        numero = 1
        instante_anterior = -1.0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            timestamp = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if not np.isfinite(timestamp) or timestamp < 0 or (numero > 1 and timestamp <= instante_anterior):
                timestamp = max((numero - 1) / fps, instante_anterior + 1 / fps)
            yield numero, timestamp, frame
            instante_anterior = timestamp
            numero += 1
