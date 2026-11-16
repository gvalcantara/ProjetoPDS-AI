"""Per-frame image-space tracking evaluation using IoU and Hungarian assignment."""

import numpy as np
from scipy.optimize import linear_sum_assignment


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


class AvaliadorRastreamento:
    def __init__(self, limiar_iou=0.5):
        if not 0 < limiar_iou <= 1:
            raise ValueError("IoU threshold must be greater than 0 and at most 1.")
        self.limiar = limiar_iou
        self.quadros_avaliados = self.quadros_sem_rotulo = 0
        self.referencias = self.correspondencias = self.falsos_positivos = self.falsos_negativos = 0
        self.trocas_id = 0
        self.soma_iou = 0.0
        self._ultimo_id = {}

    def avaliar(self, referencias, rastreamentos):
        if referencias is None:
            self.quadros_sem_rotulo += 1
            return
        self.quadros_avaliados += 1
        self.referencias += len(referencias)
        scores = np.array([[iou(ref.caixa, track.deteccao.caixa) for track in rastreamentos]
                           for ref in referencias], dtype=float).reshape(len(referencias), len(rastreamentos))
        pairs = []
        if scores.size:
            # Prioritize number of valid associations before maximizing total IoU.
            valid = scores >= self.limiar
            rows, cols = linear_sum_assignment(-(valid * (scores + min(scores.shape) + 1)))
            pairs = [(r, c) for r, c in zip(rows, cols) if valid[r, c]]
        self.correspondencias += len(pairs)
        self.falsos_negativos += len(referencias) - len(pairs)
        self.falsos_positivos += len(rastreamentos) - len(pairs)
        for r, c in pairs:
            ref_id, track_id = referencias[r].id, rastreamentos[c].id
            self.soma_iou += float(scores[r, c])
            if ref_id in self._ultimo_id and self._ultimo_id[ref_id] != track_id:
                self.trocas_id += 1
            self._ultimo_id[ref_id] = track_id

    def resultado(self):
        tp, fp, fn = self.correspondencias, self.falsos_positivos, self.falsos_negativos
        return {"limiar_iou": self.limiar, "quadros_avaliados": self.quadros_avaliados,
                "quadros_sem_rotulo": self.quadros_sem_rotulo, "referencias": self.referencias,
                "correspondencias": tp, "falsos_positivos": fp, "falsos_negativos": fn,
                "trocas_id": self.trocas_id, "iou_medio": self.soma_iou / tp if tp else None,
                "precisao": tp / (tp + fp) if tp + fp else None,
                "revocacao": tp / (tp + fn) if tp + fn else None,
                "mota_aproximado": 1 - (fn + fp + self.trocas_id) / self.referencias if self.referencias else None}
