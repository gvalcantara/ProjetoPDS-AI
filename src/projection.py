"""Project foot points to the normalized field coordinate system."""

import numpy as np

from .field_geometry import project_points
from .models import Posicao


class ProjetorCampo:
    def __init__(self, calibracao):
        self.calibracao = calibracao

    def projetar(self, rastreamentos):
        if not rastreamentos:
            return []
        calibration = self.calibracao
        feet = np.array([track.deteccao.ponto_pes for track in rastreamentos], dtype=float)
        points = project_points(feet, calibration.centro_optico, calibration.escala,
                                calibration.parametro_lente, calibration.homografia,
                                calibration.modelo_camera)
        return [Posicao(track.deteccao.instante, float(x), float(y))
                for track, (x, y) in zip(rastreamentos, points)]
