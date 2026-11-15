"""Associate player detections across frames using Ultralytics trackers."""

import numpy as np

from .models import Deteccao, Rastreamento


class RastreadorJogadores:
    """One tracker instance per video; every frame, including empty ones, must be submitted."""

    def __init__(self, configuracao="fasttrack.yaml", rastreador=None):
        if rastreador is None:
            from ultralytics.trackers.track import TRACKER_MAP
            from ultralytics.utils import IterableSimpleNamespace, YAML
            from ultralytics.utils.checks import check_yaml
            cfg = IterableSimpleNamespace(**YAML.load(check_yaml(str(configuracao))))
            if cfg.tracker_type not in ("fasttrack", "bytetrack"):
                raise ValueError("Only fasttrack and bytetrack configurations are supported.")
            rastreador = TRACKER_MAP[cfg.tracker_type](args=cfg)
        self._rastreador = rastreador

    def atualizar(self, deteccoes, quadro, instante):
        from ultralytics.engine.results import Boxes
        data = np.asarray([(*d.caixa, d.confianca, d.classe) for d in deteccoes], dtype=np.float32).reshape(-1, 6)
        boxes = Boxes(data, quadro.shape[:2])
        tracks = self._rastreador.update(boxes, quadro)
        resultados = []
        for row in tracks:
            x1, y1, x2, y2, track_id, confidence, class_id, _ = row
            detection = Deteccao((float(x1), float(y1), float(x2), float(y2)),
                                  float(confidence), int(class_id), float(instante))
            resultados.append(Rastreamento(int(track_id), detection))
        return resultados
