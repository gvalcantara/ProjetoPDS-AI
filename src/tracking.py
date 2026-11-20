"""Associate player detections across frames using Ultralytics trackers."""

import numpy as np
from importlib.resources import as_file, files

from .models import Deteccao, Rastreamento


class _CodificadorEmLotes:
    """Limit ReID inference memory while preserving detection order."""

    def __init__(self, encoder, tamanho_lote=1):
        self.encoder = encoder
        self.tamanho_lote = tamanho_lote

    def __call__(self, imagem, caixas):
        features = []
        for inicio in range(0, len(caixas), self.tamanho_lote):
            features.extend(self.encoder(imagem, caixas[inicio:inicio + self.tamanho_lote]))
        return features


class RastreadorJogadores:
    """One tracker instance per video; every frame, including empty ones, must be submitted."""

    def __init__(self, configuracao="botsort.yaml", rastreador=None, dispositivo=None):
        if rastreador is None:
            from ultralytics.trackers.track import TRACKER_MAP
            from ultralytics.utils import IterableSimpleNamespace, YAML
            from ultralytics.utils.checks import check_yaml
            if str(configuracao) == "botsort.yaml":
                recurso_cfg = files(__package__).joinpath("data", "botsort_reid.yaml")
                with as_file(recurso_cfg) as caminho_cfg:
                    cfg = IterableSimpleNamespace(**YAML.load(str(caminho_cfg)))
            else:
                cfg = IterableSimpleNamespace(**YAML.load(check_yaml(str(configuracao))))
            if cfg.tracker_type not in ("fasttrack", "bytetrack", "botsort"):
                raise ValueError("Only fasttrack, bytetrack and botsort configurations are supported.")
            if cfg.tracker_type == "botsort":
                # The standalone tracker receives frames, not detector feature maps.
                if cfg.with_reid and cfg.model == "auto":
                    raise ValueError("Standalone BoT-SORT requires an explicit ReID model when with_reid is true.")
                if dispositivo is not None:
                    device = str(dispositivo)
                    cfg.device = f"cuda:{device}" if device.isdigit() else device
                if cfg.with_reid and cfg.model == "bundled_reid":
                    recurso = files(__package__).joinpath("data", "yolo26n-reid.onnx")
                    with as_file(recurso) as modelo:
                        cfg.model = str(modelo)
                        rastreador = TRACKER_MAP[cfg.tracker_type](args=cfg)
                else:
                    rastreador = TRACKER_MAP[cfg.tracker_type](args=cfg)
            else:
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
