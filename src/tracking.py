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
            cfg = IterableSimpleNamespace(**YAML.load(check_yaml(str(configuracao))))
            if cfg.tracker_type not in ("fasttrack", "bytetrack", "botsort"):
                raise ValueError("Only fasttrack, bytetrack and botsort configurations are supported.")
            if cfg.tracker_type == "botsort":
                # The standalone tracker receives frames, not detector feature maps.
                # Use the bundled detector as an experimental appearance encoder.
                cfg.with_reid = True
                if dispositivo is not None:
                    device = str(dispositivo)
                    cfg.device = f"cuda:{device}" if device.isdigit() else device
                recurso = files(__package__).joinpath("data", "players_n_1280_baseline.pt")
                with as_file(recurso) as modelo:
                    cfg.model = str(modelo)
                    rastreador = TRACKER_MAP[cfg.tracker_type](args=cfg)
                # Ultralytics otherwise sends every crop at once through a
                # 640px predictor, which can exhaust small GPUs.
                encoder = rastreador.encoder
                encoder.model.predictor.args.imgsz = 224
                rastreador.encoder = _CodificadorEmLotes(encoder)
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
