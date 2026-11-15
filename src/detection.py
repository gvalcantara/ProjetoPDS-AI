"""YOLO player detection independent of the tracking algorithm."""

from .models import Deteccao


class DetectorJogadores:
    def __init__(self, pesos, confianca=0.1, tamanho_imagem=640, classe_id=None, dispositivo=None, modelo=None):
        if not 0 <= confianca <= 1:
            raise ValueError("Confidence must be between 0 and 1.")
        if tamanho_imagem <= 0:
            raise ValueError("Image size must be positive.")
        if modelo is None:
            from ultralytics import YOLO
            modelo = YOLO(str(pesos))
        if dispositivo is None:
            import torch
            dispositivo = 0 if torch.cuda.is_available() else "cpu"
        self.modelo = modelo
        self.confianca = confianca
        self.tamanho_imagem = tamanho_imagem
        self.dispositivo = dispositivo
        names = modelo.names
        if classe_id is not None:
            if classe_id not in names:
                raise ValueError(f"Class ID {classe_id} not found: {names}")
            self.classes = [classe_id]
        else:
            self.classes = [i for i, name in names.items() if str(name).lower() in ("player", "person")]
            if not self.classes:
                raise ValueError(f"No player/person class found in model {names}; specify --class-id.")

    def detectar(self, quadro, instante):
        result = self.modelo.predict(quadro, classes=self.classes, conf=self.confianca,
                                     imgsz=self.tamanho_imagem, device=self.dispositivo, verbose=False)[0]
        boxes = result.boxes.cpu().numpy()
        return [Deteccao(tuple(map(float, box)), float(conf), int(cls), float(instante))
                for box, conf, cls in zip(boxes.xyxy, boxes.conf, boxes.cls)]
