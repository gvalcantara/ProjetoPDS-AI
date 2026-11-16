"""Field annotation and calibration using the existing projection model."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from .field_geometry import fit_field_projection, load_camera_parameters, save_camera_parameters


def validate_field(field):
    required = {"top", "bottom", "left", "right"}
    if not isinstance(field, dict) or not required <= set(field) or set(field) - required - {"center", "center_spot"}:
        raise ValueError("Field YAML must contain top, bottom, left and right, optionally center and center_spot.")
    for name, points in field.items():
        if name == "center_spot":
            if (not isinstance(points, (list, tuple)) or len(points) != 2 or
                    any(type(value) is not int for value in points)):
                raise ValueError("center_spot must be a pair of integers.")
            continue
        if not isinstance(points, list) or len(points) < 2:
            raise ValueError(f"Field line '{name}' must contain at least two points.")
        for point in points:
            if (not isinstance(point, (list, tuple)) or len(point) != 2
                    or any(type(value) is not int for value in point)):
                raise ValueError(f"Points in '{name}' must be pairs of integers.")
    return field


def connect_field_lines(field):
    """Connect adjacent extreme points using their midpoint, regardless of click order."""
    validate_field(field)
    # Canonical order: horizontal lines left to right; vertical lines top to bottom.
    lines = {name: sorted((list(p) for p in points), key=lambda p: p[0 if name in ("top", "bottom") else 1])
             for name, points in field.items() if name != "center_spot"}
    if "center_spot" in field:
        lines["center_spot"] = list(field["center_spot"])
    for horizontal, hi, vertical, vi in (("top", 0, "left", 0),
                                          ("top", -1, "right", 0),
                                          ("bottom", 0, "left", -1),
                                          ("bottom", -1, "right", -1)):
        a, b = lines[horizontal][hi], lines[vertical][vi]
        corner = [round((a[0] + b[0]) / 2), round((a[1] + b[1]) / 2)]
        lines[horizontal][hi] = corner.copy()
        lines[vertical][vi] = corner.copy()
    if "center" in lines:
        # The marked center endpoints define the intersections with the sidelines.
        # Insert them by x so each connected line contains the exact same point.
        for horizontal, endpoint in (("top", lines["center"][0]),
                                      ("bottom", lines["center"][-1])):
            if not lines[horizontal][0][0] < endpoint[0] < lines[horizontal][-1][0]:
                raise ValueError(f"The center endpoint must lie between the corners of {horizontal}.")
            if endpoint not in lines[horizontal]:
                lines[horizontal].append(endpoint.copy())
                lines[horizontal].sort(key=lambda point: point[0])
    return lines


@dataclass(frozen=True)
class CalibracaoCampo:
    modelo_camera: str
    dimensoes_imagem: tuple
    centro_optico: np.ndarray
    escala: float
    parametro_lente: float
    homografia: np.ndarray
    erro_rms: np.ndarray | None = None
    atingiu_limite: bool | None = None

    def salvar(self, caminho):
        save_camera_parameters(caminho, self.modelo_camera, self.dimensoes_imagem,
                               self.centro_optico, self.escala, self.parametro_lente,
                               self.homografia)


class CalibradorCampo:
    """Validate markings and estimate the existing radial or fisheye model."""

    validar_marcacoes = staticmethod(validate_field)
    conectar_linhas = staticmethod(connect_field_lines)

    def carregar_marcacoes(self, caminho):
        with Path(caminho).open(encoding="utf-8") as file:
            return self.validar_marcacoes(yaml.safe_load(file))

    def salvar_marcacoes(self, caminho, marcacoes):
        self.validar_marcacoes(marcacoes)
        caminho = Path(caminho)
        caminho.parent.mkdir(parents=True, exist_ok=True)
        with caminho.open("w", encoding="utf-8") as file:
            yaml.safe_dump(marcacoes, file, sort_keys=False)

    def calibrar(self, marcacoes, dimensoes_imagem, modelo_camera="radial", **opcoes):
        self.validar_marcacoes(marcacoes)
        linhas = [marcacoes[name] for name in ("bottom", "right", "top", "left")]
        centro, escala, parametro, homografia, erros, limite = fit_field_projection(
            linhas, dimensoes_imagem, cameramodel=modelo_camera,
            center_line=marcacoes.get("center"), center_spot=marcacoes.get("center_spot"),
            **opcoes)
        return CalibracaoCampo(modelo_camera, tuple(dimensoes_imagem), centro, escala,
                               parametro, homografia, erros, limite)

    def carregar_calibracao(self, caminho, dimensoes_imagem):
        modelo, parametro, homografia = load_camera_parameters(caminho, dimensoes_imagem)
        largura, altura = dimensoes_imagem
        return CalibracaoCampo(modelo, tuple(dimensoes_imagem),
                               np.array([largura / 2, altura / 2]),
                               float(np.hypot(largura, altura) / 2), parametro, homografia)
