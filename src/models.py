"""Data passed between detection, tracking and field projection."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Deteccao:
    caixa: tuple[float, float, float, float]
    confianca: float
    classe: int
    instante: float

    @property
    def ponto_pes(self):
        x1, _, x2, y2 = self.caixa
        return ((x1 + x2) / 2, y2)


@dataclass(frozen=True)
class Rastreamento:
    id: int
    deteccao: Deteccao


@dataclass(frozen=True)
class Posicao:
    instante: float
    x: float
    y: float

    @property
    def fora_do_campo(self):
        return abs(self.x) > 1 or abs(self.y) > 1


@dataclass
class Trajetoria:
    id_rastreamento: int
    posicoes: list[Posicao] = field(default_factory=list)
