"""Build time-ordered trajectories from projected tracks."""

from .models import Trajetoria


class ConstrutorTrajetorias:
    def __init__(self):
        self._trajetorias = {}

    def adicionar(self, id_rastreamento, posicao):
        trajectory = self._trajetorias.setdefault(id_rastreamento, Trajetoria(id_rastreamento))
        if trajectory.posicoes and posicao.instante <= trajectory.posicoes[-1].instante:
            raise ValueError("Positions for a track must have strictly increasing timestamps.")
        trajectory.posicoes.append(posicao)

    def finalizar(self):
        """Return independent snapshots, ordered by tracking ID."""
        return [Trajetoria(id_, list(t.posicoes)) for id_, t in sorted(self._trajetorias.items())]
