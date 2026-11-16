"""Process a video into player trajectories on the calibrated field."""

from importlib.resources import as_file, files

from .calibration import CalibradorCampo
from .detection import DetectorJogadores
from .projection import ProjetorCampo
from .tracking import RastreadorJogadores
from .trajectory import ConstrutorTrajetorias
from .video import LeitorVideo


class ExtratorTrajetorias:
    """Extract normalized field trajectories using the bundled model and camera calibration.

    A new tracker is created for each video so tracking IDs and state do not
    carry over between calls. The camera calibration must match the video size.
    """

    def __init__(self, parametros_camera, confianca=0.1, tamanho_imagem=1280,
                 classe_id=None, dispositivo=None, configuracao_rastreador="botsort.yaml"):
        self.parametros_camera = parametros_camera
        self.configuracao_rastreador = configuracao_rastreador
        recurso = files(__package__).joinpath("data", "players_n_1280_baseline.pt")
        with as_file(recurso) as pesos:
            self.detector = DetectorJogadores(pesos, confianca=confianca,
                                             tamanho_imagem=tamanho_imagem,
                                             classe_id=classe_id, dispositivo=dispositivo)

    def processar(self, caminho_video, ao_processar_quadro=None, fps_processamento=None):
        """Return trajectories ordered by track ID; times are in seconds.

        Positions use normalized field coordinates, with field boundaries at
        -1 and 1. Positions outside those bounds are retained.
        """
        with LeitorVideo(caminho_video) as video:
            metadata = video.metadados()
            if fps_processamento is not None and fps_processamento <= 0:
                raise ValueError("Processing FPS must be positive.")
            fps_alvo = min(fps_processamento or metadata.fps, metadata.fps)
            calibracao = CalibradorCampo().carregar_calibracao(
                self.parametros_camera, (metadata.largura, metadata.altura))
            projetor = ProjetorCampo(calibracao)
            rastreador = RastreadorJogadores(self.configuracao_rastreador, dispositivo=self.detector.dispositivo)
            trajetorias = ConstrutorTrajetorias()
            frames = 0
            proximo_instante = 0.0
            for _, instante, quadro in video.ler_quadros():
                if instante + 1e-9 < proximo_instante:
                    continue
                proximo_instante = (int(instante * fps_alvo + 1e-9) + 1) / fps_alvo
                deteccoes = self.detector.detectar(quadro, instante)
                rastreamentos = rastreador.atualizar(deteccoes, quadro, instante)
                posicoes = projetor.projetar(rastreamentos)
                if ao_processar_quadro is not None:
                    ao_processar_quadro(instante, rastreamentos, posicoes)
                for rastreamento, posicao in zip(rastreamentos, posicoes):
                    trajetorias.adicionar(rastreamento.id, posicao)
                frames += 1
            if not frames:
                raise RuntimeError("No video frames were read.")
            return trajetorias.finalizar()
