"""Tests for detection, tracking, projection and trajectory construction."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import (CalibracaoCampo, ConstrutorTrajetorias, Deteccao, DetectorJogadores,
                 Posicao, ProjetorCampo, RastreadorJogadores, Rastreamento)
from tools import process_video
from src import pipeline


class FakeBoxes:
    def __init__(self, rows):
        rows = np.asarray(rows, dtype=float).reshape(-1, 6)
        self.xyxy, self.conf, self.cls = rows[:, :4], rows[:, 4], rows[:, 5]

    def cpu(self):
        return self

    def numpy(self):
        return self


class FakeModel:
    names = {0: "person", 1: "ball"}

    def __init__(self):
        self.calls = []

    def predict(self, frame, **kwargs):
        self.calls.append(kwargs)
        return [SimpleNamespace(boxes=FakeBoxes([[10, 20, 30, 50, .9, 0]]) )]


def test_botsort_reid_uses_appearance_embeddings():
    tracker = RastreadorJogadores(dispositivo="cpu")
    assert tracker._rastreador.args.with_reid
    frame = np.full((100, 100, 3), 127, dtype=np.uint8)
    detections = [Deteccao((10, 10, 30, 50), .9, 0, 0)]
    first = tracker.atualizar(detections, frame, 0)
    assert len(first) == 1
    assert tracker._rastreador.tracked_stracks[0].curr_feat is not None
    second = tracker.atualizar([Deteccao((11, 10, 31, 50), .9, 0, .1)], frame, .1)
    assert second[0].id == first[0].id


def test_reid_limits_batch_and_preserves_feature_order():
    from src.tracking import _CodificadorEmLotes
    calls = []
    def encoder(image, boxes):
        calls.append(len(boxes))
        return [box[0] for box in boxes]
    result = _CodificadorEmLotes(encoder)(None, np.array([[1], [2], [3]]))
    assert calls == [1, 1, 1]
    assert result == [1, 2, 3]


def calibration():
    return CalibracaoCampo("radial", (100, 100), np.array([50., 50.]),
                           np.hypot(100, 100) / 2, 0.,
                           np.array([[.02, 0, -1], [0, .02, -1], [0, 0, 1.]]))


def test_detector_returns_detection_and_forwards_configuration():
    model = FakeModel()
    detector = DetectorJogadores(None, confianca=.3, tamanho_imagem=320, modelo=model, dispositivo="cpu")
    detections = detector.detectar(np.zeros((100, 100, 3), dtype=np.uint8), 1.5)
    assert detections == [Deteccao((10., 20., 30., 50.), .9, 0, 1.5)]
    assert detections[0].ponto_pes == (20, 50)
    assert model.calls[0]["classes"] == [0]
    assert model.calls[0]["conf"] == .3
    assert model.calls[0]["imgsz"] == 320
    with pytest.raises(ValueError, match="Class ID"):
        DetectorJogadores(None, classe_id=8, modelo=model, dispositivo="cpu")


@pytest.mark.parametrize("config", ["fasttrack.yaml", "bytetrack.yaml"])
def test_real_tracker_keeps_id_and_handles_empty_frame(config):
    tracker = RastreadorJogadores(config)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    first = tracker.atualizar([Deteccao((10, 10, 30, 50), .9, 0, 0)], frame, 0)
    second = tracker.atualizar([Deteccao((11, 10, 31, 50), .9, 0, .1)], frame, .1)
    assert len(first) == len(second) == 1
    assert first[0].id == second[0].id
    assert second[0].deteccao.instante == .1
    assert tracker.atualizar([], frame, .2) == []


def test_projector_and_trajectory_builder():
    projector = ProjetorCampo(calibration())
    tracks = [Rastreamento(8, Deteccao((10, 20, 30, 50), .9, 0, .5)),
              Rastreamento(3, Deteccao((100, 80, 120, 120), .8, 0, .5))]
    positions = projector.projetar(tracks)
    assert (positions[0].x, positions[0].y) == pytest.approx((-.6, 0))
    assert positions[1].fora_do_campo
    assert projector.projetar([]) == []
    builder = ConstrutorTrajetorias()
    for track, pos in zip(tracks, positions):
        builder.adicionar(track.id, pos)
    builder.adicionar(8, Posicao(1, 0, 0))
    with pytest.raises(ValueError, match="increasing"):
        builder.adicionar(8, Posicao(1, 0, 0))
    result = builder.finalizar()
    assert [t.id_rastreamento for t in result] == [3, 8]
    assert [p.instante for p in result[1].posicoes] == [.5, 1]
    result[1].posicoes.clear()
    assert len(builder.finalizar()[1].posicoes) == 2


def test_extrator_processes_video_and_resets_tracker(tmp_path, monkeypatch):
    path = tmp_path / "input.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 5, (100, 100))
    if not writer.isOpened():
        pytest.skip("MJPG video writer is unavailable")
    for _ in range(3):
        writer.write(np.zeros((100, 100, 3), dtype=np.uint8))
    writer.release()
    camera = tmp_path / "camera.yaml"
    calibration().salvar(camera)

    class Detector:
        dispositivo = "cpu"
        def __init__(self, *args, **kwargs):
            pass

        def detectar(self, frame, instant):
            return [Deteccao((10, 20, 30, 50), .9, 0, instant)]

    monkeypatch.setattr(pipeline, "DetectorJogadores", Detector)
    extrator = pipeline.ExtratorTrajetorias(camera, configuracao_rastreador="bytetrack.yaml")
    for _ in range(2):
        trajectories = extrator.processar(path)
        assert len(trajectories) == 1
        assert [p.instante for p in trajectories[0].posicoes] == [0, .2, .4]
        for posicao in trajectories[0].posicoes:
            assert (posicao.x, posicao.y) == pytest.approx((-.6, 0))


def test_process_video_end_to_end_with_simulated_detector(tmp_path, monkeypatch):
    path = tmp_path / "input.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 5, (100, 100))
    if not writer.isOpened():
        pytest.skip("MJPG video writer is unavailable")
    for _ in range(3):
        writer.write(np.zeros((100, 100, 3), dtype=np.uint8))
    writer.release()
    cam_path = tmp_path / "camera.yaml"
    calibration().salvar(cam_path)
    class Detector:
        dispositivo = "cpu"
        def __init__(self, *args, **kwargs):
            pass
        def detectar(self, frame, instant):
            return [Deteccao((10, 20, 30, 50), .9, 0, instant)]
    monkeypatch.setattr(process_video, "DetectorJogadores", Detector)
    trajectory_path = tmp_path / "trajectories.json"
    output_dir = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["process_video.py", "--video", str(path), "--weights", "unused.pt",
                                     "--camera-params", str(cam_path), "--output", str(output_dir),
                                     "--trajectories", str(trajectory_path), "--tracker", "bytetrack.yaml"])
    process_video.main()
    data = json.loads(trajectory_path.read_text())
    assert len(data["trajetorias"]) == 1
    assert [p["instante"] for p in data["trajetorias"][0]["posicoes"]] == [0, .2, .4]
    capture = cv2.VideoCapture(str(output_dir / "input_field.mp4"))
    try:
        assert capture.isOpened()
        assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 3
    finally:
        capture.release()
