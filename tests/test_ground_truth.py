import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import CalibracaoCampo, Deteccao, Rastreamento
from src.evaluation import AvaliadorRastreamento
from src.ground_truth import LeitorGSR, ReferenciaJogador
from tools import process_video


def gsr_file(tmp_path, fps=5):
    images = [{"image_id": str(n), "file_name": f"{n:06}.jpg", "width": 64, "height": 48,
               "has_labeled_person": n != 1} for n in range(1, 5)]
    def annotation(image, track, x, role="player"):
        return {"image_id": str(image), "track_id": track, "supercategory": "object",
                "attributes": {"role": role}, "bbox_image": {"x": x, "y": 10, "w": 10, "h": 20}}
    data = {"info": {"frame_rate": fps}, "images": images,
            "annotations": [annotation(3, 7, 20), annotation(2, 7, 10),
                            annotation(2, 9, 30, "goalkeeper"), annotation(2, 10, 0, "referee")]}
    path = tmp_path / "gt.json"
    path.write_text(json.dumps(data))
    return path


def track(id_, box, time=0):
    return Rastreamento(id_, Deteccao(box, .9, 0, time))


def test_gsr_index_filters_interval_roles_and_unlabeled(tmp_path):
    with LeitorGSR(gsr_file(tmp_path), 1, 3, (64, 48), 5) as gt:
        assert gt.quadro(1) is None
        assert gt.quadro(2) == [ReferenciaJogador("7", (10, 10, 20, 30)),
                                 ReferenciaJogador("9", (30, 10, 40, 30))]
        assert gt.quadro(3) == [ReferenciaJogador("7", (20, 10, 30, 30))]
        with pytest.raises(ValueError, match="not indexed"):
            gt.quadro(4)


def test_gsr_validates_alignment_and_cleans_up(tmp_path):
    path = gsr_file(tmp_path)
    for dims, fps, last in [((100, 48), 5, 3), ((64, 48), 25, 3), ((64, 48), 5, 5)]:
        reader = LeitorGSR(path, 1, last, dims, fps)
        with pytest.raises(ValueError):
            with reader:
                pass
        assert reader._db is None
        assert reader._temporary is None


def test_metrics_include_empty_frames_unlabeled_and_id_switches():
    ev = AvaliadorRastreamento(.5)
    ref = ReferenciaJogador("player", (0, 0, 10, 10))
    ev.avaliar(None, [track(1, (0, 0, 10, 10))])
    ev.avaliar([ref], [track(1, (0, 0, 10, 10)), track(3, (20, 20, 30, 30))])
    ev.avaliar([ref], [])
    ev.avaliar([ref], [track(2, (0, 0, 10, 10))])
    ev.avaliar([], [track(8, (0, 0, 10, 10))])
    result = ev.resultado()
    assert (result["quadros_avaliados"], result["quadros_sem_rotulo"]) == (4, 1)
    assert (result["referencias"], result["correspondencias"]) == (3, 2)
    assert (result["falsos_positivos"], result["falsos_negativos"], result["trocas_id"]) == (2, 1, 1)
    assert result["precisao"] == .5
    assert result["revocacao"] == pytest.approx(2 / 3)
    assert result["mota_aproximado"] == pytest.approx(-1 / 3)


def test_process_video_gt_report_and_offset(tmp_path, monkeypatch):
    path = tmp_path / "video.avi"
    video_writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 5, (64, 48))
    if not video_writer.isOpened():
        pytest.skip("MJPG video writer is unavailable")
    for _ in range(2):
        video_writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
    video_writer.release()
    camera = tmp_path / "camera.yaml"
    CalibracaoCampo("radial", (64, 48), np.array([32., 24.]), 40., 0.,
                    np.array([[1/32, 0, -1], [0, 1/24, -1], [0, 0, 1.]])).salvar(camera)
    class Detector:
        def __init__(self, *args, **kwargs): pass
        def detectar(self, frame, instant): return [Deteccao((10, 10, 20, 30), .9, 0, instant)]
    class Tracker:
        def __init__(self, *args): pass
        def atualizar(self, detections, frame, instant): return [Rastreamento(1, detections[0])]
    monkeypatch.setattr(process_video, "DetectorJogadores", Detector)
    monkeypatch.setattr(process_video, "RastreadorJogadores", Tracker)
    report = tmp_path / "report.json"
    monkeypatch.setattr(sys, "argv", ["process_video.py", "--video", str(path), "--weights", "weights.pt",
                                      "--camera-params", str(camera), "--output", str(tmp_path / "out"),
                                      "--gt", str(gsr_file(tmp_path)), "--gt-frame-offset", "1",
                                      "--gt-report", str(report)])
    process_video.main()
    data = json.loads(report.read_text())
    assert data["video_frames_processed"] == 2
    assert data["gt_frame_offset"] == 1
    assert data["metricas"]["correspondencias"] == 1
    assert data["metricas"]["falsos_negativos"] == 2
    assert data["metricas"]["falsos_positivos"] == 1
