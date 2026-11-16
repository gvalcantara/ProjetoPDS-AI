import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import CalibradorCampo, LeitorVideo
from tools import annotate_field
from src.field_geometry import project_points


@pytest.fixture
def video(tmp_path):
    path = tmp_path / "video.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 5, (64, 48))
    if not writer.isOpened():
        pytest.skip("MJPG video writer is unavailable")
    for value in (30, 120, 220):
        writer.write(np.full((48, 64, 3), value, dtype=np.uint8))
    writer.release()
    return path


@pytest.fixture
def markings():
    return {"top": [[10, 10], [50, 10], [90, 10]],
            "bottom": [[10, 70], [50, 70], [90, 70]],
            "left": [[10, 10], [10, 70]],
            "right": [[90, 10], [90, 70]],
            "center": [[50, 10], [50, 70]], "center_spot": [50, 40]}


def test_video_reader_metadata_random_access_and_iteration(video):
    reader = LeitorVideo(video)
    with reader:
        metadata = reader.metadados()
        assert (metadata.largura, metadata.altura, metadata.total_quadros) == (64, 48, 3)
        assert metadata.fps == 5
        assert metadata.duracao_segundos == pytest.approx(0.6)
        assert reader.ler_quadro(2).mean() == pytest.approx(120, abs=3)
        assert [(n, t) for n, t, _ in reader.ler_quadros()] == [(1, 0), (2, 0.2), (3, 0.4)]
        with pytest.raises(ValueError):
            reader.ler_quadro(0)
        with pytest.raises(ValueError):
            reader.ler_quadro(4)
    with pytest.raises(RuntimeError):
        reader.metadados()


def test_video_reader_missing_file(tmp_path):
    with pytest.raises(ValueError, match="Could not open"):
        LeitorVideo(tmp_path / "missing.avi").abrir()


def test_reader_handles_unknown_frame_count_and_variable_timestamps(monkeypatch):
    class Capture:
        def __init__(self, path): self.index = 0
        def isOpened(self): return True
        def release(self): pass
        def set(self, prop, value): self.index = 0; return True
        def read(self):
            self.index += 1
            return (True, np.zeros((2, 2, 3), dtype=np.uint8)) if self.index <= 3 else (False, None)
        def get(self, prop):
            values = {cv2.CAP_PROP_FRAME_WIDTH: 2, cv2.CAP_PROP_FRAME_HEIGHT: 2,
                      cv2.CAP_PROP_FPS: 25, cv2.CAP_PROP_FRAME_COUNT: -100,
                      cv2.CAP_PROP_POS_MSEC: [0, 0, 171, 243][min(self.index, 3)]}
            return values[prop]
    monkeypatch.setattr(cv2, 'VideoCapture', Capture)
    with LeitorVideo('recording.webm') as reader:
        assert reader.metadados().total_quadros == 0
        assert [t for _, t, _ in reader.ler_quadros()] == pytest.approx([0, .171, .243])


def test_calibration_and_yaml_roundtrip(tmp_path, markings):
    calibrator = CalibradorCampo()
    markings_path = tmp_path / "field.yaml"
    calibrator.salvar_marcacoes(markings_path, markings)
    assert calibrator.carregar_marcacoes(markings_path) == markings
    calibration = calibrator.calibrar(markings, (100, 80), lens_parameter=0)
    points = np.array([[10, 10], [90, 70], [50, 40]])
    projected = project_points(points, calibration.centro_optico, calibration.escala,
                               calibration.parametro_lente, calibration.homografia,
                               calibration.modelo_camera)
    np.testing.assert_allclose(projected, [[-1, -1], [1, 1], [0, 0]], atol=1e-6)
    path = tmp_path / "camera.yaml"
    calibration.salvar(path)
    loaded = calibrator.carregar_calibracao(path, (100, 80))
    np.testing.assert_allclose(loaded.homografia, calibration.homografia)
    assert loaded.erro_rms is None
    with pytest.raises(ValueError, match="dimensions"):
        calibrator.carregar_calibracao(path, (200, 80))


def test_annotation_uses_reader_and_calibrator(monkeypatch, video, markings):
    monkeypatch.setattr(cv2, "namedWindow", lambda *args: None)
    monkeypatch.setattr(cv2, "setMouseCallback", lambda *args: None)
    monkeypatch.setattr(cv2, "imshow", lambda *args: None)
    monkeypatch.setattr(cv2, "waitKey", lambda *args: 13)
    monkeypatch.setattr(cv2, "destroyWindow", lambda *args: None)
    result = annotate_field.select_field_contour(video, markings, 2)
    assert result == markings


@pytest.mark.parametrize("camera_model", ["radial", "fisheye"])
def test_combined_cli_calibrates_existing_annotations(tmp_path, video, markings, monkeypatch, camera_model):
    field_path = tmp_path / "field.yaml"
    camera_path = tmp_path / "camera.yaml"
    CalibradorCampo().salvar_marcacoes(field_path, markings)
    monkeypatch.setattr(sys, "argv", ["annotate_field.py", "--video", str(video),
                                     "--output", str(field_path), "--frame", "1",
                                     "--use-existing", "--camera-model", camera_model,
                                     "--camera-params", str(camera_path)])
    annotate_field.main()
    loaded = CalibradorCampo().carregar_calibracao(camera_path, (64, 48))
    assert loaded.modelo_camera == camera_model
    assert CalibradorCampo().carregar_marcacoes(field_path) == markings


def test_combined_cli_projects_detections(tmp_path, video, markings, monkeypatch):
    from types import SimpleNamespace

    field_path = tmp_path / "field.yaml"
    CalibradorCampo().salvar_marcacoes(field_path, markings)
    class Boxes:
        xyxy = SimpleNamespace(cpu=lambda: SimpleNamespace(numpy=lambda: np.array([[20, 10, 40, 30]])))
    class FakeModel:
        names = {0: "player"}
        def __init__(self, path):
            pass
        def predict(self, frame, **kwargs):
            return [SimpleNamespace(boxes=Boxes())]
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=FakeModel))
    captured = []
    monkeypatch.setattr(annotate_field, "show_comparison", lambda *args: captured.append(args))
    monkeypatch.setattr(sys, "argv", ["annotate_field.py", "--video", str(video),
                                     "--output", str(field_path), "--frame", "1",
                                     "--use-existing", "--model", "weights.pt"])
    annotate_field.main()
    assert len(captured) == 1
    assert captured[0][1].shape == (1, 4)
    assert captured[0][2].shape == (1, 2)


def test_combined_cli_rejects_output_collision(tmp_path, video, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["annotate_field.py", "--video", str(video),
                                     "--output", str(video)])
    with pytest.raises(SystemExit):
        annotate_field.main()
    assert video.exists()
