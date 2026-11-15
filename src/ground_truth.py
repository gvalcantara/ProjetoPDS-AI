"""Stream SoccerTrack GSR annotations into a temporary, video-sized SQLite index."""

from dataclasses import dataclass
import math
from pathlib import Path
import sqlite3
import tempfile

import ijson
from ijson.common import ObjectBuilder


@dataclass(frozen=True)
class ReferenciaJogador:
    id: str
    caixa: tuple[float, float, float, float]


class LeitorGSR:
    """Index only the selected frame interval, with SQLite storage on disk.

    The GSR file is parsed once; no ordering of annotation records is assumed,
    but the images section must precede annotations. Frame numbers are one-based.
    """

    def __init__(self, caminho, primeiro_quadro, ultimo_quadro, dimensoes, fps):
        if primeiro_quadro < 1 or ultimo_quadro < primeiro_quadro:
            raise ValueError("Invalid ground-truth frame interval.")
        self.caminho = Path(caminho)
        self.primeiro = primeiro_quadro
        self.ultimo = ultimo_quadro
        self.dimensoes = tuple(dimensoes)
        self.fps = fps
        self._temporary = None
        self._db = None

    def __enter__(self):
        self._temporary = tempfile.TemporaryDirectory(prefix="pds_gsr_")
        try:
            self._db = sqlite3.connect(str(Path(self._temporary.name) / "gt.sqlite"))
            self._db.executescript('''
                CREATE TABLE frames (image_id TEXT PRIMARY KEY, numero INTEGER UNIQUE NOT NULL, rotulado INTEGER NOT NULL);
                CREATE TABLE anotacoes (image_id TEXT NOT NULL, track_id TEXT NOT NULL,
                                        x1 REAL NOT NULL, y1 REAL NOT NULL, x2 REAL NOT NULL, y2 REAL NOT NULL,
                                        UNIQUE(image_id, track_id));
                CREATE INDEX anotacoes_image_id ON anotacoes(image_id);
            ''')
            self._indexar()
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *args):
        if self._db is not None:
            self._db.close()
            self._db = None
        if self._temporary is not None:
            self._temporary.cleanup()
            self._temporary = None

    def _indexar(self):
        builder = None
        active = None
        depth = 0
        seen_info = seen_images = seen_annotations = False
        with self.caminho.open("rb") as source:
            for prefix, event, value in ijson.parse(source, use_float=True):
                if builder is None:
                    if (prefix, event) in (("info", "start_map"), ("images.item", "start_map"),
                                           ("annotations.item", "start_map")):
                        active = prefix
                        builder = ObjectBuilder()
                        depth = 0
                    elif prefix == "images" and event == "start_array":
                        seen_images = True
                        continue
                    elif prefix == "annotations" and event == "start_array":
                        if not seen_images:
                            raise ValueError("GSR images must precede annotations.")
                        seen_annotations = True
                        continue
                    else:
                        continue
                builder.event(event, value)
                if event in ("start_map", "start_array"):
                    depth += 1
                elif event in ("end_map", "end_array"):
                    depth -= 1
                if depth == 0:
                    item = builder.value
                    if active == "info":
                        self._validar_info(item)
                        seen_info = True
                    elif active == "images.item":
                        self._adicionar_imagem(item)
                    else:
                        self._adicionar_anotacao(item)
                    builder = None
        if not (seen_info and seen_images and seen_annotations):
            raise ValueError("GSR JSON requires info, images and annotations sections.")
        count, minimum, maximum = self._db.execute("SELECT COUNT(*), MIN(numero), MAX(numero) FROM frames").fetchone()
        if count != self.ultimo - self.primeiro + 1 or minimum != self.primeiro or maximum != self.ultimo:
            raise ValueError("GSR does not contain every frame in the requested interval; check --gt-frame-offset.")
        self._db.commit()

    def _validar_info(self, info):
        fps = float(info.get("frame_rate", 0))
        if not math.isfinite(fps) or fps <= 0 or not math.isclose(fps, self.fps, rel_tol=0.001, abs_tol=0.01):
            raise ValueError(f"Video FPS {self.fps} does not match GSR frame rate {fps}.")

    def _adicionar_imagem(self, image):
        try:
            number = int(Path(image["file_name"]).stem)
            if not self.primeiro <= number <= self.ultimo:
                return
            if (int(image["width"]), int(image["height"])) != self.dimensoes:
                raise ValueError("Video dimensions do not match GSR image dimensions.")
            labeled = image.get("has_labeled_person")
            if not isinstance(labeled, bool):
                raise ValueError("GSR image lacks a valid has_labeled_person flag.")
            self._db.execute("INSERT INTO frames VALUES (?, ?, ?)", (str(image["image_id"]), number, int(labeled)))
        except (KeyError, TypeError) as exc:
            raise ValueError("Invalid GSR image record.") from exc

    def _adicionar_anotacao(self, annotation):
        if annotation.get("supercategory") != "object" or annotation.get("attributes", {}).get("role") not in ("player", "goalkeeper"):
            return
        try:
            image_id = str(annotation["image_id"])
            frame = self._db.execute("SELECT rotulado FROM frames WHERE image_id = ?", (image_id,)).fetchone()
            if frame is None or not frame[0]:
                return
            box = annotation["bbox_image"]
            x, y, w, h = (float(box[key]) for key in ("x", "y", "w", "h"))
            if not all(map(math.isfinite, (x, y, w, h))) or w <= 0 or h <= 0:
                raise ValueError("GSR contains an invalid player bounding box.")
            track_id = annotation["track_id"]
            if track_id is None:
                raise ValueError("GSR player is missing track_id.")
            self._db.execute("INSERT INTO anotacoes VALUES (?, ?, ?, ?, ?, ?)",
                             (image_id, str(track_id), x, y, x + w, y + h))
        except (KeyError, TypeError) as exc:
            raise ValueError("Invalid GSR player annotation.") from exc

    def quadro(self, numero):
        """Return None for an unlabelled frame, otherwise a list (possibly empty)."""
        frame = self._db.execute("SELECT image_id, rotulado FROM frames WHERE numero = ?", (numero,)).fetchone()
        if frame is None:
            raise ValueError(f"GSR frame {numero} was not indexed.")
        if not frame[1]:
            return None
        return [ReferenciaJogador(row[0], tuple(row[1:])) for row in self._db.execute(
            "SELECT track_id, x1, y1, x2, y2 FROM anotacoes WHERE image_id = ? ORDER BY track_id", (frame[0],))]
