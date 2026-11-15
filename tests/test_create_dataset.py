import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from src.create_dataset import create_dataset


def fixtures(tmp_path):
    root = tmp_path / "mot"
    (root / "clips").mkdir(parents=True)
    for match in ("a", "b", "c"):
        video = cv2.VideoWriter(str(root / "clips" / f"{match}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
        assert video.isOpened()
        for _ in range(4):
            video.write(np.zeros((48, 64, 3), dtype=np.uint8))
        video.release()
        base = 1 if match == "c" else 0
        (root / f"{match}.txt").write_text("".join(f"{i + base},1,8,4,16,12,-1,-1,-1,-1\n" for i in range(4)))
    return root, {"train": ["a"], "val": ["b"], "test": ["c"]}


def test_generation_and_one_based_frames(tmp_path):
    root, splits = fixtures(tmp_path)
    out = tmp_path / "out"
    report = create_dataset(root, out, splits, 2, {"c": 1})
    assert report["matches"]["c"]["images"] == 2
    assert (out / "labels/test/c_frame_000000.txt").read_text() == "0 0.250000 0.208333 0.250000 0.250000\n"
    assert len(list((out / "images/train").glob("*.jpg"))) == 2
    assert yaml.safe_load((out / "dataset.yaml").read_text())["test"] == "images/test"
    assert json.loads((out / "report.json").read_text())["frame_bases"]["c"] == 1
    with pytest.raises(ValueError, match="empty"):
        create_dataset(root, out, splits)


def test_rejects_invalid_boxes_and_duplicate_split(tmp_path):
    root, splits = fixtures(tmp_path)
    with pytest.raises(ValueError, match="more than once"):
        create_dataset(root, tmp_path / "duplicate", {"train": ["a"], "val": ["a"], "test": ["c"]})
    (root / "a.txt").write_text("0,1,60,0,10,10\n")
    with pytest.raises(ValueError, match="invalid box"):
        create_dataset(root, tmp_path / "bad", splits)
