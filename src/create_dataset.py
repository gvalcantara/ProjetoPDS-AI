"""Convert SoccerTrack MOT clips to a reproducible YOLO detection dataset."""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import cv2
import yaml

# Official challenge groups; challenge clips serve as validation.
DEFAULT_SPLITS = {
    "train": ["117092", "117093", "118575", "118576", "118577", "128058"],
    "val": ["128057", "132831"],
    "test": ["118578", "132877"],
}


def load_annotations(path, frame_base):
    frames = defaultdict(list)
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            fields = line.strip().split(",")
            try:
                frame = int(fields[0]) - frame_base
                box = tuple(float(value) for value in fields[2:6])
                if len(box) != 4 or frame < 0 or not all(math.isfinite(v) for v in box):
                    raise ValueError("invalid frame or box")
            except (ValueError, IndexError) as error:
                raise ValueError(f"{path}:{line_number}: invalid annotation") from error
            frames[frame].append(box)
    if not frames:
        raise ValueError(f"No annotations in {path}")
    return frames


def create_dataset(mot_root, output, splits=DEFAULT_SPLITS, interval=36, frame_bases=None):
    mot_root, output = Path(mot_root), Path(output)
    frame_bases = frame_bases or {}
    if interval <= 0:
        raise ValueError("Frame interval must be positive")
    if set(splits) != {"train", "val", "test"} or any(not ids for ids in splits.values()):
        raise ValueError("train, val and test must each contain at least one match")
    ids = [match for group in splits.values() for match in group]
    if len(ids) != len(set(ids)):
        raise ValueError("A match cannot appear more than once in the splits")
    if set(frame_bases) - set(ids) or any(base not in (0, 1) for base in frame_bases.values()):
        raise ValueError("Frame-base overrides must refer to selected matches and be 0 or 1")
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output directory must be empty: {output}")
    for match in ids:
        for path in (mot_root / f"{match}.txt", mot_root / "clips" / f"{match}.mp4"):
            if not path.is_file():
                raise FileNotFoundError(path)
    output.mkdir(parents=True, exist_ok=True)
    report = {"frame_interval": interval, "splits": splits, "frame_bases": {m: frame_bases.get(m, 0) for m in ids}, "matches": {}}
    for split, matches in splits.items():
        images, labels = output / "images" / split, output / "labels" / split
        images.mkdir(parents=True)
        labels.mkdir(parents=True)
        for match in matches:
            annotations = load_annotations(mot_root / f"{match}.txt", frame_bases.get(match, 0))
            cap = cv2.VideoCapture(str(mot_root / "clips" / f"{match}.mp4"))
            if not cap.isOpened():
                cap.release()
                raise RuntimeError(f"Cannot open video for {match}")
            saved = boxes = index = 0
            try:
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    if index % interval == 0:
                        if index not in annotations:
                            raise ValueError(f"{match}: missing annotations for frame {index}")
                        height, width = frame.shape[:2]
                        rows = []
                        for x, y, w, h in annotations[index]:
                            if w <= 0 or h <= 0 or x < 0 or y < 0 or x + w > width or y + h > height:
                                raise ValueError(f"{match}: invalid box at frame {index}: {(x, y, w, h)}")
                            rows.append(f"0 {(x + w / 2) / width:.6f} {(y + h / 2) / height:.6f} {w / width:.6f} {h / height:.6f}\n")
                        name = f"{match}_frame_{index:06d}"
                        if not cv2.imwrite(str(images / f"{name}.jpg"), frame):
                            raise OSError(f"Failed to write image for {match}, frame {index}")
                        (labels / f"{name}.txt").write_text("".join(rows), encoding="utf-8")
                        saved += 1
                        boxes += len(rows)
                    index += 1
            finally:
                cap.release()
            if index == 0 or max(annotations) >= index:
                raise ValueError(f"{match}: video is empty or annotations extend beyond its {index} frames")
            report["matches"][match] = {"split": split, "video_frames": index, "images": saved, "boxes": boxes}
            print(f"{match}: {saved} images, {boxes} boxes ({split})")
    (output / "dataset.yaml").write_text(yaml.safe_dump({"path": str(output.resolve()), "train": "images/train", "val": "images/val", "test": "images/test", "names": {0: "player"}}, sort_keys=False), encoding="utf-8")
    (output / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mot-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval", type=int, default=36)
    parser.add_argument("--splits", type=Path, help="JSON mapping train, val and test to lists of match IDs")
    parser.add_argument("--one-based", nargs="*", default=[], metavar="MATCH", help="Matches whose source annotations start at frame 1")
    args = parser.parse_args()
    splits = json.loads(args.splits.read_text(encoding="utf-8")) if args.splits else DEFAULT_SPLITS
    create_dataset(args.mot_root, args.output, splits, args.interval, {match: 1 for match in args.one_based})


if __name__ == "__main__":
    main()
