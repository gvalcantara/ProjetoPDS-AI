#!/usr/bin/env python3
"""Extract per-player image crops from SoccerTrack MOT clips."""
import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import cv2
from create_dataset import DEFAULT_SPLITS


def prepare(mot_root, output, splits, interval, one_based):
    if interval <= 0:
        raise ValueError('interval must be positive')
    if set(splits) != {'train', 'val', 'test'} or any(not v for v in splits.values()):
        raise ValueError('train, val and test must be non-empty')
    ids = [str(m) for group in splits.values() for m in group]
    if len(ids) != len(set(ids)) or set(one_based) - set(ids):
        raise ValueError('Duplicate matches or unknown frame-base override')
    if output.exists() and any(output.iterdir()):
        raise ValueError('Output must be an empty directory')
    for m in ids:
        for path in (mot_root / f'{m}.txt', mot_root / 'clips' / f'{m}.mp4'):
            if not path.is_file(): raise FileNotFoundError(path)
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'metadata.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['path', 'split', 'match', 'player_id', 'identity', 'frame'])
        for split, matches in splits.items():
            for match in map(str, matches):
                frames = defaultdict(list)
                with (mot_root / f'{match}.txt').open(encoding='utf-8') as annotations:
                    for line_number, line in enumerate(annotations, 1):
                        if not line.strip(): continue
                        try:
                            fields = line.strip().split(',')
                            frame = int(fields[0]) - (1 if match in one_based else 0)
                            player = int(fields[1]); x, y, w, h = map(float, fields[2:6])
                            if frame < 0 or player < 0 or not all(map(math.isfinite, (x,y,w,h))) or w <= 0 or h <= 0:
                                raise ValueError('invalid annotation')
                        except (ValueError, IndexError) as exc:
                            raise ValueError(f'{match}:{line_number}: invalid annotation') from exc
                        if frame % interval == 0: frames[frame].append((player,x,y,w,h))
                if not frames: raise ValueError(f'{match}: no sampled annotations')
                video = cv2.VideoCapture(str(mot_root / 'clips' / f'{match}.mp4'))
                if not video.isOpened(): raise RuntimeError(f'Cannot open video {match}')
                index = count = 0
                try:
                    while True:
                        ok, image = video.read()
                        if not ok: break
                        if index % interval == 0:
                            if index not in frames: raise ValueError(f'{match}: missing annotations for frame {index}')
                            height, width = image.shape[:2]
                            seen = set()
                            for player, x, y, w, h in frames[index]:
                                if player in seen: raise ValueError(f'{match}: duplicate ID {player} in frame {index}')
                                seen.add(player)
                                if x < 0 or y < 0 or x + w > width or y + h > height:
                                    raise ValueError(f'{match}: box outside frame {index}')
                                crop = image[math.floor(y):math.ceil(y+h), math.floor(x):math.ceil(x+w)]
                                identity = f'{match}_{player}'
                                relative = Path(split) / identity / f'{index:06d}.jpg'
                                path = output / relative
                                path.parent.mkdir(parents=True, exist_ok=True)
                                ok, encoded = cv2.imencode('.jpg', crop)
                                if not ok: raise OSError(f'Unable to encode {path}')
                                path.write_bytes(encoded.tobytes())
                                writer.writerow([relative.as_posix(), split, match, player, identity, index])
                                count += 1
                        index += 1
                finally:
                    video.release()
                if not index or max(frames) >= index: raise ValueError(f'{match}: annotations exceed video')
                print(f'{match}: {count} crops ({split})')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mot-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--interval', type=int, default=36)
    parser.add_argument('--splits', type=Path)
    parser.add_argument('--one-based', nargs='*', default=[], metavar='MATCH')
    args = parser.parse_args()
    splits = json.loads(args.splits.read_text(encoding='utf-8')) if args.splits else DEFAULT_SPLITS
    prepare(args.mot_root, args.output, splits, args.interval, set(args.one_based))


if __name__ == '__main__': main()
