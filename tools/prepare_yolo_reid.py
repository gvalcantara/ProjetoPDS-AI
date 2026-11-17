#!/usr/bin/env python3
"""Prepare YOLO detection and player ReID datasets from SoccerTrack MOT clips."""
import argparse
import json
from pathlib import Path

from create_dataset import DEFAULT_SPLITS, create_dataset
from create_reid_dataset import prepare as prepare_reid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mot-root', type=Path, required=True, help='Directory containing <match>.txt and clips/<match>.mp4')
    parser.add_argument('--output', type=Path, required=True, help='Empty output directory; creates yolo/ and reid/')
    parser.add_argument('--interval', type=int, default=36, help='Sample one frame every N frames (default: 36)')
    parser.add_argument('--splits', type=Path, help='JSON with disjoint train, val, test match ID lists')
    parser.add_argument('--one-based', nargs='*', default=[], metavar='MATCH', help='Matches with annotations indexed from 1')
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error('--interval must be positive')
    splits = json.loads(args.splits.read_text(encoding='utf-8')) if args.splits else DEFAULT_SPLITS
    if not isinstance(splits, dict) or set(splits) != {'train', 'val', 'test'} or any(not isinstance(v, list) or not v for v in splits.values()):
        parser.error('--splits must contain nonempty train, val and test lists')
    splits = {name: list(map(str, ids)) for name, ids in splits.items()}
    ids = [m for group in splits.values() for m in group]
    if any(not m.isdigit() for m in ids) or len(ids) != len(set(ids)):
        parser.error('Match IDs must be numeric and unique across all splits')
    if set(args.one_based) - set(ids):
        parser.error('--one-based contains a match outside the selected splits')
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        parser.error('--output must be a new or empty directory')
    for match in ids:
        for path in (args.mot_root / f'{match}.txt', args.mot_root / 'clips' / f'{match}.mp4'):
            if not path.is_file():
                parser.error(f'Missing input: {path}')
    print('Preparing YOLO dataset...', flush=True)
    create_dataset(args.mot_root, args.output / 'yolo', splits, args.interval, {m: 1 for m in args.one_based})
    print('Preparing ReID dataset...', flush=True)
    prepare_reid(args.mot_root, args.output / 'reid', splits, args.interval, set(args.one_based))
    print(f'YOLO: {args.output / "yolo" / "dataset.yaml"}\nReID: {args.output / "reid" / "metadata.csv"}')


if __name__ == '__main__':
    main()
