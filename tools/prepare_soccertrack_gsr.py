#!/usr/bin/env python3
"""Prepare released videos and externally generated full-half MOT for the GSR baseline."""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def copy_new(src, dst):
    if dst.exists():
        raise FileExistsError(f'Recusing to overwrite {dst}')
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True, help='Downloaded dataset root')
    p.add_argument('--dest', type=Path, required=True, help='GSR baseline dataset root')
    p.add_argument('--soccertrack', type=Path, required=True, help='Clone of SoccerTrack-v2')
    p.add_argument('--mot-root', type=Path, required=True, help='Full-half calibrated MOT files, not challenge clips')
    p.add_argument('--train', nargs='+', required=True, metavar='ID')
    p.add_argument('--val', nargs='+', required=True, metavar='ID')
    p.add_argument('--test', nargs='+', required=True, metavar='ID')
    args = p.parse_args()
    source, dest, repo, mot_root = (x.resolve() for x in (args.source, args.dest, args.soccertrack, args.mot_root))
    groups = [set(args.train), set(args.val), set(args.test)]
    if any(not s for s in groups) or any(groups[i] & groups[j] for i in range(3) for j in range(i + 1, 3)):
        p.error('Train, validation and test matches must be disjoint')
    matches = sorted(set.union(*groups))
    if not all(m.isdigit() for m in matches): p.error('Match IDs must be numeric')
    if source == dest: p.error('--source and --dest must be different directories')
    if not (repo / 'src/calibration/calibrate_camera_from_mappings.py').is_file():
        p.error('--soccertrack must be a SoccerTrack-v2 checkout')
    required = []
    for m in matches:
        required += [source / 'raw' / m / f'{m}_{suffix}.npy' for suffix in ('mapx', 'mapy', 'homography')]
        required += [source / 'videos' / m / f'{m}_panorama_{h}_half.mp4' for h in ('1st', '2nd')]
        if m in groups[0] | groups[1]:
            required += [mot_root / m / f'{h}_half' / 'gt' / 'gt.txt' for h in ('1st', '2nd')]
        if m in groups[2]:
            required += [source / 'gsr' / m / f'{m}_{h}.json' for h in ('1st', '2nd')]
    missing = [str(x) for x in required if not x.is_file()]
    if missing: raise FileNotFoundError('Missing inputs:\n' + '\n'.join(missing))
    planned = []
    for m in matches:
        planned.append(dest / 'interim/homography' / m / f'{m}_homography.npy')
        planned += [dest / 'interim' / m / f'{m}_calibrated_panorama_{h}_half.mp4' for h in ('1st','2nd')]
        if m in groups[0] | groups[1]: planned += [dest / 'mot' / m / f'{h}_half/gt/gt.txt' for h in ('1st','2nd')]
        if m in groups[2]: planned += [dest / 'gsr' / m / f'{m}_{h}.json' for h in ('1st','2nd')]
    existing = [str(x) for x in planned if x.exists()]
    if existing: raise FileExistsError('Output exists; use a new destination:\n' + '\n'.join(existing))
    # Invoke the official calibration function with explicit mapping paths.
    snippet = ('from src.calibration.calibrate_camera_from_mappings import calibrate_video; '
               'import sys; calibrate_video(*sys.argv[1:])')
    for m in matches:
        raw = source / 'raw' / m
        copy_new(raw / f'{m}_homography.npy', dest / 'interim/homography' / m / f'{m}_homography.npy')
        for h in ('1st', '2nd'):
            out = dest / 'interim' / m / f'{m}_calibrated_panorama_{h}_half.mp4'
            out.parent.mkdir(parents=True, exist_ok=True)
            video = source / 'videos' / m / f'{m}_panorama_{h}_half.mp4'
            print(f'Calibrando {video} -> {out}', flush=True)
            subprocess.run([sys.executable, '-c', snippet, str(video), str(raw / f'{m}_mapx.npy'), str(raw / f'{m}_mapy.npy'), str(out)], cwd=repo, check=True)
            if not out.is_file() or not out.stat().st_size: raise RuntimeError(f'Calibrated video missing or empty: {out}')
            if m in groups[0] | groups[1]:
                copy_new(mot_root / m / f'{h}_half/gt/gt.txt', dest / 'mot' / m / f'{h}_half/gt/gt.txt')
            if m in groups[2]: copy_new(source / 'gsr' / m / f'{m}_{h}.json', dest / 'gsr' / m / f'{m}_{h}.json')
    print(f'Arquivos preparados em {dest}. Confirme geometria e alinhamento temporal dos MOT antes de treinar.')


if __name__ == '__main__': main()
