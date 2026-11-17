#!/usr/bin/env python3
"""Download only the SoccerTrack inputs needed for the GSR baseline."""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dest', required=True, type=Path, help='Dataset download directory')
    p.add_argument('--train', nargs='+', required=True, metavar='ID')
    p.add_argument('--val', nargs='+', required=True, metavar='ID')
    p.add_argument('--test', nargs='+', required=True, metavar='ID')
    p.add_argument('--revision', default='main', help='Hugging Face dataset revision')
    p.add_argument('--dry-run', action='store_true', help='Show files and sizes without downloading')
    args = p.parse_args()
    groups = [set(args.train), set(args.val), set(args.test)]
    if any(not s for s in groups) or any(groups[i] & groups[j] for i in range(3) for j in range(i + 1, 3)):
        p.error('Train, validation and test matches must be non-empty and disjoint')
    matches = sorted(set.union(*groups))
    if not all(m.isdigit() for m in matches):
        p.error('Match IDs must be numeric')
    if not shutil.which('hf'):
        p.error('Install the Hugging Face CLI: pip install -U huggingface_hub')
    patterns = []
    for match in matches:
        patterns += [f'videos/{match}/*', f'raw/{match}/*']
    for match in sorted(groups[2]):
        patterns.append(f'gsr/{match}/*')
    cmd = ['hf', 'download', 'atomscott/soccertrack-v2', '--type', 'dataset', '--revision', args.revision,
           '--local-dir', str(args.dest.resolve()), '--include', *patterns]
    if args.dry_run:
        cmd.append('--dry-run')
    print(' '.join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    print('Dados brutos obtidos. Os arquivos MOT de quatro minutos não são GT para os vídeos completos.')


if __name__ == '__main__':
    try:
        main()
    except subprocess.CalledProcessError as exc:
        sys.exit(exc.returncode)
