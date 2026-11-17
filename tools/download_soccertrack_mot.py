#!/usr/bin/env python3
"""Download only the SoccerTrack v2 MOT annotations and matching clips."""
import argparse
import shutil
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dest', required=True, help='Dataset root; MOT files are saved under mot/')
    parser.add_argument('--match', action='append', default=[], help='Match ID; repeat to restrict the download')
    parser.add_argument('--revision', default='main', help='Dataset revision')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if any(not m.isdigit() for m in args.match):
        parser.error('Match IDs must be numeric')
    if not shutil.which('hf'):
        parser.error('Install the CLI with: pip install -U huggingface_hub')
    patterns = [p for m in args.match for p in (f'mot/{m}.txt', f'mot/clips/{m}.mp4')] if args.match else ['mot/*']
    cmd = ['hf', 'download', 'atomscott/soccertrack-v2', '--type', 'dataset', '--revision', args.revision,
           '--local-dir', args.dest, '--include', *patterns]
    if args.dry_run:
        cmd.append('--dry-run')
    subprocess.run(cmd, check=True)


if __name__ == '__main__':
    main()
