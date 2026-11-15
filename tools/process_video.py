"""Track players and render a field projection beside the video."""

import argparse
from pathlib import Path
import sys
import json
from dataclasses import asdict

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import (CalibradorCampo, ConstrutorTrajetorias, DetectorJogadores,
                 LeitorVideo, ProjetorCampo, RastreadorJogadores)

PLAYER_COLOR = (0, 255, 255)  # Yellow in OpenCV's BGR format.


def render_field(height, positions, track_ids):
    width = max(320, int(height * 1.2))
    panel = np.full((height, width, 3), (42, 105, 50), dtype=np.uint8)
    margin = max(20, min(height, width) // 12)
    left, right, top, bottom = margin, width - margin, margin, height - margin
    cv2.rectangle(panel, (left, top), (right, bottom), (255, 255, 255), 2)
    mid_x = (left + right) // 2
    cv2.line(panel, (mid_x, top), (mid_x, bottom), (255, 255, 255), 2)
    cv2.circle(panel, (mid_x, (top + bottom) // 2), max(5, int((right - left) * .09)), (255, 255, 255), 2)
    for track_id, (x, y) in zip(track_ids, positions):
        if not (-1 <= x <= 1 and -1 <= y <= 1):
            continue
        px = int(left + (x + 1) * (right - left) / 2)
        py = int(top + (y + 1) * (bottom - top) / 2)
        cv2.circle(panel, (px, py), 7, PLAYER_COLOR, -1)
        cv2.putText(panel, str(track_id), (px + 8, py - 5), cv2.FONT_HERSHEY_SIMPLEX, .5, PLAYER_COLOR, 1, cv2.LINE_AA)
    return panel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--camera-params", type=Path, required=True, help="Camera parameters YAML")
    parser.add_argument("--output", type=Path, default=Path("yolo_testes"), help="Output directory")
    parser.add_argument("--show", action="store_true", help="Display video; press q to stop")
    parser.add_argument("--conf", type=float, default=0.1,
                        help="Detector confidence threshold (default: 0.1, to allow ByteTrack low-score matching)")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--class-id", type=int)
    parser.add_argument("--tracker", default="fasttrack.yaml",
                        help="Ultralytics tracker YAML (default: fasttrack.yaml; use bytetrack.yaml to compare)")
    parser.add_argument("--trajectories", type=Path, help="Optional JSON output containing normalized trajectories")
    args = parser.parse_args()

    output_path = args.output / f"{args.video.stem}_field.mp4"
    inputs = {args.video.resolve(), args.weights.resolve(), args.camera_params.resolve()}
    if output_path.resolve() in inputs:
        raise ValueError("Output video path must be different from input files.")
    if args.trajectories and args.trajectories.resolve() in inputs | {output_path.resolve()}:
        raise ValueError("Trajectory output path must be different from input files and output video.")

    writer = None
    try:
        with LeitorVideo(args.video) as video:
            metadata = video.metadados()
            width, height, fps = metadata.largura, metadata.altura, metadata.fps
            calibration = CalibradorCampo().carregar_calibracao(args.camera_params, (width, height))
            print(f"Camera: {calibration.modelo_camera}; using homography from {args.camera_params}")
            detector = DetectorJogadores(args.weights, confianca=args.conf, tamanho_imagem=args.imgsz,
                                          classe_id=args.class_id)
            tracker = RastreadorJogadores(args.tracker)
            projector = ProjetorCampo(calibration)
            trajectories = ConstrutorTrajetorias()
            args.output.mkdir(parents=True, exist_ok=True)
            panel_width = max(320, int(height * 1.2))
            writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                     (width + panel_width, height))
            if not writer.isOpened():
                raise RuntimeError(f"Could not create output video: {output_path}")
            if args.show:
                cv2.namedWindow("Players and field", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
                cv2.resizeWindow("Players and field", 960, 540)
            frame_count = 0
            outside_count = 0
            for _, instant, frame in video.ler_quadros():
                detections = detector.detectar(frame, instant)
                tracks = tracker.atualizar(detections, frame, instant)
                positions = projector.projetar(tracks)
                for track, position in zip(tracks, positions):
                    trajectories.adicionar(track.id, position)
                    x1, y1, x2, y2 = np.rint(track.deteccao.caixa).astype(int)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), PLAYER_COLOR, 2)
                    outside_count += int(position.fora_do_campo)
                    cv2.putText(frame, f"{track.id}{' outside' if position.fora_do_campo else ''}",
                                (x1, max(20, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, .65, PLAYER_COLOR, 2, cv2.LINE_AA)
                    cv2.circle(frame, ((x1 + x2) // 2, y2), 4, PLAYER_COLOR, -1)
                combined = np.hstack((frame, render_field(height, [(p.x, p.y) for p in positions],
                                                           [track.id for track in tracks])))
                writer.write(combined)
                frame_count += 1
                if args.show:
                    cv2.imshow("Players and field", combined)
                    if cv2.waitKey(1) & 0xff == ord("q"):
                        break
            if frame_count == 0:
                raise RuntimeError("No video frames were read.")
            if args.trajectories:
                args.trajectories.parent.mkdir(parents=True, exist_ok=True)
                payload = {"unidades": "coordenadas normalizadas [-1, 1]; tempo em segundos",
                           "trajetorias": [asdict(t) for t in trajectories.finalizar()]}
                args.trajectories.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"Saved {frame_count} frames to {output_path}; outside-field detections: {outside_count}")
    finally:
        if writer is not None:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
