"""Track players and render a field projection beside the video."""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from tools.field_projection import load_camera_parameters, project_points

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


def player_classes(model, class_id):
    names = model.names
    if class_id is not None:
        if class_id not in names:
            raise ValueError(f"Class ID {class_id} not found: {names}")
        return [class_id]
    ids = [i for i, name in names.items() if str(name).lower() in ("player", "person")]
    if not ids:
        raise ValueError(f"No player/person class found in model {names}; specify --class-id.")
    return ids


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
    args = parser.parse_args()

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {args.video}")
    writer = None
    try:
        width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = capture.get(cv2.CAP_PROP_FPS)
        if width <= 0 or height <= 0 or not np.isfinite(fps) or fps <= 0:
            raise ValueError("Video dimensions or FPS are invalid.")
        camera_model, parameter, homography = load_camera_parameters(args.camera_params, (width, height))
        center = np.array([width / 2, height / 2])
        scale = np.hypot(width, height) / 2
        print(f"Camera: {camera_model}; using homography from {args.camera_params}")
        model = YOLO(str(args.weights))
        classes = player_classes(model, args.class_id)
        device = 0 if torch.cuda.is_available() else "cpu"
        args.output.mkdir(parents=True, exist_ok=True)
        output_path = args.output / f"{args.video.stem}_field.mp4"
        if output_path.resolve() in {args.video.resolve(), args.weights.resolve(), args.camera_params.resolve()}:
            raise ValueError("Output video path must be different from input files.")
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
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            result = model.track(frame, classes=classes, conf=args.conf, imgsz=args.imgsz,
                                 device=device, tracker=args.tracker, persist=True, verbose=False)[0]
            if result.boxes.id is None:
                boxes = np.empty((0, 4))
                track_ids = np.empty(0, dtype=int)
            else:
                boxes = result.boxes.xyxy.cpu().numpy().reshape(-1, 4)
                track_ids = result.boxes.id.int().cpu().numpy()
            feet = np.column_stack(((boxes[:, 0] + boxes[:, 2]) / 2, boxes[:, 3]))
            positions = project_points(feet, center, scale, parameter, homography, camera_model)
            for track_id, box, position in zip(track_ids, boxes, positions):
                color = PLAYER_COLOR
                x1, y1, x2, y2 = np.rint(box).astype(int)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                outside = np.any(np.abs(position) > 1)
                outside_count += int(outside)
                cv2.putText(frame, f"{track_id}{' outside' if outside else ''}", (x1, max(20, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, .65, color, 2, cv2.LINE_AA)
                cv2.circle(frame, ((x1 + x2) // 2, y2), 4, color, -1)
            combined = np.hstack((frame, render_field(height, positions, track_ids)))
            writer.write(combined)
            frame_count += 1
            if args.show:
                cv2.imshow("Players and field", combined)
                if cv2.waitKey(1) & 0xff == ord("q"):
                    break
        if frame_count == 0:
            raise RuntimeError("No video frames were read.")
        print(f"Saved {frame_count} frames to {output_path}; outside-field detections: {outside_count}")
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
