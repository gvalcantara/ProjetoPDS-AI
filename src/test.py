import argparse
from pathlib import Path
from datetime import datetime

import cv2
import torch
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(
        description="YOLO object detection on video with bounding boxes."
    )

    parser.add_argument(
        "--weights",
        type=Path,
        required=True,
        help="Path to YOLO model weights (.pt)."
    )

    parser.add_argument(
        "--video",
        type=Path,
        required=True,
        help="Path to input video."
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the video with bounding boxes during inference."
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Minimum detection confidence (default: 0.25)."
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Network input image size (default: 640)."
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("yolo_testes"),
        help="Output directory (default: yolo_testes)."
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Validate input files
    if not args.weights.is_file():
        raise FileNotFoundError(f"Model weights not found: {args.weights}")

    if not args.video.is_file():
        raise FileNotFoundError(f"Video not found: {args.video}")

    # Select inference device
    device = 0 if torch.cuda.is_available() else "cpu"

    print(f"Loading model: {args.weights}")
    print(f"Device: {'GPU' if device == 0 else 'CPU'}")

    model = YOLO(str(args.weights))

    # Generate experiment name
    nome = (
        f"{args.video.stem}"
    )

    # Run inference
    results = model.predict(
        source=str(args.video),
        imgsz=args.imgsz,
        conf=args.conf,
        device=device,
        save=True,
        stream=True,
        show=args.show,
        project=str(args.output),
        name=nome,
        verbose=False,
    )

    try:
        for _ in results:
            pass

    except KeyboardInterrupt:
        print("\nInference interrupted by user.")

    finally:
        if args.show:
            cv2.destroyAllWindows()

    print("\nProcessing finished.")
    print("Results saved in:", args.output / f"{nome}.mp4")


if __name__ == "__main__":
    main()