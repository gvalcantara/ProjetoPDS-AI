import argparse
from pathlib import Path

import cv2
import yaml

FIELD_KEYS = {ord("w"): "top", ord("s"): "bottom", ord("a"): "left",
              ord("d"): "right", ord("c"): "center", ord("p"): "center_spot"}


def validate_field(field):
    required = {"top", "bottom", "left", "right"}
    if not isinstance(field, dict) or not required <= set(field) or set(field) - required - {"center", "center_spot"}:
        raise ValueError("Field YAML must contain top, bottom, left and right, optionally center and center_spot.")
    for name, points in field.items():
        if name == "center_spot":
            if (not isinstance(points, (list, tuple)) or len(points) != 2 or
                    any(type(value) is not int for value in points)):
                raise ValueError("center_spot must be a pair of integers.")
            continue
        if not isinstance(points, list) or len(points) < 2:
            raise ValueError(f"Field line '{name}' must contain at least two points.")
        for point in points:
            if (not isinstance(point, (list, tuple)) or len(point) != 2
                    or any(type(value) is not int for value in point)):
                raise ValueError(f"Points in '{name}' must be pairs of integers.")
    return field


def connect_field_lines(field):
    """Connect adjacent extreme points using their midpoint, regardless of click order."""
    validate_field(field)
    # Canonical order: horizontal lines left to right; vertical lines top to bottom.
    lines = {name: sorted((list(p) for p in points), key=lambda p: p[0 if name in ("top", "bottom") else 1])
             for name, points in field.items() if name != "center_spot"}
    if "center_spot" in field:
        lines["center_spot"] = list(field["center_spot"])
    for horizontal, hi, vertical, vi in (("top", 0, "left", 0),
                                          ("top", -1, "right", 0),
                                          ("bottom", 0, "left", -1),
                                          ("bottom", -1, "right", -1)):
        a, b = lines[horizontal][hi], lines[vertical][vi]
        corner = [round((a[0] + b[0]) / 2), round((a[1] + b[1]) / 2)]
        lines[horizontal][hi] = corner.copy()
        lines[vertical][vi] = corner.copy()
    if "center" in lines:
        # The marked center endpoints define the intersections with the sidelines.
        # Insert them by x so each connected line contains the exact same point.
        for horizontal, endpoint in (("top", lines["center"][0]),
                                      ("bottom", lines["center"][-1])):
            if not lines[horizontal][0][0] < endpoint[0] < lines[horizontal][-1][0]:
                raise ValueError(f"The center endpoint must lie between the corners of {horizontal}.")
            if endpoint not in lines[horizontal]:
                lines[horizontal].append(endpoint.copy())
                lines[horizontal].sort(key=lambda point: point[0])
    return lines


def select_field_contour(video_path, initial_field=None, frame_number=60):
    """Select field points in original video coordinates; frames start at 1."""
    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        # OpenCV indexes frames from zero; frame 60 is index 59.
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f"Could not read frame {frame_number} of the video.")
    finally:
        capture.release()

    window = f"Field contour - frame {frame_number}"
    points = {name: [list(p) for p in (initial_field or {}).get(name, [])]
              for name in FIELD_KEYS.values() if name != "center_spot"}
    spot = (initial_field or {}).get("center_spot")
    points["center_spot"] = [list(spot)] if spot is not None else []
    selected = "top"
    history = []

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            history.append((selected, [point.copy() for point in points[selected]]))
            if selected == "center_spot":
                points[selected].clear()
            points[selected].append([x, y])

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_mouse)
    print("W: superior; S: inferior; A: esquerda; D: direita; C: linha central; P: ponto central.")
    print("Marque ao menos dois pontos por linha, incluindo seus extremos.")
    print("Enter confirma; U desfaz o último ponto; R limpa a linha atual; Esc cancela.")
    try:
        while True:
            preview = frame.copy()
            colors = {"top": (0, 255, 0), "bottom": (0, 255, 255),
                      "left": (255, 0, 0), "right": (0, 0, 255), "center": (255, 0, 255),
                      "center_spot": (255, 255, 255)}
            for name, line_points in points.items():
                for index, point in enumerate(line_points):
                    cv2.circle(preview, tuple(point), 5, colors[name], -1)
                    if index:
                        cv2.line(preview, tuple(line_points[index - 1]), tuple(point), colors[name], 2)
            cv2.putText(preview, f"Selection: {selected} (W/A/S/D/C/P)", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.imshow(window, preview)
            key = cv2.waitKey(20) & 0xFF
            if key in (10, 13):
                if all(len(points[name]) >= 2 for name in FIELD_KEYS.values() if name != "center_spot"):
                    try:
                        field = {name: line for name, line in points.items() if name != "center_spot"}
                        if points["center_spot"]:
                            field["center_spot"] = points["center_spot"][0]
                        points = connect_field_lines(field)
                    except ValueError as exc:
                        print(f"Não foi possível conectar as linhas: {exc}")
                    else:
                        break
                print("Marque pelo menos dois pontos em cada linha.")
            elif key in FIELD_KEYS:
                selected = FIELD_KEYS[key]
                print(f"Linha selecionada: {selected}")
            elif key == ord("u") and history:
                name, previous = history.pop()
                points[name] = previous
            elif key == ord("r"):
                points[selected].clear()
                history[:] = [entry for entry in history if entry[0] != selected]
            elif key == 27:
                raise SystemExit("Seleção do contorno cancelada.")
            try:
                visible = cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE)
            except cv2.error:
                # The Qt backend can destroy the window before this query.
                visible = 0
            if visible < 1:
                raise SystemExit("Seleção do contorno cancelada.")
    finally:
        try:
            cv2.destroyWindow(window)
        except cv2.error:
            # Closing the window via its title bar may already have destroyed it.
            pass

    print(f"Coordenadas do contorno do campo: {points}")
    return points


def main():
    parser = argparse.ArgumentParser(description="Annotate field lines in a video frame.")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Field YAML to create or edit")
    parser.add_argument("--frame", type=int, default=60, help="Frame number, starting at 1")
    parser.add_argument("--edit", action="store_true", help="Load existing annotations from output")
    args = parser.parse_args()
    if args.frame < 1:
        parser.error("--frame must be at least 1")
    field = None
    if args.edit:
        with args.output.open(encoding="utf-8") as file:
            field = validate_field(yaml.safe_load(file))
    result = select_field_contour(args.video, field, args.frame)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        yaml.safe_dump(result, file, sort_keys=False)
    print(f"Field annotations saved to: {args.output}")


if __name__ == "__main__":
    main()
