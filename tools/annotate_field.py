import argparse
from pathlib import Path
import sys

import cv2
import numpy as np

# Support execution as a script and imports from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import CalibradorCampo, LeitorVideo
from src.field_geometry import project_points

FIELD_KEYS = {ord("w"): "top", ord("s"): "bottom", ord("a"): "left",
              ord("d"): "right", ord("c"): "center", ord("p"): "center_spot"}


def select_field_contour(video_path, initial_field=None, frame_number=60):
    """Select field points in original video coordinates; frames start at 1."""
    with LeitorVideo(video_path) as leitor:
        frame = leitor.ler_quadro(frame_number)

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
                        points = CalibradorCampo().conectar_linhas(field)
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



def show_comparison(frame, boxes, positions, frame_number):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Circle

    fig, (image_ax, field_ax) = plt.subplots(1, 2, figsize=(16, 7),
                                              gridspec_kw={"width_ratios": [1.5, 1]})
    image_ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    image_ax.set_title(f"Frame {frame_number}: jogadores detectados")
    image_ax.axis("off")
    field_ax.set_facecolor("#347b46")
    field_ax.add_patch(Rectangle((-1, -1), 2, 2, fill=False, edgecolor="white", linewidth=2))
    field_ax.axvline(0, color="white")
    field_ax.add_patch(Circle((0, 0), .18, fill=False, edgecolor="white"))
    field_ax.set_aspect("equal")
    field_ax.set_xlabel("x normalizado")
    field_ax.set_ylabel("y normalizado")
    field_ax.set_title("Posições no campo")
    # Keep out-of-field detections visible instead of silently clipping them.
    all_positions = np.vstack((np.array([[-1, -1], [1, 1]]), positions))
    field_ax.set_xlim(min(-1.1, all_positions[:, 0].min() - .1), max(1.1, all_positions[:, 0].max() + .1))
    field_ax.set_ylim(max(1.1, all_positions[:, 1].max() + .1), min(-1.1, all_positions[:, 1].min() - .1))
    for index, (box, position) in enumerate(zip(boxes, positions), start=1):
        x1, y1, x2, y2 = box
        color = "yellow"
        image_ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1,
                                      fill=False, edgecolor=color, linewidth=2))
        image_ax.scatter([(x1 + x2) / 2], [y2], color=[color], s=30)
        image_ax.text(x1, y1, str(index), color="white", bbox={"facecolor": color, "edgecolor": "none"})
        field_ax.scatter(*position, color=[color], edgecolor="black", zorder=3)
        field_ax.annotate(str(index), position, xytext=(4, 4), textcoords="offset points", color="white")
    fig.tight_layout()
    plt.show()



def main():
    parser = argparse.ArgumentParser(description="Annotate the field, calibrate the camera and optionally project detected players.")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Field annotation YAML to create or use")
    parser.add_argument("--frame", type=int, default=60, help="Frame number, starting at 1")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--edit", action="store_true", help="Edit annotations from output")
    selection.add_argument("--use-existing", action="store_true", help="Use existing annotations without opening the editor")
    parser.add_argument("--camera-model", choices=("fisheye", "radial"), default="radial",
                        help="Lens model used for calibration (default: radial)")
    parser.add_argument("--camera-params", type=Path, help="Save calibrated camera parameters to YAML")
    parser.add_argument("--model", type=Path, help="YOLO weights; when provided, show the projected detections")
    parser.add_argument("--class-id", type=int, help="Player class ID; defaults to player/person")
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()
    if args.frame < 1:
        parser.error("--frame must be at least 1")
    destinations = [args.output] + ([args.camera_params] if args.camera_params else [])
    input_paths = [args.video] + ([args.model] if args.model else [])
    if any(destination.resolve() == source.resolve() for destination in destinations for source in input_paths):
        parser.error("Output paths must be different from the video and model files")
    if args.camera_params and args.output.resolve() == args.camera_params.resolve():
        parser.error("--camera-params must be different from --output")
    calibrador = CalibradorCampo()
    field = calibrador.carregar_marcacoes(args.output) if args.edit or args.use_existing else None
    if not args.use_existing:
        field = select_field_contour(args.video, field, args.frame)
        calibrador.salvar_marcacoes(args.output, field)
        print(f"Field annotations saved to: {args.output}")
    if not (args.camera_params or args.model):
        return

    with LeitorVideo(args.video) as reader:
        frame = reader.ler_quadro(args.frame)
    height, width = frame.shape[:2]
    calibration = calibrador.calibrar(field, (width, height), modelo_camera=args.camera_model)
    center, scale, parameter, homography = (calibration.centro_optico, calibration.escala,
                                             calibration.parametro_lente, calibration.homografia)
    description = "normalized focal length" if args.camera_model == "fisheye" else "radial coefficient"
    print(f"Camera model: {args.camera_model}; {description}: {parameter:.6f}; "
          f"line RMS pixels (bottom/right/top/left{'/center' if field.get('center') else ''}): {calibration.erro_rms}")
    if field.get("center") is None:
        print("Warning: no center line in YAML; left/right projection is less constrained.")
    else:
        projected_center = project_points(field["center"], center, scale, parameter, homography, args.camera_model)
        print(f"Center line x RMS: {np.sqrt(np.mean(projected_center[:, 0] ** 2)):.4f} (ideal: 0)")
    if field.get("center_spot") is None:
        print("Warning: no center_spot in YAML; the field center is not independently constrained.")
    else:
        print("Center spot projection:", project_points(field["center_spot"], center, scale, parameter, homography, args.camera_model))
    if calibration.atingiu_limite:
        print("Warning: lens fit reached its bound; check the field points and lens model.")
    if args.camera_params:
        calibration.salvar(args.camera_params)
        print(f"Estimated camera parameters saved to: {args.camera_params}")
    if not args.model:
        return

    from ultralytics import YOLO
    model = YOLO(str(args.model))
    if args.class_id is None:
        class_ids = [i for i, name in model.names.items() if str(name).lower() in ("player", "person")]
        if not class_ids:
            raise ValueError("No player/person class found in model; specify --class-id.")
    else:
        if args.class_id not in model.names:
            raise ValueError(f"Class ID {args.class_id} not found: {model.names}")
        class_ids = [args.class_id]
    result = model.predict(frame, classes=class_ids, conf=args.conf, verbose=False)[0]
    boxes = result.boxes.xyxy.cpu().numpy().reshape(-1, 4)
    feet = np.column_stack(((boxes[:, 0] + boxes[:, 2]) / 2, boxes[:, 3]))
    positions = project_points(feet, center, scale, parameter, homography, args.camera_model)
    for index, position in enumerate(positions, start=1):
        print(f"Player {index}: x={position[0]:.4f}, y={position[1]:.4f}")
    show_comparison(frame, boxes, positions, args.frame)


if __name__ == "__main__":
    main()
