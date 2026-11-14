"""Estimate lens correction from field lines and project image points."""

import argparse
from pathlib import Path

import cv2
import numpy as np
import yaml


def correct_points(points, center, scale, radial, cameramodel="radial"):
    points = np.asarray(points, dtype=np.float64)
    if points.shape[-1:] != (2,) or not np.isfinite(points).all():
        raise ValueError("Points must be finite and have shape (..., 2).")
    p = (points - center) / scale
    radius = np.linalg.norm(p, axis=-1, keepdims=True)
    if cameramodel == "radial":
        denominator = 1 + radial * radius ** 2
        if np.any(denominator <= 1e-8):
            raise ValueError("Point is outside the valid domain of the radial model.")
        corrected = p / denominator
    elif cameramodel == "fisheye":
        # Equidistant fisheye: image radius = focal length * ray angle.
        # Its inverse pinhole radius is focal length * tan(angle).
        theta = radius / radial
        if np.any(theta >= np.pi / 2 - 1e-4):
            raise ValueError("Point is outside the valid fisheye rectification domain.")
        factor = np.ones_like(theta)
        np.divide(np.tan(theta), theta, out=factor, where=theta != 0)
        corrected = p * factor
    else:
        raise ValueError(f"Unknown camera model: {cameramodel}")
    return center + scale * corrected


def project_points(points, center, scale, radial, homography, cameramodel="radial"):
    corrected = correct_points(points, center, scale, radial, cameramodel)
    shape = corrected.shape
    flat = corrected.reshape(-1, 2)
    homogeneous = np.column_stack((flat, np.ones(len(flat)))) @ homography.T
    if np.any(np.abs(homogeneous[:, 2]) < 1e-10):
        raise ValueError("Point projects to infinity.")
    return (homogeneous[:, :2] / homogeneous[:, 2, None]).reshape(shape)


def _line(points):
    center = points.mean(axis=0)
    _, singular, vh = np.linalg.svd(points - center, full_matrices=False)
    if singular[0] < 1e-10:
        raise ValueError("A field line has coincident points.")
    normal = vh[-1]
    return np.r_[normal, -normal @ center]


def fit_field_projection(lines, image_size, field_corners=None, radial_bounds=None,
                         cameramodel="radial", center_line=None, center_spot=None,
                         lens_parameter=None):
    """Fit projection from four ordered lists of distorted image points.

    image_size is (width, height). The optical center is fixed at the image
    center; radii are normalized by half the image diagonal. The fisheye model
    is equidistant with one fitted focal length, not a full lens calibration.
    """
    if len(lines) != 4:
        raise ValueError("Exactly four ordered field lines are required.")
    groups = [np.asarray(line, dtype=np.float64) for line in lines]
    if any(g.ndim != 2 or g.shape[1] != 2 or len(g) < 2 or
           not np.isfinite(g).all() for g in groups):
        raise ValueError("Each line needs at least two finite (x, y) points.")
    if lens_parameter is None and sum(len(g) - 2 for g in groups) < 1:
        raise ValueError("At least one line needs three points to estimate distortion.")
    if center_line is not None:
        middle = np.asarray(center_line, dtype=np.float64)
        if middle.ndim != 2 or middle.shape[1] != 2 or len(middle) < 2 or not np.isfinite(middle).all():
            raise ValueError("The center line needs at least two finite (x, y) points.")
        groups.append(middle)
    width, height = np.asarray(image_size, dtype=float)
    if not np.isfinite([width, height]).all() or min(width, height) <= 0:
        raise ValueError("Image dimensions must be positive and finite.")
    center = np.array([width / 2, height / 2])
    scale = np.hypot(width, height) / 2
    normalized = [(g - center) / scale for g in groups]
    # These bounds keep the denominator positive and radial mapping monotonic
    # throughout the image and for all supplied points.
    max_r2 = max(1.0, *(np.sum(g * g, axis=1).max() for g in normalized))
    limit = 0.95 / max_r2
    if cameramodel == "radial":
        lower, upper = (-limit, limit) if radial_bounds is None else radial_bounds
        if not (-limit <= lower < upper <= limit):
            raise ValueError(f"radial_bounds must lie within [{-limit}, {limit}].")
    elif cameramodel == "fisheye":
        if radial_bounds is not None:
            raise ValueError("radial_bounds applies only to the radial model.")
        # Cover the full image with rays safely below 90 degrees.
        lower = np.sqrt(max_r2) / 1.45
        upper = max(10.0, 5 * lower)
    else:
        raise ValueError(f"Unknown camera model: {cameramodel}")

    def corrected(group, radial):
        return (correct_points(center + scale * group, center, scale, radial,
                               cameramodel) - center) / scale

    def residual(parameters):
        result = []
        for group in normalized:
            points = corrected(group, parameters[0])
            line = _line(points)
            # Equal weight per line, regardless of how densely it was marked.
            result.extend((points @ line[:2] + line[2]) / np.sqrt(len(points)))
        return result

    def cost(value):
        errors = np.asarray(residual([value]))
        return float(errors @ errors)

    # Scan the interval before refining, avoiding reliance on a single initial
    # guess for this one-dimensional, potentially non-convex fit.
    grid = np.linspace(lower, upper, 257) if lens_parameter is None else np.array([lens_parameter])
    if lens_parameter is not None and (not np.isfinite(lens_parameter) or not lower <= lens_parameter <= upper):
        raise ValueError("Camera parameter is outside the valid range for this image.")
    costs = np.array([cost(value) for value in grid])
    best = int(np.argmin(costs))
    left, right = grid[max(best - 1, 0)], grid[min(best + 1, len(grid) - 1)]
    ratio = (np.sqrt(5) - 1) / 2
    a, b = right - ratio * (right - left), left + ratio * (right - left)
    fa, fb = cost(a), cost(b)
    for _ in range(80):
        if fa < fb:
            right, b, fb = b, a, fa
            a = right - ratio * (right - left)
            fa = cost(a)
        else:
            left, a, fa = a, b, fb
            b = left + ratio * (right - left)
            fb = cost(b)
    candidates = [lower, upper, (left + right) / 2]
    radial = float(min(candidates, key=cost) if lens_parameter is None else lens_parameter)
    fitted_lines = [_line(corrected(g, radial)) for g in normalized]
    corners = []
    for i in range(4):
        intersection = np.cross(fitted_lines[(i - 1) % 4], fitted_lines[i])
        if abs(intersection[2]) < 1e-10:
            raise ValueError("Adjacent field lines are parallel or nearly parallel.")
        corners.append(center + scale * intersection[:2] / intersection[2])
    if field_corners is None:
        field_corners = [[-1, 1], [1, 1], [1, -1], [-1, -1]]
    target = np.asarray(field_corners, dtype=np.float32)
    if target.shape != (4, 2) or not np.isfinite(target).all():
        raise ValueError("field_corners must contain four finite (x, y) points.")
    homography = cv2.getPerspectiveTransform(np.asarray(corners, dtype=np.float32), target)
    if abs(np.linalg.det(homography)) < 1e-15:
        raise ValueError("Degenerate field corners; cannot calculate homography.")
    if center_line is not None:
        # Map the fitted center line to x=0. The four corners and the measured
        # center line need not agree exactly under an approximate lens model;
        # this prioritizes the left/right classification requested by the user.
        line_normalized = fitted_lines[4]
        line_pixels = np.array([line_normalized[0] / scale, line_normalized[1] / scale,
                                line_normalized[2] - line_normalized[:2] @ center / scale])
        line_field = np.linalg.solve(homography.T, line_pixels)
        if abs(line_field[0]) < 1e-10:
            raise ValueError("The center line cannot be mapped to x=0; check the markings.")
        a, b = line_field[1:] / line_field[0]
        homography = np.array([[1, a, b], [0, 1, 0], [0, 0, 1]]) @ homography
    if center_spot is not None:
        if center_line is None:
            raise ValueError("center_spot requires a marked center line.")
        spot = np.asarray(center_spot, dtype=float)
        if spot.shape != (2,) or not np.isfinite(spot).all():
            raise ValueError("center_spot must be a finite (x, y) point.")
        x0, y0 = project_points(spot, center, scale, radial, homography, cameramodel)
        if abs(y0) >= 0.95:
            raise ValueError("The center spot projects too close to or outside a sideline; check the markings and lens model.")
        # Fix the marked spot at the origin while preserving y = +/-1. With
        # imperfect markings, the center-line residual reveals any x mismatch.
        homography = np.array([[1, 0, -x0], [0, 1, -y0], [0, -y0, 1]]) @ homography
    rms = np.array([np.sqrt(np.mean((corrected(g, radial) @ line[:2] + line[2]) ** 2)) * scale
                    for g, line in zip(normalized, fitted_lines)])
    return center, scale, radial, homography, rms, bool(
        min(radial - lower, upper - radial) < 1e-4 * (upper - lower))


def load_field(path):
    with open(path, encoding="utf-8") as file:
        field = yaml.safe_load(file)
    required = {"top", "bottom", "left", "right"}
    if not isinstance(field, dict) or not required <= set(field) or set(field) - required - {"center", "center_spot"}:
        raise ValueError("Field YAML must contain top, bottom, left and right, optionally center and center_spot.")
    # bottom-left, bottom-right, top-right, top-left corners respectively.
    return ([field[name] for name in ("bottom", "right", "top", "left")],
            field.get("center"), field.get("center_spot"))


def save_camera_parameters(path, cameramodel, image_size, center, scale, parameter, homography):
    """Save lens parameters and the image-to-field homography."""
    homography = np.asarray(homography, dtype=float)
    if homography.shape != (3, 3) or not np.isfinite(homography).all() or np.linalg.matrix_rank(homography) < 3:
        raise ValueError("Homography must be a finite, nonsingular 3x3 matrix.")
    data = {
        "camera_model": cameramodel,
        "image_size": {"width": int(image_size[0]), "height": int(image_size[1])},
        "optical_center_pixels": [float(value) for value in center],
        "normalization_scale_pixels": float(scale),
        "homography": homography.tolist(),
    }
    if cameramodel == "fisheye":
        # Equidistant model corresponds to OpenCV fisheye with D = 0;
        # these zeros are fixed assumptions, not estimated coefficients.
        focal = float(parameter * scale)
        data["camera_matrix"] = [[focal, 0.0, float(center[0])],
                                 [0.0, focal, float(center[1])], [0.0, 0.0, 1.0]]
        data["distortion_coefficients"] = [0.0, 0.0, 0.0, 0.0]
        data["distortion_coefficients_status"] = "fixed_not_estimated"
    elif cameramodel == "radial":
        data["distortion_coefficients"] = {"division_k1": float(parameter)}
        data["distortion_model"] = "division_normalized_radius"
    else:
        raise ValueError(f"Unknown camera model: {cameramodel}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(data, file, sort_keys=False)


def load_camera_parameters(path, image_size):
    """Load parameters produced by save_camera_parameters for this resolution."""
    with Path(path).open(encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict):
        raise ValueError("Camera parameters YAML must be a mapping.")
    width, height = image_size
    if data.get("image_size") != {"width": width, "height": height}:
        raise ValueError("Camera YAML image dimensions do not match the video.")
    center = np.asarray(data.get("optical_center_pixels"), dtype=float)
    scale = float(data.get("normalization_scale_pixels", 0))
    if center.shape != (2,) or not np.allclose(center, [width / 2, height / 2]) or not np.isclose(scale, np.hypot(width, height) / 2):
        raise ValueError("Camera YAML center or normalization scale is incompatible with this model.")
    model = data.get("camera_model")
    if model == "radial":
        if data.get("distortion_model") != "division_normalized_radius":
            raise ValueError("Unsupported radial distortion model.")
        parameter = float(data["distortion_coefficients"]["division_k1"])
    elif model == "fisheye":
        matrix = np.asarray(data.get("camera_matrix"), dtype=float)
        coefficients = np.asarray(data.get("distortion_coefficients"), dtype=float)
        if matrix.shape != (3, 3) or coefficients.shape != (4,) or not np.allclose(coefficients, 0):
            raise ValueError("Only the equidistant fisheye model with four zero coefficients is supported.")
        focal = matrix[0, 0]
        expected = np.array([[focal, 0, center[0]], [0, focal, center[1]], [0, 0, 1]])
        if not np.isfinite(matrix).all() or focal <= 0 or not np.allclose(matrix, expected):
            raise ValueError("Fisheye camera matrix is incompatible with the equidistant model.")
        parameter = focal / scale
    else:
        raise ValueError("Camera model must be radial or fisheye.")
    if not np.isfinite(parameter):
        raise ValueError("Camera parameter must be finite.")
    homography = np.asarray(data.get("homography"), dtype=float)
    if homography.shape != (3, 3) or not np.isfinite(homography).all() or np.linalg.matrix_rank(homography) < 3:
        raise ValueError("Camera YAML requires a finite, nonsingular 3x3 homography; regenerate it with field_projection.py.")
    return model, parameter, homography


def read_frame(video, frame_number):
    if frame_number < 1:
        raise ValueError("Frame number must be at least 1 (one-based).")
    capture = cv2.VideoCapture(str(video))
    try:
        if not capture.isOpened():
            raise ValueError(f"Could not open video: {video}")
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
        ok, frame = capture.read()
        if not ok:
            raise ValueError(f"Could not read frame {frame_number} from {video}.")
        return frame
    finally:
        capture.release()


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
    parser = argparse.ArgumentParser(description="Detect players in a video frame and project them onto a normalized field.")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--field", type=Path, required=True, help="YAML with top, bottom, left and right field lines")
    parser.add_argument("--model", type=Path, required=True, help="Ultralytics YOLO weights")
    parser.add_argument("--frame", type=int, required=True, help="Frame number, starting at 1")
    parser.add_argument("--camera-model", choices=("fisheye", "radial"), default="radial",
                        help="Lens model (default: radial)")
    parser.add_argument("--camera-params", type=Path,
                        help="Save camera model and distortion coefficients to YAML (overwrites it)")
    parser.add_argument("--class-id", type=int, help="Player class ID; defaults to a class named player or person")
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()

    if args.camera_params and args.camera_params.resolve() in {
            args.field.resolve(), args.video.resolve(), args.model.resolve()}:
        parser.error("--camera-params must be different from the input files")

    frame = read_frame(args.video, args.frame)
    lines, center_line, center_spot = load_field(args.field)
    height, width = frame.shape[:2]
    center, scale, radial, homography, rms, at_bound = fit_field_projection(
        lines, (width, height), cameramodel=args.camera_model, center_line=center_line,
        center_spot=center_spot)
    parameter = "normalized focal length" if args.camera_model == "fisheye" else "radial coefficient"
    print(f"Camera model: {args.camera_model}; {parameter}: {radial:.6f}; "
          f"line RMS pixels (bottom/right/top/left{'/center' if center_line else ''}): {rms}")
    if center_line is None:
        print("Warning: no center line in YAML; left/right projection is less constrained.")
    else:
        center_positions = project_points(center_line, center, scale, radial, homography, args.camera_model)
        print(f"Center line x RMS: {np.sqrt(np.mean(center_positions[:, 0] ** 2)):.4f} "
              "(ideal: 0)")
    if center_spot is not None:
        print("Center spot projection:", project_points(center_spot, center, scale, radial, homography, args.camera_model))
    else:
        print("Warning: no center_spot in YAML; the field center is not independently constrained.")
    if at_bound:
        print("Warning: lens fit reached its bound; check the field points and lens model.")
    if args.camera_params:
        save_camera_parameters(args.camera_params, args.camera_model, (width, height),
                               center, scale, radial, homography)
        print(f"Estimated camera parameters saved to: {args.camera_params}")

    from ultralytics import YOLO
    model = YOLO(str(args.model))
    names = model.names
    if args.class_id is None:
        matches = [i for i, name in names.items() if str(name).lower() in ("player", "person")]
        if not matches:
            raise ValueError(f"No player/person class found in model {names}; specify --class-id.")
        class_ids = matches
    else:
        if args.class_id not in names:
            raise ValueError(f"Class ID {args.class_id} not found in model: {names}")
        class_ids = [args.class_id]
    result = model.predict(frame, classes=class_ids, conf=args.conf, verbose=False)[0]
    boxes = result.boxes.xyxy.cpu().numpy().reshape(-1, 4)
    feet = np.column_stack(((boxes[:, 0] + boxes[:, 2]) / 2, boxes[:, 3]))
    positions = project_points(feet, center, scale, radial, homography, args.camera_model)
    for index, position in enumerate(positions, start=1):
        print(f"Player {index}: x={position[0]:.4f}, y={position[1]:.4f}")
    show_comparison(frame, boxes, positions, args.frame)


if __name__ == "__main__":
    main()
