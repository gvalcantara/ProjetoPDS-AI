from pathlib import Path

import cv2
import ijson
import numpy as np
import matplotlib.pyplot as plt


# ============================================================================
# CONFIGURAÇÃO
# ============================================================================

MATCH_ID = "117092"
FRAME_INDEX = 100

ROOT = Path(__file__).resolve().parents[1]

GSR_PATH = ROOT / "data" / "gsr" / MATCH_ID / f"{MATCH_ID}_1st.json"
VIDEO_PATH = (
    ROOT
    / "data"
    / "videos"
    / MATCH_ID
    / f"{MATCH_ID}_panorama_1st_half.mp4"
)

H_PATH = ROOT / "data" / "raw" / MATCH_ID / f"{MATCH_ID}_homography.npy"
MAPX_PATH = ROOT / "data" / "raw" / MATCH_ID / f"{MATCH_ID}_mapx.npy"
MAPY_PATH = ROOT / "data" / "raw" / MATCH_ID / f"{MATCH_ID}_mapy.npy"


# ============================================================================
# GSR
# ============================================================================

def get_image_id_for_frame(gsr_path: Path, frame_index: int) -> str:
    with open(gsr_path, "rb") as f:
        for index, image in enumerate(
            ijson.items(f, "images.item")
        ):
            if index == frame_index:
                return str(image["image_id"])

    raise IndexError(f"Frame {frame_index} não encontrado.")


def get_players_for_frame(
    gsr_path: Path,
    image_id: str,
):
    players = []

    with open(gsr_path, "rb") as f:
        for ann in ijson.items(f, "annotations.item"):

            if str(ann.get("image_id")) != str(image_id):
                continue

            if ann.get("supercategory") != "object":
                continue

            attributes = ann.get("attributes", {})

            role = attributes.get("role")

            if role not in {
                "player",
                "goalkeeper",
                "referee",
            }:
                continue

            bbox_pitch = ann.get("bbox_pitch", {})
            bbox_image = ann.get("bbox_image", {})

            pitch_x = bbox_pitch.get("x_bottom_middle")
            pitch_y = bbox_pitch.get("y_bottom_middle")

            if pitch_x is None or pitch_y is None:
                continue

            x = bbox_image.get("x")
            y = bbox_image.get("y")
            w = bbox_image.get("w")
            h = bbox_image.get("h")

            if None in (x, y, w, h):
                continue

            # Posição dos pés na imagem original
            image_x = float(x) + float(w) / 2.0
            image_y = float(y) + float(h)

            players.append(
                {
                    "track_id": ann.get("track_id"),
                    "role": role,
                    "team": attributes.get("team"),
                    "jersey": attributes.get("jersey"),
                    "player_id": attributes.get("player_id"),
                    "pitch_x": float(pitch_x),
                    "pitch_y": float(pitch_y),
                    "image_x": image_x,
                    "image_y": image_y,
                }
            )

    return players


# ============================================================================
# MAPX / MAPY
# ============================================================================

def load_maps():
    """
    Converte os mapas OpenCV CV_16SC2 + CV_16UC1 para coordenadas float.
    """

    mapx = np.load(MAPX_PATH)
    mapy = np.load(MAPY_PATH)

    print("\nMapa:")
    print("mapx:", mapx.shape, mapx.dtype)
    print("mapy:", mapy.shape, mapy.dtype)

    map_x_float, map_y_float = cv2.convertMaps(
        mapx,
        mapy,
        cv2.CV_32FC1,
    )

    return map_x_float, map_y_float


def sample_map_bilinear(
    map_x,
    map_y,
    points,
):
    """
    Amostra map_x/map_y nas coordenadas fornecidas usando interpolação bilinear.

    points:
        [[x1, y1],
         [x2, y2],
         ...]
    """

    points = np.asarray(points, dtype=np.float64)

    height, width = map_x.shape

    results = []

    for px, py in points:

        px = np.clip(px, 0, width - 1.001)
        py = np.clip(py, 0, height - 1.001)

        x0 = int(np.floor(px))
        y0 = int(np.floor(py))

        x1 = min(x0 + 1, width - 1)
        y1 = min(y0 + 1, height - 1)

        dx = px - x0
        dy = py - y0

        mx00 = map_x[y0, x0]
        mx10 = map_x[y0, x1]
        mx01 = map_x[y1, x0]
        mx11 = map_x[y1, x1]

        my00 = map_y[y0, x0]
        my10 = map_y[y0, x1]
        my01 = map_y[y1, x0]
        my11 = map_y[y1, x1]

        raw_x = (
            (1 - dx) * (1 - dy) * mx00
            + dx * (1 - dy) * mx10
            + (1 - dx) * dy * mx01
            + dx * dy * mx11
        )

        raw_y = (
            (1 - dx) * (1 - dy) * my00
            + dx * (1 - dy) * my10
            + (1 - dx) * dy * my01
            + dx * dy * my11
        )

        results.append([raw_x, raw_y])

    return np.asarray(results, dtype=np.float64)


# ============================================================================
# HOMOGRAFIA
# ============================================================================

def apply_homography(H, pitch_points):
    return cv2.perspectiveTransform(
        pitch_points.reshape(-1, 1, 2).astype(np.float32),
        H,
    ).reshape(-1, 2)


# ============================================================================
# FRAME
# ============================================================================

def load_video_frame(frame_index):
    cap = cv2.VideoCapture(str(VIDEO_PATH))

    if not cap.isOpened():
        raise RuntimeError(
            f"Não foi possível abrir:\n{VIDEO_PATH}"
        )

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

    success, frame = cap.read()

    cap.release()

    if not success:
        raise RuntimeError(
            f"Não foi possível ler o frame {frame_index}."
        )

    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


# ============================================================================
# PLOT
# ============================================================================

def plot_comparison(
    frame,
    players,
    homography_points,
    pitch_to_map_points,
    raw_to_map_points,
):

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(20, 8),
    )

    # ------------------------------------------------------------------------
    # DADOS GSR
    # ------------------------------------------------------------------------

    gsr_points = np.array(
        [
            [p["image_x"], p["image_y"]]
            for p in players
        ],
        dtype=np.float64,
    )

    # ------------------------------------------------------------------------
    # ERROS
    # ------------------------------------------------------------------------

    error_pitch_to_map = np.linalg.norm(
        gsr_points - pitch_to_map_points,
        axis=1,
    )

    error_raw_to_map_vs_H = np.linalg.norm(
        homography_points - raw_to_map_points,
        axis=1,
    )

    # ========================================================================
    # PAINEL 1
    # PITCH -> H -> MAP
    # ========================================================================

    ax = axes[0]

    ax.imshow(frame)

    ax.scatter(
        gsr_points[:, 0],
        gsr_points[:, 1],
        marker="o",
        s=45,
        label="GSR",
    )

    ax.scatter(
        pitch_to_map_points[:, 0],
        pitch_to_map_points[:, 1],
        marker="x",
        s=60,
        label="Pitch → H → Map",
    )

    for i, player in enumerate(players):

        x1, y1 = gsr_points[i]
        x2, y2 = pitch_to_map_points[i]

        ax.plot(
            [x1, x2],
            [y1, y2],
            linewidth=0.8,
        )

        ax.text(
            x1 + 8,
            y1 - 8,
            str(player["track_id"]),
            fontsize=8,
        )

    ax.set_title(
        "Transformação 1\n"
        "Pitch → Homografia → MapX/MapY\n"
        f"Erro médio: {error_pitch_to_map.mean():.2f} px"
    )

    ax.set_xlim(0, frame.shape[1])
    ax.set_ylim(frame.shape[0], 0)

    ax.set_xlabel("X [px]")
    ax.set_ylabel("Y [px]")

    ax.legend()

    # ========================================================================
    # PAINEL 2
    # GSR RAW -> MAP VS H
    # ========================================================================

    ax = axes[1]

    ax.imshow(frame)

    # GSR original
    ax.scatter(
        gsr_points[:, 0],
        gsr_points[:, 1],
        marker="o",
        s=45,
        label="GSR raw",
    )

    # H(pitch)
    ax.scatter(
        homography_points[:, 0],
        homography_points[:, 1],
        marker="x",
        s=60,
        label="Pitch → H",
    )

    # GSR -> map
    ax.scatter(
        raw_to_map_points[:, 0],
        raw_to_map_points[:, 1],
        marker="+",
        s=80,
        label="GSR raw → Map",
    )

    for i, player in enumerate(players):

        x1, y1 = homography_points[i]
        x2, y2 = raw_to_map_points[i]

        ax.plot(
            [x1, x2],
            [y1, y2],
            linewidth=0.8,
        )

        ax.text(
            homography_points[i, 0] + 8,
            homography_points[i, 1] - 8,
            str(player["track_id"]),
            fontsize=8,
        )

    ax.set_title(
        "Transformação 2\n"
        "GSR raw → MapX/MapY  vs  Pitch → H\n"
        f"Erro médio: {error_raw_to_map_vs_H.mean():.2f} px"
    )

    ax.set_xlim(0, frame.shape[1])
    ax.set_ylim(frame.shape[0], 0)

    ax.set_xlabel("X [px]")
    ax.set_ylabel("Y [px]")

    ax.legend()

    # ========================================================================
    # FIGURA
    # ========================================================================

    fig.suptitle(
        f"SoccerTrack v2 — Match {MATCH_ID} — Frame {FRAME_INDEX}",
        fontsize=16,
    )

    plt.tight_layout()

    plt.show()

    # ========================================================================
    # ESTATÍSTICAS
    # ========================================================================

    print("\n")
    print("=" * 75)
    print("COMPARAÇÃO DAS DUAS TRANSFORMAÇÕES")
    print("=" * 75)

    print("\n1) PITCH -> H -> MAP")
    print("-" * 75)
    print(f"mean   : {error_pitch_to_map.mean():.2f} px")
    print(f"median : {np.median(error_pitch_to_map):.2f} px")
    print(f"min    : {error_pitch_to_map.min():.2f} px")
    print(f"max    : {error_pitch_to_map.max():.2f} px")

    print("\n2) GSR RAW -> MAP  VS  H(PITCH)")
    print("-" * 75)
    print(f"mean   : {error_raw_to_map_vs_H.mean():.2f} px")
    print(f"median : {np.median(error_raw_to_map_vs_H):.2f} px")
    print(f"min    : {error_raw_to_map_vs_H.min():.2f} px")
    print(f"max    : {error_raw_to_map_vs_H.max():.2f} px")


# ============================================================================
# MAIN
# ============================================================================

def main():

    print("=" * 75)
    print("SOCCERTRACK V2 - COMPARAÇÃO DAS TRANSFORMAÇÕES")
    print("=" * 75)

    print(f"Match: {MATCH_ID}")
    print(f"Frame: {FRAME_INDEX}")

    # ------------------------------------------------------------------------
    # GSR
    # ------------------------------------------------------------------------

    image_id = get_image_id_for_frame(
        GSR_PATH,
        FRAME_INDEX,
    )

    players = get_players_for_frame(
        GSR_PATH,
        image_id,
    )

    print(f"\nimage_id: {image_id}")
    print(f"Entidades: {len(players)}")

    # ------------------------------------------------------------------------
    # PONTOS
    # ------------------------------------------------------------------------

    pitch_points = np.array(
        [
            [p["pitch_x"], p["pitch_y"]]
            for p in players
        ],
        dtype=np.float32,
    )

    gsr_points = np.array(
        [
            [p["image_x"], p["image_y"]]
            for p in players
        ],
        dtype=np.float64,
    )

    # ------------------------------------------------------------------------
    # HOMOGRAFIA
    # ------------------------------------------------------------------------

    H = np.load(H_PATH)

    print("\nHomografia:")
    print(H)

    homography_points = apply_homography(
        H,
        pitch_points,
    )

    # ------------------------------------------------------------------------
    # MAPAS
    # ------------------------------------------------------------------------

    map_x, map_y = load_maps()

    # ========================================================================
    # TRANSFORMAÇÃO 1
    #
    # pitch -> H -> map
    # ========================================================================

    pitch_to_map_points = sample_map_bilinear(
        map_x,
        map_y,
        homography_points,
    )

    # ========================================================================
    # TRANSFORMAÇÃO 2
    #
    # GSR raw -> map
    # ========================================================================

    raw_to_map_points = sample_map_bilinear(
        map_x,
        map_y,
        gsr_points,
    )

    # ------------------------------------------------------------------------
    # FRAME
    # ------------------------------------------------------------------------

    print("\nCarregando frame do vídeo...")

    frame = load_video_frame(
        FRAME_INDEX,
    )

    print(
        f"Frame: {frame.shape[1]} x {frame.shape[0]}"
    )

    # ------------------------------------------------------------------------
    # PLOT
    # ------------------------------------------------------------------------

    plot_comparison(
        frame,
        players,
        homography_points,
        pitch_to_map_points,
        raw_to_map_points,
    )


if __name__ == "__main__":
    main()
