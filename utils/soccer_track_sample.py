import cv2

VIDEO_PATH = "./data/videos/117092_panorama_1st_half.mp4"

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError(f"Não foi possível abrir o vídeo: {VIDEO_PATH}")

ret, frame = cap.read()

if not ret:
    cap.release()
    raise RuntimeError("Não foi possível ler o primeiro frame do vídeo.")

cv2.imshow("Frame", frame)

cv2.waitKey(0)
cv2.destroyAllWindows()
cap.release()