import cv2
import numpy as np

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

kernel = np.ones((5, 5), np.uint8)


COLOR_RANGES = {
    "RED": [
        (np.array([0, 120, 70]),  np.array([10, 255, 255])),
        (np.array([170, 120, 70]), np.array([179, 255, 255])),
    ],
    "GREEN": [
        (np.array([35, 80, 40]),  np.array([85, 255, 255])),
    ],
    "BLUE": [
        (np.array([90, 80, 40]),  np.array([130, 255, 255])),
    ],
    "YELLOW": [
        (np.array([20, 120, 80]), np.array([35, 255, 255])),
    ],
}

def make_mask(hsv, ranges):
    
    mask_total = None
    for (low, up) in ranges:
        m = cv2.inRange(hsv, low, up)
        mask_total = m if mask_total is None else cv2.bitwise_or(mask_total, m)
    return mask_total

while True:
    ret, frame = cap.read()
    if not ret:
        print("Kamera görüntüsü okunamadı.")
        break

    frame = cv2.flip(frame, 1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    best_color = None
    best_area = 0
    best_contour = None
    best_mask = None

    for color_name, ranges in COLOR_RANGES.items():
        mask = make_mask(hsv, ranges)

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)

        if area > best_area:
            best_area = area
            best_color = color_name
            best_contour = c
            best_mask = mask

    
    if best_contour is not None and best_area > 1500:
        x, y, w, h = cv2.boundingRect(best_contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cx, cy = x + w // 2, y + h // 2
        cv2.circle(frame, (cx, cy), 7, (0, 0, 255), -1)

        cv2.putText(frame, f"Color: {best_color}  Area: {int(best_area)}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 255), 2)

        cv2.imshow("Mask", best_mask)
    else:
        cv2.putText(frame, "Color: -",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 255), 2)
        cv2.imshow("Mask", np.zeros(frame.shape[:2], dtype=np.uint8))

    cv2.imshow("Frame", frame)

    if (cv2.waitKey(1) & 0xFF) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
