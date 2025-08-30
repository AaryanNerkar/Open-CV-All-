import cv2
import numpy as np

cap = cv2.VideoCapture(
    "mixkit-man-and-woman-jogging-together-on-the-street-40881-hd-ready.mp4"
)
ret, frame = cap.read()
x, y, w, h = 280, 60, 271, 790
t = (x, y, w, h)
roi = frame[y : y + h, x : x + w]
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(
    hsv_roi, np.array((0.0, 60.0, 32.0)), np.array((180.0, 255.0, 255.0))
)
roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
tr = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
while cap.isOpened():
    ret, frame = cap.read()
    if ret == True:
        hsv_f = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        d = cv2.calcBackProject([hsv_f], [0], roi_hist, [0, 180], 1)
        # movement of object with video automatically(mean shift)
        ret, tp = cv2.meanShift(d, t, tr)
        x, y, w, h = tp
        final = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 4)
        cv2.imshow("final", final)
        if cv2.waitKey(20) & 0xFF == ord("q"):
            break

    else:
        break

cap.release()
cv2.destroyAllWindows()
