import cv2
import mediapipe as mp
import numpy as np
import time

cap = cv2.VideoCapture(0)
ptime = 0
while cap.isOpened():
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    if ret == True:

        ctime = time.time()
        fps = 1 / (ctime - ptime)
        ptime = ctime
        cv2.putText(
            frame, str(int(fps)), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
        )

        cv2.imshow("frame", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()         
