import cv2

cap = cv2.VideoCapture(
    "mixkit-man-and-woman-jogging-together-on-the-street-40881-hd-ready.mp4"
)
algo1 = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
algo2 = cv2.createBackgroundSubtractorKNN(dist2Threshold=2000, detectShadows=True)
while True:
    ret, frame = cap.read()
    if ret == True:
        frame = cv2.resize(frame, (500, 400))
        r1 = algo1.apply(frame)
        r2 = algo2.apply(frame)
        cv2.imshow("algo1", r1)
        cv2.imshow("algo2", r2)

        cv2.imshow("MY", frame)

        if cv2.waitKey(20) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
