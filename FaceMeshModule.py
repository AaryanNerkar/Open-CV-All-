import cv2
import mediapipe as mp
import numpy as np
import time


class FaceMeshDetector:
    def __init__(
        self, staticMode=False, maxFaces=2, minDetectionCon=0.5, minTrackCon=0.5
    ):
        # Initialisation
        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.minDetectionCon = minDetectionCon
        self.minTrackCon = minTrackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh

        self.faceMesh = self.mpFaceMesh.FaceMesh(
            static_image_mode=self.staticMode,
            max_num_faces=self.maxFaces,
            min_detection_confidence=self.minDetectionCon,
            min_tracking_confidence=self.minTrackCon,
        )

        self.FACE_CONNECTIONS = self.mpFaceMesh.FACEMESH_TESSELATION
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=1)

    def findFaceMesh(self, frame, draw_True):

        self.imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(self.imgRGB)
        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                self.mpDraw.draw_landmarks(
                    frame, faceLms, self.FACE_CONNECTIONS, self.drawSpec, self.drawSpec
                )

            for id, lm in enumerate(faceLms.landmark):
                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                print(id, cx, cy)

        return frame


def main():
    cap = cv2.VideoCapture(0)
    ptime = 0
    detector = FaceMeshDetector()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = detector.findFaceMesh(frame, draw_True=True)

        frame = cv2.flip(frame, 1)

        ctime = time.time()
        fps = 1 / (ctime - ptime)
        ptime = ctime
        cv2.putText(
            frame, str(int(fps)), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
        )
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
