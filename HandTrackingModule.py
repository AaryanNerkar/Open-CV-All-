import cv2
import mediapipe as mp
import time

class HandDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        # Initialization
        #fixed
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )
        self.mpDraw = mp.solutions.drawing_utils
        self.drawSpec = self.mpDraw.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2)
        self.drawSpec= self.mpDraw.DrawingSpec(color=(0, 255, 255), thickness=2)

    def findHands(self, frame, draw_True=True):
        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw_True:
                    self.mpDraw.draw_landmarks(
                        frame, handLms, self.mpHands.HAND_CONNECTIONS, 
                        self.drawSpec, self.drawSpec
                    )
                
        return frame
    def findPosition(self, frame, draw=True):
         
        lmList = []
        if self.results and self.results.multi_hand_landmarks:
           for handLms in self.results.multi_hand_landmarks: 
               for id, lm in enumerate(handLms.landmark):
                   h, w, c = frame.shape
                   cx, cy = int(lm.x * w), int(lm.y * h)
                   lmList.append((id, cx, cy))
                   if draw:  
                      cv2.circle(frame, (cx, cy),2, (255, 0, 0), cv2.FILLED)
        return lmList

def main():
    cap = cv2.VideoCapture(0)
    ptime = 0
    detector = HandDetector()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # Flip first for consistency
        frame = detector.findHands(frame, draw_True=True)
        lmList = detector.findPosition(frame)
        if len(lmList) != 0:
            print(lmList)  # Print the coordinates of the tip of the index finger
        
        # FPS calculation
        ctime = time.time()
        fps = 1 / (ctime - ptime)
        ptime = ctime
        cv2.putText(frame, f'FPS: {int(fps)}', (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Hand Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
