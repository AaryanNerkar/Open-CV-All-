import cv2
import mediapipe as mp
mp_face_det=mp.solutions.face_detection
mp_draw=mp.solutions.drawing_utils
face_dect=mp_face_det.FaceDetection(min_detection_confidence=1,model_selection=0)

cap=cv2.VideoCapture(0)
while True:
    ret,frame=cap.read()
    frame=cv2.flip(frame,1)
    frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    result=face_dect.process(frame)
    frame=cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
    if ret==True:
        for cr in result.detections:
            mp_draw.draw_detection(frame,cr)
            

        cv2.imshow('frame',frame)
        if cv2.waitKey(10)&0xFF==ord('q'):
            break
    else:
        break


cap.release()
cv2.destroyAllWindows