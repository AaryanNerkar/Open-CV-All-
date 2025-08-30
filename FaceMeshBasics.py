import cv2
import mediapipe as mp
import numpy as np
import time
mpDraw=mp.solutions.drawing_utils
mpFaceMesh=mp.solutions.face_mesh
faceMesh=mpFaceMesh.FaceMesh()
FACE_CONNECTIONS = mpFaceMesh.FACEMESH_TESSELATION

drawSpec=mpDraw.DrawingSpec(thickness=1,circle_radius=1)

ptime=0
cap=cv2.VideoCapture(0)
while cap.isOpened():
    ret,frame=cap.read()
    frame=cv2.flip(frame,1)
    if ret==True:
        imgRGB=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        results=faceMesh.process(imgRGB)
        if results.multi_face_landmarks:
            for faceLms in results.multi_face_landmarks:
                 mpDraw.draw_landmarks(frame, faceLms, FACE_CONNECTIONS, drawSpec, drawSpec)

            for id,lm in enumerate(faceLms.landmark):
                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                print(id,cx,cy)
                
        ctime=time.time()
        fps=1/(ctime-ptime)
        ptime=ctime
        cv2.putText(frame,str(int(fps)), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (0, 255, 0), 2)
        
        cv2.imshow("frame",frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()