import cv2
import mediapipe as mp
import numpy as np
import time
#Draws connections
mp_drawing=mp.solutions.drawing_utils
mpPose=mp.solutions.pose
pose=mpPose.Pose()
cap=cv2.VideoCapture(0)
ptime=0
ctime=0
while cap.isOpened():
    ret,frame=cap.read()
    frame=cv2.flip(frame,1)
    if ret==True:
        img_rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        results=pose.process(img_rgb)
        print(results.pose_landmarks)
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame,results.pose_landmarks,mpPose.POSE_CONNECTIONS)
            
            #unique id and specific colour for each landmark
            for id,lm in enumerate(results.pose_landmarks.landmark):
                #frame's height, width, and channels (color depth)
                # Needed because MediaPipe gives landmark positions as normalized values (from 0.0 to 1.0).
                h,w,c=frame.shape
                print(id,lm)
                cx,cy=lm.x* w,lm.y*h
                cv2.circle(frame,(int(cx),int(cy)),5,(0,0,255),1,cv2.FILLED)
                           
        
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