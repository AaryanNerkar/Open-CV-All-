import cv2
import mediapipe as mp
import numpy as np
import time
import HandTrackingModule as htm
import autopy

#####################333
wCam,HCam=640,480
bbox=0

#######################3
cap=cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)
ptime=0
detector=htm.HandDetector(detectionCon=0.9)


while cap.isOpened():
    #Find the hand lamdmark
    
    #get the tip of index and middle finger
    #check which fingers are up
    #only index finger-Moving Mode
    #Convert Coordinates
    #smoothen values
    #Move Mouse
    #both index and middle fingers up-clicking mode
    #find distance between fingers
    #click mouse if distance short
    
    
    ret,frame=cap.read()
    frame=cv2.flip(frame,1)
    frame=detector.findHands(frame,draw_True=True)
    result = detector.findPosition(frame)
    if result and len(result) == 2:
       lmList, bbox = result
    else:
        lmList, bbox = [], None  # or handle the case accordingly
    if len(lmList) != 0:
        print(lmList)
    if ret==True:
        
        
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